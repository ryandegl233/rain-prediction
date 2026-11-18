import inspect
import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from torch import nn

from src.baselines_stage3.contracts import latest_rain, make_inputs, sample_masks
from src.baselines_stage3.evaluation import evaluate, write_report
from src.baselines_stage3.models import Persistence, Stage3Reference
from src.baselines_stage3.runtime import check_data, load_reference, validate_config
from src.tests.baselines_stage3.test_contracts import batch

ROOT = Path(__file__).resolve().parents[3]


def config() -> DictConfig:
    with initialize_config_dir(config_dir=str(ROOT / "src/config/ts_rain_train"), version_base=None):
        return compose(config_name="baseline_stage3_train")


class RecordingNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.frame_patch_size = 1
        self.use_time_embedding = True
        self.use_modality_mask_token = False
        self.contexts: list[torch.Tensor] = []

    def forward_ar(self, context_x: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
        self.contexts.append(context_x.detach().clone())
        return {
            "radar": context_x[:, :1, -1:],
            "satellite": context_x[:, 1:11, -1:],
            "rain": context_x[:, :1, -1:] * self.scale,
        }


def test_final_stage3_settings_and_loss_weights() -> None:
    cfg = config()
    validate_config(cfg)
    assert cfg.train.max_steps == cfg.train.scheduler.num_training_steps == 70000
    assert cfg.train.next_pred.rollout_branch.detach_history
    assert not cfg.train.next_pred.rollout_branch.detach_rain_anchor
    assert cfg.train.next_pred.rollout_branch.use_gt_future_modalities
    assert cfg.val.rollout_use_gt_future_modalities
    assert list(cfg.train.loss_weights.values()) == [1, 1, 2]
    assert cfg.train.next_pred.rollout_branch.weight == 0.5
    assert cfg.dataset.train.batch_size * cfg.accelerator.gradient_accumulation_plugin.num_steps == 2
    assert cfg.dataset.rain_ratio_filter.file_name == "metadata_rain_ratio.parquet"


def test_auxiliary_updates_are_causal_and_future_rain_is_never_read() -> None:
    cfg, data = config(), batch(count=1)
    mask = sample_masks(data["time_past_timestamp"], "satellite", 2025, 0.6)
    network = RecordingNetwork()
    wrapper = Stage3Reference(network, cfg)
    original = wrapper(make_inputs(data, mask, True))
    data["radar_future"][:, :, 1] += 7
    data["rain_future"].fill_(float("nan"))
    changed = wrapper(make_inputs(data, mask, True))
    torch.testing.assert_close(original[:, :, :2], changed[:, :, :2])
    assert not torch.equal(original[:, :, 2:], changed[:, :, 2:])
    assert [x.shape[2] for x in network.contexts[:4]] == [4, 5, 6, 7]


def test_metrics_masks_and_tail_batch_are_batch_size_independent(tmp_path: Path) -> None:
    cfg, data = config(), batch()
    cfg.evaluation.max_batches = -1
    cfg.evaluation.progress_every = 0
    model = Persistence().train()
    single = evaluate({"persistence": model}, [data], cfg)
    split = [{name: value[i:j] for name, value in data.items()} for i, j in ((0, 2), (2, 3))]
    split_report = evaluate({"persistence": model}, split, cfg)
    assert single["results"] == split_report["results"]
    assert single["window_mask_sha256"] == split_report["window_mask_sha256"]
    assert split_report["samples"] == 3 and model.training
    write_report(split_report, tmp_path)
    assert json.loads((tmp_path / "metrics.json").read_text())["future_rain_as_input"] is False


def test_missing_filter_metadata_fails_before_upstream_can_disable_it(tmp_path: Path) -> None:
    cfg = config()
    cfg.dataset.val.inp_dirs = [str(tmp_path)]
    with pytest.raises(FileNotFoundError, match="refusing to silently disable"):
        check_data(cfg, ("val",))


def test_common_evaluation_applies_and_reports_requested_precision() -> None:
    class PrecisionProbe(nn.Module):
        uses_auxiliary_observations = False

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.eye(2))
            self.observed_dtype = None

        def forward(self, inputs: object) -> torch.Tensor:
            self.observed_dtype = (self.weight @ self.weight).dtype
            return Persistence()(inputs)

    cfg, model = config(), PrecisionProbe()
    cfg.evaluation.mixed_precision = "bf16"
    report = evaluate({"probe": model}, [batch()], cfg)
    assert model.observed_dtype == torch.bfloat16
    assert report["results"]["full"]["probe"]["inference_precision"] == "bf16"


def test_original_stage3_training_gradients_rollout_and_checkpoints(tmp_path: Path) -> None:
    from src.baselines_stage3.trainer import Stage3ComparisonTrainer
    from src.trainer.rain_trainer_ts_next_frame_missing_modality import RainCausalPatchTransformerMissingModality

    assert Path(inspect.getfile(RainCausalPatchTransformerMissingModality)).is_relative_to(ROOT)
    cfg = config()
    cfg.train.init_model_path = None
    cfg.train.proj_dir = str(tmp_path)
    cfg.train.max_steps = 3
    cfg.train.save_every = 1
    cfg.train.checkpoint.keep_recent = 1
    cfg.train.checkpoint.archive_every = 100
    cfg.train.log.log_every = 1
    cfg.val.val_duration = 3
    cfg.evaluation.max_batches = -1
    cfg.evaluation.progress_every = 0
    cfg.dataset.rain_ratio_filter.enabled = True
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / cfg.dataset.rain_ratio_filter.file_name).touch()
    cfg.dataset.img_size = 16
    OmegaConf.update(cfg, "accelerator.cpu", True, force_add=True)
    cfg.accelerator.mixed_precision = "no"
    cfg.accelerator.log_with = None
    for split in ("train", "val"):
        cfg.dataset[split] = {
            "_target_": "src.tests.baselines_stage3.fixtures.make_loader",
            "count": 3,
            "batch_size": 2,
            "iter_index_mode": "sequential",
            "shuffle": False,
            "drop_last": False,
            "inp_dirs": [str(data_dir)],
            "rain_ratio_filter_enabled": True,
            "rain_ratio_filter_file_name": cfg.dataset.rain_ratio_filter.file_name,
        }
    for name, value in {
        "input_size": 16,
        "patch_size": 2,
        "stem_channels": 8,
        "dim": 8,
        "depth": 1,
        "num_heads": 2,
        "decoder_base_channels": 8,
        "decoder_k_size": 3,
        "activation_checkpoint": False,
    }.items():
        cfg.rain_prediction_model[name] = value
    trainer = Stage3ComparisonTrainer(cfg)
    assert trainer.cfg.dataset.rain_ratio_filter.enabled
    assert trainer.cfg.dataset.train.rain_ratio_filter_enabled
    assert trainer.cfg.dataset.val.rain_ratio_filter_enabled
    data = next(iter(trainer.val_dataloader))
    trainer.model.eval()
    for scenario in ("full", "satellite"):
        mask = sample_masks(data["time_past_timestamp"], scenario, 2025, 0.6)
        inputs = make_inputs(data, mask, True)
        context = trainer._apply_per_frame_mask_tokens(inputs.history, inputs.available)
        anchor = latest_rain(inputs)[0].detach().requires_grad_()
        options = dict(
            masked_context=context,
            initial_rain_anchor=anchor,
            total_future_frames=4,
            detach_history=True,
            history_window_frames=7,
            context_time=inputs.time,
            future_time=data["time_future"],
            future_modalities={"radar": data["radar_future"], "satellite": data["satellite_future"]},
            use_gt_future_modalities=True,
        )
        original = trainer._rollout_predict_from_masked_context(**options, detach_rain_anchor=False)["rain"]
        torch.testing.assert_close(Stage3Reference(trainer.model, cfg)(inputs), original)
        gradient = torch.autograd.grad(original[:, :, -1].sum(), anchor)[0]
        torch.testing.assert_close(gradient, torch.ones_like(anchor))
        detached = trainer._rollout_predict_from_masked_context(**options, detach_rain_anchor=True)["rain"]
        blocked = torch.autograd.grad(detached[:, :, -1].sum(), anchor, allow_unused=True)[0]
        assert blocked is None or not blocked.any()
    before = trainer.model.patch_embed.weight.detach().clone()
    trainer.run()
    assert trainer.global_step == 3 and not torch.equal(before, trainer.model.patch_embed.weight)
    logged = trainer.last_train_metrics["values"]
    assert logged["loss"] == pytest.approx(logged["loss/rec_teacher_forced"] + logged["loss/rollout_branch"])
    assert logged["loss/rollout_branch"] == pytest.approx(0.5 * logged["loss/rollout_branch_raw"])
    checkpoints = sorted(tmp_path.rglob("step_*/metadata.json"))
    assert len(checkpoints) == 3
    assert all(json.loads(path.read_text())["retention"]["automatic_deletion"] is False for path in checkpoints)
    reports = list(tmp_path.rglob("baseline_validation/step_00000003/metrics.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["samples"] == 3 and report["complete"]
    assert set(report["results"]["sampled"]) == {"reference", "persistence", "lk"}
    cfg.evaluation.reference_checkpoint = str(checkpoints[-1].parent)
    restored = load_reference(cfg)
    trainer.model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs), Stage3Reference(trainer.model, cfg)(inputs))
    cfg.evaluation.checkpoint_kind = "ema"
    restored_ema = load_reference(cfg)
    trainer.ema_model.ema_model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored_ema(inputs), Stage3Reference(trainer.ema_model.ema_model, cfg)(inputs))

    cfg.train.init_model_path = str(tmp_path / "no-longer-present-stage2.safetensors")
    cfg.train.resume_path = str(checkpoints[-1].parent)
    cfg.train.max_steps = 4
    cfg.train.proj_dir = str(tmp_path / "resumed")
    cfg.val.val_duration = 4
    resumed = Stage3ComparisonTrainer(cfg)
    resumed.run()
    assert resumed.global_step == 4
    assert resumed.cfg.train.init_model_path is None
    assert len(list((tmp_path / "resumed").rglob("step_00000004/model.safetensors"))) == 1

    cfg.train.max_steps = 3
    cfg.train.proj_dir = str(tmp_path / "already-complete")
    completed = Stage3ComparisonTrainer(cfg)
    completed.run()
    assert completed.global_step == 3
    assert not list((tmp_path / "already-complete").rglob("model.safetensors"))
