import copy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.rain_priority.evaluation import RainLeadMetrics
from src.rain_priority.trainer import RainPriorityTrainer


def samples() -> tuple[list[dict[str, torch.Tensor]], DataLoader]:
    generator = torch.Generator().manual_seed(17)
    item = {
        f"{m}_{period}": torch.rand(c, 4, 8, 8, generator=generator)
        for m, c in (("radar", 1), ("satellite", 10), ("rain", 1))
        for period in ("past", "future")
    }
    stamps = 1_700_000_000 + torch.arange(8, dtype=torch.float64) * 1800
    item.update(time_past_timestamp=stamps[:4], time_future_timestamp=stamps[4:], aug_time_reversed=torch.tensor(0))
    dataset = [item] * 4
    return dataset, DataLoader(dataset, batch_size=1)


def tiny_config(path: Path) -> DictConfig:
    config_dir = Path(__file__).resolve().parents[3] / "src/config/ts_rain_train"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="rain_priority", overrides=["+accelerator.cpu=true"])
    cfg.accelerator.mixed_precision = "no"
    cfg.accelerator.log_with = None
    cfg.accelerator.gradient_accumulation_plugin.num_steps = 2
    cfg.ema.beta = 0
    cfg.train.proj_dir = str(path)
    cfg.train.log.log_with_time = False
    cfg.train.log.run_comment = ""
    cfg.train.debug = True
    cfg.train.priority.gradient_diagnostics_every = 1
    cfg.train.priority.horizon.warmup_steps = 2
    cfg.train.priority.horizon.update_every = 2
    cfg.train.next_pred.missing_modality.enabled = False
    cfg.dataset.train = {"_target_": f"{__name__}.samples"}
    cfg.dataset.val = {"_target_": f"{__name__}.samples"}
    cfg.rain_prediction_model.update(
        input_size=8,
        patch_size=2,
        stem_channels=8,
        dim=16,
        depth=1,
        num_heads=2,
        decoder_base_channels=8,
        decoder_k_size=3,
        decoder_pad_mode="zeros",
        stem_pad_mode="zeros",
        drop_path=0.0,
        dropout=0.0,
    )
    cfg.val.max_val_iters = 1
    return cfg


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_real_training_accumulation_checkpoint_and_validation(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path / "run")
    trainer = RainPriorityTrainer(cfg)
    batch = next(iter(trainer.train_dataloader))
    before = trainer.model.patch_embed.weight.detach().clone()
    _, first = trainer.train_step(batch)
    assert not first and trainer.global_step == 0 and trainer.controller.step == 0
    logs, second = trainer.train_step(batch)
    assert second and trainer.global_step == 1 and trainer.controller.step == 1
    assert not torch.equal(before, trainer.model.patch_embed.weight)
    assert torch.count_nonzero(trainer.model.spatial_modality_gate.net[-1].weight)
    assert all(torch.isfinite(v).all() for v in logs.values())
    assert logs["gradient/gate_norm_before_clip"] > 0
    assert "horizon/neutral_reference_count" in logs
    trainer._save_checkpoint()
    checkpoint = trainer.proj_dir / "checkpoint-00000001"
    assert (checkpoint / "custom_checkpoint_0.pkl").is_file()

    cfg2 = tiny_config(tmp_path / "resume")
    cfg2.train.resume_path = str(checkpoint)
    resumed = RainPriorityTrainer(cfg2)
    resumed._resume_if_needed()
    assert resumed.global_step == resumed.controller.step == 1
    torch.testing.assert_close(resumed.controller.window_sum, trainer.controller.window_sum, rtol=0, atol=0)
    state = copy.deepcopy(resumed.controller.state_dict())
    resumed._run_val()
    assert resumed.controller.step == state["step"]
    torch.testing.assert_close(resumed.controller.weights, state["weights"], rtol=0, atol=0)
    assert (resumed.proj_dir / "validation" / "step-00000001.json").is_file()
    assert (resumed.proj_dir / "validation" / "step-00000001-raw.png").is_file()
    for _ in range(2):
        trainer.train_step(batch)
        resumed.train_step(batch)
    for key, parameter in trainer.model.state_dict().items():
        torch.testing.assert_close(parameter, resumed.model.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(trainer.controller.reference, resumed.controller.reference, rtol=0, atol=0)
    trainer._close_tensorboard_writer()
    resumed._close_tensorboard_writer()


def test_future_weather_labels_do_not_change_fixed_origin_predictions(tmp_path: Path) -> None:
    trainer = RainPriorityTrainer(tiny_config(tmp_path))
    trainer.model.eval()
    batch = next(iter(trainer.train_dataloader))
    changed = {k: (v + 100 if k.endswith("future") else v) for k, v in batch.items()}
    with torch.no_grad():
        p1, _, _, _ = trainer._val_inference_step(batch)
        p2, _, _, _ = trainer._val_inference_step(changed)
    for name in p1:
        torch.testing.assert_close(p1[name], p2[name], rtol=0, atol=0)
    trainer._close_tensorboard_writer()


def test_per_lead_metrics_aggregate_counts_and_keep_empty_events_explicit() -> None:
    metrics = RainLeadMetrics([0.5])
    pred = torch.tensor([1.0, 1.0, 0.0, 0.0]).reshape(1, 1, 1, 1, 4).expand(1, 1, 4, 1, 4)
    target = torch.tensor([1.0, 0.0, 1.0, 0.0]).reshape(1, 1, 1, 1, 4).expand_as(pred)
    metrics.update(pred, target)
    result = metrics.compute()
    assert result["h1/csi@0.5"] == pytest.approx(1 / 3)
    assert result["h1/far@0.5"] == pytest.approx(0.5)
    assert result["h1/pod@0.5"] == pytest.approx(0.5)
    assert result["h1/mse"] == pytest.approx(0.5)
    assert result["h1/dry_pred_p50_hist"] == pytest.approx(0.0, abs=0.001)
    assert result["h1/dry_pred_p90_hist"] == pytest.approx(1.0, abs=0.001)
    empty = RainLeadMetrics([0.5])
    empty.update(torch.zeros_like(pred), torch.zeros_like(pred))
    assert empty.compute()["h1/csi@0.5"] is None


def test_dry_quantile_histogram_reports_overflow_instead_of_clipping_it_away() -> None:
    metrics = RainLeadMetrics([0.5])
    pred = torch.full((1, 1, 4, 1, 1), 10.0)
    metrics.update(pred, torch.zeros_like(pred))
    result = metrics.compute()
    assert result["h1/dry_pred_p90_hist"] is None
    assert result["h1/dry_hist_overflow_fraction"] == 1


def test_unsafe_or_obsolete_loss_switches_fail_before_loading_data(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path)
    cfg.train.loss.temporal_diff.enabled = True
    with pytest.raises(ValueError, match="temporal_diff"):
        RainPriorityTrainer(cfg)


@pytest.mark.parametrize("override", ["train.strict_target_isolation", "val.rollout_history_detach"])
def test_unsupported_protocol_switches_fail_before_loading_data(tmp_path: Path, override: str) -> None:
    from omegaconf import OmegaConf

    cfg = tiny_config(tmp_path)
    OmegaConf.update(cfg, override, override.startswith("train"))
    with pytest.raises(ValueError, match="protocol"):
        RainPriorityTrainer(cfg)


def test_batch_time_checks_accept_reverse_augmentation_but_reject_gaps() -> None:
    trainer = object.__new__(RainPriorityTrainer)
    trainer.device = torch.device("cpu")
    trainer.accelerator = SimpleNamespace(non_blocking=False)
    _, loader = samples()
    batch = next(iter(loader))
    trainer._device_batch(batch)
    stamps = torch.cat([batch["time_past_timestamp"], batch["time_future_timestamp"]], dim=1).flip(1)
    reversed_batch = {
        **batch,
        "time_past_timestamp": stamps[:, :4],
        "time_future_timestamp": stamps[:, 4:],
        "aug_time_reversed": torch.ones(1, dtype=torch.long),
    }
    trainer._device_batch(reversed_batch)
    for offset in (0, 900, 3600):
        changed = {k: v.clone() for k, v in batch.items()}
        changed["time_past_timestamp"][:, 1] = changed["time_past_timestamp"][:, 0] + offset
        with pytest.raises(ValueError, match="1800"):
            trainer._device_batch(changed)
    with pytest.raises(ValueError, match="timestamp"):
        trainer._device_batch({k: v for k, v in batch.items() if not k.endswith("timestamp")})


def test_cli_resolves_original_decoder_and_new_objective_without_loading_data() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.trainer.rain_priority_trainer", "--cfg", "job", "--resolve"],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(result.stdout)
    assert cfg.rain_prediction_model.decoder_type == "3d"
    assert cfg.train.next_pred.sequence_loss.enabled
    assert cfg.dataset.n_past == cfg.dataset.n_futures == 4
    assert not cfg.train.loss.temporal_diff.enabled
    assert cfg.val.max_val_iters == -1 and not cfg.dataset.val.drop_last


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for BF16 integration")
def test_cuda_bf16_routed_backward_and_checkpoint(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.tests.rain_priority.test_trainer", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    cfg = tiny_config(Path(sys.argv[1]))
    cfg.accelerator.cpu = False
    cfg.accelerator.mixed_precision = "bf16"
    trainer = RainPriorityTrainer(cfg)
    assert trainer.device.type == "cuda"
    batch = next(iter(trainer.train_dataloader))
    before = trainer.model.spatial_modality_gate.net[-1].weight.detach().clone()
    trainer.train_step(batch)
    logs, stepped = trainer.train_step(batch)
    assert stepped and all(torch.isfinite(v).all() for v in logs.values())
    assert not torch.equal(before, trainer.model.spatial_modality_gate.net[-1].weight)
    trainer._save_checkpoint()
    trainer._run_val()
    trainer._close_tensorboard_writer()
