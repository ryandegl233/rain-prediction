import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from torch import nn

from src.baselines_stage3.contracts import make_inputs, sample_masks
from src.baselines_stage3.deep.common import DeepForecaster, objective_terms
from src.baselines_stage3.deep.training import DeepTrainer, load_deep
from src.tests.baselines_stage3.test_contracts import batch

ROOT = Path(__file__).resolve().parents[3]
METHODS = ("convlstm", "predrnn_v2", "simvp", "earthformer", "alphapre", "npm")
TINY = {
    "convlstm": {"num_hidden": [8, 8], "patch_size": 2, "filter_size": 3},
    "predrnn_v2": {"num_hidden": [8, 8], "patch_size": 2, "filter_size": 3},
    "simvp": {"hid_S": 8, "hid_T": 16, "N_S": 2, "N_T": 2},
    "earthformer": {"base_units": 16, "stem_dims": [4, 8, 16], "stem_scales": [2, 2, 2]},
    "alphapre": {"hidden_dim": 8, "spec_num": 4},
    "npm": {
        "hidden": 4,
        "spatial_depth": 2,
        "temporal_depth": 2,
        "translator_width": 2,
        "translator_downsamples": 2,
        "translator_residual_blocks": 1,
    },
}


def deep_config(method: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(ROOT / "src/config/ts_rain_train"), version_base=None):
        cfg = compose(config_name="baseline_deep_train", overrides=[f"baseline_model={method}"])
    cfg.dataset.img_size = 16
    for name, value in TINY[method].items():
        cfg.baseline_model.options[name] = value
    cfg.deep.device = "cpu"
    cfg.accelerator.mixed_precision = "no"
    cfg.evaluation.device = "cpu"
    cfg.evaluation.max_batches = -1
    cfg.evaluation.progress_every = 0
    cfg.train.max_steps = 2
    cfg.train.save_every = 1
    cfg.train.optim.lr = 1e-3
    cfg.train.scheduler.num_warmup_steps = 0
    cfg.train.log.log_every = 1
    cfg.val.val_duration = 2
    cfg.dataset.rain_ratio_filter.enabled = False
    cfg.baseline.methods = []
    for split in ("train", "val"):
        cfg.dataset[split] = {
            "_target_": "src.tests.baselines_stage3.fixtures.make_loader",
            "count": 2,
            "batch_size": 2,
            "iter_index_mode": "sequential",
            "shuffle": False,
            "drop_last": False,
        }
    return cfg


@pytest.mark.parametrize("method", METHODS)
def test_actual_baseline_training_resume_ema_and_common_evaluation(method: str, tmp_path: Path) -> None:
    torch.set_num_threads(1)
    cfg = deep_config(method)
    trainer = DeepTrainer(cfg, tmp_path / "first")
    before = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    trainer.run(stop_after=1)
    assert trainer.step == 1 and all(torch.isfinite(torch.tensor(v)) for v in trainer.last_logs.values())
    assert any(not torch.equal(value, before[key]) for key, value in trainer.model.state_dict().items())
    checkpoint = tmp_path / "first/checkpoints/step_00000001"
    cfg.train.resume_path = str(checkpoint)
    if method == "simvp":
        changed = OmegaConf.merge(cfg)
        changed.train.scheduler.num_warmup_steps = 1
        with pytest.raises(ValueError, match="signature differs"):
            DeepTrainer(changed, tmp_path / "wrong-schedule")
        changed = OmegaConf.merge(cfg)
        changed.evaluation.reference_checkpoint = str(checkpoint)
        changed.train.loss.rain_residual.enabled = False
        with pytest.raises(ValueError, match="residual output convention"):
            load_deep(changed)
    resumed = DeepTrainer(cfg, tmp_path / "resumed")
    assert resumed.step == 1 and resumed.ema.step == trainer.ema.step
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[key], value, rtol=0, atol=0)
    resumed.run()
    assert resumed.step == 2
    assert (checkpoint / "metadata.json").is_file()
    report_path = tmp_path / "resumed/baseline_validation/step_00000002/metrics.json"
    report = json.loads(report_path.read_text())
    assert report["complete"] and report["samples"] == 2
    assert report["protocol"] == (
        "fixed_origin_satellite_only" if method == "npm" else "progressively_observed_radar_satellite"
    )
    cfg.evaluation.reference_checkpoint = str(tmp_path / "resumed/checkpoints/step_00000002")
    data = batch(count=1)
    scenario = "satellite_only" if method == "npm" else "sampled"
    inputs = make_inputs(data, sample_masks(data["time_past_timestamp"], scenario, 2025, 0.6), method != "npm")
    for kind, original in (("model", resumed.model), ("ema", resumed.ema.ema_model)):
        cfg.evaluation.checkpoint_kind = kind
        restored = load_deep(cfg)
        original.eval()
        for name, value in original.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(restored(inputs), original(inputs), rtol=1e-5, atol=1e-5)
    cfg.train.resume_path = cfg.evaluation.reference_checkpoint
    completed = DeepTrainer(cfg, tmp_path / "complete")
    completed.run()
    assert completed.step == 2 and not (tmp_path / "complete/checkpoints").exists()


def test_npm_ignores_all_rain_radar_and_future_observations() -> None:
    cfg = deep_config("npm")
    model = DeepForecaster(cfg).eval()
    data = batch(count=1)
    mask = sample_masks(data["time_past_timestamp"], "satellite_only", 2025, 0.6)
    with torch.no_grad():
        first = model(make_inputs(data, mask, False))
        for name in ("radar_past", "rain_past", "radar_future", "rain_future", "satellite_future"):
            data.pop(name)
        second = model(make_inputs(data, mask, False))
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_deep_rollout_gt_arrival_and_rain_anchor_gradients() -> None:
    class Probe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.1))

        def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, dict]:
            rain = history[:, :1, -1:] * self.scale
            return torch.cat([history[:, :11, -1:], rain], dim=1), {}

    cfg, data = deep_config("simvp"), batch(count=1)
    model = DeepForecaster(cfg, Probe())
    mask = sample_masks(data["time_past_timestamp"], "full", 2025, 0.6)
    inputs = make_inputs(data, mask, True)
    original = model(inputs)
    data["radar_future"][:, :, 1] += 10
    data["rain_future"].fill_(float("nan"))
    modified = model(make_inputs(data, mask, True))
    torch.testing.assert_close(original[:, :, :2], modified[:, :, :2])
    assert not torch.equal(original[:, :, 2:], modified[:, :, 2:])
    history = inputs.history.clone().requires_grad_()
    result = model.rollout(history, mask, inputs.future_aux, training=True)
    gradient = torch.autograd.grad(result[:, :, -1].sum(), history)[0]
    torch.testing.assert_close(gradient[:, 11:12, -1], torch.ones_like(gradient[:, 11:12, -1]))


def test_shared_loss_coefficients_match_stage3_sequence_and_rollout() -> None:
    class Zero(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value = nn.Parameter(torch.tensor(0.0))
            self.targets = []

        def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, dict]:
            return history[:, :, -1:] * 0 + self.value, {}

        def supervised_losses(self, auxiliary: dict, target: torch.Tensor, step: int) -> dict:
            self.targets.append(target.detach().clone())
            return {}

    cfg, data = deep_config("simvp"), batch(count=1)
    cfg.deep.training_scenario = "full"
    for name, value in data.items():
        if name.startswith(("radar_", "satellite_", "rain_")):
            value.fill_(1)
    model = DeepForecaster(cfg, Zero())
    values = list(objective_terms(model, data, 0))
    sequence = sum(logs.get("loss/sequence", 0) for _, logs in values)
    assert sequence == pytest.approx(2.0)
    assert values[-1][1]["loss/rollout"] == pytest.approx(0)
    assert len(model.core.targets) == 7
    assert all(torch.equal(target[:, 11:12], torch.zeros_like(target[:, 11:12])) for target in model.core.targets)


def test_alphapre_rain_amplitude_has_native_residual_domain() -> None:
    cfg = deep_config("alphapre")
    core = DeepForecaster(cfg).core
    history = torch.randn(1, 12, 7, 16, 16)
    raw = core.network.amplinet(history.transpose(1, 2))
    prediction, auxiliary = core(history)
    amplitude = auxiliary["amplitude_prediction"]
    torch.testing.assert_close(amplitude[:, :, :11], raw[:, :, :11].sigmoid())
    torch.testing.assert_close(amplitude[:, :, 11:12], raw[:, :, 11:12])
    assert torch.isfinite(sum(core.supervised_losses(auxiliary, -torch.ones_like(prediction), 0).values()))


def test_unsupported_training_precision_and_objectives_fail_explicitly(tmp_path: Path) -> None:
    cfg = deep_config("simvp")
    cfg.accelerator.mixed_precision = "fp16"
    with pytest.raises(ValueError, match="no or bf16"):
        DeepTrainer(cfg, tmp_path)
    cfg.accelerator.mixed_precision = "no"
    cfg.train.next_pred.rollout_branch.enabled = False
    with pytest.raises(ValueError, match="branches enabled"):
        DeepTrainer(cfg, tmp_path)
