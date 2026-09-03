import hashlib
import inspect
from pathlib import Path

import torch
from omegaconf import OmegaConf

import src.gated_modality_rain.model as independent_model_module
from src.gated_modality_rain.model import GatedModalityRainModel, _SpatialModalityGate
from src.gated_modality_rain.trainer import GatedModalityRainTrainer
from src.networks.time_series.causal_patch_transformer_next_frame import RainCausalPatchTransformerNextFrame
from src.tests.time_series.test_spatial_modality_gate import build_baseline_model, build_model
from src.trainer.rain_trainer_ts_next_frame import RainTSNextFrameTrainer

def normalized_git_blob(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()


def test_independent_framework_package_exists() -> None:
    root = Path(__file__).resolve().parents[3]
    assert (root / "src/gated_modality_rain/model.py").is_file()
    assert (root / "src/gated_modality_rain/trainer.py").is_file()


def test_original_baseline_sources_are_unchanged() -> None:
    root = Path(__file__).resolve().parents[3]
    assert normalized_git_blob(root / "src/networks/time_series/causal_patch_transformer_next_frame.py") == (
        "7a60fbf3da6723eb220cb9c793dbd091318b35b9"
    )
    assert normalized_git_blob(root / "src/trainer/rain_trainer_ts_next_frame.py") == (
        "d81d4b13db01c98af7c1a00c881784797b2da485"
    )


def test_independent_classes_own_their_implementations() -> None:
    assert GatedModalityRainModel is not RainCausalPatchTransformerNextFrame
    assert GatedModalityRainTrainer is not RainTSNextFrameTrainer
    assert not issubclass(GatedModalityRainModel, RainCausalPatchTransformerNextFrame)
    assert not issubclass(GatedModalityRainTrainer, RainTSNextFrameTrainer)
    assert _SpatialModalityGate.__module__ == "src.gated_modality_rain.model"
    for name, helper_class in inspect.getmembers(independent_model_module, inspect.isclass):
        if name.startswith("_"):
            assert helper_class.__module__ == "src.gated_modality_rain.model"
    for _name, method in inspect.getmembers(GatedModalityRainTrainer, inspect.isfunction):
        assert method.__module__ == "src.gated_modality_rain.trainer"


def test_disabled_model_and_trainer_loss_gradients_match_baseline() -> None:
    torch.manual_seed(19)
    baseline_model = build_baseline_model().eval()
    independent_model = build_model().eval()
    independent_model.load_state_dict(baseline_model.state_dict(), strict=True)
    x = torch.randn(1, 12, 4, 16, 16)
    cfg = OmegaConf.create(
        {"loss_weights": {"radar": 0.7, "satellite": 1.1, "rain": 1.3}, "strict_target_isolation": True}
    )
    baseline_trainer = object.__new__(RainTSNextFrameTrainer)
    independent_trainer = object.__new__(GatedModalityRainTrainer)
    for trainer in (baseline_trainer, independent_trainer):
        trainer.radar_c, trainer.satellite_c, trainer.rain_c = 1, 10, 1
        trainer.train_cfg = cfg
    baseline_pred = baseline_model(x)
    independent_pred = independent_model(x)
    target = {name: torch.randn_like(value) for name, value in baseline_pred.items()}
    baseline_loss, baseline_logs = baseline_trainer._next_prediction_loss(baseline_pred, target)
    independent_loss, independent_logs = independent_trainer._next_prediction_loss(independent_pred, target)
    torch.testing.assert_close(independent_loss, baseline_loss, rtol=0, atol=0)
    assert independent_logs.keys() == baseline_logs.keys()
    for name in baseline_logs:
        torch.testing.assert_close(independent_logs[name], baseline_logs[name], rtol=0, atol=0)
    baseline_loss.backward()
    independent_loss.backward()
    for name, baseline_parameter in baseline_model.named_parameters():
        independent_gradient = dict(independent_model.named_parameters())[name].grad
        torch.testing.assert_close(independent_gradient, baseline_parameter.grad, rtol=0, atol=0)


def test_disabled_trainer_optimizer_step_matches_baseline() -> None:
    torch.manual_seed(29)
    baseline_model = build_baseline_model().train()
    independent_model = build_model().train()
    independent_model.load_state_dict(baseline_model.state_dict(), strict=True)
    cfg = OmegaConf.create(
        {
            "next_pred": {
                "target_mode": "next_frame",
                "block_size": 1,
                "missing_modality": {"enabled": False},
                "rollout_branch": {"enabled": False},
            },
            "loss_weights": {"radar": 0.9, "satellite": 1.0, "rain": 1.2},
            "strict_target_isolation": True,
            "max_grad_norm": 0.0,
        }
    )
    batch = {
        "radar_past": torch.randn(1, 1, 2, 16, 16),
        "satellite_past": torch.randn(1, 10, 2, 16, 16),
        "rain_past": torch.randn(1, 1, 2, 16, 16),
        "radar_future": torch.randn(1, 1, 1, 16, 16),
        "satellite_future": torch.randn(1, 10, 1, 16, 16),
        "rain_future": torch.randn(1, 1, 1, 16, 16),
    }
    trainers = []
    for trainer_class, model in (
        (RainTSNextFrameTrainer, baseline_model),
        (GatedModalityRainTrainer, independent_model),
    ):
        trainer = object.__new__(trainer_class)
        trainer.device = torch.device("cpu")
        trainer.radar_c, trainer.satellite_c, trainer.rain_c = 1, 10, 1
        trainer.train_cfg = cfg
        trainer.model = model
        trainer.accelerator = _ParityAccelerator()
        trainer.optim = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer.sched = torch.optim.lr_scheduler.LambdaLR(trainer.optim, lr_lambda=lambda _step: 1.0)
        trainer.use_gan = False
        trainer.ema_model = None
        trainer.global_step = 0
        trainers.append(trainer)
    baseline_logs, baseline_did_step = trainers[0].train_step(batch)
    independent_logs, independent_did_step = trainers[1].train_step(batch)
    assert baseline_did_step and independent_did_step
    assert baseline_logs.keys() == independent_logs.keys()
    for name in baseline_logs:
        torch.testing.assert_close(independent_logs[name], baseline_logs[name], rtol=0, atol=0)
    for name, baseline_parameter in baseline_model.named_parameters():
        torch.testing.assert_close(independent_model.state_dict()[name], baseline_parameter, rtol=0, atol=0)


class _ParityAccelerator:
    sync_gradients = True

    @staticmethod
    def autocast() -> object:
        from contextlib import nullcontext

        return nullcontext()

    @staticmethod
    def accumulate(_model: torch.nn.Module) -> object:
        from contextlib import nullcontext

        return nullcontext()

    @staticmethod
    def backward(loss: torch.Tensor) -> None:
        loss.backward()

    @staticmethod
    def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        return model
