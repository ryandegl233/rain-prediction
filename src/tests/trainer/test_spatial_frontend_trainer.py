from contextlib import nullcontext

import torch
from omegaconf import OmegaConf

from src.networks.spatial_rain_upsample.upsampler import MultimodalSpatialEnhancementFrontend
from src.trainer.spatial_frontend_trainer import SpatialFrontendTrainer


class DummyAccelerator:
    is_main_process = True
    sync_gradients = True

    def accumulate(self, _model):
        return nullcontext()

    def autocast(self):
        return nullcontext()

    @staticmethod
    def backward(loss: torch.Tensor) -> None:
        loss.backward()

    @staticmethod
    def clip_grad_norm_(parameters, max_grad_norm: float) -> None:
        torch.nn.utils.clip_grad_norm_(list(parameters), max_grad_norm)

    @staticmethod
    def unwrap_model(model):
        return model


def _make_trainer(tmp_path) -> SpatialFrontendTrainer:
    trainer = object.__new__(SpatialFrontendTrainer)
    trainer.device = torch.device("cpu")
    trainer.accelerator = DummyAccelerator()
    trainer.frontend = MultimodalSpatialEnhancementFrontend(
        feature_channels=4,
        growth_channels=2,
        dense_blocks=1,
        dense_layers=2,
        shared_depth=1,
        output_size=(16, 16),
    )
    trainer.optim = torch.optim.AdamW(trainer.frontend.parameters(), lr=1.0e-4)
    trainer.sched = torch.optim.lr_scheduler.LambdaLR(trainer.optim, lambda _step: 1.0)
    trainer.global_step = 0
    trainer.proj_dir = tmp_path
    trainer.frontend_cfg = OmegaConf.create(
        {
            "input_size": 8,
            "model": {
                "radar_channels": 1,
                "satellite_channels": 10,
                "rain_channels": 1,
                "output_size": [16, 16],
            },
        }
    )
    trainer.loss_cfg = OmegaConf.create(
        {
            "rain_hr_weight": 1.0,
            "rain_detail_weight": 0.25,
            "degradation_weight": 0.1,
            "residual_weight": 1.0e-4,
        }
    )
    trainer.metric_cfg = OmegaConf.create({"data_range": 1.0})
    trainer.train_cfg = OmegaConf.create({"max_grad_norm": 1.0})
    trainer.cfg = OmegaConf.create({"test": True})
    trainer.log_msg = lambda *_args, **_kwargs: None
    return trainer


def _make_batch() -> dict[str, torch.Tensor]:
    return {
        "radar_past": torch.rand(1, 1, 1, 8, 8),
        "satellite_past": torch.rand(1, 10, 1, 8, 8),
        "rain_past": torch.rand(1, 1, 1, 8, 8),
        "radar_past_hr": torch.rand(1, 1, 1, 16, 16),
        "satellite_past_hr": torch.rand(1, 10, 1, 16, 16),
        "rain_past_hr": torch.rand(1, 1, 1, 16, 16),
    }


def test_frontend_trainer_one_step_smoke(tmp_path) -> None:
    trainer = _make_trainer(tmp_path)

    logs, did_step = trainer.train_step(_make_batch())

    assert did_step
    assert trainer.global_step == 1
    assert torch.isfinite(logs["loss/frontend_total"])
    assert torch.isfinite(logs["frontend/rain_enhanced_hr_psnr"])
    assert torch.isfinite(logs["frontend/rain_base_hr_psnr"])
    assert torch.isfinite(logs["frontend/rain_psnr_gain"])
    assert torch.isfinite(logs["frontend/rain_gate_mean"])


def test_frontend_checkpoint_contains_required_state(tmp_path) -> None:
    trainer = _make_trainer(tmp_path)
    trainer.global_step = 7

    trainer._save_checkpoint()
    ckpt_path = tmp_path / "checkpoint-00000007" / "frontend.pt"

    state = torch.load(ckpt_path, map_location="cpu")
    assert {"frontend", "optimizer", "scheduler", "global_step", "config"}.issubset(state.keys())
    assert state["global_step"] == 7

    reloaded = MultimodalSpatialEnhancementFrontend(
        feature_channels=4,
        growth_channels=2,
        dense_blocks=1,
        dense_layers=2,
        shared_depth=1,
        output_size=(16, 16),
    )
    reloaded.load_state_dict(state["frontend"], strict=True)
