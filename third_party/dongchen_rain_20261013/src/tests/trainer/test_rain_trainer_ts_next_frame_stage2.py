import pytest
import torch
from omegaconf import OmegaConf

from src.trainer.rain_trainer_ts_next_frame_stage2 import RainTSNextFrameStage2Trainer


class DummyARModel:
    def forward_ar(
        self,
        target_x: torch.Tensor,
        context_x: torch.Tensor | None,
        predict_frames: int,
        strict_target_isolation: bool,
        return_modality_dict: bool,
    ) -> dict[str, torch.Tensor]:
        _ = context_x, predict_frames, strict_target_isolation, return_modality_dict
        out = target_x + 1.0
        return {
            "radar": out[:, :1],
            "satellite": out[:, 1:11],
            "rain": out[:, 11:12],
        }


def _make_stage2_trainer(
    roll_n: int = 1,
    block_size: int = 2,
    detach_history: bool = True,
) -> RainTSNextFrameStage2Trainer:
    trainer = object.__new__(RainTSNextFrameStage2Trainer)
    trainer.device = torch.device("cpu")
    trainer.radar_c = 1
    trainer.satellite_c = 10
    trainer.rain_c = 1
    trainer.model = DummyARModel()
    trainer.train_cfg = OmegaConf.create(
        {
            "strict_target_isolation": True,
            "loss_weights": {"radar": 1.0, "satellite": 1.0, "rain": 1.0},
            "next_pred": {"target_mode": "block", "block_size": block_size},
        }
    )
    trainer.stage2_block_size = block_size
    trainer.stage2_roll_n = roll_n
    trainer.stage2_detach_history = detach_history
    return trainer


def _make_context_and_target() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    context = torch.zeros(1, 12, 2, 4, 4)
    radar = torch.zeros(1, 1, 3, 4, 4)
    satellite = torch.zeros(1, 10, 3, 4, 4)
    rain = torch.zeros(1, 1, 3, 4, 4)

    radar[:, :, 2] = 3.0
    satellite[:, :, 2] = 3.0
    rain[:, :, 2] = 3.0
    target = {
        "radar": radar,
        "satellite": satellite,
        "rain": rain,
    }
    return context, target


def test_stage2_future_requirement_check() -> None:
    trainer = _make_stage2_trainer(roll_n=2, block_size=2)
    with pytest.raises(ValueError):
        trainer._check_stage2_future_requirement(4)
    trainer._check_stage2_future_requirement(5)


def test_compute_roll_next_loss_zero_when_target_matches_dummy_roll() -> None:
    trainer = _make_stage2_trainer(roll_n=1, block_size=2, detach_history=True)
    context, target = _make_context_and_target()

    loss_roll_next, pred_next, target_next = trainer._compute_roll_next_loss(context=context, target=target)

    assert torch.isfinite(loss_roll_next)
    assert float(loss_roll_next.item()) == pytest.approx(0.0, abs=1e-6)
    assert pred_next["rain"].shape == (1, 1, 1, 4, 4)
    assert target_next["rain"].shape == (1, 1, 1, 4, 4)


def test_compute_roll_next_loss_runs_without_detach() -> None:
    trainer = _make_stage2_trainer(roll_n=1, block_size=2, detach_history=False)
    context, target = _make_context_and_target()

    loss_roll_next, pred_next, target_next = trainer._compute_roll_next_loss(context=context, target=target)

    assert torch.isfinite(loss_roll_next)
    assert pred_next["radar"].shape == target_next["radar"].shape
