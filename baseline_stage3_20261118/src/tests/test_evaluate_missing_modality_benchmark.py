from contextlib import nullcontext
import sys
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import pytest
except ModuleNotFoundError:
    pytest = None

import torch
import torch.nn as nn

from src.tests.evaluate_missing_modality_benchmark import (
    MODALITY_COMBINATIONS,
    apply_historical_modality_mask,
    evaluate_combinations_on_batches,
    rollout_with_modality_combination,
)


class _DummyAutocastAccelerator:
    @staticmethod
    def unwrap_model(model: nn.Module) -> nn.Module:
        return model

    @staticmethod
    def autocast():
        return nullcontext()


class _DummyCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frame_patch_size = 1
        self.radar_mask_token = nn.Parameter(torch.full((1, 1, 1, 1, 1), 0.11))
        self.satellite_mask_token = nn.Parameter(torch.full((1, 2, 1, 1, 1), -0.22))
        self.rain_mask_token = nn.Parameter(torch.full((1, 1, 1, 1, 1), 0.33))

    def forward_ar(
        self,
        context_x: torch.Tensor,
        target_x: torch.Tensor,
        predict_frames: int = 1,
        strict_target_isolation: bool = False,
        return_modality_dict: bool = True,
        context_time: torch.Tensor | None = None,
        target_time: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del strict_target_isolation, return_modality_dict, context_time, target_time
        last_frame = target_x[:, :, -predict_frames:]
        radar = last_frame[:, :1] * 0.9
        satellite = last_frame[:, 1:3] * 0.95
        rain_delta = last_frame[:, :1] * 0.05
        return {"radar": radar, "satellite": satellite, "rain": rain_delta}


def _make_trainer_stub(residual_enabled: bool = True) -> SimpleNamespace:
    model = _DummyCausalModel()
    train_cfg = SimpleNamespace(
        strict_target_isolation=False,
        loss={"rain_residual": {"enabled": residual_enabled, "clamp_output": False, "min_value": 0.0}},
    )

    def _split_modalities(x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"radar": x[:, :1], "satellite": x[:, 1:3], "rain": x[:, 3:4]}

    def _merge_modalities(radar: torch.Tensor, satellite: torch.Tensor, rain: torch.Tensor) -> torch.Tensor:
        return torch.cat([radar, satellite, rain], dim=1)

    def _resolve_rain_residual_cfg() -> tuple[bool, bool, float]:
        return residual_enabled, False, 0.0

    def _apply_rain_residual_output(pred_delta: torch.Tensor, rain_ref: torch.Tensor) -> torch.Tensor:
        return rain_ref + pred_delta

    def _prepare_val_inference_batch(
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        context = torch.cat([batch["radar_past"], batch["satellite_past"], batch["rain_past"]], dim=1)
        target = {
            "radar": batch["radar_future"],
            "satellite": batch["satellite_future"],
            "rain": batch["rain_future"],
        }
        return context, target, batch["time_past"], batch["time_future"]

    return SimpleNamespace(
        model=model,
        accelerator=_DummyAutocastAccelerator(),
        device=torch.device("cpu"),
        radar_c=1,
        satellite_c=2,
        rain_c=1,
        train_cfg=train_cfg,
        _resolve_frame_patch_size=lambda: 1,
        _resolve_rain_residual_cfg=_resolve_rain_residual_cfg,
        _apply_rain_residual_output=_apply_rain_residual_output,
        _split_modalities=_split_modalities,
        _merge_modalities=_merge_modalities,
        _prepare_val_inference_batch=_prepare_val_inference_batch,
        _denormalize_rain_for_metrics=lambda rain: rain,
    )


def test_apply_historical_modality_mask_keeps_zero_rain_anchor_under_mask_token() -> None:
    context = torch.ones((2, 4, 3, 4, 4), dtype=torch.float32)
    radar_token = torch.full((1, 1, 1, 1, 1), 0.25)
    sat_token = torch.full((1, 2, 1, 1, 1), -0.5)
    rain_token = torch.full((1, 1, 1, 1, 1), 0.17)

    masked, rain_anchor = apply_historical_modality_mask(
        context,
        available=(True, False, False),
        radar_channels=1,
        satellite_channels=2,
        rain_channels=1,
        mask_mode="mask_token",
        radar_mask_token=radar_token,
        satellite_mask_token=sat_token,
        rain_mask_token=rain_token,
    )

    assert torch.allclose(masked[:, :1], torch.ones_like(masked[:, :1]))
    assert torch.allclose(masked[:, 1:3], torch.full_like(masked[:, 1:3], -0.5))
    assert torch.allclose(masked[:, 3:4], torch.full_like(masked[:, 3:4], 0.17))
    assert torch.count_nonzero(rain_anchor).item() == 0


def test_rollout_with_modality_combination_supports_all_seven_combinations() -> None:
    trainer = _make_trainer_stub(residual_enabled=True)
    batch = {
        "radar_past": torch.full((2, 1, 4, 8, 8), 0.4),
        "satellite_past": torch.full((2, 2, 4, 8, 8), 0.5),
        "rain_past": torch.full((2, 1, 4, 8, 8), 0.3),
        "radar_future": torch.full((2, 1, 4, 8, 8), 0.4),
        "satellite_future": torch.full((2, 2, 4, 8, 8), 0.5),
        "rain_future": torch.full((2, 1, 4, 8, 8), 0.35),
        "time_past": torch.linspace(0.1, 0.4, 4).unsqueeze(0).repeat(2, 1),
        "time_future": torch.linspace(0.5, 0.8, 4).unsqueeze(0).repeat(2, 1),
    }

    summary = evaluate_combinations_on_batches(
        trainer,
        [batch],
        combination_names=list(MODALITY_COMBINATIONS.keys()),
        thresholds=[0.1, 0.3, 0.5],
        use_gt_future_modalities=False,
        mask_mode="mask_token",
        include_persistence=True,
    )

    assert set(summary.keys()) == {*MODALITY_COMBINATIONS.keys(), "persistence"}
    for combo_name in MODALITY_COMBINATIONS:
        overall = summary[combo_name]["overall"]
        assert "threshold@0.1" in overall
        assert set(overall["threshold@0.1"].keys()) >= {"csi", "pod", "far", "hss"}
        assert len(summary[combo_name]["by_lead"]) == 4

    context, target, context_time, target_time = trainer._prepare_val_inference_batch(batch)
    pred_no_rain = rollout_with_modality_combination(
        trainer,
        context,
        target,
        available=MODALITY_COMBINATIONS["rad_sat"],
        context_time=context_time,
        future_time=target_time,
        use_gt_future_modalities=False,
        mask_mode="mask_token",
    )
    assert pred_no_rain["rain"].shape == (2, 1, 4, 8, 8)
    assert torch.allclose(pred_no_rain["rain"][:, :, 0], torch.full((2, 1, 8, 8), 0.02))


def test_apply_historical_modality_mask_rejects_invalid_inputs() -> None:
    try:
        apply_historical_modality_mask(
            torch.zeros((1, 3, 2, 4, 4)),
            available=(True, True, True),
            radar_channels=1,
            satellite_channels=1,
            rain_channels=1,
            mask_mode="invalid",
        )
    except ValueError as exc:
        assert "mask_mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid mask_mode")


def test_apply_historical_modality_mask_anchors_first_frame_and_carries_forward_rain() -> None:
    context = torch.ones((2, 4, 4, 4, 4), dtype=torch.float32)
    context[:, 3:4, 0] = 0.42
    context[:, 3:4, 1:] = 0.88
    radar_token = torch.full((1, 1, 1, 1, 1), 0.25)
    sat_token = torch.full((1, 2, 1, 1, 1), -0.5)
    rain_token = torch.full((1, 1, 1, 1, 1), 0.17)

    masked, rain_anchor = apply_historical_modality_mask(
        context,
        available=(True, False, False),
        radar_channels=1,
        satellite_channels=2,
        rain_channels=1,
        mask_mode="mask_token",
        radar_mask_token=radar_token,
        satellite_mask_token=sat_token,
        rain_mask_token=rain_token,
        anchor_first_frame=True,
    )

    assert torch.allclose(masked[:, 1:3, :1], torch.ones_like(masked[:, 1:3, :1]))
    assert torch.allclose(masked[:, 1:3, 1:], torch.full_like(masked[:, 1:3, 1:], -0.5))
    assert torch.allclose(masked[:, 3:4, :1], torch.full_like(masked[:, 3:4, :1], 0.42))
    assert torch.allclose(masked[:, 3:4, 1:], torch.full_like(masked[:, 3:4, 1:], 0.17))
    assert torch.allclose(rain_anchor, torch.full_like(rain_anchor, 0.42))


if __name__ == "__main__":
    test_apply_historical_modality_mask_keeps_zero_rain_anchor_under_mask_token()
    test_apply_historical_modality_mask_anchors_first_frame_and_carries_forward_rain()
    test_rollout_with_modality_combination_supports_all_seven_combinations()
    test_apply_historical_modality_mask_rejects_invalid_inputs()
    print("All tests passed.")
