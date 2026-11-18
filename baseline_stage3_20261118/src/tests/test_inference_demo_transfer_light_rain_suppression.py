from types import SimpleNamespace

import pytest
import torch

from src.tests.inference_demo_transfer_light_rain_suppression import (
    rollout_predict_with_light_rain_suppression,
    suppress_light_rain,
)


class AcceleratorStub:
    @staticmethod
    def unwrap_model(model: object) -> object:
        return model


class ZeroDeltaModel:
    frame_patch_size = 1
    max_frames = 8

    def forward_ar(self, *, target_x: torch.Tensor, **_: object) -> torch.Tensor:
        return torch.zeros_like(target_x[:, -1:])


class ResidualTrainerStub:
    def __init__(self) -> None:
        self.model = ZeroDeltaModel()
        self.accelerator = AcceleratorStub()
        self.train_cfg = SimpleNamespace(strict_target_isolation=False)
        self.rain_c = 1

    @staticmethod
    def _resolve_rain_residual_cfg() -> tuple[bool, bool, float]:
        return True, False, 0.0

    @staticmethod
    def _apply_rain_residual_output(delta: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return reference + delta

    @staticmethod
    def _merge_modalities(
        radar: torch.Tensor,
        satellite: torch.Tensor,
        rain: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([radar, satellite, rain], dim=1)


def test_suppress_light_rain_clears_scales_and_preserves_expected_ranges() -> None:
    rain = torch.tensor([-0.1, 0.0, 0.019, 0.02, 0.05, 0.099, 0.1, 0.5])

    suppressed = suppress_light_rain(
        rain,
        clear_threshold=0.02,
        light_upper=0.1,
        light_scale=0.7,
    )

    expected = torch.tensor([0.0, 0.0, 0.0, 0.014, 0.035, 0.0693, 0.1, 0.5])
    torch.testing.assert_close(suppressed, expected)


@pytest.mark.parametrize(
    ("clear_threshold", "light_upper", "light_scale"),
    [
        (-0.01, 0.1, 0.7),
        (0.1, 0.1, 0.7),
        (0.2, 0.1, 0.7),
        (0.02, 0.1, -0.1),
        (0.02, 0.1, 1.1),
        (float("nan"), 0.1, 0.7),
    ],
)
def test_suppress_light_rain_rejects_invalid_settings(
    clear_threshold: float,
    light_upper: float,
    light_scale: float,
) -> None:
    with pytest.raises(ValueError):
        suppress_light_rain(
            torch.ones(1),
            clear_threshold=clear_threshold,
            light_upper=light_upper,
            light_scale=light_scale,
        )


def test_rollout_feeds_suppressed_rain_back_into_history() -> None:
    trainer = ResidualTrainerStub()
    context = torch.zeros(1, 3, 2, 1, 1)
    context[:, -1, -1] = 0.05
    target = {
        "radar": torch.zeros(1, 1, 3, 1, 1),
        "satellite": torch.zeros(1, 1, 3, 1, 1),
        "rain": torch.zeros(1, 1, 3, 1, 1),
    }

    prediction, model_input_frames = rollout_predict_with_light_rain_suppression(
        trainer,
        context,
        target,
        context_time=None,
        future_time=None,
        history_window_frames=2,
        clear_threshold=0.02,
        light_upper=0.1,
        light_scale=0.5,
    )

    torch.testing.assert_close(
        prediction["rain"].flatten(),
        torch.tensor([0.025, 0.0125, 0.0]),
    )
    torch.testing.assert_close(
        prediction["rain_delta"].flatten(),
        torch.tensor([-0.025, -0.0125, -0.0125]),
    )
    assert model_input_frames == [3, 3, 3]
