import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.gated_modality_rain.model import GatedModalityRainModel
from src.tools.gated_sichuan_case import CaseRuntime, build_case_times, find_case_window, save_case_figures


def test_case_times_cross_midnight_and_reject_off_grid() -> None:
    past, future = build_case_times("2025-08-10 23:30:00", 4, 12, 30)
    assert past[0] == datetime(2025, 8, 10, 21, 30)
    assert past[-1] == datetime(2025, 8, 10, 23)
    assert future[-1] == datetime(2025, 8, 11, 5)
    with pytest.raises(ValueError, match="interval"):
        build_case_times("2025-08-10 23:35:00", 4, 12, 30)


def test_case_window_requires_all_timestamps() -> None:
    past, future = build_case_times("2025-08-10 23:30:00", 4, 12, 30)
    shifted = ([t - timedelta(minutes=30) for t in past], [t - timedelta(minutes=30) for t in future])
    assert find_case_window([shifted, (past, future)], past, future) == 1
    with pytest.raises(ValueError, match="continuous"):
        find_case_window([shifted], past, future)


class RolloutProbe(torch.nn.Module):
    frame_patch_size = 1
    max_frames = 16
    radar_out_channels = 1
    satellite_out_channels = 10
    rain_out_channels = 1

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward_ar(
        self,
        context_x: torch.Tensor,
        target_x: torch.Tensor,
        context_time: torch.Tensor,
        target_time: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        torch.testing.assert_close(target_x, context_x[:, :, -1:])
        torch.testing.assert_close(target_time, context_time[:, -1:])
        self.calls.append((context_x.clone(), context_time.clone()))
        return {
            "radar": target_x[:, :1] + 1,
            "satellite": target_x[:, 1:11] + 2,
            "rain": torch.ones_like(target_x[:, 11:12]) * 0.1,
        }


def test_rollout_uses_predictions_and_accumulates_rain_delta() -> None:
    cfg = OmegaConf.create(
        {
            "train": {"strict_target_isolation": False, "loss": {"rain_residual": {"enabled": True}}},
            "dataset": {"train": {}, "val": {}},
        }
    )
    model = RolloutProbe()
    runtime = CaseRuntime(cfg, model, device="cpu", precision="no")
    context = torch.zeros(1, 12, 4, 2, 2)
    past, future = build_case_times("2025-08-10 23:30:00", 4, 12, 30)
    output = runtime.predict(context, past, future)
    torch.testing.assert_close(output["rain"][0, 0, :, 0, 0], torch.arange(1, 13) * 0.1)
    torch.testing.assert_close(output["radar"][0, 0, :, 0, 0], torch.arange(1, 13).float())
    assert [int(x.shape[2]) for x, _ in model.calls] == list(range(4, 16))
    assert model.calls[2][1][0, -1] == 0
    assert not output["rain"].requires_grad
    with pytest.raises(ValueError, match="max_frames"):
        runtime.predict(context, past, future + [future[-1] + timedelta(minutes=30)])


def test_plots_preserve_raw_values_and_record_times(tmp_path: Path) -> None:
    boundary = tmp_path / "boundary.json"
    boundary.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[97.3, 26.1], [108.4, 26.1], [108.4, 34.25], [97.3, 34.25], [97.3, 26.1]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    past, future = build_case_times("2025-08-10 23:30:00", 4, 2, 30)
    prediction = np.linspace(-0.5, 3, 128, dtype=np.float32).reshape(2, 8, 8)
    original = prediction.copy()
    save_case_figures(
        tmp_path / "figures",
        np.zeros((4, 8, 8)),
        prediction,
        np.ones((2, 8, 8)),
        past,
        future,
        boundary,
        (97.3, 108.4, 26.1, 34.25),
        dpi=40,
    )
    np.testing.assert_array_equal(prediction, original)
    for name in ("prediction_grid.png", "comparison_grid.png", "comparison.gif", "history_grid.png"):
        assert (tmp_path / "figures" / name).stat().st_size > 0
    assert (tmp_path / "figures/prediction/lead_02_20250811_0000.png").exists()


def test_real_gated_model_twelve_steps_preserve_first_four_forecasts() -> None:
    torch.set_num_threads(1)
    model = GatedModalityRainModel(
        input_size=8,
        patch_size=2,
        stem_channels=8,
        dim=16,
        depth=1,
        num_heads=2,
        decoder_base_channels=8,
        decoder_type="3d",
        decoder_k_size=3,
        decoder_pad_mode="zeros",
        stem_pad_mode="zeros",
        max_frames=16,
        spatial_modality_gate_enabled=True,
        spatial_modality_gate_hidden_channels=4,
        dropout=0.0,
        drop_path=0.0,
    )
    cfg = OmegaConf.create(
        {
            "train": {"strict_target_isolation": False, "loss": {"rain_residual": {"enabled": True}}},
            "dataset": {"train": {}, "val": {}},
        }
    )
    runtime = CaseRuntime(cfg, model, device="cpu", precision="no")
    past, future = build_case_times("2025-08-10 23:30:00", 4, 12, 30)
    context = torch.rand(1, 12, 4, 8, 8)
    short = runtime.predict(context, past, future[:4])
    long = runtime.predict(context, past, future)
    for name, channels in (("radar", 1), ("satellite", 10), ("rain", 1)):
        assert long[name].shape == (1, channels, 12, 8, 8)
        assert torch.isfinite(long[name]).all()
        torch.testing.assert_close(short[name], long[name][:, :, :4], rtol=0, atol=0)
