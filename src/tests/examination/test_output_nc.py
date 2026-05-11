import torch
from pytest import MonkeyPatch

import src.examination.output_nc as output_nc_module
import src.tools.optical_flow_interpolator as optical_flow_module
from src.examination.output_nc import tensor_output_to_xarray


def test_tensor_output_to_xarray_default_spatial_resample_for_5d() -> None:
    data = torch.arange(8, dtype=torch.float32).reshape(1, 2, 1, 2, 2)
    da = tensor_output_to_xarray(
        data,
        geobound=(100.0, 110.0, 20.0, 30.0),
        spatial_sampler_kwargs={"size": (4, 4)},
    )

    assert da.shape == (1, 2, 1, 4, 4)
    assert da.dims == ("batch", "time", "channel", "lat", "lon")
    assert da.lon.size == 4
    assert da.lat.size == 4
    assert float(da.lon.values[0]) == 100.0
    assert float(da.lon.values[-1]) == 110.0
    assert float(da.lat.values[0]) == 20.0
    assert float(da.lat.values[-1]) == 30.0


def test_tensor_output_to_xarray_uses_optical_flow_temporal_upsampler(monkeypatch: MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeInterpolator:
        def __init__(self, modality: str, interp_n_frames: int, params: dict[str, object] | None = None):
            calls["modality"] = modality
            calls["interp_n_frames"] = interp_n_frames
            calls["params"] = params

        def __call__(self, frames: torch.Tensor) -> torch.Tensor:
            calls["called"] = True
            return torch.cat([frames, frames[:, :, -1:, :, :]], dim=2)

    monkeypatch.setattr(optical_flow_module, "AnyModalityAnyFramesInterpolation", FakeInterpolator)

    data = torch.arange(8, dtype=torch.float32).reshape(1, 2, 1, 2, 2)
    da = tensor_output_to_xarray(
        data,
        spatial_sampler_kwargs={"size": (2, 2)},
        temporal_sampler_kwargs={"frames": 1},
    )

    assert calls.get("called") is True
    assert calls.get("modality") == "satellite"
    assert calls.get("interp_n_frames") == 1
    assert da.shape == (1, 3, 1, 2, 2)


def test_tensor_output_to_xarray_exact_nx_rule(monkeypatch: MonkeyPatch) -> None:
    def fake_temp_interp(data: torch.Tensor, frames: int, params: dict[str, object] | None = None) -> torch.Tensor:
        assert frames == 1
        batch, time, channels, height, width = data.shape
        out = torch.zeros((batch, time * 2 - 1, channels, height, width), dtype=data.dtype, device=data.device)
        out[:, 0::2] = data
        out[:, 1::2] = 0.5 * (data[:, :-1] + data[:, 1:])
        return out

    monkeypatch.setattr(output_nc_module, "_temp_interp_data_array", fake_temp_interp)

    data = torch.arange(12, dtype=torch.float32).reshape(1, 3, 1, 2, 2)
    da = tensor_output_to_xarray(
        data,
        spatial_upsampler=lambda x: x,
        temporal_sampler_kwargs={"exact_nx": 2},
    )

    out = torch.from_numpy(da.values)
    assert da.shape == (1, 6, 1, 2, 2)
    assert torch.allclose(out[:, 0::2], data)
    assert torch.allclose(out[:, -1], out[:, -2])


def test_tensor_output_to_xarray_exact_nx_time_bound(monkeypatch: MonkeyPatch) -> None:
    def fake_temp_interp(data: torch.Tensor, frames: int, params: dict[str, object] | None = None) -> torch.Tensor:
        assert frames == 1
        batch, time, channels, height, width = data.shape
        out = torch.zeros((batch, time * 2 - 1, channels, height, width), dtype=data.dtype, device=data.device)
        out[:, 0::2] = data
        out[:, 1::2] = 0.5 * (data[:, :-1] + data[:, 1:])
        return out

    monkeypatch.setattr(output_nc_module, "_temp_interp_data_array", fake_temp_interp)

    data = torch.arange(12, dtype=torch.float32).reshape(1, 3, 1, 2, 2)
    da = tensor_output_to_xarray(
        data,
        spatial_upsampler=lambda x: x,
        temporal_sampler_kwargs={"exact_nx": 2},
        time_bound=(0.0, 20.0),
    )

    assert da.shape[1] == 6
    assert da.time.size == 6
    assert da.time.values.tolist() == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]


def test_tensor_output_to_xarray_frames_time_bound(monkeypatch: MonkeyPatch) -> None:
    def fake_temp_interp(data: torch.Tensor, frames: int, params: dict[str, object] | None = None) -> torch.Tensor:
        assert frames == 1
        batch, time, channels, height, width = data.shape
        out = torch.zeros((batch, time * 2 - 1, channels, height, width), dtype=data.dtype, device=data.device)
        out[:, 0::2] = data
        out[:, 1::2] = 0.5 * (data[:, :-1] + data[:, 1:])
        return out

    monkeypatch.setattr(output_nc_module, "_temp_interp_data_array", fake_temp_interp)

    data = torch.arange(12, dtype=torch.float32).reshape(1, 3, 1, 2, 2)
    da = tensor_output_to_xarray(
        data,
        spatial_upsampler=lambda x: x,
        temporal_sampler_kwargs={"frames": 1},
        time_bound=(0.0, 20.0),
    )

    assert da.shape[1] == 5
    assert da.time.values.tolist() == [0.0, 5.0, 10.0, 15.0, 20.0]
