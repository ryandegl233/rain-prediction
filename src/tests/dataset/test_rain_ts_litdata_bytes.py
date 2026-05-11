import io

import numpy as np
import pytest
import tifffile
import torch

from src.dataset.rain_ts_litdata import (
    RainTimeSeriesDataset,
    denormalize_rain_linear,
    normalize_rain_linear,
)


def _to_tiff_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    tifffile.imwrite(buf, array)
    return buf.getvalue()


def test_to_float_tensor_decodes_tiff_bytes_2d() -> None:
    dataset = object.__new__(RainTimeSeriesDataset)
    arr = np.arange(12, dtype=np.uint16).reshape(3, 4)
    raw_bytes = _to_tiff_bytes(arr)

    out = dataset._to_float_tensor(raw_bytes, field_name="radar", index=17)

    assert out.dtype == torch.float32
    assert out.shape == (3, 4)
    assert torch.allclose(out, torch.tensor(arr, dtype=torch.float32))


def test_to_float_tensor_decodes_tiff_bytes_3d_to_chw() -> None:
    dataset = object.__new__(RainTimeSeriesDataset)
    arr = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    raw_bytes = _to_tiff_bytes(arr)

    out = dataset._to_float_tensor(raw_bytes, field_name="satellite", index=23)

    assert out.dtype == torch.float32
    assert out.shape == (5, 3, 4)
    expected = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32)
    assert torch.allclose(out, expected)


def test_to_float_tensor_raises_clear_error_for_non_tensorable_values() -> None:
    dataset = object.__new__(RainTimeSeriesDataset)

    with pytest.raises(TypeError, match="field 'radar'.*type=dict"):
        dataset._to_float_tensor({"unexpected": "value"}, field_name="radar", index=99)


def test_rain_linear_norm_round_trip() -> None:
    rain = torch.tensor([0.0, 0.2, 1.0, 3.0], dtype=torch.float32)
    mean = 0.4
    std = 0.2

    rain_norm = normalize_rain_linear(rain, mean=mean, std=std)
    rain_recovered = denormalize_rain_linear(rain_norm, mean=mean, std=std)

    assert torch.allclose(rain_recovered, rain)


def test_rain_linear_norm_requires_positive_std() -> None:
    rain = torch.tensor([0.0, 1.0], dtype=torch.float32)
    with pytest.raises(ValueError, match="std must be > 0"):
        normalize_rain_linear(rain, mean=0.0, std=0.0)
    with pytest.raises(ValueError, match="std must be > 0"):
        denormalize_rain_linear(rain, mean=0.0, std=-1.0)
