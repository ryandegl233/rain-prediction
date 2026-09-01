import torch

from src.dataset.rain_ts_litdata_swinnet import RainLitData_CLS_Filter


def test_item_has_bad_value_ignores_timestamp_tensors() -> None:
    dataset = RainLitData_CLS_Filter.__new__(RainLitData_CLS_Filter)
    item = {
        "rain_future": torch.zeros(1, 1, 4, 4),
        "time_future_timestamp": torch.tensor([1_755_993_000.0], dtype=torch.float64),
    }

    assert not dataset._item_has_bad_value(item)


def test_item_has_bad_value_still_rejects_huge_modality_tensor() -> None:
    dataset = RainLitData_CLS_Filter.__new__(RainLitData_CLS_Filter)
    item = {
        "rain_future": torch.tensor([[1.0e6]], dtype=torch.float32),
        "time_future_timestamp": torch.tensor([1_755_993_000.0], dtype=torch.float64),
    }

    assert dataset._item_has_bad_value(item)
