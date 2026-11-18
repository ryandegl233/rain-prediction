from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

from src.dataset import rain_ts_litdata
from src.dataset.rain_ts_litdata import RainTimeSeriesDataset


def test_len_returns_window_count() -> None:
    dataset = object.__new__(RainTimeSeriesDataset)
    dataset.times_pairs = [1, 2, 3, 4]

    assert len(dataset) == 4


def test_construct_group_pairs_filters_short_groups_without_inplace_pop(monkeypatch) -> None:
    dataset = object.__new__(RainTimeSeriesDataset)
    dataset.metadata = pd.DataFrame({"time": ["dummy"]})
    dataset.time_interval = 30
    dataset.n_past = 4
    dataset.n_futures = 1
    dataset.times_pairs = []
    dataset.indices_pairs = []

    t0 = datetime(2025, 8, 1, 0, 0)
    groups = [
        ([t0 + timedelta(minutes=30 * i) for i in range(4)], [0, 1, 2, 3]),
        ([t0 + timedelta(days=1)], [4]),
        ([t0 + timedelta(days=2, minutes=30 * i) for i in range(6)], [5, 6, 7, 8, 9, 10]),
    ]

    def fake_find_consecutive_time(*args, **kwargs):
        return groups

    captured: dict[str, list[int]] = {}

    def fake_window_partition(consecutive_times, consecutive_indices):
        captured["time_group_lens"] = [len(v) for v in consecutive_times]
        captured["index_group_lens"] = [len(v) for v in consecutive_indices]
        return [("past", "future")], [([7, 8, 9, 10], [11])]

    monkeypatch.setattr(rain_ts_litdata, "find_consecutive_time", fake_find_consecutive_time)
    dataset._window_sliding_partition = fake_window_partition

    dataset._construct_group_pairs()

    assert captured["time_group_lens"] == [6]
    assert captured["index_group_lens"] == [6]
    assert dataset.times_pairs == [("past", "future")]
    assert dataset.indices_pairs == [([7, 8, 9, 10], [11])]


def test_sanitize_and_clip_handles_non_finite_and_range() -> None:
    dataset = object.__new__(RainTimeSeriesDataset)
    dataset.clip_values = True

    x = torch.tensor([float("nan"), float("inf"), float("-inf"), -2.0, 8.0], dtype=torch.float32)
    out = dataset._sanitize_and_clip(x, min_value=0.0, max_value=5.0, fill_value=1.0)

    assert torch.allclose(out, torch.tensor([1.0, 5.0, 0.0, 0.0, 5.0], dtype=torch.float32))

    dataset.clip_values = False
    out_no_clip = dataset._sanitize_and_clip(x, min_value=0.0, max_value=5.0, fill_value=1.0)
    assert torch.allclose(out_no_clip, torch.tensor([1.0, 5.0, 0.0, -2.0, 8.0], dtype=torch.float32))


def test_rain_ratio_window_filter_future_any_keeps_expected_windows() -> None:
    dataset = object.__new__(RainTimeSeriesDataset)
    dataset.rain_ratio_values = pd.Series(np.asarray([0.01, 0.2, 0.05, 0.15], dtype=np.float32))
    dataset.rain_ratio_filter_mode = "future_any"
    dataset.rain_ratio_filter_min_value = 0.1
    dataset.rain_ratio_filter_column = "rain_ratio_gt_0p1"

    t0 = datetime(2025, 8, 1, 0, 0)
    times_pairs = [
        ([t0], [t0 + timedelta(minutes=30)]),
        ([t0 + timedelta(minutes=30)], [t0 + timedelta(minutes=60)]),
        ([t0 + timedelta(minutes=60)], [t0 + timedelta(minutes=90)]),
    ]
    indices_pairs = [
        ([0], [1]),
        ([1], [2]),
        ([2], [3]),
    ]

    kept_times, kept_indices = dataset._apply_rain_ratio_window_filter(
        times_pairs=times_pairs,
        indices_pairs=indices_pairs,
    )

    assert len(kept_times) == 2
    assert kept_indices == [([0], [1]), ([2], [3])]


def test_rain_ratio_window_filter_future_all_requires_all_future_frames() -> None:
    dataset = object.__new__(RainTimeSeriesDataset)
    dataset.rain_ratio_values = pd.Series(np.asarray([0.2, 0.15, 0.09, 0.12, 0.11, 0.2], dtype=np.float32))
    dataset.rain_ratio_filter_mode = "future_all"
    dataset.rain_ratio_filter_min_value = 0.1
    dataset.rain_ratio_filter_column = "rain_ratio_gt_0p1"

    t0 = datetime(2025, 8, 2, 0, 0)
    times_pairs = [
        ([t0], [t0 + timedelta(minutes=30), t0 + timedelta(minutes=60)]),
        ([t0 + timedelta(minutes=30)], [t0 + timedelta(minutes=60), t0 + timedelta(minutes=90)]),
        ([t0 + timedelta(minutes=60)], [t0 + timedelta(minutes=90), t0 + timedelta(minutes=120)]),
    ]
    indices_pairs = [
        ([0], [1, 2]),
        ([1], [2, 3]),
        ([2], [4, 5]),
    ]

    _, kept_indices = dataset._apply_rain_ratio_window_filter(
        times_pairs=times_pairs,
        indices_pairs=indices_pairs,
    )

    assert kept_indices == [([2], [4, 5])]
