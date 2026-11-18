from datetime import timedelta
import json
from pathlib import Path

import matplotlib
import pandas as pd
import pytest
import torch
from litdata import StreamingDataLoader

from src.dataset.rain_ts_litdata import RainTimeSeriesDataset
from src.utils.visualization.plot import plot_any_modality

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _resolve_litdata_dirs() -> list[str]:
    root = Path("data2/litdata_train/litdata_interval_30")
    candidates = [
        root / "202508",
        root / "202509",
        root / "202507",
    ]
    return [str(path) for path in candidates if path.exists()]


def _find_strict_continuous_index(dataset: RainTimeSeriesDataset) -> int:
    expected_delta = timedelta(minutes=int(dataset.time_interval))
    for idx, (past_times, future_times) in enumerate(dataset.times_pairs):
        all_times = [*past_times, *future_times]
        if len(all_times) < 2:
            continue
        if all((cur - prev) == expected_delta for prev, cur in zip(all_times[:-1], all_times[1:])):
            return idx
    raise AssertionError("No strictly continuous time window found in dataset.")


def _cat_sequence(sample: dict[str, torch.Tensor], past_key: str, future_key: str) -> torch.Tensor:
    past = sample[past_key]
    future = sample[future_key]
    if past.ndim == 3:
        past = past.unsqueeze(0)
    if future.ndim == 3:
        future = future.unsqueeze(0)
    if past.ndim != 4 or future.ndim != 4:
        raise AssertionError(f"Expected [C,T,H,W] or [T,H,W], got {past.shape} and {future.shape}")
    return torch.cat([past, future], dim=1)


def _render_sequence_figure(
    radar_seq: torch.Tensor,
    satellite_seq: torch.Tensor,
    rain_seq: torch.Tensor,
    time_labels: list[str],
    output_path: Path,
) -> None:
    total_frames = radar_seq.shape[1]
    fig, axes = plt.subplots(
        nrows=3,
        ncols=total_frames,
        figsize=(max(16, total_frames * 2.3), 7.8),
        squeeze=False,
    )

    for frame_idx in range(total_frames):
        radar_img = plot_any_modality(radar_seq[:, frame_idx], modality_name="radar", to_PIL=False)
        satellite_img = plot_any_modality(satellite_seq[:, frame_idx], modality_name="satellite", to_PIL=False)
        rain_img = plot_any_modality(rain_seq[:, frame_idx], modality_name="rain", to_PIL=False)

        axes[0, frame_idx].imshow(radar_img)
        axes[1, frame_idx].imshow(satellite_img)
        axes[2, frame_idx].imshow(rain_img)

        axes[0, frame_idx].set_title(time_labels[frame_idx], fontsize=8)
        axes[0, frame_idx].axis("off")
        axes[1, frame_idx].axis("off")
        axes[2, frame_idx].axis("off")

    axes[0, 0].set_ylabel("Radar", fontsize=11)
    axes[1, 0].set_ylabel("Satellite", fontsize=11)
    axes[2, 0].set_ylabel("Rain", fontsize=11)
    fig.suptitle("Continuous Multimodal Sequence (Past + Future)", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def test_plot_one_continuous_multimodal_sequence() -> None:
    inp_dirs = _resolve_litdata_dirs()
    if not inp_dirs:
        pytest.skip("No litdata directories found under data2/litdata_train/litdata_interval_30.")

    dataset = RainTimeSeriesDataset(
        inp_dirs=inp_dirs[:2],
        time_interval=30,
        n_past=6,
        n_futures=6,
        img_resize=256,
        stack_data=True,
        is_cycled=False,
        index_file_name=None,
        modality_zero_centering=False,
        batching_method="per_stream",
        iterate_over_all=True,
    )
    if len(dataset.times_pairs) == 0:
        pytest.skip("Dataset has no available windows.")

    sample_index = _find_strict_continuous_index(dataset)
    past_times, future_times = dataset.times_pairs[sample_index]
    all_times = [*past_times, *future_times]
    expected_delta = timedelta(minutes=30)
    assert all((cur - prev) == expected_delta for prev, cur in zip(all_times[:-1], all_times[1:]))

    sample = dataset[sample_index]
    radar_seq = _cat_sequence(sample, "radar_past", "radar_future")
    satellite_seq = _cat_sequence(sample, "satellite_past", "satellite_future")
    rain_seq = _cat_sequence(sample, "rain_past", "rain_future")
    assert radar_seq.shape[1] == len(all_times)
    assert satellite_seq.shape[1] == len(all_times)
    assert rain_seq.shape[1] == len(all_times)

    time_labels = [time_item.strftime("%m-%d %H:%M") for time_item in all_times]
    out_path = Path("runs/test_outputs/dataset_continuous_multimodal_sequence.png")
    _render_sequence_figure(
        radar_seq=radar_seq,
        satellite_seq=satellite_seq,
        rain_seq=rain_seq,
        time_labels=time_labels,
        output_path=out_path,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def _window_is_strictly_continuous(dataset: RainTimeSeriesDataset, sample_index: int) -> bool:
    expected_delta = timedelta(minutes=int(dataset.time_interval))
    past_times, future_times = dataset.times_pairs[sample_index]
    all_times = [*past_times, *future_times]
    if len(all_times) < 2:
        return False
    return all((cur - prev) == expected_delta for prev, cur in zip(all_times[:-1], all_times[1:]))


class _IndexedRainTimeSeriesDataset(RainTimeSeriesDataset):
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(index)
        sample["sample_index"] = torch.tensor(index, dtype=torch.int64)
        return sample


def _extract_cthw_from_batch(tensor: torch.Tensor, sample_index: int) -> torch.Tensor:
    if tensor.ndim == 4:
        # [B,T,H,W] -> [C=1,T,H,W]
        return tensor[sample_index].unsqueeze(0)
    if tensor.ndim == 5:
        # [B,C,T,H,W]
        return tensor[sample_index]
    raise AssertionError(f"Unsupported tensor shape for batch extraction: {tuple(tensor.shape)}")


def _get_sample_times(dataset: RainTimeSeriesDataset, sample_index: int) -> list:
    past_times, future_times = dataset.times_pairs[sample_index]
    return [*past_times, *future_times]


def _render_dataloader_batch_panel(
    dataset: RainTimeSeriesDataset,
    batch: dict[str, torch.Tensor],
    output_path: Path,
    max_samples: int | None = None,
) -> None:
    sample_indices = [int(v) for v in batch["sample_index"].tolist()]
    shown_samples = len(sample_indices) if max_samples is None else min(max_samples, len(sample_indices))
    if shown_samples <= 0:
        raise AssertionError("No samples in dataloader batch.")

    # Build one sample to get the number of frames.
    radar_first = torch.cat(
        [
            _extract_cthw_from_batch(batch["radar_past"], 0),
            _extract_cthw_from_batch(batch["radar_future"], 0),
        ],
        dim=1,
    )
    total_frames = int(radar_first.shape[1])

    fig, axes = plt.subplots(
        nrows=shown_samples * 3,
        ncols=total_frames,
        figsize=(max(18, total_frames * 2.2), shown_samples * 6.0),
        squeeze=False,
    )

    for sample_slot in range(shown_samples):
        sample_idx = sample_indices[sample_slot]
        radar_seq = torch.cat(
            [
                _extract_cthw_from_batch(batch["radar_past"], sample_slot),
                _extract_cthw_from_batch(batch["radar_future"], sample_slot),
            ],
            dim=1,
        )
        satellite_seq = torch.cat(
            [
                _extract_cthw_from_batch(batch["satellite_past"], sample_slot),
                _extract_cthw_from_batch(batch["satellite_future"], sample_slot),
            ],
            dim=1,
        )
        rain_seq = torch.cat(
            [
                _extract_cthw_from_batch(batch["rain_past"], sample_slot),
                _extract_cthw_from_batch(batch["rain_future"], sample_slot),
            ],
            dim=1,
        )

        sample_times = _get_sample_times(dataset=dataset, sample_index=sample_idx)
        expected_delta = timedelta(minutes=int(dataset.time_interval))
        is_continuous = all(
            (cur - prev) == expected_delta for prev, cur in zip(sample_times[:-1], sample_times[1:])
        )
        state_text = "OK" if is_continuous else "FAIL"
        label_color = "green" if is_continuous else "red"
        time_labels = [time_item.strftime("%m-%d %H:%M") for time_item in sample_times]

        row_base = sample_slot * 3
        for frame_idx in range(total_frames):
            radar_img = plot_any_modality(radar_seq[:, frame_idx], modality_name="radar", to_PIL=False)
            satellite_img = plot_any_modality(satellite_seq[:, frame_idx], modality_name="satellite", to_PIL=False)
            rain_img = plot_any_modality(rain_seq[:, frame_idx], modality_name="rain", to_PIL=False)

            axes[row_base, frame_idx].imshow(radar_img)
            axes[row_base + 1, frame_idx].imshow(satellite_img)
            axes[row_base + 2, frame_idx].imshow(rain_img)

            axes[row_base, frame_idx].axis("off")
            axes[row_base + 1, frame_idx].axis("off")
            axes[row_base + 2, frame_idx].axis("off")

            if frame_idx < len(time_labels):
                axes[row_base, frame_idx].set_title(time_labels[frame_idx], fontsize=8)

        axes[row_base, 0].set_ylabel(
            f"S{sample_slot} idx={sample_idx}\nRadar [{state_text}]",
            fontsize=9,
            color=label_color,
        )
        axes[row_base + 1, 0].set_ylabel("Satellite", fontsize=9)
        axes[row_base + 2, 0].set_ylabel("Rain", fontsize=9)

    fig.suptitle(
        "Dataloader Batch Sequence (num_workers=4, shuffle=True)\n"
        f"Continuity is checked per sample window by real timestamps | batch_size={shown_samples}",
        fontsize=13,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def test_dataloader_num_workers_shuffle_continuity_report() -> None:
    inp_dirs = _resolve_litdata_dirs()
    if not inp_dirs:
        pytest.skip("No litdata directories found under data2/litdata_train/litdata_interval_30.")

    dataset = _IndexedRainTimeSeriesDataset(
        inp_dirs=inp_dirs[:2],
        time_interval=30,
        n_past=3,
        n_futures=3,
        img_resize=128,
        stack_data=True,
        is_cycled=False,
        index_file_name=None,
        modality_zero_centering=False,
        batching_method="per_stream",
        iterate_over_all=True,
    )
    if len(dataset.times_pairs) == 0:
        pytest.skip("Dataset has no available windows.")

    loader = StreamingDataLoader(
        dataset,
        batch_size=4,
        num_workers=4,
        shuffle=True,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
    )

    sampled_indices: list[int] = []
    sampled_continuity: list[bool] = []
    max_batches = 20
    for batch_idx, batch in enumerate(loader):
        indices = batch["sample_index"].tolist()
        for index_value in indices:
            sample_index = int(index_value)
            sampled_indices.append(sample_index)
            sampled_continuity.append(_window_is_strictly_continuous(dataset=dataset, sample_index=sample_index))
        if batch_idx + 1 >= max_batches:
            break

    assert len(sampled_indices) > 0, "No samples were collected from dataloader."
    assert all(sampled_continuity), "Found non-continuous windows in dataloader output."

    diffs = [sampled_indices[i + 1] - sampled_indices[i] for i in range(len(sampled_indices) - 1)]
    strictly_increasing = all(diff > 0 for diff in diffs)
    report = {
        "num_samples_checked": len(sampled_indices),
        "num_workers": 4,
        "shuffle": True,
        "all_windows_strictly_continuous": all(sampled_continuity),
        "strictly_increasing_index_order": strictly_increasing,
        "first_40_indices": sampled_indices[:40],
        "first_40_diffs": diffs[:40],
    }

    out_dir = Path("runs/test_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "dataloader_shuffle_num_workers_continuity_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(range(len(sampled_indices)), sampled_indices, marker="o", linewidth=1.2, markersize=2.6)
    ax.set_title("Dataloader Sample Index Order (num_workers=4, shuffle=True)")
    ax.set_xlabel("Sample Order")
    ax.set_ylabel("Sample Index")
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    order_plot_path = out_dir / "dataloader_shuffle_num_workers_index_order.png"
    fig.savefig(order_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    assert report_path.exists() and report_path.stat().st_size > 0
    assert order_plot_path.exists() and order_plot_path.stat().st_size > 0


def test_dataloader_num_workers_shuffle_visual_time_continuity() -> None:
    inp_dirs = _resolve_litdata_dirs()
    if not inp_dirs:
        pytest.skip("No litdata directories found under data2/litdata_train/litdata_interval_30.")

    dataset = _IndexedRainTimeSeriesDataset(
        inp_dirs=inp_dirs[:2],
        time_interval=30,
        n_past=6,
        n_futures=6,
        img_resize=128,
        stack_data=True,
        is_cycled=False,
        index_file_name=None,
        modality_zero_centering=False,
        batching_method="per_stream",
        iterate_over_all=True,
    )
    if len(dataset.times_pairs) == 0:
        pytest.skip("Dataset has no available windows.")

    loader = StreamingDataLoader(
        dataset,
        batch_size=4,
        num_workers=4,
        shuffle=True,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
    )
    batch = next(iter(loader))

    batch_indices = [int(v) for v in batch["sample_index"].tolist()]
    for sample_index in batch_indices:
        assert _window_is_strictly_continuous(dataset=dataset, sample_index=sample_index)

    out_path = Path("runs/test_outputs/dataloader_shuffle_num_workers_time_continuity_panel.png")
    _render_dataloader_batch_panel(
        dataset=dataset,
        batch=batch,
        output_path=out_path,
        max_samples=4,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def _to_cthw(frame_tensor: torch.Tensor) -> torch.Tensor:
    if frame_tensor.ndim == 2:
        return frame_tensor.unsqueeze(0).unsqueeze(1)
    if frame_tensor.ndim == 3:
        return frame_tensor.unsqueeze(1)
    raise AssertionError(f"Expected [H,W] or [C,H,W], got {tuple(frame_tensor.shape)}")


def test_plot_july_heavy_rain_single_frame_512() -> None:
    july_dir = Path("data2/litdata_train/litdata_interval_30/202507")
    metadata_path = july_dir / "metadata.parquet"
    if not july_dir.exists() or not metadata_path.exists():
        pytest.skip("July 202507 litdata or metadata is missing.")

    metadata = pd.read_parquet(metadata_path).reset_index(drop=True)
    if len(metadata) == 0:
        pytest.skip("July metadata is empty.")
    metadata["time"] = pd.to_datetime(metadata["time"])

    july_day = metadata[metadata["time"].dt.date == pd.Timestamp("2025-07-03").date()]
    if len(july_day) > 0:
        target_row = july_day.loc[july_day["rain_range_max"].astype(float).idxmax()]
    else:
        target_row = metadata.loc[metadata["rain_range_max"].astype(float).idxmax()]

    target_index = int(target_row.name)
    target_time = pd.Timestamp(target_row["time"])
    target_rain_max = float(target_row["rain_range_max"])

    dataset = RainTimeSeriesDataset(
        inp_dirs=[str(july_dir)],
        time_interval=30,
        n_past=2,
        n_futures=2,
        img_resize=512,
        stack_data=True,
        is_cycled=False,
        index_file_name=None,
        modality_zero_centering=False,
        batching_method="per_stream",
        iterate_over_all=True,
    )

    radar_frame, satellite_frame, rain_frame = dataset._get_sample(target_index)
    radar_img = plot_any_modality(_to_cthw(radar_frame)[:, 0], modality_name="radar", to_PIL=False)
    satellite_img = plot_any_modality(_to_cthw(satellite_frame)[:, 0], modality_name="satellite", to_PIL=False)
    rain_img = plot_any_modality(_to_cthw(rain_frame)[:, 0], modality_name="rain", to_PIL=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), squeeze=False)
    axes[0, 0].imshow(radar_img)
    axes[0, 1].imshow(satellite_img)
    axes[0, 2].imshow(rain_img)
    axes[0, 0].set_title("Radar")
    axes[0, 1].set_title("Satellite")
    axes[0, 2].set_title("Rain")
    for idx in range(3):
        axes[0, idx].set_xlim(0, 511)
        axes[0, idx].set_ylim(511, 0)
    fig.suptitle(
        f"Time: {target_time.strftime('%Y-%m-%d %H:%M:%S')} | rain_range_max={target_rain_max:.3f}",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path("runs/test_outputs/july_heavy_rain_single_frame_512.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_full_batch_sequence_512() -> None:
    july_dir = Path("data2/litdata_train/litdata_interval_30/202507")
    if not july_dir.exists():
        pytest.skip("July 202507 litdata is missing.")

    dataset = _IndexedRainTimeSeriesDataset(
        inp_dirs=[str(july_dir)],
        time_interval=30,
        n_past=6,
        n_futures=6,
        img_resize=512,
        stack_data=True,
        is_cycled=False,
        index_file_name=None,
        modality_zero_centering=False,
        batching_method="per_stream",
        iterate_over_all=True,
    )
    if len(dataset.times_pairs) == 0:
        pytest.skip("Dataset has no available windows.")

    loader = StreamingDataLoader(
        dataset,
        batch_size=4,
        num_workers=4,
        shuffle=True,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
    )

    best_batch: dict[str, torch.Tensor] | None = None
    best_score = float("-inf")
    max_search_batches = 20
    for batch_idx, batch in enumerate(loader):
        rain_future = batch["rain_future"].float()
        score = float(rain_future.max().item())
        if score > best_score:
            best_score = score
            best_batch = batch
        if batch_idx + 1 >= max_search_batches:
            break

    if best_batch is None:
        raise AssertionError("No batch was sampled from dataloader.")

    for sample_index in [int(v) for v in best_batch["sample_index"].tolist()]:
        assert _window_is_strictly_continuous(dataset=dataset, sample_index=sample_index)

    out_dir = Path("runs/test_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "july_full_batch_sequence_512.png"
    _render_dataloader_batch_panel(
        dataset=dataset,
        batch=best_batch,
        output_path=out_path,
        max_samples=None,
    )

    info_path = out_dir / "july_full_batch_sequence_512_info.json"
    info = {
        "img_resize": 512,
        "num_workers": 4,
        "shuffle": True,
        "batch_size": int(best_batch["sample_index"].shape[0]),
        "sample_indices": [int(v) for v in best_batch["sample_index"].tolist()],
        "selected_batch_peak_rain_future": best_score,
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert info_path.exists()
    assert info_path.stat().st_size > 0
