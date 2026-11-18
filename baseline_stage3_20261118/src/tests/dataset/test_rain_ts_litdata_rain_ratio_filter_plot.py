from pathlib import Path
import json

import matplotlib
import pandas as pd
import pytest
import torch

from src.dataset.rain_ts_litdata import RainTimeSeriesDataset
from src.utils.visualization.plot import plot_any_modality

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _resolve_target_month_dir() -> Path:
    return Path("data2/litdata_train/litdata_interval_30/202307")


def _find_rain_ratio_column(metadata: pd.DataFrame) -> str:
    ratio_cols = [col for col in metadata.columns if col.startswith("rain_ratio_gt_")]
    if not ratio_cols:
        raise AssertionError("metadata_rain_ratio.parquet has no rain_ratio_gt_* column.")
    return ratio_cols[0]


def _threshold_from_ratio_col(ratio_col: str) -> float:
    token = ratio_col.removeprefix("rain_ratio_gt_")
    token = token.replace("m", "-").replace("p", ".")
    return float(token)


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


def _render_context_future_panel(
    radar_seq: torch.Tensor,
    satellite_seq: torch.Tensor,
    rain_seq: torch.Tensor,
    time_labels: list[str],
    future_start: int,
    output_path: Path,
) -> None:
    total_frames = int(radar_seq.shape[1])
    fig, axes = plt.subplots(
        nrows=3,
        ncols=total_frames,
        figsize=(max(14, total_frames * 2.4), 8.0),
        squeeze=False,
    )

    for frame_idx in range(total_frames):
        radar_img = plot_any_modality(radar_seq[:, frame_idx], modality_name="radar", to_PIL=False)
        satellite_img = plot_any_modality(satellite_seq[:, frame_idx], modality_name="satellite", to_PIL=False)
        rain_img = plot_any_modality(rain_seq[:, frame_idx], modality_name="rain", to_PIL=False)

        axes[0, frame_idx].imshow(radar_img)
        axes[1, frame_idx].imshow(satellite_img)
        axes[2, frame_idx].imshow(rain_img)

        prefix = "C" if frame_idx < future_start else "F"
        axes[0, frame_idx].set_title(f"{prefix}{frame_idx + 1} | {time_labels[frame_idx]}", fontsize=8)
        axes[0, frame_idx].axis("off")
        axes[1, frame_idx].axis("off")
        axes[2, frame_idx].axis("off")

    axes[0, 0].set_ylabel("Radar", fontsize=11)
    axes[1, 0].set_ylabel("Satellite", fontsize=11)
    axes[2, 0].set_ylabel("Rain", fontsize=11)

    if future_start > 0:
        for row_idx in range(3):
            axes[row_idx, future_start - 1].axvline(
                x=radar_img.shape[1] - 0.5,
                color="white",
                linestyle="--",
                linewidth=1.0,
            )

    fig.suptitle("Rain Ratio Filter Hit Check: Context + Future(GT)", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def test_rain_ratio_filter_hits_future_and_plot_context_future() -> None:
    data_dir = _resolve_target_month_dir()
    ratio_parquet = data_dir / "metadata_rain_ratio.parquet"

    if not data_dir.exists() or not ratio_parquet.exists():
        pytest.skip("Target directory or metadata_rain_ratio.parquet is missing.")

    ratio_metadata = pd.read_parquet(ratio_parquet).reset_index(drop=True)
    ratio_col = _find_rain_ratio_column(ratio_metadata)
    ratio_threshold = _threshold_from_ratio_col(ratio_col)

    filtered_mask = ratio_metadata[ratio_col].astype(float) > ratio_threshold
    filtered_indices = set(ratio_metadata.index[filtered_mask].tolist())
    if len(filtered_indices) == 0:
        pytest.skip(f"No rows satisfy {ratio_col} > {ratio_threshold} in {ratio_parquet}.")

    n_past = 6
    n_futures = 1
    dataset = RainTimeSeriesDataset(
        inp_dirs=[str(data_dir)],
        time_interval=30,
        n_past=n_past,
        n_futures=n_futures,
        img_resize=512,
        stack_data=True,
        is_cycled=False,
        index_file_name=None,
        modality_zero_centering=False,
        batching_method="per_stream",
        iterate_over_all=True,
        iter_index_mode="shuffle_each_epoch",
    )

    if len(dataset.times_pairs) == 0:
        pytest.skip("No available windows in RainTimeSeriesDataset.")

    candidate_windows: list[tuple[int, int, float]] = []
    for sample_index, (_, future_indices) in enumerate(dataset.indices_pairs):
        future_index = int(future_indices[0])
        if future_index not in filtered_indices:
            continue
        future_ratio = float(ratio_metadata.at[future_index, ratio_col])
        candidate_windows.append((sample_index, future_index, future_ratio))

    assert len(candidate_windows) > 0, "No dataset window has future frame hit by rain_ratio filter."

    best_sample_index, best_future_index, best_future_ratio = max(candidate_windows, key=lambda item: item[2])

    sample = dataset[best_sample_index]
    radar_seq = _cat_sequence(sample, "radar_past", "radar_future")
    satellite_seq = _cat_sequence(sample, "satellite_past", "satellite_future")
    rain_seq = _cat_sequence(sample, "rain_past", "rain_future")

    past_times, future_times = dataset.times_pairs[best_sample_index]
    all_times = [*past_times, *future_times]
    assert len(all_times) == n_past + n_futures
    time_labels = [time_item.strftime("%m-%d %H:%M") for time_item in all_times]

    out_dir = Path("runs/test_outputs")
    out_path = out_dir / "rain_ratio_filter_context_future_202308.png"
    _render_context_future_panel(
        radar_seq=radar_seq,
        satellite_seq=satellite_seq,
        rain_seq=rain_seq,
        time_labels=time_labels,
        future_start=n_past,
        output_path=out_path,
    )

    report_path = out_dir / "rain_ratio_filter_context_future_202308.json"
    report = {
        "month": data_dir.name,
        "ratio_parquet": str(ratio_parquet),
        "ratio_column": ratio_col,
        "ratio_threshold": ratio_threshold,
        "num_filtered_rows": len(filtered_indices),
        "num_total_rows": int(len(ratio_metadata)),
        "selected_sample_index": best_sample_index,
        "selected_future_index": best_future_index,
        "selected_future_ratio": best_future_ratio,
        "times": [time_item.strftime("%Y-%m-%d %H:%M:%S") for time_item in all_times],
        "figure_path": str(out_path),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    assert out_path.exists() and out_path.stat().st_size > 0
    assert report_path.exists() and report_path.stat().st_size > 0
