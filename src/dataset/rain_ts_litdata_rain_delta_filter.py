"""
for m in 202305 202306 202307 202308 202309 202506 202507; do
  python src/dataset/rain_ts_litdata_rain_delta_filter.py \
    --input-dir /home/rainpred/RainPrediction/data2/litdata_train_2025/litdata_interval_30/${m} \
    --output-parquet /home/rainpred/RainPrediction/data2/litdata_train_2025/litdata_interval_30/${m}/metadata_rain_delta_filter.parquet \
    --vis-dir /home/rainpred/RainPrediction/vis_show/rain_delta_filter/${m} \
    --n-past 4 \
    --n-futures 6 \
    --quantile 0.8 \
    --max-visual-samples 8
done
python src/dataset/rain_ts_litdata_rain_delta_filter.py \
  --input-dir /home/rainpred/RainPrediction/data2/litdata_train_2025/litdata_interval_30/202305/pairs/chunk-0-0.bin \
  --output-parquet /home/rainpred/RainPrediction/vis_show/exsample \
  --vis-dir /home/rainpred/RainPrediction/vis_show/exsample \
  --n-past 4 \
  --n-futures 6 \
  --quantile 0.8 \
  --max-visual-samples 8 \
  --use-rain-ratio-filter \
  --rain-ratio-filter-min-value 0 \
  --overwrite
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

# Ensure `src.*` imports work when running this file directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.dataset.rain_ts_litdata import RainTimeSeriesDataset, find_consecutive_time
from src.utils.visualization.plot import plot_any_modality

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _resolve_input_dir(input_dir: str) -> Path:
    path = Path(input_dir)
    if path.exists():
        if path.is_file() and path.name.startswith("chunk-"):
            return path.parent.parent
        return path

    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / path
    if candidate.exists():
        if candidate.is_file() and candidate.name.startswith("chunk-"):
            return candidate.parent.parent
        return candidate

    raise FileNotFoundError(f"Input directory does not exist: {input_dir}")


def _load_low_resolution_sample(dataset: RainTimeSeriesDataset, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    low, _ = dataset._get_sample_with_target(index)
    radar, satellite, rain = low
    return radar, satellite, rain


def _compute_frame_delta_scores(
    dataset: RainTimeSeriesDataset,
    *,
    base_mask: pd.Series | None = None,
) -> pd.DataFrame:
    metadata = dataset.metadata.reset_index(drop=True).copy()
    metadata["rain_delta_source_index"] = np.arange(len(metadata), dtype=np.int64)
    metadata["rain_delta_prev_index"] = -1
    metadata["rain_delta_abs_mean"] = np.nan
    metadata["rain_delta_abs_max"] = np.nan
    metadata["rain_delta_abs_sum"] = np.nan
    metadata["rain_delta_is_valid"] = False

    consecutive_groups = find_consecutive_time(
        metadata["time"].tolist(),
        time_format="%Y-%m-%d %H:%M:%S",
        time_interval=int(dataset.time_interval),
    )

    total_valid = 0
    for _times, indices in tqdm(consecutive_groups, desc="Scan time groups", leave=True):
        if len(indices) <= 1:
            continue

        prev_rain: torch.Tensor | None = None
        prev_index: int | None = None
        for raw_index in tqdm(indices, desc="Read frames", leave=False):
            if base_mask is not None and not bool(base_mask.iloc[int(raw_index)]):
                prev_rain = None
                prev_index = None
                continue
            try:
                _radar, _satellite, rain = _load_low_resolution_sample(dataset, int(raw_index))
            except Exception as exc:
                logger.warning(f"[RainDeltaFilter] sample_index={raw_index} decode failed: {exc}")
                prev_rain = None
                prev_index = None
                continue

            if prev_rain is None or prev_index is None:
                prev_rain = rain
                prev_index = int(raw_index)
                continue

            delta = rain - prev_rain
            abs_delta = delta.abs()

            metadata.at[int(raw_index), "rain_delta_prev_index"] = int(prev_index)
            metadata.at[int(raw_index), "rain_delta_abs_mean"] = float(abs_delta.mean().item())
            metadata.at[int(raw_index), "rain_delta_abs_max"] = float(abs_delta.max().item())
            metadata.at[int(raw_index), "rain_delta_abs_sum"] = float(abs_delta.sum().item())
            metadata.at[int(raw_index), "rain_delta_is_valid"] = True
            total_valid += 1

            prev_rain = rain
            prev_index = int(raw_index)

    logger.info(
        f"[RainDeltaFilter] computed delta scores for {total_valid}/{len(metadata)} frames "
        f"under time_interval={dataset.time_interval}"
    )
    return metadata


def _load_ratio_filter_mask(
    resolved_dir: Path,
    *,
    file_name: str,
    column_name: str | None,
    min_value: float,
) -> tuple[pd.Series, str]:
    ratio_path = resolved_dir / file_name
    if not ratio_path.exists():
        raise FileNotFoundError(f"Rain ratio parquet does not exist: {ratio_path}")

    ratio_metadata = pd.read_parquet(ratio_path).reset_index(drop=True)
    if len(ratio_metadata) == 0:
        raise ValueError(f"Rain ratio parquet is empty: {ratio_path}")

    ratio_column = column_name
    if ratio_column is None:
        ratio_candidates = [col for col in ratio_metadata.columns if str(col).startswith("rain_ratio_gt_")]
        if len(ratio_candidates) == 0:
            raise ValueError(f"No rain_ratio_gt_* column found in {ratio_path}")
        ratio_column = str(ratio_candidates[0])
    if ratio_column not in ratio_metadata.columns:
        raise ValueError(f"rain_ratio_filter_column={ratio_column} not found in {ratio_path}")

    ratio_values = pd.to_numeric(ratio_metadata[ratio_column], errors="coerce")
    ratio_mask = ratio_values >= float(min_value)
    ratio_mask = ratio_mask.fillna(False)
    logger.info(
        f"[RainDeltaFilter] ratio filter loaded | file={ratio_path} | column={ratio_column} | "
        f"min_value={min_value} | kept={int(ratio_mask.sum())}/{len(ratio_mask)}"
    )
    return ratio_mask, ratio_column


def _render_sequence_panel(
    radar_seq: torch.Tensor,
    rain_seq: torch.Tensor,
    time_labels: list[str],
    output_path: Path,
    sample_title: str,
    highlight_frame: int | None = None,
) -> None:
    total_frames = int(radar_seq.shape[1])
    rain_diff_seq = torch.zeros_like(rain_seq)
    radar_diff_seq = torch.zeros_like(radar_seq)
    if total_frames > 1:
        rain_diff_seq[:, 1:] = (rain_seq[:, 1:] - rain_seq[:, :-1]).abs()
        radar_diff_seq[:, 1:] = (radar_seq[:, 1:] - radar_seq[:, :-1]).abs()

    rain_diff_max = float(rain_diff_seq.max().item()) if total_frames > 0 else 0.0
    radar_diff_max = float(radar_diff_seq.max().item()) if total_frames > 0 else 0.0

    fig, axes = plt.subplots(
        nrows=4,
        ncols=total_frames,
        figsize=(max(16, total_frames * 2.2), 10.0),
        squeeze=False,
    )

    for frame_idx in range(total_frames):
        radar_img = plot_any_modality(radar_seq[:, frame_idx], modality_name="radar", to_PIL=False)
        rain_img = plot_any_modality(rain_seq[:, frame_idx], modality_name="rain", to_PIL=False)

        axes[0, frame_idx].imshow(rain_img)
        axes[1, frame_idx].imshow(
            rain_diff_seq[0, frame_idx],
            cmap="magma",
            vmin=0.0,
            vmax=max(rain_diff_max, 1.0e-8),
        )
        axes[2, frame_idx].imshow(radar_img)
        axes[3, frame_idx].imshow(
            radar_diff_seq[0, frame_idx],
            cmap="magma",
            vmin=0.0,
            vmax=max(radar_diff_max, 1.0e-8),
        )

        axes[0, frame_idx].set_title(time_labels[frame_idx], fontsize=8)
        for row_idx in range(4):
            axes[row_idx, frame_idx].axis("off")
            if highlight_frame is not None and frame_idx == highlight_frame:
                axes[row_idx, frame_idx].set_facecolor("#fff0f0")
            if highlight_frame is not None and frame_idx == highlight_frame:
                for spine in axes[row_idx, frame_idx].spines.values():
                    spine.set_visible(True)
                    spine.set_color("red")
                    spine.set_linewidth(2.0)

    axes[0, 0].set_ylabel("Rain", fontsize=11)
    axes[1, 0].set_ylabel("Rain Δ", fontsize=11)
    axes[2, 0].set_ylabel("Radar", fontsize=11)
    axes[3, 0].set_ylabel("Radar Δ", fontsize=11)

    fig.suptitle(sample_title, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _select_visual_windows(
    dataset: RainTimeSeriesDataset,
    score_metadata: pd.DataFrame,
    score_column: str,
    *,
    max_samples: int,
) -> list[tuple[int, int, float]]:
    valid_scores = score_metadata[score_column].astype(float)
    selected_indices = valid_scores[valid_scores.notna()].sort_values(ascending=False).index.tolist()
    if len(selected_indices) == 0:
        return []

    selected_windows: list[tuple[int, int, float]] = []
    used_windows: set[int] = set()

    for raw_index in selected_indices:
        raw_score = float(score_metadata.at[raw_index, score_column])
        candidate_windows: list[tuple[int, int, float]] = []
        for sample_index, (_past_indices, future_indices) in enumerate(dataset.indices_pairs):
            if int(raw_index) not in future_indices:
                continue
            future_scores = score_metadata.loc[list(future_indices), score_column].astype(float)
            max_future_score = float(future_scores.max(skipna=True))
            if np.isfinite(max_future_score):
                candidate_windows.append((sample_index, int(raw_index), max(max_future_score, raw_score)))

        if not candidate_windows:
            continue

        best_sample_index, best_raw_index, best_score = max(candidate_windows, key=lambda item: item[2])
        if best_sample_index in used_windows:
            continue

        used_windows.add(best_sample_index)
        selected_windows.append((best_sample_index, best_raw_index, best_score))
        if len(selected_windows) >= max_samples:
            break

    return selected_windows


def build_rain_delta_filter(
    input_dir: str,
    *,
    output_parquet_path: str,
    vis_dir: str,
    time_interval: int,
    n_past: int,
    n_futures: int,
    img_resize: int,
    threshold: float | None,
    quantile: float,
    max_visual_samples: int,
    overwrite: bool,
    use_rain_ratio_filter: bool,
    rain_ratio_filter_file_name: str,
    rain_ratio_filter_column: str | None,
    rain_ratio_filter_min_value: float,
) -> dict[str, object]:
    resolved_dir = _resolve_input_dir(input_dir)
    output_path = Path(output_parquet_path)
    if output_path.suffix != ".parquet":
        if output_path.exists() and output_path.is_dir():
            output_path = output_path / "metadata_rain_delta.parquet"
        elif not output_path.exists() and output_parquet_path.endswith("/"):
            output_path = output_path / "metadata_rain_delta.parquet"
        else:
            output_path = output_path.with_suffix(".parquet")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output parquet already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = RainTimeSeriesDataset(
        inp_dirs=[str(resolved_dir)],
        time_interval=time_interval,
        n_past=n_past,
        n_futures=n_futures,
        img_resize=img_resize,
        stack_data=True,
        is_cycled=False,
        index_file_name=None,
        modality_zero_centering=False,
        batching_method="per_stream",
        iterate_over_all=True,
    )

    ratio_mask: pd.Series | None = None
    ratio_column: str | None = None
    if use_rain_ratio_filter:
        ratio_mask, ratio_column = _load_ratio_filter_mask(
            resolved_dir,
            file_name=rain_ratio_filter_file_name,
            column_name=rain_ratio_filter_column,
            min_value=rain_ratio_filter_min_value,
        )

    score_metadata = _compute_frame_delta_scores(dataset, base_mask=ratio_mask)
    score_column = "rain_delta_abs_mean"
    valid_scores = pd.to_numeric(score_metadata[score_column], errors="coerce")
    valid_scores_array = valid_scores.to_numpy(dtype=np.float32, na_value=np.nan)
    finite_scores = valid_scores_array[np.isfinite(valid_scores_array)]
    if finite_scores.size == 0:
        raise RuntimeError("No valid rain delta scores were computed.")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")

    if threshold is not None:
        selected_threshold = float(threshold)
    else:
        selected_threshold = float(np.quantile(finite_scores, quantile))
    selected_mask = valid_scores >= selected_threshold
    score_metadata["rain_delta_selected"] = selected_mask.fillna(False)
    score_metadata["rain_delta_threshold"] = selected_threshold
    if ratio_mask is not None:
        score_metadata["rain_ratio_selected"] = ratio_mask.reindex(score_metadata.index, fill_value=False)
        score_metadata["rain_ratio_filter_file_name"] = rain_ratio_filter_file_name
        score_metadata["rain_ratio_filter_column"] = ratio_column
        score_metadata["rain_ratio_filter_min_value"] = float(rain_ratio_filter_min_value)

    score_metadata.to_parquet(output_path, index=False)

    vis_output_dir = Path(vis_dir)
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    selected_windows = _select_visual_windows(
        dataset,
        score_metadata,
        score_column,
        max_samples=max_visual_samples,
    )

    if len(selected_windows) == 0:
        logger.warning("[RainDeltaFilter] No visual windows matched the selected frames.")

    figure_paths: list[str] = []
    for rank, (sample_index, raw_index, sample_score) in enumerate(selected_windows):
        sample = dataset[sample_index]
        radar_seq = torch.cat([sample["radar_past"], sample["radar_future"]], dim=1)
        rain_seq = torch.cat([sample["rain_past"], sample["rain_future"]], dim=1)
        past_times, future_times = dataset.times_pairs[sample_index]
        all_times = [*past_times, *future_times]
        time_labels = [time_item.strftime("%m-%d %H:%M") for time_item in all_times]
        highlight_frame = None
        future_indices = dataset.indices_pairs[sample_index][1]
        if raw_index in future_indices:
            highlight_frame = len(past_times) + future_indices.index(raw_index)

        figure_path = vis_output_dir / f"rain_delta_sample_{rank:03d}_idx_{raw_index:06d}.png"
        sample_title = (
            f"Rain Delta Selection | sample_index={sample_index} | raw_index={raw_index} | "
            f"score={sample_score:.6f} | threshold={selected_threshold:.6f}"
        )
        _render_sequence_panel(
            radar_seq=radar_seq,
            rain_seq=rain_seq,
            time_labels=time_labels,
            output_path=figure_path,
            sample_title=sample_title,
            highlight_frame=highlight_frame,
        )
        figure_paths.append(str(figure_path))

    summary = {
        "input_dir": str(resolved_dir),
        "output_parquet": str(output_path),
        "score_column": score_column,
        "threshold": selected_threshold,
        "quantile": quantile,
        "n_total_rows": int(len(score_metadata)),
        "n_selected_rows": int(selected_mask.fillna(False).sum()),
        "use_rain_ratio_filter": use_rain_ratio_filter,
        "rain_ratio_filter_file_name": rain_ratio_filter_file_name if use_rain_ratio_filter else None,
        "rain_ratio_filter_column": ratio_column if use_rain_ratio_filter else None,
        "rain_ratio_filter_min_value": rain_ratio_filter_min_value if use_rain_ratio_filter else None,
        "n_ratio_selected_rows": int(ratio_mask.sum()) if ratio_mask is not None else None,
        "n_visual_samples": len(figure_paths),
        "figure_paths": figure_paths,
    }
    summary_path = vis_output_dir / "rain_delta_selection_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info(f"[RainDeltaFilter] Saved parquet: {output_path}")
    logger.info(f"[RainDeltaFilter] Saved summary: {summary_path}")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build rain delta metadata parquet and visual samples.")
    parser.add_argument("--input-dir", type=str, required=True, help="Input litdata month directory.")
    parser.add_argument(
        "--output-parquet",
        type=str,
        default=None,
        help="Output parquet path. Default: <input-dir>/metadata_rain_delta.parquet",
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default="/home/rainpred/RainPrediction/vis_show/exsample",
        help="Directory for visual samples.",
    )
    parser.add_argument("--time-interval", type=int, default=30)
    parser.add_argument("--n-past", type=int, default=4)
    parser.add_argument("--n-futures", type=int, default=6)
    parser.add_argument("--img-resize", type=int, default=512)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Absolute threshold on rain_delta_abs_mean. If omitted, use quantile.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.8,
        help="Quantile used to derive the threshold when --threshold is not set.",
    )
    parser.add_argument("--max-visual-samples", type=int, default=8)
    parser.add_argument("--use-rain-ratio-filter", action="store_true")
    parser.add_argument(
        "--rain-ratio-filter-file-name",
        type=str,
        default="metadata_rain_ratio.parquet",
    )
    parser.add_argument("--rain-ratio-filter-column", type=str, default=None)
    parser.add_argument("--rain-ratio-filter-min-value", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    input_dir = str(args.input_dir)
    output_parquet = args.output_parquet
    if output_parquet is None:
        output_parquet = str(_resolve_input_dir(input_dir) / "metadata_rain_delta.parquet")
    else:
        output_path = Path(output_parquet)
        if output_path.suffix != ".parquet":
            if output_path.exists() and output_path.is_dir():
                output_parquet = str(output_path / "metadata_rain_delta.parquet")
            elif output_parquet.endswith("/"):
                output_parquet = str(output_path / "metadata_rain_delta.parquet")
            else:
                output_parquet = str(output_path.with_suffix(".parquet"))

    summary = build_rain_delta_filter(
        input_dir=input_dir,
        output_parquet_path=output_parquet,
        vis_dir=str(args.vis_dir),
        time_interval=int(args.time_interval),
        n_past=int(args.n_past),
        n_futures=int(args.n_futures),
        img_resize=int(args.img_resize),
        threshold=args.threshold,
        quantile=float(args.quantile),
        max_visual_samples=int(args.max_visual_samples),
        overwrite=bool(args.overwrite),
        use_rain_ratio_filter=bool(args.use_rain_ratio_filter),
        rain_ratio_filter_file_name=str(args.rain_ratio_filter_file_name),
        rain_ratio_filter_column=args.rain_ratio_filter_column,
        rain_ratio_filter_min_value=float(args.rain_ratio_filter_min_value),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
