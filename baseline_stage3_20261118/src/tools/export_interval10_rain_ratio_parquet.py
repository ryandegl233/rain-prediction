"""Export `metadata_rain_ratio.parquet` (containing `rain_ratio_gt_0p02` and `rain_ratio_gt_0p1`)
for all valid 10-minute dataset month directories under `litdata_interval_10`.
"""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from src.dataset.litdata_base import IndexedCombinedStreamingDataset
from src.dataset.rain_ts_litdata import RainTimeSeriesDataset

DEFAULT_INTERVAL10_DIRS: list[str] = [
    "/home/dongchenwang/RainPrediction/data2/litdata_train_2023_2025/litdata_interval_10/202306",
    "/home/dongchenwang/RainPrediction/data2/litdata_train_2023_2025/litdata_interval_10/202307",
    "/home/dongchenwang/RainPrediction/data2/litdata_train_2023_2025/litdata_interval_10/202308",
    "/home/dongchenwang/RainPrediction/data2/litdata_train_2023_2025/litdata_interval_10/202309",
    "/home/dongchenwang/RainPrediction/data2/litdata_train_2023_2025/litdata_interval_10/202310",
    "/home/dongchenwang/RainPrediction/data2/litdata_train_2023_2025/litdata_interval_10/202506",
    "/home/dongchenwang/RainPrediction/data2/litdata_train_2023_2025/litdata_interval_10/202507",
    "/home/dongchenwang/RainPrediction/data2/litdata_train_2023_2025/litdata_interval_10/202508",
    "/home/dongchenwang/RainPrediction/data2/litdata_train_2023_2025/litdata_interval_10/202509",
]


def _format_threshold_col(threshold: float) -> str:
    token = f"{float(threshold):.6f}".rstrip("0").rstrip(".")
    if token == "":
        token = "0"
    return f"rain_ratio_gt_{token.replace('-', 'm').replace('.', 'p')}"


def export_month_rain_ratio_parquet(
    month_dir: Path,
    *,
    thresholds: list[float],
    img_resize: int = 256,
    time_interval: int = 10,
    overwrite: bool = False,
) -> Path:
    output_path = month_dir / "metadata_rain_ratio.parquet"
    if output_path.exists() and not overwrite:
        logger.info(f"[Skip] {output_path} already exists (use --overwrite to recompute).")
        return output_path

    dataset = RainTimeSeriesDataset(
        inp_dirs=[str(month_dir)],
        time_interval=time_interval,
        n_past=1,
        n_futures=1,
        img_resize=img_resize,
        stack_data=True,
        is_cycled=False,
        modality_zero_centering=False,
        clip_values=True,
        rain_clip_min=0.0,
        rain_clip_max=None,
        rain_ratio_filter_enabled=False,
        batching_method="per_stream",
        iterate_over_all=True,
    )

    total = int(len(dataset.metadata))
    col_names = [_format_threshold_col(thr) for thr in thresholds]
    ratio_buffers: dict[str, list[float]] = {col: [] for col in col_names}

    for sample_index in tqdm(range(total), desc=f"RainRatio {month_dir.name}", dynamic_ncols=True):
        try:
            raw = IndexedCombinedStreamingDataset.__getitem__(dataset, sample_index)
            rain_raw = raw.get("rain_interpolated", None) if isinstance(raw, dict) else None
            rain = dataset._to_float_tensor(rain_raw, field_name="rain_interpolated", index=sample_index)
            rain = dataset.resizer(rain)
            rain = dataset._sanitize_and_clip(
                rain,
                min_value=dataset.rain_clip_min,
                max_value=dataset.rain_clip_max,
                fill_value=0.0,
            )
            for thr, col in zip(thresholds, col_names):
                val = float((rain > float(thr)).to(torch.float32).mean().item())
                ratio_buffers[col].append(val)
        except Exception as exc:
            logger.warning(f"[RainRatio] {month_dir.name} sample_index={sample_index} failed: {exc}")
            for col in col_names:
                ratio_buffers[col].append(float("nan"))

    metadata_out = dataset.metadata.copy()
    for col in col_names:
        metadata_out[col] = np.asarray(ratio_buffers[col], dtype=np.float32)

    metadata_out.to_parquet(output_path, index=False)
    valid_count = int(np.isfinite(metadata_out[col_names[0]].to_numpy(dtype=np.float32)).sum())
    logger.info(
        f"[Saved] {output_path} | rows={valid_count}/{total} | columns={col_names}"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export metadata_rain_ratio.parquet for 10-min dataset")
    parser.add_argument(
        "--dirs",
        nargs="*",
        default=DEFAULT_INTERVAL10_DIRS,
        help="Month directories to process",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.1, 0.02],
        help="Rain thresholds (mm) for computing rain_ratio_gt_* columns",
    )
    parser.add_argument("--img-resize", type=int, default=256)
    parser.add_argument("--time-interval", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for d in args.dirs:
        month_dir = Path(d)
        if not month_dir.is_dir():
            logger.warning(f"Directory not found, skipping: {month_dir}")
            continue
        export_month_rain_ratio_parquet(
            month_dir,
            thresholds=args.thresholds,
            img_resize=args.img_resize,
            time_interval=args.time_interval,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
