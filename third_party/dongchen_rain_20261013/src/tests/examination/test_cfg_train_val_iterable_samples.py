from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.dataset.rain_ts_litdata import RainTimeSeriesDataset


def _load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_dataset_from_cfg(cfg: dict[str, Any], split: str) -> RainTimeSeriesDataset:
    dataset_cfg = cfg["dataset"]
    aug_cfg = dataset_cfg["augmentation"]
    clip_cfg = dataset_cfg["value_clip"]
    ratio_cfg = dataset_cfg["rain_ratio_filter"]
    rain_norm = dataset_cfg["rain_norm"]
    split_cfg = dataset_cfg[split]
    inp_key = f"{split}_inp_dirs"

    return RainTimeSeriesDataset(
        inp_dirs=dataset_cfg[inp_key],
        time_interval=int(dataset_cfg["time_interval"]),
        n_past=int(dataset_cfg["n_past"]),
        n_futures=int(dataset_cfg["n_futures"]),
        img_resize=int(dataset_cfg["img_size"]),
        stack_data=True,
        is_cycled=False,
        index_file_name=None,
        modality_zero_centering=bool(dataset_cfg["modality_zero_centering"]),
        rain_norm_mean=rain_norm.get("mean"),
        rain_norm_std=rain_norm.get("std"),
        clip_values=bool(clip_cfg["enabled"]),
        radar_clip_min=clip_cfg.get("radar_min"),
        radar_clip_max=clip_cfg.get("radar_max"),
        satellite_clip_min=clip_cfg.get("satellite_min"),
        satellite_clip_max=clip_cfg.get("satellite_max"),
        rain_clip_min=clip_cfg.get("rain_min"),
        rain_clip_max=clip_cfg.get("rain_max"),
        iter_index_mode=str(split_cfg.get("iter_index_mode")),
        iter_index_seed=2025,
        rain_ratio_filter_enabled=bool(ratio_cfg["enabled"]),
        rain_ratio_filter_file_name=str(ratio_cfg["file_name"]),
        rain_ratio_filter_column=ratio_cfg.get("column"),
        rain_ratio_filter_min_value=float(ratio_cfg["min_value"]),
        rain_ratio_filter_mode=str(ratio_cfg["mode"]),
        aug_enabled=bool(aug_cfg[f"{split}_enabled"]),
        aug_random_crop_prob=float(aug_cfg["random_crop_prob"]) if split == "train" else 0.0,
        aug_random_crop_min_scale=float(aug_cfg["random_crop_min_scale"]) if split == "train" else 1.0,
        aug_random_crop_max_scale=float(aug_cfg["random_crop_max_scale"]) if split == "train" else 1.0,
        aug_random_crop_keep_size=bool(aug_cfg["random_crop_keep_size"]),
        aug_temporal_reverse_prob=float(aug_cfg["temporal_reverse_prob"]) if split == "train" else 0.0,
        batching_method="per_stream",
        iterate_over_all=True,
    )


def _effective_iterable_samples(total_samples: int, batch_size: int, drop_last: bool) -> int:
    if not drop_last:
        return total_samples
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    return (total_samples // batch_size) * batch_size


def test_cfg_train_val_iterable_sample_count() -> None:
    cfg_path = Path("src/config/ts_rain_train/rain_trainer_ts_next_frame.yaml")
    cfg = _load_cfg(cfg_path)

    train_ds = _build_dataset_from_cfg(cfg, split="train")
    val_ds = _build_dataset_from_cfg(cfg, split="val")

    train_total = int(len(train_ds))
    val_total = int(len(val_ds))

    train_batch_size = int(cfg["dataset"]["train"]["batch_size"])
    val_batch_size = int(cfg["dataset"]["val"]["batch_size"])
    train_drop_last = bool(cfg["dataset"]["train"]["drop_last"])
    val_drop_last = bool(cfg["dataset"]["val"]["drop_last"])

    train_iterable = _effective_iterable_samples(train_total, train_batch_size, train_drop_last)
    val_iterable = _effective_iterable_samples(val_total, val_batch_size, val_drop_last)

    report_lines = [
        f"timestamp={datetime.now().isoformat(timespec='seconds')}",
        f"cfg={cfg_path}",
        f"train_total_samples={train_total}",
        f"train_batch_size={train_batch_size}",
        f"train_drop_last={train_drop_last}",
        f"train_iterable_samples={train_iterable}",
        f"val_total_samples={val_total}",
        f"val_batch_size={val_batch_size}",
        f"val_drop_last={val_drop_last}",
        f"val_iterable_samples={val_iterable}",
    ]

    out_dir = Path("runs/examination")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"cfg_train_val_iterable_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    out_file.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n".join(report_lines))
    print(f"saved_report={out_file}")

    assert train_total > 0
    assert val_total > 0
