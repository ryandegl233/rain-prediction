from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from src.dataset.rain_ts_litdata import RainTimeSeriesDataset
from src.utils.visualization.plot import plot_any_modality


def _load_dataset_cfg() -> dict[str, Any]:
    cfg_path = Path("src/config/ts_rain_train/rain_trainer_ts_next_frame.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["dataset"]


def _build_dataset_for_gif(dataset_cfg: dict[str, Any]) -> RainTimeSeriesDataset:
    train_cfg = dataset_cfg["train"]
    clip_cfg = dataset_cfg["value_clip"]
    rain_norm = dataset_cfg["rain_norm"]

    return RainTimeSeriesDataset(
        inp_dirs=dataset_cfg["train_inp_dirs"],
        time_interval=int(dataset_cfg["time_interval"]),
        n_past=int(dataset_cfg["n_past"]),
        n_futures=int(dataset_cfg["n_futures"]),
        img_resize=int(dataset_cfg["img_size"]),
        stack_data=bool(train_cfg.get("stack_data", True)),
        is_cycled=bool(train_cfg.get("is_cycled", False)),
        index_file_name=train_cfg.get("index_file_name"),
        modality_zero_centering=bool(dataset_cfg.get("modality_zero_centering", False)),
        rain_norm_mean=rain_norm.get("mean"),
        rain_norm_std=rain_norm.get("std"),
        clip_values=bool(clip_cfg.get("enabled", True)),
        radar_clip_min=clip_cfg.get("radar_min"),
        radar_clip_max=clip_cfg.get("radar_max"),
        satellite_clip_min=clip_cfg.get("satellite_min"),
        satellite_clip_max=clip_cfg.get("satellite_max"),
        rain_clip_min=clip_cfg.get("rain_min"),
        rain_clip_max=clip_cfg.get("rain_max"),
        iter_index_mode=str(train_cfg.get("iter_index_mode", "shuffle_each_epoch")),
        iter_index_seed=2025,
        rain_ratio_filter_enabled=False,
        rain_ratio_filter_file_name="metadata_rain_ratio.parquet",
        rain_ratio_filter_column=None,
        rain_ratio_filter_min_value=0.0,
        rain_ratio_filter_mode="future_any",
        aug_enabled=False,
        batching_method=str(train_cfg.get("batching_method", "per_stream")),
        iterate_over_all=bool(train_cfg.get("iterate_over_all", True)),
    )


def test_export_100_filtered_rain_frames_to_gif() -> None:
    dataset_cfg = _load_dataset_cfg()
    dataset = _build_dataset_for_gif(dataset_cfg)

    threshold = 0.2
    rain_frames: list[Image.Image] = []
    selected_indices: list[int] = []
    selected_times: list[datetime] = []
    all_time_strs = dataset.metadata["time"].tolist()

    for sample_index, time_str in enumerate(all_time_strs):
        _, _, rain = dataset._get_sample(int(sample_index))
        rain = rain.detach().float().cpu().clamp_min(0.0)
        if float(rain.max().item()) <= threshold:
            continue

        if rain.ndim == 2:
            rain_data_for_plot = rain.numpy()
        elif rain.ndim == 3:
            rain_data_for_plot = rain
        else:
            raise ValueError(f"Unexpected rain frame shape: {tuple(rain.shape)}")

        rain_img = plot_any_modality(rain_data_for_plot, modality_name="rain", to_PIL=True)
        if not isinstance(rain_img, Image.Image):
            raise TypeError("plot_any_modality should return PIL Image when to_PIL=True.")
        rain_frames.append(rain_img)
        selected_indices.append(int(sample_index))
        selected_times.append(datetime.strptime(str(time_str), "%Y-%m-%d %H:%M:%S"))
        if len(rain_frames) >= 100:
            break

    assert len(rain_frames) == 100, f"Frames with rain max > {threshold} are fewer than 100."

    output_dir = Path("runs/examination")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = output_dir / f"rain_filtered_gt_0p2_100_{timestamp}.gif"
    meta_path = output_dir / f"rain_filtered_gt_0p2_100_{timestamp}.txt"

    rain_frames[0].save(
        gif_path,
        save_all=True,
        append_images=rain_frames[1:],
        duration=180,
        loop=0,
        optimize=False,
        disposal=2,
    )

    meta_path.write_text(
        "\n".join(
            [
                f"gif_path={gif_path}",
                f"n_frames={len(rain_frames)}",
                f"filter_rule=rain_max_gt_{threshold}",
                f"time_start={selected_times[0].strftime('%Y-%m-%d %H:%M:%S')}",
                f"time_end={selected_times[-1].strftime('%Y-%m-%d %H:%M:%S')}",
                f"first_index={selected_indices[0]}",
                f"last_index={selected_indices[-1]}",
            ]
        ),
        encoding="utf-8",
    )

    assert gif_path.exists(), f"GIF not found: {gif_path}"
    with Image.open(gif_path) as gif:
        assert int(getattr(gif, "n_frames", 1)) == 100
