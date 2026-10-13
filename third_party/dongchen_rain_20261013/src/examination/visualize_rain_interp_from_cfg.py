from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from src.dataset.rain_ts_litdata import RainTimeSeriesDataset
from src.examination.output_nc import (
    draw_rain_spatial_before_after,
    draw_rain_temporal_before_after,
    tensor_output_to_xarray,
)
from src.tools.optical_flow_interpolator import AnyModalityAnyFramesInterpolation


def _load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_dataset_from_cfg(cfg: dict[str, Any]) -> RainTimeSeriesDataset:
    dcfg = cfg["dataset"]
    clip_cfg = dcfg["value_clip"]
    ratio_cfg = dcfg["rain_ratio_filter"]
    rain_norm = dcfg["rain_norm"]

    return RainTimeSeriesDataset(
        inp_dirs=dcfg["train_inp_dirs"],
        time_interval=int(dcfg["time_interval"]),
        n_past=int(dcfg["n_past"]),
        n_futures=int(dcfg["n_futures"]),
        img_resize=int(dcfg["img_size"]),
        stack_data=True,
        is_cycled=False,
        index_file_name=None,
        modality_zero_centering=bool(dcfg["modality_zero_centering"]),
        rain_norm_mean=rain_norm.get("mean"),
        rain_norm_std=rain_norm.get("std"),
        clip_values=bool(clip_cfg["enabled"]),
        radar_clip_min=clip_cfg.get("radar_min"),
        radar_clip_max=clip_cfg.get("radar_max"),
        satellite_clip_min=clip_cfg.get("satellite_min"),
        satellite_clip_max=clip_cfg.get("satellite_max"),
        rain_clip_min=clip_cfg.get("rain_min"),
        rain_clip_max=clip_cfg.get("rain_max"),
        iter_index_mode="shuffle_each_epoch",
        iter_index_seed=2025,
        rain_ratio_filter_enabled=bool(ratio_cfg["enabled"]),
        rain_ratio_filter_file_name=str(ratio_cfg["file_name"]),
        rain_ratio_filter_column=ratio_cfg.get("column"),
        rain_ratio_filter_min_value=float(ratio_cfg["min_value"]),
        rain_ratio_filter_mode=str(ratio_cfg["mode"]),
        aug_enabled=False,
        batching_method="per_stream",
        iterate_over_all=True,
    )


def _spatial_interp_2x_btchw(data: torch.Tensor) -> torch.Tensor:
    if data.ndim != 5:
        raise ValueError(f"Expected BTCHW tensor, got shape={tuple(data.shape)}")
    b, t, c, h, w = data.shape
    resized = F.interpolate(
        data.reshape(b * t, c, h, w).float(),
        scale_factor=2.0,
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(b, t, c, h * 2, w * 2).to(dtype=data.dtype)


def _temporal_interp_btchw(data: torch.Tensor, interp_n_frames: int) -> torch.Tensor:
    if data.ndim != 5:
        raise ValueError(f"Expected BTCHW tensor, got shape={tuple(data.shape)}")
    if interp_n_frames <= 0:
        return data

    interpolator = AnyModalityAnyFramesInterpolation(
        modality="satellite",
        interp_n_frames=interp_n_frames,
    )
    data_bcthw = data.permute(0, 2, 1, 3, 4).contiguous()
    out = interpolator(data_bcthw)
    if not isinstance(out, torch.Tensor):
        out = torch.from_numpy(out)
    out = out.to(device=data.device, dtype=data.dtype)
    return out.permute(0, 2, 1, 3, 4).contiguous()


def _temporal_interp_exact_nx_btchw(data: torch.Tensor, temporal_nx: int) -> torch.Tensor:
    if temporal_nx < 1:
        raise ValueError(f"temporal_nx must be >= 1, got {temporal_nx}")
    if temporal_nx == 1:
        return data

    interp_n_frames = temporal_nx - 1
    interpolated = _temporal_interp_btchw(data=data, interp_n_frames=interp_n_frames)
    tail = interpolated[:, -1:, :, :, :].repeat(1, interp_n_frames, 1, 1, 1)
    return torch.cat([interpolated, tail], dim=1)


def main() -> None:
    cfg_path = Path("src/config/ts_rain_train/rain_trainer_ts_next_frame.yaml")
    cfg = _load_cfg(cfg_path)
    dataset = _build_dataset_from_cfg(cfg)

    sample = dataset[17]
    rain_past = sample["rain_past"]
    rain_future = sample["rain_future"]
    rain_tchw = torch.cat([rain_past, rain_future], dim=0).float()
    if rain_tchw.ndim == 3:
        rain_tchw = rain_tchw.unsqueeze(1)
    elif rain_tchw.ndim != 4:
        raise ValueError(f"Expected rain sequence to be TCHW/THW, got shape={tuple(rain_tchw.shape)}")
    rain_btchw = rain_tchw.unsqueeze(0)
    time_interval = int(cfg["dataset"]["time_interval"])
    time_bound = (0.0, float((rain_btchw.shape[1] - 1) * time_interval))

    temporal_nx = 2
    rain_spatial = _spatial_interp_2x_btchw(rain_btchw)
    rain_temporal = _temporal_interp_exact_nx_btchw(rain_btchw, temporal_nx=temporal_nx)
    rain_both = _temporal_interp_exact_nx_btchw(rain_spatial, temporal_nx=temporal_nx)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs/examination") / f"interp_viz_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    spatial_png = out_dir / "spatial_before_after.png"
    temporal_png = out_dir / "temporal_before_after.png"
    draw_rain_spatial_before_after(before_btchw=rain_btchw, after_btchw=rain_spatial, out_path=spatial_png)
    draw_rain_temporal_before_after(
        before_btchw=rain_btchw,
        after_btchw=rain_temporal,
        out_path=temporal_png,
        temporal_nx=temporal_nx,
    )

    da_before = tensor_output_to_xarray(
        rain_btchw,
        time_bound=time_bound,
        spatial_upsampler=lambda x: x,
        temporal_sampler_kwargs={"frames": 0},
    )
    da_after_both = tensor_output_to_xarray(
        rain_btchw,
        time_bound=time_bound,
        spatial_sampler_kwargs={"size": (rain_btchw.shape[-2] * 2, rain_btchw.shape[-1] * 2)},
        temporal_sampler_kwargs={"exact_nx": temporal_nx},
    )

    before_nc = out_dir / "rain_before_interp.nc"
    after_nc = out_dir / f"rain_after_spatial2x_temporal_exact{temporal_nx}x.nc"
    da_before.to_netcdf(before_nc)
    da_after_both.to_netcdf(after_nc)

    (out_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"rain_btchw_shape={tuple(rain_btchw.shape)}",
                f"rain_spatial_shape={tuple(rain_spatial.shape)}",
                f"rain_temporal_shape={tuple(rain_temporal.shape)}",
                f"rain_both_shape={tuple(rain_both.shape)}",
                f"spatial_png={spatial_png}",
                f"temporal_png={temporal_png}",
                f"before_nc={before_nc}",
                f"after_nc={after_nc}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Saved outputs to: {out_dir}")
    print(f"Spatial compare image: {spatial_png}")
    print(f"Temporal compare image: {temporal_png}")
    print(f"NC before: {before_nc}")
    print(f"NC after : {after_nc}")


if __name__ == "__main__":
    main()
