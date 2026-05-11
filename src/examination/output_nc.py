from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from src.utils.visualization.plot import plot_any_modality


def _time_dim_size(data: torch.Tensor) -> int | None:
    if data.dim() == 3:
        return int(data.shape[0])
    if data.dim() in {4, 5}:
        return int(data.shape[1])
    return None


def _parse_datetime_like(value: Any) -> np.datetime64 | None:
    if isinstance(value, (datetime, np.datetime64)):
        return np.datetime64(value, "ns")
    if isinstance(value, str):
        try:
            return np.datetime64(value, "ns")
        except ValueError:
            return None
    return None


def _build_time_coords(
    input_time_size: int,
    output_time_size: int,
    time_bound: tuple[Any, Any] | None,
    exact_nx: int | None,
    frames: int | None,
) -> np.ndarray:
    if output_time_size < 1:
        raise ValueError(f"output_time_size must be >= 1, got {output_time_size}")

    if time_bound is None:
        return np.arange(output_time_size, dtype=np.int64)

    if len(time_bound) != 2:
        raise ValueError(f"time_bound must be a 2-tuple, got {time_bound}")

    start_value, end_value = time_bound
    start_dt = _parse_datetime_like(start_value)
    end_dt = _parse_datetime_like(end_value)

    if (start_dt is None) != (end_dt is None):
        raise ValueError("time_bound start/end must both be numeric or both be datetime-like values.")

    if output_time_size == 1:
        if start_dt is not None:
            return np.array([start_dt], dtype="datetime64[ns]")
        return np.array([float(start_value)], dtype=np.float64)

    divisor = 1
    if exact_nx is not None and exact_nx > 0:
        divisor = int(exact_nx)
    elif frames is not None and frames >= 0:
        divisor = int(frames) + 1

    if start_dt is not None:
        start_ns = start_dt.astype("datetime64[ns]").astype(np.int64)
        end_ns = end_dt.astype("datetime64[ns]").astype(np.int64)

        if input_time_size <= 1:
            steps = np.linspace(start_ns, end_ns, output_time_size, dtype=np.float64)
        else:
            base_delta_ns = (end_ns - start_ns) / (input_time_size - 1)
            step_ns = base_delta_ns / divisor
            steps = start_ns + np.arange(output_time_size, dtype=np.float64) * step_ns

        return np.rint(steps).astype(np.int64).astype("datetime64[ns]")

    start_num = float(start_value)
    end_num = float(end_value)
    if input_time_size <= 1:
        return np.linspace(start_num, end_num, output_time_size, dtype=np.float64)

    base_delta = (end_num - start_num) / (input_time_size - 1)
    step = base_delta / divisor
    return start_num + np.arange(output_time_size, dtype=np.float64) * step


def _resample_data_array(data: torch.Tensor, size: tuple[int, int] | None = None) -> torch.Tensor:
    target_size = size or (1024, 1024)

    if data.dim() == 2:
        return (
            F.interpolate(
                data.unsqueeze(0).unsqueeze(0).float(),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

    if data.dim() == 3:
        return F.interpolate(
            data.unsqueeze(1).float(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    if data.dim() == 4:
        return F.interpolate(
            data.float(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    if data.dim() == 5:
        batch, time, channels, height, width = data.shape
        reshaped = data.reshape(batch * time, channels, height, width).float()
        upsampled = F.interpolate(
            reshaped,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        return upsampled.reshape(batch, time, channels, *target_size)

    raise ValueError(f"Expected 2D/3D/4D/5D rain data, but got shape {tuple(data.shape)}")


def _temp_interp_data_array(data: torch.Tensor, frames: int, params: dict[str, Any] | None = None) -> torch.Tensor:
    if frames <= 0:
        return data

    from src.tools.optical_flow_interpolator import AnyModalityAnyFramesInterpolation

    if data.dim() == 5:
        interpolator = AnyModalityAnyFramesInterpolation(
            modality="satellite",
            interp_n_frames=frames,
            params=params,
        )
        data_bcthw = data.permute(0, 2, 1, 3, 4)
        interpolated = interpolator(data_bcthw)
        return interpolated.permute(0, 2, 1, 3, 4)

    if data.dim() in {3, 4}:
        interpolator = AnyModalityAnyFramesInterpolation(
            modality="radar",
            interp_n_frames=frames,
            params=params,
        )
        return interpolator(data)

    raise ValueError(f"Expected 3D/4D/5D rain data for temporal interpolation, but got shape {tuple(data.shape)}")


def _repeat_last_time_steps(data: torch.Tensor, n_repeat: int) -> torch.Tensor:
    if n_repeat <= 0:
        return data

    if data.dim() == 3:
        time_dim = 0
    elif data.dim() in {4, 5}:
        time_dim = 1
    else:
        raise ValueError(f"Expected 3D/4D/5D rain data, but got shape {tuple(data.shape)}")

    last = data.select(time_dim, data.shape[time_dim] - 1).unsqueeze(time_dim)
    repeat_shape = [1] * data.dim()
    repeat_shape[time_dim] = n_repeat
    tail = last.repeat(*repeat_shape)
    return torch.cat([data, tail], dim=time_dim)


def _temp_interp_data_array_exact_nx(data: torch.Tensor, nx: int, params: dict[str, Any] | None = None) -> torch.Tensor:
    if nx < 1:
        raise ValueError(f"exact_nx must be >= 1, got {nx}")
    if nx == 1:
        return data

    interp_n_frames = nx - 1
    interpolated = _temp_interp_data_array(data=data, frames=interp_n_frames, params=params)
    return _repeat_last_time_steps(interpolated, n_repeat=interp_n_frames)


def rain_frame_to_image(frame: torch.Tensor) -> Image.Image:
    """Convert one rain frame tensor into a RGB image.

    Args:
        frame: Rain frame tensor in ``(H, W)`` or ``(C, H, W)`` format.

    Returns:
        PIL.Image.Image: Rain colormap image.
    """
    frame = frame.detach().float().cpu().clamp_min(0.0)
    if frame.ndim == 2:
        img = plot_any_modality(frame.numpy(), modality_name="rain", to_PIL=False)
    elif frame.ndim == 3:
        img = plot_any_modality(frame, modality_name="rain", to_PIL=False)
    else:
        raise ValueError(f"Expected rain frame shape (H,W) or (C,H,W), got {tuple(frame.shape)}")
    return Image.fromarray(img)


def draw_rain_spatial_before_after(
    before_btchw: torch.Tensor,
    after_btchw: torch.Tensor,
    out_path: str | Path,
    batch_index: int = 0,
    time_index: int = 0,
) -> Path:
    """Draw rain spatial comparison image and save it.

    Args:
        before_btchw: Rain tensor in ``(B, T, C, H, W)`` before spatial interpolation.
        after_btchw: Rain tensor in ``(B, T, C, H, W)`` after spatial interpolation.
        out_path: Target image path.
        batch_index: Batch index used for visualization.
        time_index: Time index used for visualization.

    Returns:
        pathlib.Path: Saved image path.

    Example:
        >>> spatial_png = draw_rain_spatial_before_after(
        ...     before_btchw=rain_btchw,
        ...     after_btchw=rain_spatial,
        ...     out_path="runs/examination/spatial_before_after.png",
        ... )
    """
    if before_btchw.ndim != 5 or after_btchw.ndim != 5:
        raise ValueError("before_btchw/after_btchw must both be 5D tensors in (B,T,C,H,W).")

    before_img = rain_frame_to_image(before_btchw[batch_index, time_index])
    after_img = rain_frame_to_image(after_btchw[batch_index, time_index])

    font = ImageFont.load_default()
    margin = 16
    gap = 20
    title_h = 24
    width = margin * 2 + before_img.width + gap + after_img.width
    height = margin * 2 + title_h + max(before_img.height, after_img.height)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    x0 = margin
    y0 = margin + title_h
    x1 = x0 + before_img.width + gap
    canvas.paste(before_img, (x0, y0))
    canvas.paste(after_img, (x1, y0))
    draw.text((x0, margin), f"Spatial before (t{time_index})", fill=(0, 0, 0), font=font)
    draw.text((x1, margin), f"Spatial after (t{time_index})", fill=(0, 0, 0), font=font)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=90)
    return out


def draw_rain_temporal_before_after(
    before_btchw: torch.Tensor,
    after_btchw: torch.Tensor,
    out_path: str | Path,
    temporal_nx: int,
    batch_index: int = 0,
) -> Path:
    """Draw aligned rain temporal comparison image and save it.

    Real frames in ``before_btchw`` are aligned with columns ``0, nx, 2*nx, ...``.

    Args:
        before_btchw: Rain tensor in ``(B, T, C, H, W)`` before temporal interpolation.
        after_btchw: Rain tensor in ``(B, T, C, H, W)`` after temporal interpolation.
        out_path: Target image path.
        temporal_nx: Exact temporal factor. For ``temporal_nx=2``, real frames are aligned every 2 columns.
        batch_index: Batch index used for visualization.

    Returns:
        pathlib.Path: Saved image path.

    Example:
        >>> temporal_png = draw_rain_temporal_before_after(
        ...     before_btchw=rain_btchw,
        ...     after_btchw=rain_temporal,
        ...     out_path="runs/examination/temporal_before_after.png",
        ...     temporal_nx=2,
        ... )
    """
    if before_btchw.ndim != 5 or after_btchw.ndim != 5:
        raise ValueError("before_btchw/after_btchw must both be 5D tensors in (B,T,C,H,W).")
    if temporal_nx < 1:
        raise ValueError(f"temporal_nx must be >= 1, got {temporal_nx}")

    before_t = int(before_btchw.shape[1])
    after_t = int(after_btchw.shape[1])
    align_stride = temporal_nx
    cols = max(after_t, (before_t - 1) * align_stride + 1)

    font = ImageFont.load_default()
    tile = rain_frame_to_image(before_btchw[batch_index, 0])
    tile_w, tile_h = tile.width, tile.height
    left_w = 130
    top_h = 26
    gap = 6
    row_gap = 10
    width = left_w + cols * tile_w + max(cols - 1, 0) * gap + 16
    height = top_h + 2 * tile_h + row_gap + 16
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((8, 4), "Temporal interpolation comparison (aligned by real frames)", fill=(0, 0, 0), font=font)
    draw.text((8, top_h + tile_h // 2 - 8), "Before (real)", fill=(0, 0, 0), font=font)
    draw.text(
        (8, top_h + tile_h + row_gap + tile_h // 2 - 8), f"After (exact {temporal_nx}x)", fill=(0, 0, 0), font=font
    )

    for idx in range(before_t):
        col = idx * align_stride
        x = left_w + col * (tile_w + gap)
        y = top_h
        frame = rain_frame_to_image(before_btchw[batch_index, idx])
        canvas.paste(frame, (x, y))
        draw.text((x + 2, y - 18), f"r{idx}", fill=(0, 0, 0), font=font)

    for idx in range(after_t):
        x = left_w + idx * (tile_w + gap)
        y = top_h + tile_h + row_gap
        frame = rain_frame_to_image(after_btchw[batch_index, idx])
        canvas.paste(frame, (x, y))
        label = f"r{idx // align_stride}" if idx % align_stride == 0 else f"i{idx}"
        draw.text((x + 2, y - 18), label, fill=(0, 0, 0), font=font)

    for idx in range(before_t):
        col = idx * align_stride
        x = left_w + col * (tile_w + gap) - (gap // 2)
        draw.line((x, 0, x, height), fill=(180, 180, 180), width=1)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=90)
    return out


def tensor_output_to_xarray(
    data: torch.Tensor,  # rain data (B, T, C, H, W)
    geobound: tuple[float, float, float, float] = (97.3, 108.4, 26.1, 34.25),
    time_bound: tuple[Any, Any] | None = None,
    spatial_upsampler: nn.Module | Callable[[torch.Tensor], torch.Tensor] | None = None,
    spatial_sampler_kwargs: dict[str, Any] | None = None,
    temporal_upsampler: nn.Module | Callable[[torch.Tensor], torch.Tensor] | None = None,
    temporal_sampler_kwargs: dict[str, Any] | None = None,
) -> xr.DataArray:
    """Convert rain tensor to ``xarray.DataArray`` with optional spatial/temporal upsampling.

    Args:
        data: Rain tensor in 2D/3D/4D/5D. Common case is ``(B, T, C, H, W)``.
        geobound: Geographic bound as ``(lon_min, lon_max, lat_min, lat_max)``.
        time_bound: Optional 2-tuple ``(time_start, time_end)`` for the original sequence time range.
            You can pass numeric values (for example minutes) or datetime-like values
            (for example ``"2026-04-16T00:00:00"`` / ``np.datetime64`` / ``datetime``).
            If the tensor contains a time dimension, time coordinates are generated uniformly
            according to interpolation settings.
        spatial_upsampler: Optional custom spatial upsampler. If ``None``, use ``_resample_data_array``.
        spatial_sampler_kwargs: Keyword arguments for internal spatial resampler, e.g. ``{"size": (1024, 1024)}``.
        temporal_upsampler: Optional custom temporal upsampler. If ``None``, use optical-flow interpolator.
        temporal_sampler_kwargs: Keyword arguments for internal temporal interpolation.
            Supported keys:
            - ``exact_nx``: target exact temporal multiplier.
            - ``frames`` or ``interp_n_frames``: inserted frames between adjacent real frames.
            - ``params``: extra params forwarded to the optical-flow interpolator.

    Returns:
        xr.DataArray: Rain data with dims inferred from tensor rank and coords ``lat/lon``.

    Example:
        >>> da = tensor_output_to_xarray(
        ...     rain_btchw,
        ...     time_bound=(0.0, 50.0),
        ...     spatial_sampler_kwargs={
        ...         "size": (
        ...             512,
        ...             512,
        ...         )
        ...     },
        ...     temporal_sampler_kwargs={
        ...         "exact_nx": 2
        ...     },
        ... )
        >>> da.to_netcdf(
        ...     "runs/examination/rain_interp.nc"
        ... )
    """
    spatial_kwargs = spatial_sampler_kwargs or {}
    temporal_kwargs = temporal_sampler_kwargs or {}
    input_time_size = _time_dim_size(data)

    spatial_upsampled_data = (
        spatial_upsampler(data) if spatial_upsampler is not None else _resample_data_array(data, **spatial_kwargs)
    )

    if temporal_upsampler is not None:
        temp_upsampled_data = temporal_upsampler(spatial_upsampled_data)
    else:
        exact_nx = temporal_kwargs.get("exact_nx")
        if exact_nx is not None:
            temp_upsampled_data = _temp_interp_data_array_exact_nx(
                data=spatial_upsampled_data,
                nx=int(exact_nx),
                params=temporal_kwargs.get("params"),
            )
        else:
            frames = temporal_kwargs.get("frames")
            if frames is None:
                frames = temporal_kwargs.get("interp_n_frames", 0)

            temp_upsampled_data = _temp_interp_data_array(
                data=spatial_upsampled_data,
                frames=int(frames),
                params=temporal_kwargs.get("params"),
            )

    upsampled_data = temp_upsampled_data
    output_time_size = _time_dim_size(upsampled_data)

    if time_bound is not None and output_time_size is None:
        raise ValueError(f"time_bound is only valid for 3D/4D/5D tensors, but got shape {tuple(upsampled_data.shape)}")

    h, w = upsampled_data.shape[-2:]
    lon = np.linspace(geobound[0], geobound[1], w)
    lat = np.linspace(geobound[2], geobound[3], h)

    if upsampled_data.dim() == 2:
        dims = ["lat", "lon"]
    elif upsampled_data.dim() == 3:
        dims = ["time", "lat", "lon"]
    elif upsampled_data.dim() == 4:
        dims = ["batch", "time", "lat", "lon"]
    elif upsampled_data.dim() == 5:
        dims = ["batch", "time", "channel", "lat", "lon"]
    else:
        raise ValueError(f"Expected upsampled rain data to be 2D/3D/4D/5D, but got shape {tuple(upsampled_data.shape)}")

    coords: dict[str, np.ndarray] = {"lon": lon, "lat": lat}
    if output_time_size is not None:
        coords["time"] = _build_time_coords(
            input_time_size=input_time_size if input_time_size is not None else output_time_size,
            output_time_size=output_time_size,
            time_bound=time_bound,
            exact_nx=int(temporal_kwargs["exact_nx"]) if temporal_kwargs.get("exact_nx") is not None else None,
            frames=int(temporal_kwargs["frames"])
            if temporal_kwargs.get("frames") is not None
            else (int(temporal_kwargs["interp_n_frames"]) if temporal_kwargs.get("interp_n_frames") is not None else None),
        )

    da = xr.DataArray(
        upsampled_data.detach().cpu().numpy(),
        dims=dims,
        coords=coords,
    )
    return da
