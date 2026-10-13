from __future__ import annotations

from typing import Any, Literal

import cv2
import numpy as np
import torch

farneback_params: dict[str, Any] = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
}


def interpolate_frame(img0_norm: np.ndarray, img1_norm: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Compute optical flow per band and stack the warped results.

    img0_norm/img1_norm: (C, H, W) or (H, W) arrays already roughly in [0, 1].
    Each band is handled independently so channel mixing cannot occur.
    """

    was_2d = img0_norm.ndim == 2
    if was_2d:
        img0_norm = img0_norm[np.newaxis, ...]
        img1_norm = img1_norm[np.newaxis, ...]

    bands, height, width = img0_norm.shape
    band_results: list[np.ndarray] = []

    for band_idx in range(bands):
        ch0 = img0_norm[band_idx].astype(np.float32)
        ch1 = img1_norm[band_idx].astype(np.float32)

        ch_min = float(min(ch0.min(), ch1.min()))
        ch_max = float(max(ch0.max(), ch1.max()))
        if ch_max - ch_min < 1e-6:
            band_results.append(0.5 * (ch0 + ch1))
            continue

        ch0_scaled = (ch0 - ch_min) / (ch_max - ch_min)
        ch1_scaled = (ch1 - ch_min) / (ch_max - ch_min)

        prev_gray = np.clip(ch0_scaled * 255, 0, 255).astype(np.uint8)
        next_gray = np.clip(ch1_scaled * 255, 0, 255).astype(np.uint8)

        flow_fwd = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, **params)
        flow_bwd = cv2.calcOpticalFlowFarneback(next_gray, prev_gray, None, **params)

        grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
        map_x_fwd = grid_x - 0.5 * flow_fwd[..., 0]
        map_y_fwd = grid_y - 0.5 * flow_fwd[..., 1]
        map_x_bwd = grid_x - 0.5 * flow_bwd[..., 0]
        map_y_bwd = grid_y - 0.5 * flow_bwd[..., 1]

        warped0 = cv2.remap(ch0, map_x_fwd, map_y_fwd, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        warped1 = cv2.remap(ch1, map_x_bwd, map_y_bwd, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        band_results.append(0.5 * (warped0 + warped1))

    stacked = np.stack(band_results, axis=0)
    if was_2d:
        return stacked[0]
    return stacked


class OpticalFlowInterpolator:
    """Utility class to interpolate multiple frames between two inputs."""

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or farneback_params

    def interp_n_frames(self, img0_norm: np.ndarray, img1_norm: np.ndarray, n_frames: int) -> np.ndarray:
        """Insert n_frames between img0_norm and img1_norm using interpolate_frame."""
        if n_frames <= 0:
            return np.empty((0,) + img0_norm.shape, dtype=img0_norm.dtype)

        frames: list[np.ndarray] = [img0_norm, img1_norm]
        while len(frames) < n_frames + 2:
            refined: list[np.ndarray] = []
            for idx in range(len(frames) - 1):
                refined.append(frames[idx])
                refined.append(interpolate_frame(frames[idx], frames[idx + 1], self.params))
            refined.append(frames[-1])
            frames = refined

        start_idx = 1
        end_idx = len(frames) - 2
        pick = np.linspace(start_idx, end_idx, n_frames, dtype=int)
        return np.stack([frames[i] for i in pick], axis=0)


InterpModality = Literal["radar", "satellite"]


class AnyModalityAnyFramesInterpolation:
    def __init__(self, modality: InterpModality, interp_n_frames: int, params: dict[str, Any] | None = None):
        self.modality = modality
        self.interp_n_frames = interp_n_frames
        self.params = params or farneback_params
        self._interpolator = OpticalFlowInterpolator(self.params)

        assert self.modality in {"radar", "satellite"}
        assert self.interp_n_frames >= 1, f"Interpolation n frames={self.interp_n_frames} should >= 1"

    def _interp_sequence(self, frames: np.ndarray, time_axis: int) -> np.ndarray:
        total_frames = frames.shape[time_axis]
        if total_frames < 2:
            raise ValueError("Need at least 2 frames to interpolate.")

        output_frames: list[np.ndarray] = []
        for t in range(total_frames - 1):
            frame0 = np.take(frames, t, axis=time_axis)
            frame1 = np.take(frames, t + 1, axis=time_axis)
            output_frames.append(frame0)
            interps = self._interpolator.interp_n_frames(frame0, frame1, self.interp_n_frames)
            output_frames.extend([interps[k] for k in range(self.interp_n_frames)])
        output_frames.append(np.take(frames, total_frames - 1, axis=time_axis))

        return np.stack(output_frames, axis=time_axis)

    def _interp_radar(self, frames: Any) -> Any:
        if isinstance(frames, (tuple, list)) and len(frames) == 2:
            frames = np.stack([np.asarray(frames[0]), np.asarray(frames[1])], axis=0)

        if isinstance(frames, torch.Tensor):
            device = frames.device
            dtype = frames.dtype
            frames = frames.detach().cpu().numpy()
            out = self._interp_radar(frames)
            return torch.from_numpy(out).to(device=device, dtype=dtype)

        if frames.ndim == 4:
            return np.stack([self._interp_radar(frames[idx]) for idx in range(frames.shape[0])], axis=0)
        if frames.ndim != 3:
            raise ValueError(f"Radar frames must be (T,H,W) or (B,T,H,W); got shape {frames.shape}")

        return self._interp_sequence(frames, time_axis=0)

    def _interp_satellite(self, frames: Any) -> Any:
        if isinstance(frames, (tuple, list)) and len(frames) == 2:
            frames = np.stack([np.asarray(frames[0]), np.asarray(frames[1])], axis=1)

        if isinstance(frames, torch.Tensor):
            device = frames.device
            dtype = frames.dtype
            frames = frames.detach().cpu().numpy()
            out = self._interp_satellite(frames)
            return torch.from_numpy(out).to(device=device, dtype=dtype)

        if frames.ndim == 5:
            return np.stack([self._interp_satellite(frames[idx]) for idx in range(frames.shape[0])], axis=0)
        if frames.ndim != 4:
            raise ValueError(f"Satellite frames must be (C,T,H,W) or (B,C,T,H,W); got shape {frames.shape}")

        return self._interp_sequence(frames, time_axis=1)

    def __call__(self, frames: Any) -> Any:
        if self.modality == "radar":
            return self._interp_radar(frames)
        return self._interp_satellite(frames)

