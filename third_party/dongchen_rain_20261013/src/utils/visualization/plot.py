from typing import Any, Callable, Literal, no_type_check

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from PIL import Image

from src.utils.visualization.color import color_rain_map


def _robust_channel_stretch(
    img: np.ndarray,
    lower_q: float = 0.01,
    upper_q: float = 0.995,
    blend: float = 0.35,
    min_width: float = 0.18,
) -> np.ndarray:
    if img.ndim != 3:
        raise ValueError(f"Expected RGB-like image [H,W,C], got shape={img.shape}")

    base = np.clip(img.astype(np.float32), 0.0, 1.0)
    stretched = np.empty_like(base, dtype=np.float32)
    for ch in range(img.shape[-1]):
        band = base[..., ch]
        lo = float(np.quantile(band, lower_q))
        hi = float(np.quantile(band, upper_q))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            band = band - band.min()
            denom = band.max()
            stretched[..., ch] = band / denom if denom > 0 else np.zeros_like(band)
            continue

        if hi - lo < min_width:
            center = 0.5 * (hi + lo)
            lo = max(0.0, center - 0.5 * min_width)
            hi = min(1.0, center + 0.5 * min_width)

        band = np.clip((band - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        stretched[..., ch] = band

    blend = float(np.clip(blend, 0.0, 1.0))
    display = base * (1.0 - blend) + stretched * blend
    return np.clip(display * 255.0, 0.0, 255.0).astype(np.uint8)


def plot_any_modality(
    data: np.ndarray | th.Tensor,
    modality_name: Literal["rain", "satellite", "radar"],
    to_PIL: bool = True,
) -> Image.Image | np.ndarray:
    """
    data: ndarray [H, W, C] or [C, H, W] or [H, W] for rain
          tensor: tensor [B, C, H, W] or [B, 1, H, W] for satellite or radar
    modality_name: "rain", "satellite", or "radar"
    """

    def get_plot_fn(
        modality: Literal["rain", "satellite", "radar"],
    ) -> Callable[[np.ndarray], Image.Image | np.ndarray]:
        if modality == "rain":

            @no_type_check
            def plot_fn(img: np.ndarray):
                img = color_rain_map()[0](img.clip(min=0), return_ndarray=True)[-1][
                    ..., :3
                ]
                return Image.fromarray(img) if to_PIL else img
        elif modality == "satellite":

            def plot_fn(img: np.ndarray):
                sate_channels = [9, 8, 7]
                img = img[..., sate_channels]
                img = _robust_channel_stretch(img, lower_q=0.01, upper_q=0.995, blend=0.30, min_width=0.20)
                return Image.fromarray(img) if to_PIL else img
        elif modality == "radar":

            def plot_fn(img: np.ndarray):
                cmap = plt.get_cmap("turbo")
                norm = mcolors.Normalize(vmin=0, vmax=img.max())
                img = cmap(norm(img))[..., :3]
                img = (img * 255).astype(np.uint8)
                return Image.fromarray(img) if to_PIL else img
        else:
            raise ValueError(f"Unknown modality: {modality}")

        return plot_fn

    plot_fn = get_plot_fn(modality_name)

    # tensor to numpy array if needed
    if th.is_tensor(data):
        data = data.cpu().numpy()
        if data.ndim == 4:
            assert data.shape[0] == 1, "Only support batch size of 1 for now"
            data = data[0]  # take the first sample if batch dimension exists
        data = data.transpose(1, 2, 0)  # move channels to last dimension

    data = data.squeeze()  # remove any singleton dimensions
    assert isinstance(data, np.ndarray), "Data must be a numpy array"
    if modality_name in ("rain", "radar"):
        assert data.ndim == 2, "Rain data must be 2D"
    else:
        assert data.ndim == 3, "Satellite data must be 3D (H, W, C)"

    # plot the image
    img = plot_fn(data)
    if modality_name == "rain":
        # clear figure and axes
        plt.clf()
        plt.cla()
        plt.close()

    return img
