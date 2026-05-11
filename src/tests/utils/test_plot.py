import numpy as np

from src.utils.visualization.plot import _robust_channel_stretch, plot_any_modality


def test_robust_channel_stretch_stays_uint8_for_narrow_range_satellite() -> None:
    h, w = 16, 16
    base = np.linspace(0.72, 0.90, h * w, dtype=np.float32).reshape(h, w)
    img = np.stack([base, base + 0.01, base + 0.02], axis=-1)

    out = _robust_channel_stretch(img, lower_q=0.01, upper_q=0.995, blend=0.30, min_width=0.20)

    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8
    assert out.std() > 0


def test_plot_any_modality_satellite_returns_rgb_image() -> None:
    sat = np.zeros((8, 8, 10), dtype=np.float32)
    for idx in range(10):
        sat[..., idx] = 0.70 + idx * 0.01

    out = plot_any_modality(sat, modality_name="satellite", to_PIL=False)

    assert isinstance(out, np.ndarray)
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8
