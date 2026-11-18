from typing import cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image

from src.dataset.utils import gaussian_data

bounds = np.array([0, 0.01, 0.1, 0.2, 0.5, 1.0, 5.0, 10.0, 15.0])
# 国家标准每小时分级降水量（单位：mm/h）
# hourly_bounds = np.array([0, 0.05, 0.2, 5, 30, 70, 140,300])  # 小雨、中雨、大雨、暴雨、大暴雨、特大暴雨
# bounds = hourly_bounds / 12  # [0, 0.1667, 0.4167, 0.8333, 1.6667, 4.1667]
precipitation_colors = [
    "white",
    "lightblue",
    "blue",
    "green",
    "yellow",
    "orange",
    "red",
    "magenta",
    "purple",
]


def get_pure_colored_image(img_data, target_size=None):
    """
    将降雨数据直接转换为彩色图像，不包含任何边框或坐标轴

    Parameters:
    img_data: 2D numpy array with rainfall data
    target_size: tuple (width, height) for resizing, optional

    Returns:
    numpy array of shape (height, width, 4) with RGBA values
    """
    # 创建颜色映射
    cmap = mcolors.ListedColormap(precipitation_colors)
    norm = mcolors.BoundaryNorm(bounds, ncolors=cmap.N)

    # 应用颜色映射
    normalized_data = norm(img_data)
    rgba_image = cmap(normalized_data)

    # 转换为 0-255 范围的 uint8 类型
    rgba_image = (rgba_image * 255).astype(np.uint8)

    # 如果指定了目标尺寸，则调整大小
    if target_size:
        pil_img = Image.fromarray(rgba_image)
        pil_img = pil_img.resize(target_size, resample=Image.BILINEAR)
        rgba_image = np.array(pil_img)

    return rgba_image


def color_rain_map(
    colors: list[str] | np.ndarray = precipitation_colors,
    bounds: list[float] = bounds,
    expand_rain: bool = True,
):
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, ncolors=cmap.N)

    # 返回 cmap 和 norm 以便在外部使用（例如添加 colorbar）
    def plot(
        img: np.ndarray,
        ax: plt.Axes | None = None,
        unified_bounds: list[float] = None,
        return_ndarray=False,
    ):
        img = img.astype(np.float32)
        if ax is None:
            fig, ax = plt.subplots()

        if expand_rain:
            img = gaussian_data(  # type: ignore
                img, sigma=1.5, unchanged_amp=True, n_times=2, use_cuda=False
            )

        options = dict(cmap=cmap, norm=norm)
        if unified_bounds is not None:
            options["extent"] = [
                unified_bounds[0],
                unified_bounds[1],
                unified_bounds[2],
                unified_bounds[3],
            ]

        im_obj = ax.imshow(img, **options)
        # im_obj = ax.pcolormesh(img, **options)

        # return the ndarray if requested
        if return_ndarray:
            # 注意: get_images()[0].make_image() 在较新的 matplotlib 版本中可能不推荐
            # 更现代的方法是直接从 figure buffer 导出

            # with box
            # fig = ax.get_figure()
            # fig.canvas.draw()
            # img_rgba = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            # img_rgba = img_rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            # # 从 ARGB 转换为 RGBA
            # img_rgba = np.roll(img_rgba, -1, axis=2)

            # return ax, im_obj, img_rgba

            # without box
            fig = ax.get_figure()
            fig.canvas.draw()

            renderer = fig.canvas.get_renderer()
            image_data = ax.get_images()[0].make_image(renderer, unsampled=True)
            img_array = image_data[0]

            # 确保返回的是 RGBA 格式
            if img_array.shape[2] == 3:
                # 如果只有 RGB，添加 alpha 通道
                alpha_channel = np.full((img_array.shape[0], img_array.shape[1], 1), 255, dtype=np.uint8)
                img_rgba = np.concatenate([img_array, alpha_channel], axis=2)
            else:
                img_rgba = img_array

            return ax, im_obj, img_rgba

        return ax, im_obj

    # 将 cmap 和 norm 也返回，方便创建 colorbar
    return plot, cmap, norm

