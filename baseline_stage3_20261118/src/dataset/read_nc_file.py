import datetime
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# 加载卫星数据
radar_file = "data/satellite/202305/20230501_0800.nc"
radar_data = xr.open_dataset(radar_file)

# 提取基础数据
latitudes = radar_data["latitude"]
longitudes = radar_data["longitude"]

# 创建3x3的子图布局
fig, axes = plt.subplots(3, 3, figsize=(18, 15), subplot_kw={"projection": ccrs.PlateCarree()})
fig.suptitle("Himawari-8/9 TBB (2023-05-01 08:00 UTC)", fontsize=16)

# 要绘制的波段列表
bands = [f"tbb_{i:02d}" for i in range(8, 17)]  # tbb_08到tbb_16

# 统一设置颜色范围和色标
vmin, vmax = 200, 330  # 亮温范围(K)
levels = np.linspace(vmin, vmax, 20)

for i, band in enumerate(bands):
    ax = axes[i // 3, i % 3]  # 确定子图位置

    # 提取数据
    data = radar_data[band].values

    # 绘制等高线图
    contour = ax.contourf(
        longitudes, latitudes, data, levels=levels, cmap="viridis", transform=ccrs.PlateCarree()
    )

    # 添加地理特征
    ax.coastlines(resolution="10m", color="black", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")

    # 设置子图标题
    ax.set_title(f"{band} channels", fontsize=10)

# 添加公共色标
cbar = fig.colorbar(
    contour, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.02, pad=0.1
)
cbar.set_label("brightness (K)")

# 调整子图间距
plt.tight_layout()
plt.subplots_adjust(top=0.9, wspace=0.1, hspace=0.2)

# 显示图形
plt.show()
