import alphashape
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import Point, Polygon

# 读取CSV文件
df = pd.read_csv("data2/四川省雨量站信息.csv")
points = df[["lng", "lat"]].dropna().values

# 计算alpha shape
alpha = 4
shape = alphashape.alphashape(points, alpha)

# 设定mask分辨率
mask_size = 512
# lng_min, lng_max = df["lng"].min(), df["lng"].max()
# lat_min, lat_max = df["lat"].min(), df["lat"].max()
lng_min, lng_max, lat_min, lat_max = (97.3, 108.4, 26.1, 34.25)


# 生成网格点
lng_grid = np.linspace(lng_min, lng_max, mask_size)
lat_grid = np.linspace(lat_min, lat_max, mask_size)
mask = np.zeros((mask_size, mask_size), dtype=np.uint8)

for i, lng in enumerate(lng_grid):
    for j, lat in enumerate(lat_grid):
        pt = Point(lng, lat)
        if shape.contains(pt):
            mask[mask_size - 1 - j, i] = 1  # y轴反向，保证和imshow一致

# 保存mask为PNG
img = Image.fromarray(mask * 255)
img.save("rain_station_mask.png")
print("Mask saved as rain_station_mask.png")

# 可视化时用 extent 对齐
plt.imshow(mask, extent=[lng_min, lng_max, lat_min, lat_max], origin="lower", alpha=0.3)
