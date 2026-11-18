import json

import alphashape
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
from shapely.geometry import Polygon

# 读取CSV文件
df = pd.read_csv("data2/四川省雨量站信息.csv")
lng = df["lng"]
lat = df["lat"]

# 组合为点坐标
points = df[["lng", "lat"]].dropna().values

# 计算alpha shape（凹包），alpha参数可调整，越小越贴合
alpha = 4  # 可根据实际分布调整
shape = alphashape.alphashape(points, alpha)

plt.figure(figsize=(10, 8))
plt.scatter(lng, lat, s=1, alpha=0.5)

# 绘制alpha shape polygon
if isinstance(shape, Polygon):
    x, y = shape.exterior.xy
    plt.plot(x, y, "r-", lw=1.5)
elif hasattr(shape, "geoms"):
    for geom in shape.geoms:
        x, y = geom.exterior.xy
        plt.plot(x, y, "r-", lw=1.5)


# 自动下载四川省边界GeoJSON并绘制
try:
    url = "https://geo.datav.aliyun.com/areas_v3/bound/510000_full.json"
    r = requests.get(url, timeout=10)
    boundary = gpd.GeoDataFrame.from_features(json.loads(r.text)["features"])
    boundary.plot(ax=plt.gca(), facecolor="none", edgecolor="blue", linewidth=1.5)
    print("Sichuan boundary loaded and plotted.")
except Exception as e:
    print(f"Failed to load Sichuan boundary: {e}")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Sichuan Province Rain Stations with Alpha Shape and Boundary")
plt.xlim(lng.min(), lng.max())
plt.ylim(lat.min(), lat.max())
plt.savefig("sichuan_rain_stations_with_boundary.png", dpi=300, bbox_inches="tight")
print("Plot with alpha shape and boundary saved as sichuan_rain_stations_with_boundary.png")
