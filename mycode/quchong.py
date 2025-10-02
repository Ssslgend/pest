import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

# ----------------------
# 配置参数（按需修改）
# ----------------------
CSV_FOLDER = "H:/data_new2025/2019_2024_sd/label/sd"  # 现有CSV文件所在文件夹
OUTPUT_FILE = "H:/data_new2025/2019_2024_sd/label/sd/random.csv"  # 输出文件名
SHAPEFILE_PATH = "H:/yanyi/Stydy_data/shandong.shp"  # 山东边界文件路径
TARGET_POINTS = 300  # 需要生成的点数
MAX_ATTEMPTS = 5000  # 最大尝试次数（防止死循环）

# ----------------------
# 1. 读取现有CSV中的坐标
# ----------------------
def load_existing_coords(folder):
    existing = set()
    for file in Path(folder).glob("*.csv"):
        try:
            df = pd.read_csv(file)
            # 自动探测经纬度列名（中英文兼容）
            lon_col = next((col for col in df.columns if '发生样点经度' in col or 'lon' in col.lower()), None)
            lat_col = next((col for col in df.columns if '发生样点纬度' in col or 'lat' in col.lower()), None)
            
            if lon_col and lat_col:
                # 四舍五入到6位小数去重
                coords = set(zip(
                    df[lon_col].round(6).astype(float),
                    df[lat_col].round(6).astype(float)
                ))
                existing.update(coords)
                print(f"从 {file.name} 中加载 {len(coords)} 个现有坐标")
        except Exception as e:
            print(f"警告：跳过 {file.name}（错误：{str(e)}）")
    return existing

existing_coords = load_existing_coords(CSV_FOLDER)
print(f"共发现 {len(existing_coords)} 个需排除的坐标")

# ----------------------
# 2. 加载山东边界
# ----------------------
try:
    gdf = gpd.read_file(SHAPEFILE_PATH)
    print("字段名：", gdf.columns.tolist())  # 打印字段名
    print("省份列表：", gdf['省'].unique())  # 打印所有唯一的省份
    shandong = gdf[gdf['省'] == '山东省']
    
    if shandong.empty:
        raise ValueError("未找到山东省的边界数据")
    
    shandong = shandong.geometry.iloc[0]
    minx, miny, maxx, maxy = shandong.bounds
except Exception as e:
    print(f"边界加载失败：{str(e)}")
    exit()

# ----------------------
# 3. 生成随机点
# ----------------------
new_points = []
attempts = 0

while len(new_points) < TARGET_POINTS and attempts < MAX_ATTEMPTS:
    # 生成随机坐标
    lon = np.random.uniform(minx, maxx)
    lat = np.random.uniform(miny, maxy)
    point = (round(lon, 6), round(lat, 6))
    
    # 检查条件
    if (point not in existing_coords) and shandong.contains(Point(lon, lat)):
        new_points.append(point)
        existing_coords.add(point)  # 防止重复生成
        
    attempts += 1

# ----------------------
# 4. 结果处理与保存
# ----------------------
if len(new_points) < TARGET_POINTS:
    print(f"警告：仅生成 {len(new_points)} 个有效点（可能边界内空间不足）")

df = pd.DataFrame(new_points, columns=['经度', '纬度'])
df.to_csv(OUTPUT_FILE, index=False)
print(f"成功生成 {len(df)} 个点，已保存至 {OUTPUT_FILE}")

# 可选：可视化验证
# shandong.boundary.plot()
# plt.scatter(df['经度'], df['纬度'], s=1)
# plt.show()