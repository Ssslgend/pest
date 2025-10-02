import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ================== 配置参数 ==================
input_folder = "H:/data_new2025/2019_2024_sd/label"      # CSV文件所在文件夹
output_folder = "H:/data_new2025/2019_2024_sd/label/sd"     # 输出文件夹
boundary_shp = "H:/yanyi/Stydy_data/shandong.shp"       # 山东省边界文件路径
lon_col = "发生样点经度"                # CSV经度列名
lat_col = "发生样点纬度"                # CSV纬度列名
# =============================================

# 加载山东省边界并生成空间索引
shandong = gpd.read_file(boundary_shp).unary_union

def is_in_shandong(row):
    """判断点是否在山东省边界内"""
    try:
        point = Point(row[lon_col], row[lat_col])
        return point.within(shandong)
    except:
        return False  # 无效坐标点跳过

# 创建输出目录
os.makedirs(output_folder, exist_ok=True)

# 遍历处理CSV文件
processed = 0
for filename in os.listdir(input_folder):
    if not filename.endswith(".csv"):
        continue
    
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    
    try:
        # 读取CSV -> 转为GeoDataFrame
        df = pd.read_csv(input_path)
        geometry = df.apply(lambda row: Point(row[lon_col], row[lat_col]), axis=1)
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        
        # 空间筛选
        mask = gdf.geometry.within(shandong)
        shandong_gdf = gdf[mask]
        
        # 保存CSV (丢弃几何列)
        if not shandong_gdf.empty:
            shandong_gdf.drop(columns='geometry').to_csv(output_path, index=False, encoding="utf_8_sig")
            print(f"已保存: {filename} ({len(shandong_gdf)}个点)")
            processed += 1
        else:
            print(f"无数据: {filename}")
            
    except KeyError:
        print(f"跳过文件 {filename} (缺少经纬度列)")
    except Exception as e:
        print(f"处理失败: {filename} - {str(e)}")

print(f"\n处理完成！有效文件 {processed} 个，输出目录: {output_folder}")