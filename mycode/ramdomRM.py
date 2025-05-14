import pandas as pd
import geopandas as gpd
from dbfread import DBF
from shapely.geometry import Point
from pyproj import CRS, Transformer

# 配置参数
buffer_distance = 100000  # 邻近阈值（米）
random_dbf_path = r"H:\data_new2025\baie\ramdom.dbf"
china_csv_path = r"H:\data_new2025\baie\china_bianliang\china_19.csv"
output_path = r"H:\data_new2025\baie\china_bianliang\filtered_points.csv"

def process_spatial_data():
    # 读取China参考点数据
    china_df = pd.read_csv(china_csv_path,encoding='gbk')
    print(f"从 china_19.csv 读取到 {len(china_df)} 个点")
    china_points = [Point(xy) for xy in zip(china_df['X'], china_df['Y'])]
    
    # 创建GeoDataFrame并转换坐标系
    china_gdf = gpd.GeoDataFrame(geometry=china_points, crs="EPSG:4326")
    china_gdf = china_gdf.to_crs(epsg=3857)  # 转换为Web墨卡托投影（米制单位）
    
    # 读取Random点数据
    dbf_records = DBF(random_dbf_path, encoding='utf-8')
    random_df = pd.DataFrame(iter(dbf_records))
    print(f"从 ramdom.dbf 读取到 {len(random_df)} 个点")
    random_points = [Point(xy) for xy in zip(random_df['POINT_X'], random_df['POINT_Y'])]
    
    # 创建GeoDataFrame并转换坐标系
    random_gdf = gpd.GeoDataFrame(random_df, geometry=random_points, crs="EPSG:4326")
    random_gdf = random_gdf.to_crs(epsg=3857)
    
    # 创建空间索引
    spatial_index = random_gdf.sindex
    
    # 标记需要删除的点
    to_delete = set()
    for china_idx, china_point_geom in china_gdf.geometry.items(): # china_point_geom is a Shapely Point
        # 创建围绕当前 china_point_geom 的缓冲区
        buffer_around_china_point = china_point_geom.buffer(buffer_distance) # This is a Shapely Polygon

        # 使用空间索引查找其边界框与缓冲区边界框相交的候选 random_gdf 点的索引
        # random_gdf.sindex.intersection() 返回整数索引的迭代器
        candidate_random_indices = list(random_gdf.sindex.intersection(buffer_around_china_point.bounds))

        for random_idx in candidate_random_indices: # random_idx 是一个整数
            # 获取候选点的实际几何形状
            random_point_geom_candidate = random_gdf.geometry.iloc[random_idx] # 这是一个 Shapely Point

            # 精确距离检查，因为 intersection 是基于边界框的
            if random_point_geom_candidate.distance(china_point_geom) <= buffer_distance:
                to_delete.add(random_idx) # random_idx 是整数，可以直接添加到集合中
    
    print(f"总共标记了 {len(to_delete)} 个 ramdom 点待删除")
    # 过滤保留点
    filtered_gdf = random_gdf.drop(index=list(to_delete))
    
    # 转换回原始坐标系并保存
    filtered_gdf = filtered_gdf.to_crs(epsg=4326)
    filtered_gdf['POINT_X'] = filtered_gdf.geometry.x
    filtered_gdf['POINT_Y'] = filtered_gdf.geometry.y
    
    # 保存为DBF文件
    filtered_gdf.to_file(output_path, driver="ESRI Shapefile")
    print(f"过滤完成，剩余点数：{len(filtered_gdf)}，已保存至：{output_path}")

if __name__ == "__main__":
    
    process_spatial_data()