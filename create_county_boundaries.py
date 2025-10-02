#!/usr/bin/env python3
"""
山东县域边界数据创建工具
基于现有的发生点位数据创建县级行政边界
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.ops import unary_union
import json
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ShandongCountyBoundaryCreator:
    """山东县域边界创建器"""
    
    def __init__(self, data_dir: str = "pestBIstm"):
        self.data_dir = data_dir
        self.occurrence_data = None
        self.county_boundaries = None
        
    def load_occurrence_data(self) -> pd.DataFrame:
        """
        加载美国白蛾发生数据
        Returns:
            发生数据DataFrame
        """
        try:
            all_data = []
            
            # 加载多年的发生数据
            for year in range(2019, 2024):  # 2019-2023
                file_path = os.path.join(self.data_dir, f"fall_webworm_occurrences_{year}_geocoded.csv")
                if os.path.exists(file_path):
                    try:
                        data = pd.read_csv(file_path)
                        data['year'] = year
                        all_data.append(data)
                        print(f"成功加载{year}年发生数据: {len(data)}条记录")
                    except Exception as e:
                        print(f"加载{year}年数据失败: {e}")
            
            if all_data:
                self.occurrence_data = pd.concat(all_data, ignore_index=True)
                print(f"总计加载发生数据: {len(self.occurrence_data)}条记录")
            else:
                print("未找到任何发生数据")
                
            return self.occurrence_data
            
        except Exception as e:
            print(f"加载发生数据失败: {e}")
            return None
    
    def load_target_counties(self) -> List[str]:
        """
        加载目标预测县域列表
        Returns:
            县域名称列表
        """
        try:
            location_file = os.path.join(self.data_dir, "location_list_2024.csv")
            if os.path.exists(location_file):
                locations = pd.read_csv(location_file)
                counties = locations['location'].tolist()
                print(f"加载目标县域: {len(counties)}个")
                return counties
            else:
                print("未找到目标县域文件")
                return []
        except Exception as e:
            print(f"加载目标县域失败: {e}")
            return []
    
    def create_county_boundaries_from_points(self, buffer_size: float = 0.05) -> gpd.GeoDataFrame:
        """
        基于发生点位创建县域边界
        Args:
            buffer_size: 缓冲区大小（度）
        Returns:
            县域边界GeoDataFrame
        """
        if self.occurrence_data is None:
            print("请先加载发生数据")
            return None
        
        try:
            # 按县域分组
            county_groups = self.occurrence_data.groupby("原始行政区名称")
            
            boundaries = []
            county_names = []
            occurrence_counts = []
            year_ranges = []
            
            print("创建县域边界...")
            
            for county_name, group in county_groups:
                # 创建该县域所有点的几何对象
                points = []
                for _, row in group.iterrows():
                    try:
                        lon = float(row['发生样点经度'])
                        lat = float(row['发生样点纬度'])
                        points.append(Point(lon, lat))
                    except (ValueError, TypeError):
                        continue
                
                if len(points) >= 1:
                    if len(points) == 1:
                        # 单点：创建圆形缓冲区
                        boundary = points[0].buffer(buffer_size)
                    elif len(points) == 2:
                        # 两点：创建包络线缓冲区
                        line = MultiPoint(points).convex_hull
                        boundary = line.buffer(buffer_size)
                    else:
                        # 多点：创建凸包缓冲区
                        convex_hull = MultiPoint(points).convex_hull
                        boundary = convex_hull.buffer(buffer_size)
                    
                    # 确保边界是有效的多边形
                    if not boundary.is_valid:
                        boundary = boundary.buffer(0.001)
                    
                    boundaries.append(boundary)
                    county_names.append(county_name)
                    occurrence_counts.append(len(points))
                    
                    # 记录发生年份范围
                    years = sorted(group['year'].unique())
                    year_ranges.append(f"{min(years)}-{max(years)}")
            
            # 创建GeoDataFrame
            if boundaries:
                self.county_boundaries = gpd.GeoDataFrame({
                    'county_name': county_names,
                    'geometry': boundaries,
                    'occurrence_count': occurrence_counts,
                    'year_range': year_ranges
                }, crs='EPSG:4326')
                
                print(f"成功创建{len(self.county_boundaries)}个县域边界")
                return self.county_boundaries
            else:
                print("未能创建任何县域边界")
                return None
                
        except Exception as e:
            print(f"创建县域边界失败: {e}")
            return None
    
    def add_missing_counties(self, target_counties: List[str], default_buffer: float = 0.1) -> gpd.GeoDataFrame:
        """
        为缺失的县域添加默认边界
        Args:
            target_counties: 目标县域列表
            default_buffer: 默认缓冲区大小
        Returns:
            完整的县域边界GeoDataFrame
        """
        if self.county_boundaries is None:
            print("请先创建基础边界")
            return None
        
        try:
            existing_counties = set(self.county_boundaries['county_name'].tolist())
            missing_counties = set(target_counties) - existing_counties
            
            if not missing_counties:
                print("所有目标县域都有边界数据")
                return self.county_boundaries
            
            print(f"发现{len(missing_counties)}个缺失县域，正在添加默认边界...")
            
            # 山东县域中心坐标（近似值）
            shandong_county_centers = {
                # 济南市
                '历下区': (117.03, 36.67),
                '市中区': (117.00, 36.65),
                '槐荫区': (116.90, 36.65),
                '天桥区': (116.98, 36.68),
                '历城区': (117.07, 36.68),
                '长清区': (116.75, 36.55),
                '章丘区': (117.53, 36.72),
                '平阴县': (116.45, 36.29),
                '济阳县': (117.20, 36.97),
                '商河县': (117.15, 37.31),
                
                # 青岛市
                '市南区': (120.38, 36.06),
                '市北区': (120.37, 36.08),
                '黄岛区': (120.18, 35.97),
                '崂山区': (120.47, 36.15),
                '李沧区': (120.43, 36.18),
                '城阳区': (120.38, 36.30),
                '即墨区': (120.45, 36.38),
                '胶州市': (120.03, 36.28),
                '平度市': (119.97, 36.78),
                '莱西市': (120.53, 36.87),
                
                # 淄博市
                '淄川区': (117.97, 36.64),
                '张店区': (118.05, 36.81),
                '博山区': (117.86, 36.50),
                '临淄区': (118.31, 36.82),
                '周村区': (117.87, 36.80),
                '桓台县': (118.10, 36.96),
                '高青县': (117.83, 37.18),
                '沂源县': (118.17, 36.18),
                
                # 添加更多县域坐标...
            }
            
            missing_boundaries = []
            missing_names = []
            missing_counts = []
            missing_years = []
            
            for county_name in missing_counties:
                # 获取县域中心坐标
                if county_name in shandong_county_centers:
                    center = Point(shandong_county_centers[county_name])
                else:
                    # 使用山东省中心坐标作为默认值
                    center = Point(117.00, 36.67)
                
                # 创建默认边界
                boundary = center.buffer(default_buffer)
                
                missing_boundaries.append(boundary)
                missing_names.append(county_name)
                missing_counts.append(0)  # 无发生记录
                missing_years.append("无数据")
            
            # 创建缺失县域的GeoDataFrame
            missing_gdf = gpd.GeoDataFrame({
                'county_name': missing_names,
                'geometry': missing_boundaries,
                'occurrence_count': missing_counts,
                'year_range': missing_years
            }, crs='EPSG:4326')
            
            # 合并边界数据
            complete_boundaries = pd.concat([self.county_boundaries, missing_gdf], ignore_index=True)
            
            print(f"完整县域边界: {len(complete_boundaries)}个")
            self.county_boundaries = complete_boundaries
            
            return complete_boundaries
            
        except Exception as e:
            print(f"添加缺失县域失败: {e}")
            return None
    
    def save_boundaries(self, output_dir: str = None):
        """
        保存县域边界数据
        Args:
            output_dir: 输出目录
        """
        if self.county_boundaries is None:
            print("没有边界数据可保存")
            return
        
        try:
            if output_dir is None:
                output_dir = os.path.join(self.data_dir, "data", "county_boundaries")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存为Shapefile
            shapefile_path = os.path.join(output_dir, "shandong_county_boundaries.shp")
            self.county_boundaries.to_file(shapefile_path, encoding='utf-8')
            print(f"Shapefile已保存至: {shapefile_path}")
            
            # 保存为GeoJSON
            geojson_path = os.path.join(output_dir, "shandong_county_boundaries.geojson")
            self.county_boundaries.to_file(geojson_path, driver='GeoJSON', encoding='utf-8')
            print(f"GeoJSON已保存至: {geojson_path}")
            
            # 保存为CSV（带边界WKT）
            csv_path = os.path.join(output_dir, "shandong_county_boundaries.csv")
            csv_data = self.county_boundaries.copy()
            csv_data['geometry_wkt'] = csv_data['geometry'].apply(lambda x: x.wkt)
            csv_data = csv_data.drop('geometry', axis=1)
            csv_data.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"CSV已保存至: {csv_path}")
            
            # 保存边界信息摘要
            summary_path = os.path.join(output_dir, "county_boundaries_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("山东县域边界数据摘要\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"总县域数: {len(self.county_boundaries)}\n")
                f.write(f"有发生记录的县域: {len(self.county_boundaries[self.county_boundaries['occurrence_count'] > 0])}\n")
                f.write(f"无发生记录的县域: {len(self.county_boundaries[self.county_boundaries['occurrence_count'] == 0])}\n\n")
                
                f.write("详细信息:\n")
                f.write("-" * 30 + "\n")
                for _, row in self.county_boundaries.iterrows():
                    f.write(f"{row['county_name']}: {row['occurrence_count']}个发生点 ({row['year_range']})\n")
            
            print(f"摘要已保存至: {summary_path}")
            
        except Exception as e:
            print(f"保存边界数据失败: {e}")
    
    def run_boundary_creation(self, target_counties: List[str] = None) -> gpd.GeoDataFrame:
        """
        运行完整的边界创建流程
        Args:
            target_counties: 目标县域列表
        Returns:
            完整的县域边界GeoDataFrame
        """
        print("开始创建山东县域边界数据...")
        
        # 1. 加载发生数据
        self.load_occurrence_data()
        
        # 2. 加载目标县域
        if target_counties is None:
            target_counties = self.load_target_counties()
        
        # 3. 基于发生点创建边界
        self.create_county_boundaries_from_points()
        
        # 4. 添加缺失县域
        if target_counties:
            self.add_missing_counties(target_counties)
        
        # 5. 保存边界数据
        self.save_boundaries()
        
        print("县域边界数据创建完成!")
        return self.county_boundaries

def main():
    """主函数"""
    creator = ShandongCountyBoundaryCreator()
    
    # 运行边界创建
    boundaries = creator.run_boundary_creation()
    
    if boundaries is not None:
        print(f"\n边界创建完成!")
        print(f"总县域数: {len(boundaries)}")
        print(f"数据格式: GeoDataFrame")
        print(f"坐标系: EPSG:4326")
        
        # 显示前5个县域信息
        print(f"\n前5个县域:")
        for _, row in boundaries.head().iterrows():
            print(f"{row['county_name']}: {row['occurrence_count']}个发生点")

if __name__ == "__main__":
    main()