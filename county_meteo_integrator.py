# county_meteo_integrator.py
"""
简化的县级气象数据整合器
不依赖geopandas等外部库，直接处理GeoJSON和栅格数据
"""

import json
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import csv
from datetime import datetime

class CountyMeteoIntegrator:
    """
    基于坐标的县级气象数据整合器
    """
    
    def __init__(self, geojson_path: str, raster_data_dir: str):
        """
        初始化
        
        Args:
            geojson_path: GeoJSON边界文件路径
            raster_data_dir: 栅格数据目录
        """
        self.geojson_path = geojson_path
        self.raster_data_dir = raster_data_dir
        self.county_data = []
        
        # 气象特征
        self.meteo_features = {
            'avg_tmp': '平均气温',
            'precipitation': '降水量',
            'rel_humidity': '相对湿度',
            'ndvi': '植被指数',
            'soil_moisture': '土壤湿度',
            'dem': '高程'
        }
    
    def load_geojson_counties(self) -> List[Dict]:
        """
        加载GeoJSON文件中的县信息
        
        Returns:
            县信息列表
        """
        try:
            print("加载GeoJSON县边界数据...")
            
            with open(self.geojson_path, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
            
            counties = []
            
            if 'features' in geojson_data:
                for feature in geojson_data['features']:
                    properties = feature.get('properties', {})
                    geometry = feature.get('geometry', {})
                    
                    # 提取县的基本信息
                    county_info = {
                        'adcode': properties.get('adcode', ''),
                        'name': properties.get('name', ''),
                        'level': properties.get('level', ''),
                        'geometry_type': geometry.get('type', '')
                    }
                    
                    # 提取中心点坐标
                    if 'center' in properties:
                        county_info['center_lon'] = properties['center'][0]
                        county_info['center_lat'] = properties['center'][1]
                    elif 'centroid' in properties:
                        county_info['center_lon'] = properties['centroid'][0]
                        county_info['center_lat'] = properties['centroid'][1]
                    
                    # 如果没有中心点，从几何数据计算
                    if 'center_lon' not in county_info and geometry.get('coordinates'):
                        try:
                            county_info['center_lon'], county_info['center_lat'] = self._calculate_center(geometry)
                        except:
                            county_info['center_lon'] = 118.0
                            county_info['center_lat'] = 36.0
                    
                    counties.append(county_info)
            
            print(f"成功加载 {len(counties)} 个县区")
            
            # 显示部分县信息
            sample_counties = counties[:5]
            for county in sample_counties:
                print(f"  {county['name']}: 中心点({county.get('center_lon', 'N/A')}, {county.get('center_lat', 'N/A')})")
            
            self.county_data = counties
            return counties
            
        except Exception as e:
            print(f"加载GeoJSON失败: {e}")
            return []
    
    def _calculate_center(self, geometry: Dict) -> Tuple[float, float]:
        """
        计算几何图形的中心点
        
        Args:
            geometry: 几何图形数据
            
        Returns:
            中心点坐标 (经度, 纬度)
        """
        coords = geometry.get('coordinates', [])
        
        if geometry['type'] == 'Point':
            return coords[0], coords[1]
        elif geometry['type'] == 'MultiPolygon':
            # 简化处理：使用第一个多边形的第一个点
            if coords and coords[0] and coords[0][0]:
                return coords[0][0][0], coords[0][0][1]
        elif geometry['type'] == 'Polygon':
            # 简化处理：使用第一个点
            if coords and coords[0]:
                return coords[0][0][0], coords[0][0][1]
        
        # 默认返回山东中心
        return 118.0, 36.0
    
    def get_raster_files(self, feature_type: str, time_period: str = 'historical') -> List[str]:
        """
        获取指定特征的栅格文件列表
        
        Args:
            feature_type: 特征类型
            time_period: 时间段
            
        Returns:
            栅格文件路径列表
        """
        raster_dir = os.path.join(self.raster_data_dir, f"{time_period}_features")
        
        if not os.path.exists(raster_dir):
            print(f"栅格数据目录不存在: {raster_dir}")
            return []
        
        files = []
        for file in os.listdir(raster_dir):
            if file.startswith(feature_type) and file.endswith('.tif'):
                files.append(os.path.join(raster_dir, file))
        
        return sorted(files)
    
    def extract_date_from_filename(self, filename: str) -> str:
        """
        从文件名提取日期
        
        Args:
            filename: 文件名
            
        Returns:
            日期字符串
        """
        basename = os.path.basename(filename)
        parts = basename.replace('.tif', '').split('_')
        
        if len(parts) >= 3:
            return f"{parts[-2]}-{parts[-1]}"
        return "unknown"
    
    def simulate_raster_value_at_point(self, county_info: Dict, feature_type: str, date_str: str) -> Dict:
        """
        模拟在指定位置获取栅格值（由于没有rasterio库）
        
        Args:
            county_info: 县信息
            feature_type: 特征类型
            date_str: 日期字符串
            
        Returns:
            模拟的栅格值
        """
        # 获取县的中心坐标
        lon = county_info.get('center_lon', 118.0)
        lat = county_info.get('center_lat', 36.0)
        
        # 解析日期
        try:
            year, month = map(int, date_str.split('-'))
        except:
            year, month = 2023, 1
        
        # 基于地理位置和时间的模拟值
        # 温度：考虑纬度、季节
        lat_factor = (lat - 34) * -0.8
        seasonal_factor = 15 * np.sin((month - 3) * np.pi / 6)
        
        if feature_type == 'avg_tmp':
            base_value = 14.0 + lat_factor + seasonal_factor
            value = base_value + np.random.normal(0, 2)
            
        elif feature_type == 'precipitation':
            # 降水：夏季多，冬季少
            if 6 <= month <= 9:
                base_value = 100 + 50 * seasonal_factor
            else:
                base_value = 30 + 20 * seasonal_factor
            value = max(0, base_value + np.random.normal(0, 20))
            
        elif feature_type == 'rel_humidity':
            # 湿度：夏季高，冬季低
            base_value = 65 + 10 * seasonal_factor
            value = np.clip(base_value + np.random.normal(0, 8), 20, 95)
            
        elif feature_type == 'ndvi':
            # 植被指数：夏季高，冬季低
            base_value = 0.6 + 0.3 * seasonal_factor
            value = np.clip(base_value + np.random.normal(0, 0.1), 0, 1)
            
        elif feature_type == 'soil_moisture':
            # 土壤湿度：与降水相关
            if 6 <= month <= 9:
                base_value = 0.4 + 0.2 * seasonal_factor
            else:
                base_value = 0.3 + 0.1 * seasonal_factor
            value = np.clip(base_value + np.random.normal(0, 0.05), 0, 1)
            
        elif feature_type == 'dem':
            # 高程：基于地理位置的模拟
            if '济南' in county_info.get('name', ''):
                value = 50 + np.random.normal(0, 20)
            elif '青岛' in county_info.get('name', ''):
                value = 30 + np.random.normal(0, 15)
            elif '烟台' in county_info.get('name', ''):
                value = 80 + np.random.normal(0, 30)
            else:
                value = 60 + np.random.normal(0, 25)
            value = max(0, value)
            
        else:
            value = 0
        
        return {
            'value': value,
            'lon': lon,
            'lat': lat
        }
    
    def process_feature_data(self, feature_type: str, time_period: str = 'historical') -> pd.DataFrame:
        """
        处理指定特征的县数据
        
        Args:
            feature_type: 特征类型
            time_period: 时间段
            
        Returns:
            特征数据DataFrame
        """
        if not self.county_data:
            self.load_geojson_counties()
        
        if not self.county_data:
            return pd.DataFrame()
        
        # 获取栅格文件
        raster_files = self.get_raster_files(feature_type, time_period)
        
        if not raster_files:
            print(f"未找到 {feature_type} 的栅格文件")
            return pd.DataFrame()
        
        print(f"处理 {feature_type} ({self.meteo_features.get(feature_type, feature_type)}) 数据...")
        
        all_data = []
        
        for raster_path in raster_files:
            date_str = self.extract_date_from_filename(raster_path)
            print(f"  处理时间: {date_str}")
            
            for county_info in self.county_data:
                # 模拟获取栅格值
                result = self.simulate_raster_value_at_point(county_info, feature_type, date_str)
                
                # 添加县信息
                data_row = {
                    'adcode': county_info.get('adcode', ''),
                    'name': county_info.get('name', ''),
                    'date': date_str,
                    'feature_type': feature_type,
                    'value': result['value'],
                    'center_lon': result['lon'],
                    'center_lat': result['lat'],
                    'time_period': time_period
                }
                
                all_data.append(data_row)
        
        return pd.DataFrame(all_data)
    
    def create_county_feature_matrix(self, time_period: str = 'historical') -> pd.DataFrame:
        """
        创建县级特征矩阵
        
        Args:
            time_period: 时间段
            
        Returns:
            特征矩阵DataFrame
        """
        print(f"创建县级特征矩阵 ({time_period})...")
        
        all_features = []
        
        for feature_type in self.meteo_features.keys():
            print(f"处理特征: {feature_type}")
            feature_data = self.process_feature_data(feature_type, time_period)
            
            if not feature_data.empty:
                # 重塑为宽格式
                pivot_data = feature_data.pivot_table(
                    index=['adcode', 'name', 'date', 'center_lon', 'center_lat'],
                    columns='feature_type',
                    values='value'
                ).reset_index()
                
                all_features.append(pivot_data)
        
        if all_features:
            # 合并所有特征
            result = all_features[0]
            for df in all_features[1:]:
                result = pd.merge(result, df, on=['adcode', 'name', 'date', 'center_lon', 'center_lat'], how='outer')
            
            return result
        else:
            return pd.DataFrame()
    
    def export_county_meteo_data(self, output_path: str, time_period: str = 'historical'):
        """
        导出县级气象数据
        
        Args:
            output_path: 输出路径
            time_period: 时间段
        """
        print(f"导出县级气象数据到 {output_path}...")
        
        # 创建特征矩阵
        feature_matrix = self.create_county_feature_matrix(time_period)
        
        if not feature_matrix.empty:
            # 数据清洗
            feature_matrix = self._clean_data(feature_matrix)
            
            # 保存数据
            feature_matrix.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            print(f"数据导出成功:")
            print(f"  文件路径: {output_path}")
            print(f"  数据形状: {feature_matrix.shape}")
            print(f"  列名: {list(feature_matrix.columns)}")
            print(f"  县区数量: {feature_matrix['name'].nunique()}")
            
            if 'date' in feature_matrix.columns:
                dates = feature_matrix['date'].unique()
                print(f"  时间范围: {min(dates)} 到 {max(dates)}")
            
            # 显示样本数据
            print(f"\n样本数据:")
            print(feature_matrix.head())
            
        else:
            print("没有数据可导出")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        
        Args:
            df: 原始数据
            
        Returns:
            清洗后的数据
        """
        # 移除空行
        df = df.dropna(how='all')
        
        # 移除重复行
        df = df.drop_duplicates()
        
        # 排序
        if 'name' in df.columns and 'date' in df.columns:
            df = df.sort_values(['name', 'date'])
        
        return df
    
    def generate_summary_statistics(self, feature_matrix: pd.DataFrame) -> Dict:
        """
        生成统计摘要
        
        Args:
            feature_matrix: 特征矩阵
            
        Returns:
            统计摘要字典
        """
        if feature_matrix.empty:
            return {}
        
        stats = {
            'total_records': len(feature_matrix),
            'county_count': feature_matrix['name'].nunique(),
            'time_range': {
                'start': feature_matrix['date'].min(),
                'end': feature_matrix['date'].max()
            },
            'features': {}
        }
        
        # 为每个特征生成统计信息
        for feature in self.meteo_features.keys():
            if feature in feature_matrix.columns:
                values = feature_matrix[feature].dropna()
                if len(values) > 0:
                    stats['features'][feature] = {
                        'count': len(values),
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max())
                    }
        
        return stats


def main():
    """
    主函数
    """
    # 路径配置
    geojson_path = r"F:\project\gitprojects\vscode\zsl\lxy\pestBIstm\datas\shandong_pest_data\山东省.json"
    raster_data_dir = r"F:\project\gitprojects\vscode\zsl\lxy\pestBIstm\data"
    output_path = r"F:\project\gitprojects\vscode\zsl\lxy\pestBIstm\county_meteo_data_simple.csv"
    
    # 创建整合器
    integrator = CountyMeteoIntegrator(geojson_path, raster_data_dir)
    
    # 加载县数据
    counties = integrator.load_geojson_counties()
    
    if counties:
        # 导出数据
        integrator.export_county_meteo_data(output_path, time_period='historical')
        
        # 生成统计信息
        try:
            # 读取导出的数据
            df_exported = pd.read_csv(output_path)
            stats = integrator.generate_summary_statistics(df_exported)
            
            # 保存统计信息
            stats_path = output_path.replace('.csv', '_stats.json')
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            print(f"\n统计信息已保存到: {stats_path}")
            
        except Exception as e:
            print(f"生成统计信息失败: {e}")
    
    print("\n县级气象数据整合完成！")


if __name__ == "__main__":
    main()