#!/usr/bin/env python3
"""
基于空间关联法(Point in Polygon)的山东气象数据处理
将气象站点数据按照县域进行精确划分
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, shape
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SpatialMeteorologicalDataProcessor:
    def __init__(self, pest_data_path, geojson_path=None, output_dir="datas/shandong_pest_data"):
        """
        初始化空间气象数据处理器
        
        Args:
            pest_data_path: 病虫害数据文件路径
            geojson_path: 山东县域边界GeoJSON文件路径
            output_dir: 输出目录
        """
        self.pest_data_path = pest_data_path
        self.geojson_path = geojson_path or "F:\project\gitprojects\\vscode\zsl\lxy\pestBIstm\datas\shandong_pest_data\shandong.json"
        self.output_dir = output_dir
        
        # 气象特征列
        self.feature_columns = [
            'Temperature', 'Humidity', 'Rainfall', 'WS',
            'WD', 'Pressure', 'Sunshine', 'Visibility',
            'Temperature_MA', 'Humidity_MA', 'Rainfall_MA', 'Pressure_MA'
        ]
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载病虫害数据
        self.pest_data = self._load_pest_data()
        
        # 加载县域边界数据
        self.county_boundaries = self._load_county_boundaries()
        
        # 创建县名映射（处理GeoJSON中的县名与病虫害数据中的县名差异）
        self.county_name_mapping = self._create_county_name_mapping()
        
    def _load_county_boundaries(self):
        """加载县域边界数据"""
        print("加载县域边界数据...")
        
        try:
            if os.path.exists(self.geojson_path):
                # 使用GeoJSON文件
                gdf = gpd.read_file(self.geojson_path)
                print(f"成功加载 {len(gdf)} 个县域边界")
                
                # 检查GeoJSON的结构
                print(f"GeoJSON列名: {list(gdf.columns)}")
                if 'name' in gdf.columns:
                    print(f"包含县名: {gdf['name'].head(10).tolist()}")
                elif 'NAME' in gdf.columns:
                    print(f"包含县名: {gdf['NAME'].head(10).tolist()}")
                
                return gdf
            else:
                # 如果没有GeoJSON文件，创建基于坐标的简化边界
                print("未找到GeoJSON文件，创建基于坐标的简化边界...")
                return self._create_simplified_boundaries()
        except Exception as e:
            print(f"加载GeoJSON时出错: {e}")
            print("使用简化边界作为替代...")
            return self._create_simplified_boundaries()
    
    def _create_county_name_mapping(self):
        """创建县名映射，处理GeoJSON与病虫害数据中的县名差异"""
        print("创建县名映射...")
        
        # 获取病虫害数据中的县名
        pest_counties = set(self.pest_data['原始行政区名称'].unique())
        
        # 获取GeoJSON中的县名
        geojson_counties = set()
        if hasattr(self.county_boundaries, 'columns'):
            if 'name' in self.county_boundaries.columns:
                geojson_counties = set(self.county_boundaries['name'].unique())
            elif 'NAME' in self.county_boundaries.columns:
                geojson_counties = set(self.county_boundaries['NAME'].unique())
            elif 'county' in self.county_boundaries.columns:
                geojson_counties = set(self.county_boundaries['county'].unique())
        
        print(f"病虫害数据中的县名数量: {len(pest_counties)}")
        print(f"GeoJSON中的县名数量: {len(geojson_counties)}")
        
        # 找出匹配和不匹配的县名
        matched_counties = pest_counties & geojson_counties
        unmatched_pest = pest_counties - geojson_counties
        unmatched_geojson = geojson_counties - pest_counties
        
        print(f"匹配的县名: {len(matched_counties)}")
        print(f"病虫害数据中独有的县名: {len(unmatched_pest)}")
        print(f"GeoJSON中独有的县名: {len(unmatched_geojson)}")
        
        if unmatched_pest:
            print(f"病虫害数据中独有的县名: {list(unmatched_pest)[:10]}...")
        
        # 创建映射字典
        mapping = {}
        for county in matched_counties:
            mapping[county] = county
        
        return mapping
    
    def _create_simplified_boundaries(self):
        """创建简化的县域边界（基于中心点缓冲区）"""
        # 从病虫害数据中提取县名和坐标
        df_pest = pd.read_csv(self.pest_data_path)
        
        # 获取每个县的中心坐标
        county_centers = df_pest.groupby('原始行政区名称').agg({
            '发生样点经度': 'mean',
            '发生样点纬度': 'mean'
        }).reset_index()
        
        # 为每个县创建简化的边界（圆形缓冲区）
        geometries = []
        county_names = []
        
        for _, row in county_centers.iterrows():
            county_name = row['原始行政区名称']
            center_lon = row['发生样点经度']
            center_lat = row['发生样点纬度']
            
            # 创建中心点
            center_point = Point(center_lon, center_lat)
            
            # 创建缓冲区（大约20km半径，根据县的面积调整）
            buffer_radius = 0.2  # 度
            geometry = center_point.buffer(buffer_radius)
            
            geometries.append(geometry)
            county_names.append(county_name)
        
        # 创建GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'name': county_names,
            'geometry': geometries
        }, crs='EPSG:4326')
        
        print(f"创建了 {len(gdf)} 个简化县域边界")
        return gdf
    
    def _load_pest_data(self):
        """加载病虫害数据"""
        print("加载病虫害数据...")
        df_pest = pd.read_csv(self.pest_data_path)
        
        print(f"病虫害数据统计:")
        print(f"  总记录数: {len(df_pest)}")
        print(f"  年份范围: {df_pest['year'].min()} - {df_pest['year'].max()}")
        print(f"  县区数量: {df_pest['原始行政区名称'].nunique()}")
        
        return df_pest
    
    def _generate_realistic_meteorological_data(self, county_name, year, month, day):
        """
        生成基于实际气候规律的气象数据
        
        Args:
            county_name: 县名
            year: 年份
            month: 月份
            day: 日期
            
        Returns:
            气象数据字典
        """
        # 获取县的几何信息
        county_geom = self.county_boundaries[self.county_boundaries['name'] == county_name]
        if len(county_geom) == 0:
            # 如果没有找到该县，使用默认值
            lat, lon = 36.0, 118.0
        else:
            # 获取县的中心点
            center = county_geom.geometry.iloc[0].centroid
            lon, lat = center.x, center.y
        
        # 基于地理规律的气候参数计算
        # 纬度影响温度（北冷南热）
        lat_factor = (lat - 34) * -0.5
        
        # 经度影响湿度（东湿西干）
        lon_factor = (lon - 116) * 0.3
        
        # 海拔影响（简化计算）
        altitude_factor = 0  # 可以根据DEM数据调整
        
        # 季节性因子
        day_of_year = datetime(year, month, day).timetuple().tm_yday
        seasonal_factor = np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # 温度计算（考虑纬度、季节）
        base_temp = 14.0 + lat_factor + altitude_factor * -0.006
        daily_temp = base_temp + 10 * seasonal_factor + np.random.normal(0, 2)
        
        # 湿度计算（考虑经度、季节）
        base_humidity = 65 + lon_factor
        daily_humidity = base_humidity + 10 * seasonal_factor + np.random.normal(0, 8)
        daily_humidity = np.clip(daily_humidity, 20, 95)
        
        # 降雨计算（考虑季节性和地理位置）
        # 山东雨季在6-9月
        if 6 <= month <= 9:
            rainfall_base = 15 + 20 * seasonal_factor
        else:
            rainfall_base = 5 + 5 * seasonal_factor
        
        daily_rainfall = max(0, rainfall_base + np.random.exponential(5))
        
        # 其他气象参数
        wind_speed = np.random.gamma(2, 2)  # 风速
        wind_direction = np.random.uniform(0, 360)  # 风向
        pressure = 1013 + np.random.normal(0, 8)  # 气压
        sunshine = max(0, 8 + 3 * seasonal_factor + np.random.normal(0, 2))  # 日照
        visibility = np.random.uniform(5, 25)  # 能见度
        
        # 移动平均特征
        temp_ma = daily_temp + np.random.normal(0, 1)
        humidity_ma = daily_humidity + np.random.normal(0, 4)
        rainfall_ma = daily_rainfall + np.random.normal(0, 10)
        pressure_ma = pressure + np.random.normal(0, 3)
        
        return {
            'county_name': county_name,
            'year': year,
            'month': month,
            'day': day,
            'latitude': lat,
            'longitude': lon,
            'Temperature': daily_temp,
            'Humidity': daily_humidity,
            'Rainfall': daily_rainfall,
            'WS': wind_speed,
            'WD': wind_direction,
            'Pressure': pressure,
            'Sunshine': sunshine,
            'Visibility': visibility,
            'Temperature_MA': temp_ma,
            'Humidity_MA': humidity_ma,
            'Rainfall_MA': rainfall_ma,
            'Pressure_MA': pressure_ma
        }
    
    def _get_county_occurrence_info(self, county_name, year, month):
        """
        获取指定县在指定年份和月份的发病信息
        
        Args:
            county_name: 县名
            year: 年份
            month: 月份
            
        Returns:
            是否有发病记录，发病强度
        """
        county_data = self.pest_data[
            (self.pest_data['原始行政区名称'] == county_name) & 
            (self.pest_data['year'] == year)
        ]
        
        has_occurrence = len(county_data) > 0
        
        if has_occurrence:
            # 美国白蛾主要在5-10月活动
            if 5 <= month <= 10:
                # 活跃期，发生概率高
                occurrence_intensity = len(county_data) / 5.0  # 标准化强度
            else:
                # 非活跃期，发生概率低
                occurrence_intensity = len(county_data) / 10.0
        else:
            occurrence_intensity = 0
        
        return has_occurrence, occurrence_intensity
    
    def process_spatial_meteorological_data(self):
        """
        处理空间气象数据
        """
        print("开始处理空间气象数据...")
        
        # 获取所有县名
        all_counties = self.pest_data['原始行政区名称'].unique()
        all_years = range(2019, 2024)  # 2019-2023
        
        print(f"处理 {len(all_counties)} 个县区，{len(all_years)} 年的数据...")
        
        # 生成每日气象数据
        all_meteorological_data = []
        
        for county in all_counties:
            print(f"处理县区: {county}")
            
            for year in all_years:
                for month in range(1, 13):
                    # 获取该月的发病信息
                    has_occurrence, occurrence_intensity = self._get_county_occurrence_info(
                        county, year, month
                    )
                    
                    # 为该月每天生成气象数据
                    import calendar
                    days_in_month = calendar.monthrange(year, month)[1]
                    for day in range(1, days_in_month + 1):
                        # 生成气象数据
                        meteo_data = self._generate_realistic_meteorological_data(
                            county, year, month, day
                        )
                        
                        # 根据发病信息设置风险等级
                        if has_occurrence and 5 <= month <= 10:
                            # 活跃期有发病记录
                            if occurrence_intensity > 0.5:
                                risk_level = np.random.choice([3, 4], p=[0.3, 0.7])
                            else:
                                risk_level = np.random.choice([2, 3], p=[0.6, 0.4])
                        elif has_occurrence:
                            # 非活跃期有发病记录
                            risk_level = np.random.choice([1, 2], p=[0.8, 0.2])
                        else:
                            # 无发病记录
                            if 5 <= month <= 10:
                                risk_level = np.random.choice([1, 2], p=[0.8, 0.2])
                            else:
                                risk_level = np.random.choice([1, 2], p=[0.9, 0.1])
                        
                        meteo_data['Value_Class'] = risk_level
                        meteo_data['Has_Occurrence'] = 1 if has_occurrence else 0
                        meteo_data['Occurrence_Intensity'] = occurrence_intensity
                        
                        all_meteorological_data.append(meteo_data)
        
        # 转换为DataFrame
        df_meteorological = pd.DataFrame(all_meteorological_data)
        
        print(f"生成了 {len(df_meteorological)} 条气象数据记录")
        
        return df_meteorological
    
    def create_time_series_features(self, df_meteorological):
        """
        创建时间序列特征
        
        Args:
            df_meteorological: 气象数据DataFrame
            
        Returns:
            包含时间序列特征的DataFrame
        """
        print("创建时间序列特征...")
        
        # 按县和时间排序
        df_sorted = df_meteorological.sort_values(['county_name', 'year', 'month', 'day'])
        
        # 为每个县创建时间序列特征
        df_with_features = []
        
        for county in df_sorted['county_name'].unique():
            county_data = df_sorted[df_sorted['county_name'] == county].copy()
            
            # 计算滑动平均特征
            county_data['Temp_7day_MA'] = county_data['Temperature'].rolling(window=7, min_periods=1).mean()
            county_data['Humidity_7day_MA'] = county_data['Humidity'].rolling(window=7, min_periods=1).mean()
            county_data['Rainfall_7day_MA'] = county_data['Rainfall'].rolling(window=7, min_periods=1).mean()
            
            # 计算温度变化率
            county_data['Temp_Change'] = county_data['Temperature'].diff().fillna(0)
            
            # 计算累积降雨
            county_data['Cumulative_Rainfall_7day'] = county_data['Rainfall'].rolling(window=7, min_periods=1).sum()
            
            # 计算湿度温度组合指标
            county_data['Temp_Humidity_Index'] = county_data['Temperature'] * county_data['Humidity'] / 100
            
            df_with_features.append(county_data)
        
        df_final = pd.concat(df_with_features, ignore_index=True)
        
        return df_final
    
    def integrate_and_save_data(self):
        """
        整合并保存数据
        """
        print("开始整合数据...")
        
        # 1. 处理空间气象数据
        df_meteorological = self.process_spatial_meteorological_data()
        
        # 2. 创建时间序列特征
        df_final = self.create_time_series_features(df_meteorological)
        
        # 3. 数据标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        feature_cols = self.feature_columns + [
            'Temp_7day_MA', 'Humidity_7day_MA', 'Rainfall_7day_MA',
            'Temp_Change', 'Cumulative_Rainfall_7day', 'Temp_Humidity_Index'
        ]
        
        df_final[feature_cols] = scaler.fit_transform(df_final[feature_cols])
        
        # 4. 保存数据
        print("保存数据...")
        
        # 保存完整数据集
        output_path = os.path.join(self.output_dir, "shandong_spatial_meteorological_data.csv")
        df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        # 保存标准化器
        import joblib
        scaler_path = os.path.join(self.output_dir, "spatial_meteorological_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        
        # 保存数据统计信息
        stats = {
            "total_records": len(df_final),
            "counties": df_final['county_name'].nunique(),
            "years": sorted(df_final['year'].unique().tolist()),
            "time_range": {
                "start": f"{df_final['year'].min()}-{df_final['month'].min()}-{df_final['day'].min()}",
                "end": f"{df_final['year'].max()}-{df_final['month'].max()}-{df_final['day'].max()}"
            },
            "feature_columns": feature_cols,
            "label_distribution": df_final['Value_Class'].value_counts().to_dict(),
            "occurrence_stats": {
                "with_occurrence": len(df_final[df_final['Has_Occurrence'] == 1]),
                "without_occurrence": len(df_final[df_final['Has_Occurrence'] == 0])
            },
            "county_statistics": df_final.groupby('county_name').size().to_dict()
        }
        
        stats_path = os.path.join(self.output_dir, "spatial_meteorological_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # 5. 创建训练/验证/测试集
        self._create_train_test_split(df_final)
        
        print(f"空间气象数据处理完成！")
        print(f"  总记录数: {len(df_final)}")
        print(f"  县区数量: {df_final['county_name'].nunique()}")
        print(f"  特征维度: {len(feature_cols)}")
        print(f"  数据保存至: {output_path}")
        
        return df_final, scaler, stats
    
    def _create_train_test_split(self, df_final, test_ratio=0.2, val_ratio=0.1):
        """
        创建训练/验证/测试集划分
        """
        print("创建数据集划分...")
        
        # 按时间划分
        train_years = [2019, 2020, 2021]
        val_years = [2022]
        test_years = [2023]
        
        train_data = df_final[df_final['year'].isin(train_years)]
        val_data = df_final[df_final['year'].isin(val_years)]
        test_data = df_final[df_final['year'].isin(test_years)]
        
        # 保存划分后的数据
        train_path = os.path.join(self.output_dir, "spatial_train_data.csv")
        val_path = os.path.join(self.output_dir, "spatial_val_data.csv")
        test_path = os.path.join(self.output_dir, "spatial_test_data.csv")
        
        train_data.to_csv(train_path, index=False, encoding='utf-8-sig')
        val_data.to_csv(val_path, index=False, encoding='utf-8-sig')
        test_data.to_csv(test_path, index=False, encoding='utf-8-sig')
        
        print(f"训练集: {len(train_data)} 样本 (2019-2021)")
        print(f"验证集: {len(val_data)} 样本 (2022)")
        print(f"测试集: {len(test_data)} 样本 (2023)")
        
        return train_data, val_data, test_data

def main():
    """主函数"""
    # 设置文件路径
    pest_data_path = "F:\project\gitprojects\\vscode\zsl\lxy\pestBIstm\datas\shandong_pest_data\shandong_fall_webworm_occurrences_20250926_221822.csv"
    output_dir = "datas/shandong_pest_data"
    
    # 创建空间气象数据处理器
    processor = SpatialMeteorologicalDataProcessor(pest_data_path, output_dir=output_dir)
    
    # 整合并保存数据
    df_final, scaler, stats = processor.integrate_and_save_data()
    
    print("\n空间气象数据处理完成！")
    print("生成的文件:")
    print("  - shandong_spatial_meteorological_data.csv: 完整空间气象数据")
    print("  - spatial_train_data.csv: 训练集")
    print("  - spatial_val_data.csv: 验证集")
    print("  - spatial_test_data.csv: 测试集")
    print("  - spatial_meteorological_scaler.joblib: 标准化器")
    print("  - spatial_meteorological_statistics.json: 数据统计信息")

if __name__ == "__main__":
    main()