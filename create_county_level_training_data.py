#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建山东省县域粒度美国白蛾第一代（5-6月）发病情况训练数据集
整合发病程度数据、县边界数据和栅格气象数据
"""

import pandas as pd
import numpy as np
import json
import geopandas as gpd
from shapely.geometry import Point, shape
import warnings
warnings.filterwarnings('ignore')

class CountyLevelTrainingDataBuilder:
    def __init__(self):
        self.pest_data = None
        self.geojson_data = None
        self.meteo_data = None
        self.training_data = None

    def load_pest_occurrence_data(self, excel_path):
        """加载美国白蛾发病程度数据"""
        print("Loading pest occurrence data...")

        # 读取Excel文件
        df = pd.read_excel(excel_path)

        # 重新命名列以便处理
        df.columns = ['Year', 'City', 'County', 'FirstGen_Severity_MayJun',
                     'SecondGen_Severity_JulAug', 'ThirdGen_Severity_SepOct']

        # 只保留第一代（5-6月）的数据
        df_first_gen = df[['Year', 'City', 'County', 'FirstGen_Severity_MayJun']].copy()
        df_first_gen = df_first_gen.rename(columns={'FirstGen_Severity_MayJun': 'Severity_Level'})

        # 添加二分类标签
        df_first_gen['Has_Occurrence'] = (df_first_gen['Severity_Level'] > 0).astype(int)

        print(f"Pest occurrence data loaded: {len(df_first_gen)} records")
        print(f"Years: {sorted(df_first_gen['Year'].unique())}")
        print(f"Counties: {df_first_gen['County'].nunique()}")
        print(f"Severity distribution:")
        print(df_first_gen['Severity_Level'].value_counts().sort_index())

        self.pest_data = df_first_gen
        return df_first_gen

    def load_county_boundaries(self, geojson_path):
        """加载县边界数据"""
        print("Loading county boundary data...")

        # 读取GeoJSON
        with open(geojson_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)

        # 转换为GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])

        # 清理县名
        gdf['county_name_clean'] = gdf['name'].str.strip()

        print(f"County boundary data loaded: {len(gdf)} counties")

        self.geojson_data = gdf
        return gdf

    def load_meteorological_data(self, csv_path):
        """加载栅格气象数据"""
        print("Loading meteorological data...")

        # 分块读取大文件
        chunks = []
        chunk_size = 100000

        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            chunks.append(chunk)
            print(f"Loaded chunk: {len(chunk)} records")

        df = pd.concat(chunks, ignore_index=True)

        print(f"Meteorological data loaded: {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df['year'].min()}-{df['month'].min()} to {df['year'].max()}-{df['month'].max()}")

        self.meteo_data = df
        return df

    def extract_first_generation_meteo(self):
        """提取第一代（5-6月）气象数据并计算统计特征"""
        print("Extracting first generation meteorological features...")

        # 筛选5-6月的数据
        first_gen_meteo = self.meteo_data[
            (self.meteo_data['month'].isin([5, 6]))
        ].copy()

        print(f"First generation meteorological records: {len(first_gen_meteo)}")

        # 按县、年分组计算统计特征
        feature_columns = ['Temperature', 'Humidity', 'Rainfall', 'WS', 'Pressure',
                          'Sunshine', 'Visibility', 'Temp_Humidity_Index']

        # 移动平均列
        ma_columns = ['Temperature_MA', 'Humidity_MA', 'Rainfall_MA', 'Pressure_MA']

        # 7天移动平均列
        ma7_columns = ['Temp_7day_MA', 'Humidity_7day_MA', 'Rainfall_7day_MA']

        # 其他特征列
        other_columns = ['Temp_Change', 'Cumulative_Rainfall_7day']

        all_feature_cols = feature_columns + ma_columns + ma7_columns + other_columns

        # 计算统计特征
        stats_list = []

        for (county, year), group in first_gen_meteo.groupby(['county_name', 'year']):
            stats_dict = {
                'County': county,
                'Year': year,
                'Meteo_Record_Count': len(group)
            }

            # 对每个特征计算统计量
            for col in all_feature_cols:
                if col in group.columns:
                    stats_dict[f'{col}_mean'] = group[col].mean()
                    stats_dict[f'{col}_std'] = group[col].std()
                    stats_dict[f'{col}_min'] = group[col].min()
                    stats_dict[f'{col}_max'] = group[col].max()
                    stats_dict[f'{col}_median'] = group[col].median()

            stats_list.append(stats_dict)

        meteo_features = pd.DataFrame(stats_list)
        print(f"Meteorological features computed for {len(meteo_features)} county-year combinations")

        return meteo_features

    def merge_datasets(self):
        """整合发病数据和气象特征数据"""
        print("Merging pest occurrence and meteorological data...")

        # 提取第一代气象特征
        meteo_features = self.extract_first_generation_meteo()

        # 合并发病数据和气象特征
        merged_data = self.pest_data.merge(
            meteo_features,
            on=['County', 'Year'],
            how='inner'
        )

        print(f"Merged dataset created: {len(merged_data)} records")
        print(f"Years in merged data: {sorted(merged_data['Year'].unique())}")
        print(f"Counties in merged data: {merged_data['County'].nunique()}")

        # 移除没有气象数据的记录
        initial_count = len(merged_data)
        merged_data = merged_data.dropna(subset=['Meteo_Record_Count'])
        final_count = len(merged_data)

        print(f"Removed {initial_count - final_count} records without meteorological data")

        self.training_data = merged_data
        return merged_data

    def add_spatial_features(self):
        """添加空间特征"""
        print("Adding spatial features...")

        if self.geojson_data is None:
            print("Warning: No boundary data available for spatial features")
            return self.training_data

        # 计算县的质心坐标
        self.geojson_data['centroid_lat'] = self.geojson_data.geometry.centroid.y
        self.geojson_data['centroid_lon'] = self.geojson_data.geometry.centroid.x

        # 将坐标添加到训练数据
        training_with_spatial = self.training_data.merge(
            self.geojson_data[['name', 'centroid_lat', 'centroid_lon']],
            left_on='County',
            right_on='name',
            how='left'
        )

        # 添加纬度带特征
        training_with_spatial['Latitude_Zone'] = pd.cut(
            training_with_spatial['centroid_lat'],
            bins=5,
            labels=['South', 'South-Central', 'Central', 'North-Central', 'North']
        )

        print(f"Spatial features added for {training_with_spatial['centroid_lat'].notna().sum()} counties")

        self.training_data = training_with_spatial
        return training_with_spatial

    def create_final_dataset(self):
        """创建最终训练数据集"""
        print("Creating final training dataset...")

        # 选择最终特征列
        feature_columns = []

        # 基础信息
        feature_columns.extend(['Year', 'County', 'City'])

        # 标签
        feature_columns.extend(['Severity_Level', 'Has_Occurrence'])

        # 气象特征统计量
        meteo_base_features = ['Temperature', 'Humidity', 'Rainfall', 'Pressure', 'Temp_Humidity_Index']
        stats = ['mean', 'std', 'min', 'max', 'median']

        for feature in meteo_base_features:
            for stat in stats:
                col_name = f'{feature}_{stat}'
                if col_name in self.training_data.columns:
                    feature_columns.append(col_name)

        # 空间特征
        if 'centroid_lat' in self.training_data.columns:
            feature_columns.extend(['centroid_lat', 'centroid_lon'])

        # 创建最终数据集
        final_data = self.training_data[feature_columns].copy()

        # 重命名列以便使用
        final_data = final_data.rename(columns={
            'centroid_lat': 'Latitude',
            'centroid_lon': 'Longitude'
        })

        # 移除包含缺失值的行
        final_data = final_data.dropna()

        print(f"Final training dataset created: {len(final_data)} records")
        print(f"Features: {len(final_data.columns)}")
        print("\nDataset summary:")
        print(f"- Years: {sorted(final_data['Year'].unique())}")
        print(f"- Counties: {final_data['County'].nunique()}")
        print(f"- Positive cases (severity > 0): {(final_data['Severity_Level'] > 0).sum()}")
        print(f"- Negative cases (severity = 0): {(final_data['Severity_Level'] == 0).sum()}")

        return final_data

    def save_datasets(self, output_dir='datas/shandong_pest_data'):
        """保存训练数据集"""
        print("Saving training datasets...")

        if self.training_data is None:
            print("Error: No training data to save")
            return

        # 创建最终数据集
        final_data = self.create_final_dataset()

        # 按年份分割数据
        years = sorted(final_data['Year'].unique())

        # 训练集（2019-2022）
        train_years = [2019, 2020, 2021, 2022]
        train_data = final_data[final_data['Year'].isin(train_years)]

        # 验证集（2023）
        val_data = final_data[final_data['Year'] == 2023]

        # 测试集（2024）
        test_data = final_data[final_data['Year'] == 2024]

        # 保存数据集
        train_data.to_csv(f'{output_dir}/county_level_firstgen_train.csv', index=False, encoding='utf-8-sig')
        val_data.to_csv(f'{output_dir}/county_level_firstgen_val.csv', index=False, encoding='utf-8-sig')
        test_data.to_csv(f'{output_dir}/county_level_firstgen_test.csv', index=False, encoding='utf-8-sig')

        # 保存完整数据集
        final_data.to_csv(f'{output_dir}/county_level_firstgen_complete.csv', index=False, encoding='utf-8-sig')

        # 保存数据集信息
        dataset_info = {
            'description': '山东省县域粒度美国白蛾第一代（5-6月）发病情况训练数据集',
            'created_time': pd.Timestamp.now().isoformat(),
            'total_records': int(len(final_data)),
            'years': [int(year) for year in years],
            'counties': int(final_data['County'].nunique()),
            'features': list(final_data.columns),
            'split_stats': {
                'train': {'years': train_years, 'records': int(len(train_data))},
                'validation': {'years': [2023], 'records': int(len(val_data))},
                'test': {'years': [2024], 'records': int(len(test_data))}
            }
        }

        with open(f'{output_dir}/county_level_firstgen_dataset_info.json', 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)

        print(f"Datasets saved to {output_dir}/")
        print(f"- Training set: {len(train_data)} records (2019-2022)")
        print(f"- Validation set: {len(val_data)} records (2023)")
        print(f"- Test set: {len(test_data)} records (2024)")
        print(f"- Complete dataset: {len(final_data)} records")

def main():
    """主函数"""
    print("=== 创建山东省县域美国白蛾第一代发病情况训练数据集 ===\n")

    # 初始化数据构建器
    builder = CountyLevelTrainingDataBuilder()

    # 数据文件路径
    excel_path = 'datas/shandong_pest_data/发病情况.xlsx'
    geojson_path = 'datas/shandong_pest_data/shandong.json'
    meteo_path = 'datas/shandong_pest_data/shandong_spatial_meteorological_data.csv'

    try:
        # 1. 加载发病数据
        builder.load_pest_occurrence_data(excel_path)

        # 2. 加载县边界数据
        builder.load_county_boundaries(geojson_path)

        # 3. 加载气象数据
        builder.load_meteorological_data(meteo_path)

        # 4. 整合数据集
        builder.merge_datasets()

        # 5. 添加空间特征
        builder.add_spatial_features()

        # 6. 创建并保存最终数据集
        builder.save_datasets()

        print("\n=== 数据集创建完成 ===")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()