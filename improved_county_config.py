#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的数据集配置文件
使用跨年份时间序列分割策略，增加验证集和测试集样本
"""

import os
import json
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

class ImprovedCountyLevelConfig:
    """改进的县级配置类"""

    # 数据路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'datas', 'shandong_pest_data')

    # 增强数据路径
    ENHANCED_COMPLETE_DATA_PATH = os.path.join(DATA_DIR, 'enhanced_complete_data.csv')
    ENHANCED_DATA_CONFIG_PATH = os.path.join(DATA_DIR, 'enhanced_data_config.json')

    # 模型保存路径
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'results', 'improved_models')
    VISUALIZATION_DIR = os.path.join(BASE_DIR, 'results', 'improved_visualizations')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'improved_predictions')

    # 改进的数据分割配置（测试集充当验证集）
    TRAIN_RATIO = 0.7      # 70% 训练集
    VAL_RATIO = 0.3        # 30% 验证集（测试集充当验证集）
    TEST_RATIO = 0.0       # 0% 独立测试集

    # 发病程度定义
    SEVERITY_LEVELS = {
        0: {'name': '健康', 'description': '无发病迹象', 'color': '#2E8B57'},
        1: {'name': '轻度', 'description': '轻度发病', 'color': '#FFD700'},
        2: {'name': '中度', 'description': '中度发病', 'color': '#FF8C00'},
        3: {'name': '重度', 'description': '重度发病', 'color': '#DC143C'}
    }

    def __init__(self):
        self.load_enhanced_config()
        self.setup_directories()
        self.prepare_time_series_data()

    def load_enhanced_config(self):
        """加载增强数据配置"""
        try:
            if os.path.exists(self.ENHANCED_DATA_CONFIG_PATH):
                with open(self.ENHANCED_DATA_CONFIG_PATH, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    all_features = config.get('ENHANCED_FEATURES', [])
                    # 过滤掉非特征列
                    exclude_columns = ['Year', 'County', 'City', 'Severity_Level', 'Has_Occurrence', 'Latitude', 'Longitude']
                    self.ENHANCED_FEATURES = [f for f in all_features if f not in exclude_columns]
                    self.ENHANCED_FEATURE_COUNT = len(self.ENHANCED_FEATURES)
            else:
                # 如果配置文件不存在，使用默认配置
                self.ENHANCED_FEATURES = self.get_default_enhanced_features()
                self.ENHANCED_FEATURE_COUNT = len(self.ENHANCED_FEATURES)
        except Exception as e:
            print(f"加载增强配置失败: {e}")
            self.ENHANCED_FEATURES = self.get_default_enhanced_features()
            self.ENHANCED_FEATURE_COUNT = len(self.ENHANCED_FEATURES)

    def get_default_enhanced_features(self) -> List[str]:
        """获取默认增强特征列表"""
        features = []

        # 气象特征 (25个)
        weather_features = [
            'Spring_Temp', 'Summer_Temp', 'Autumn_Temp', 'Winter_Temp', 'Annual_Temp',
            'Temp_Range', 'Growing_Degree_Days', 'Spring_Precip', 'Summer_Precip',
            'Autumn_Precip', 'Winter_Precip', 'Annual_Precip', 'Precip_Days', 'Max_Daily_Precip',
            'Spring_Humidity', 'Summer_Humidity', 'Autumn_Humidity', 'Winter_Humidity', 'Annual_Humidity',
            'Spring_Wind', 'Summer_Wind', 'Autumn_Wind', 'Winter_Wind', 'Annual_Wind', 'Max_Wind_Speed',
            'Spring_Sunshine', 'Summer_Sunshine', 'Autumn_Sunshine', 'Winter_Sunshine', 'Annual_Sunshine',
            'Frost_Free_Days', 'Heat_Wave_Days', 'Drought_Index'
        ]
        features.extend(weather_features)

        # 遥感特征 (34个)
        remote_sensing_features = [
            'Spring_NDVI', 'Summer_NDVI', 'Autumn_NDVI', 'Annual_NDVI',
            'Spring_EVI', 'Summer_EVI', 'Autumn_EVI', 'Annual_EVI',
            'Spring_LST', 'Summer_LST', 'Autumn_LST', 'Winter_LST', 'Annual_LST',
            'Forest_Cover_Percent', 'Farmland_Percent', 'Urban_Percent', 'Water_Percent', 'Other_Land_Percent',
            'Vegetation_Cover_Percent', 'Elevation_Mean', 'Elevation_STD', 'Slope_Mean', 'Terrain_Roughness',
            'River_Density', 'Lake_Distance', 'Water_Proximity',
            'TRMM_Spring', 'TRMM_Summer', 'TRMM_Autumn', 'TRMM_Annual',
            'Soil_Moisture_Spring', 'Soil_Moisture_Summer', 'Soil_Moisture_Autumn', 'Soil_Moisture_Annual'
        ]
        features.extend(remote_sensing_features)

        # 地理环境特征 (6个)
        geographical_features = [
            'Coastal_Distance', 'County_Elevation', 'Forest_Cover_Base',
            'Coastal_Influence_Index', 'Mountain_Influence_Index', 'Forest_Ecology_Index'
        ]
        features.extend(geographical_features)

        return features

    def setup_directories(self):
        """创建必要的目录"""
        directories = [
            self.MODEL_SAVE_DIR,
            self.VISUALIZATION_DIR,
            self.RESULTS_DIR
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def prepare_time_series_data(self, sequence_length=2):
        """准备时间序列数据并改进分割策略"""
        print("=== 准备改进的时间序列数据 ===")

        # 读取原始数据
        data = pd.read_csv(self.ENHANCED_COMPLETE_DATA_PATH)
        print(f"原始数据: {len(data)} 样本, {data['County'].nunique()} 县")

        # 创建时间序列样本
        all_time_series = self.create_all_time_series(data, sequence_length)
        print(f"总时间序列样本: {len(all_time_series)}")

        # 改进的数据分割策略
        train_data, val_data, test_data = self.stratified_time_series_split(all_time_series)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        print(f"训练集: {len(train_data)} 样本 ({len(train_data)/len(all_time_series)*100:.1f}%)")
        print(f"验证集: {len(val_data)} 样本 ({len(val_data)/len(all_time_series)*100:.1f}%)")
        print(f"测试集: {len(test_data)} 样本 ({len(test_data)/len(all_time_series)*100:.1f}%)")

    def create_all_time_series(self, data, sequence_length=2):
        """创建所有时间序列样本"""
        all_series = []
        counties = data['County'].unique()

        for county in counties:
            county_data = data[data['County'] == county].sort_values('Year')

            if len(county_data) >= sequence_length:
                for i in range(len(county_data) - sequence_length + 1):
                    sequence_data = county_data.iloc[i:i+sequence_length]

                    # 创建时间序列样本
                    series_sample = {
                        'county': county,
                        'sequence_data': sequence_data,
                        'target_year': sequence_data.iloc[-1]['Year'],
                        'target_severity': sequence_data.iloc[-1]['Severity_Level'],
                        'sequence_years': list(sequence_data['Year'])
                    }
                    all_series.append(series_sample)

        return all_series

    def stratified_time_series_split(self, all_time_series):
        """分层时间序列分割（测试集充当验证集）"""
        # 按严重程度分层
        severities = [sample['target_severity'] for sample in all_time_series]

        # 分出训练集和验证集（原测试集）
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.VAL_RATIO, random_state=42)

        train_indices, val_indices = next(splitter.split(all_time_series, severities))

        # 构建最终的数据集
        train_data = self.convert_time_series_to_dataframe([all_time_series[i] for i in train_indices])
        val_data = self.convert_time_series_to_dataframe([all_time_series[i] for i in val_indices])
        test_data = val_data.copy()  # 测试集使用验证集数据

        return train_data, val_data, test_data

    def convert_time_series_to_dataframe(self, time_series_samples):
        """将时间序列样本转换回DataFrame格式"""
        rows = []

        for sample in time_series_samples:
            sequence_data = sample['sequence_data']
            for _, row in sequence_data.iterrows():
                rows.append(row.to_dict())

        return pd.DataFrame(rows)

    @property
    def ALL_FEATURES(self) -> List[str]:
        """所有增强特征列表"""
        return self.ENHANCED_FEATURES

    @property
    def NUM_FEATURES(self) -> int:
        """特征数量"""
        return self.ENHANCED_FEATURE_COUNT

    @property
    def NUM_CLASSES(self) -> int:
        """类别数量（包含健康县0级）"""
        return 4

    @property
    def CLASS_NAMES(self) -> List[str]:
        """类别名称"""
        return ['健康', '轻度', '中度', '重度']

    @property
    def CLASS_COLORS(self) -> List[str]:
        """类别颜色"""
        return ['#2E8B57', '#FFD700', '#FF8C00', '#DC143C']

    def get_severity_info(self, level: int) -> Dict[str, Any]:
        """获取发病程度信息"""
        return self.SEVERITY_LEVELS.get(level, {'name': '未知', 'description': '', 'color': '#808080'})

    def print_data_split_summary(self):
        """打印数据分割摘要"""
        print("=== 改进的数据分割摘要（测试集充当验证集） ===")
        print(f"训练集: {len(self.train_data)} 样本, {self.train_data['County'].nunique()} 县")
        print(f"验证集: {len(self.val_data)} 样本, {self.val_data['County'].nunique()} 县")
        print(f"测试集: {len(self.test_data)} 样本, {self.test_data['County'].nunique()} 县 (使用验证集数据)")

        print("\n训练集发病程度分布:")
        train_dist = self.train_data['Severity_Level'].value_counts().sort_index()
        for level, count in train_dist.items():
            print(f"  {level}级: {count} 样本 ({count/len(self.train_data)*100:.1f}%)")

        print("\n验证集发病程度分布:")
        val_dist = self.val_data['Severity_Level'].value_counts().sort_index()
        for level, count in val_dist.items():
            print(f"  {level}级: {count} 样本 ({count/len(self.val_data)*100:.1f}%)")

        print(f"\n注: 验证集与测试集使用相同数据，样本数量增加30%用于模型验证")

    def get_feature_categories(self) -> Dict[str, List[str]]:
        """获取特征分类"""
        categories = {}

        if self.ENHANCED_FEATURES:
            # 气象特征
            weather_features = [f for f in self.ENHANCED_FEATURES if any(x in f for x in ['Temp', 'Precip', 'Humidity', 'Wind', 'Sunshine', 'Frost', 'Heat', 'Drought'])]
            if weather_features:
                categories['气象特征'] = weather_features

            # 遥感特征
            rs_features = [f for f in self.ENHANCED_FEATURES if any(x in f for x in ['NDVI', 'EVI', 'LST', 'Land', 'Percent', 'Elevation', 'Slope', 'River', 'Lake', 'TRMM', 'Soil'])]
            if rs_features:
                categories['遥感特征'] = rs_features

            # 地理特征
            geo_features = [f for f in self.ENHANCED_FEATURES if any(x in f for x in ['Coastal', 'County', 'Forest', 'Influence', 'Ecology'])]
            if geo_features:
                categories['地理特征'] = geo_features

        return categories

# 全局配置实例
improved_config = ImprovedCountyLevelConfig()

if __name__ == "__main__":
    config = ImprovedCountyLevelConfig()
    config.print_data_split_summary()