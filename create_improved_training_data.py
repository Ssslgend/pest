#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于完整发病情况数据生成训练数据集
使用新的数据处理逻辑：缺失月份直接设为未发生
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import json

class ImprovedTrainingDataGenerator:
    """改进的训练数据生成器"""

    def __init__(self, output_dir="datas/shandong_pest_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_complete_occurrence_data(self):
        """加载完整发病数据"""
        print("加载完整发病数据...")
        data_file = os.path.join(self.output_dir, 'complete_shandong_occurrence_data.csv')

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"完整发病数据文件不存在: {data_file}")

        df = pd.read_csv(data_file, encoding='utf-8-sig')
        print(f"加载发病数据: {df.shape}")
        print(f"县区数: {df['county_name'].nunique()}")
        print(f"年份范围: {df['year'].min()}-{df['year'].max()}")

        return df

    def load_meteorological_data(self):
        """加载气象数据"""
        print("加载气象数据...")
        meteo_file = "./datas/shandong_pest_data/shandong_spatial_meteorological_data.csv"

        if not os.path.exists(meteo_file):
            raise FileNotFoundError(f"气象数据文件不存在: {meteo_file}")

        df = pd.read_csv(meteo_file, encoding='utf-8-sig')
        print(f"加载气象数据: {df.shape}")
        print(f"气象数据县区数: {df['county_name'].nunique()}")

        return df

    def merge_occurrence_and_meteo_data(self, df_occurrence, df_meteo):
        """合并发病数据和气象数据"""
        print("合并发病数据和气象数据...")

        # 首先聚合气象数据到月度
        monthly_meteo = df_meteo.groupby(['county_name', 'year', 'month']).agg({
            'Temperature': ['mean', 'std', 'min', 'max'],
            'Humidity': ['mean', 'std', 'min', 'max'],
            'Rainfall': ['mean', 'sum', 'min', 'max'],
            'WS': ['mean', 'std'],
            'WD': ['mean'],
            'Pressure': ['mean', 'std'],
            'Sunshine': ['mean', 'std'],
            'Visibility': ['mean', 'std'],
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index()

        # 扁平化列名
        monthly_meteo.columns = [
            'county_name', 'year', 'month',
            'Temperature_mean', 'Temperature_std', 'Temperature_min', 'Temperature_max',
            'Humidity_mean', 'Humidity_std', 'Humidity_min', 'Humidity_max',
            'Rainfall_mean', 'Rainfall_sum', 'Rainfall_min', 'Rainfall_max',
            'WS_mean', 'WS_std',
            'WD_mean',
            'Pressure_mean', 'Pressure_std',
            'Sunshine_mean', 'Sunshine_std',
            'Visibility_mean', 'Visibility_std',
            'latitude', 'longitude'
        ]

        print(f"月度气象数据: {monthly_meteo.shape}")

        # 合并数据
        merged_df = df_occurrence.merge(
            monthly_meteo,
            on=['county_name', 'year', 'month'],
            how='inner'  # 只保留有气象数据的记录
        )

        print(f"合并后数据: {merged_df.shape}")

        return merged_df

    def add_engineered_features(self, df):
        """添加工程特征"""
        print("添加工程特征...")

        # 温度湿度指数
        df['Temp_Humidity_Index'] = df['Temperature_mean'] * df['Humidity_mean'] / 100

        # 累积降雨特征（3个月）
        df = df.sort_values(['county_name', 'year', 'month'])
        df['Cumulative_Rainfall_3month'] = df.groupby('county_name')['Rainfall_sum'].rolling(
            window=3, min_periods=1
        ).sum().reset_index(0, drop=True)

        # 温度变化趋势
        df['Temp_Trend'] = df.groupby('county_name')['Temperature_mean'].diff().fillna(0)

        # 滞后特征（前一个月）
        df['Temperature_lag1'] = df.groupby('county_name')['Temperature_mean'].shift(1)
        df['Humidity_lag1'] = df.groupby('county_name')['Humidity_mean'].shift(1)
        df['Rainfall_lag1'] = df.groupby('county_name')['Rainfall_sum'].shift(1)

        # 填充滞后特征缺失值
        lag_features = ['Temperature_lag1', 'Humidity_lag1', 'Rainfall_lag1']
        for feature in lag_features:
            if feature in df.columns:
                df[feature] = df.groupby('county_name')[feature].bfill()

        # 美国白蛾活动水平（基于月份）
        df['Moth_Activity_Level'] = df.apply(self._get_moth_activity_level, axis=1)

        print(f"添加工程特征后: {df.shape}")
        return df

    def _get_moth_activity_level(self, row):
        """根据月份确定美国白蛾活动水平"""
        month = row['month']
        temperature = row.get('Temperature_mean', 15)  # 默认温度

        if month in [11, 12, 1, 2, 3]:
            return 0  # 越冬期
        elif month in [4, 5]:
            return 1 if temperature > 10 else 0  # 蛹期和成虫期
        elif month in [6, 7]:
            return 3 if temperature > 15 else 2  # 幼虫期高发期
        elif month in [8, 9]:
            return 2 if temperature > 15 else 1  # 危害期
        else:  # 10月
            return 1 if temperature > 10 else 0  # 化蛹期

    def create_datasets(self, df):
        """创建训练数据集"""
        print("创建训练数据集...")

        # 检查可用的特征列
        available_columns = df.columns.tolist()
        print(f"可用列名: {available_columns[:20]}...")  # 显示前20个列名

        # 选择特征列
        base_features = [
            'Temperature_mean', 'Temperature_std', 'Temperature_min', 'Temperature_max',
            'Humidity_mean', 'Humidity_std', 'Humidity_min', 'Humidity_max',
            'Rainfall_mean', 'Rainfall_sum', 'Rainfall_min', 'Rainfall_max',
            'WS_mean', 'WS_std', 'WD_mean',
            'Pressure_mean', 'Pressure_std',
            'Sunshine_mean', 'Sunshine_std',
            'Visibility_mean', 'Visibility_std'
        ]

        # 检查并添加地理特征
        geo_features = []
        if 'latitude' in available_columns:
            geo_features.append('latitude')
        if 'longitude' in available_columns:
            geo_features.append('longitude')

        # 检查并添加工程特征
        engineered_features = ['Season', 'Temp_Humidity_Index',
                              'Cumulative_Rainfall_3month', 'Temp_Trend',
                              'Temperature_lag1', 'Humidity_lag1', 'Rainfall_lag1',
                              'Moth_Activity_Level']

        feature_columns = base_features + geo_features + engineered_features
        feature_columns = [col for col in feature_columns if col in available_columns]

        print(f"最终使用特征数: {len(feature_columns)}")
        print(f"特征列: {feature_columns[:10]}...")

        # 选择需要的数据列
        selected_columns = feature_columns + [
            'county_name', 'year', 'month', 'Severity', 'Has_Occurrence',
            'Period', 'Data_Source'
        ]

        df_clean = df[selected_columns].dropna()
        print(f"清洗后数据: {df_clean.shape}")

        # 特征标准化
        scaler = StandardScaler()
        df_clean[feature_columns] = scaler.fit_transform(df_clean[feature_columns])

        # 按年份划分数据集
        years = sorted(df_clean['year'].unique())
        print(f"可用年份: {years}")

        # 按时间顺序划分
        train_years = [2019, 2020, 2021]  # 前3年训练
        val_years = [2022]  # 2022年验证
        test_years = [2023]  # 2023年测试

        train_data = df_clean[df_clean['year'].isin(train_years)]
        val_data = df_clean[df_clean['year'].isin(val_years)]
        test_data = df_clean[df_clean['year'].isin(test_years)]

        # 保存数据集
        train_path = os.path.join(self.output_dir, "improved_train.csv")
        val_path = os.path.join(self.output_dir, "improved_val.csv")
        test_path = os.path.join(self.output_dir, "improved_test.csv")
        complete_path = os.path.join(self.output_dir, "improved_complete_data.csv")

        train_data.to_csv(train_path, index=False, encoding='utf-8-sig')
        val_data.to_csv(val_path, index=False, encoding='utf-8-sig')
        test_data.to_csv(test_path, index=False, encoding='utf-8-sig')
        df_clean.to_csv(complete_path, index=False, encoding='utf-8-sig')

        # 保存标准化器
        scaler_path = os.path.join(self.output_dir, "improved_scaler.joblib")
        joblib.dump(scaler, scaler_path)

        print(f"数据集已保存:")
        print(f"  训练集: {train_path} ({len(train_data)} 样本)")
        print(f"  验证集: {val_path} ({len(val_data)} 样本)")
        print(f"  测试集: {test_path} ({len(test_data)} 样本)")
        print(f"  完整数据: {complete_path} ({len(df_clean)} 样本)")
        print(f"  标准化器: {scaler_path}")

        return train_data, val_data, test_data, feature_columns, scaler

    def generate_statistics(self, train_data, val_data, test_data, feature_columns):
        """生成数据统计报告"""
        print("生成数据统计报告...")

        def convert_to_serializable(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            return obj

        stats = {
            "data_info": {
                "total_samples": int(len(train_data) + len(val_data) + len(test_data)),
                "train_samples": int(len(train_data)),
                "val_samples": int(len(val_data)),
                "test_samples": int(len(test_data)),
                "feature_count": int(len(feature_columns)),
                "counties": int(train_data['county_name'].nunique()),
                "years": [convert_to_serializable(x) for x in sorted(train_data['year'].unique())]
            },
            "dataset_info": {
                "train_years": [2019, 2020, 2021],
                "val_years": [2022],
                "test_years": [2023]
            },
            "label_distribution": {
                "train": {
                    "severity_dist": {convert_to_serializable(k): int(v) for k, v in train_data['Severity'].value_counts().to_dict().items()},
                    "occurrence_dist": {convert_to_serializable(k): int(v) for k, v in train_data['Has_Occurrence'].value_counts().to_dict().items()},
                    "occurrence_rate": float(train_data['Has_Occurrence'].mean())
                },
                "val": {
                    "severity_dist": {convert_to_serializable(k): int(v) for k, v in val_data['Severity'].value_counts().to_dict().items()},
                    "occurrence_dist": {convert_to_serializable(k): int(v) for k, v in val_data['Has_Occurrence'].value_counts().to_dict().items()},
                    "occurrence_rate": float(val_data['Has_Occurrence'].mean())
                },
                "test": {
                    "severity_dist": {convert_to_serializable(k): int(v) for k, v in test_data['Severity'].value_counts().to_dict().items()},
                    "occurrence_dist": {convert_to_serializable(k): int(v) for k, v in test_data['Has_Occurrence'].value_counts().to_dict().items()},
                    "occurrence_rate": float(test_data['Has_Occurrence'].mean())
                }
            },
            "data_sources": {
                "train": {
                    "original": int(len(train_data[train_data['Data_Source']=='Original'])),
                    "filled": int(len(train_data[train_data['Data_Source']=='Filled']))
                },
                "val": {
                    "original": int(len(val_data[val_data['Data_Source']=='Original'])),
                    "filled": int(len(val_data[val_data['Data_Source']=='Filled']))
                },
                "test": {
                    "original": int(len(test_data[test_data['Data_Source']=='Original'])),
                    "filled": int(len(test_data[test_data['Data_Source']=='Filled']))
                }
            },
            "feature_columns": feature_columns
        }

        # 保存统计报告
        stats_path = os.path.join(self.output_dir, "improved_data_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)

        print(f"统计报告已保存: {stats_path}")

        # 打印关键信息
        print(f"\n=== 数据集统计 ===")
        print(f"总样本数: {stats['data_info']['total_samples']}")
        print(f"特征数: {stats['data_info']['feature_count']}")
        print(f"县区数: {stats['data_info']['counties']}")

        print(f"\n训练集 ({stats['data_info']['train_samples']} 样本):")
        print(f"  发生率: {stats['label_distribution']['train']['occurrence_rate']:.4f}")
        print(f"  原始数据: {stats['data_sources']['train']['original']}")
        print(f"  填充数据: {stats['data_sources']['train']['filled']}")

        print(f"\n验证集 ({stats['data_info']['val_samples']} 样本):")
        print(f"  发生率: {stats['label_distribution']['val']['occurrence_rate']:.4f}")

        print(f"\n测试集 ({stats['data_info']['test_samples']} 样本):")
        print(f"  发生率: {stats['label_distribution']['test']['occurrence_rate']:.4f}")

        return stats

    def run(self):
        """运行完整的数据生成流程"""
        print("开始生成改进的训练数据...")

        try:
            # 1. 加载数据
            df_occurrence = self.load_complete_occurrence_data()
            df_meteo = self.load_meteorological_data()

            # 2. 合并数据
            merged_df = self.merge_occurrence_and_meteo_data(df_occurrence, df_meteo)

            # 3. 添加工程特征
            merged_df = self.add_engineered_features(merged_df)

            # 4. 创建数据集
            train_data, val_data, test_data, feature_columns, scaler = self.create_datasets(merged_df)

            # 5. 生成统计报告
            stats = self.generate_statistics(train_data, val_data, test_data, feature_columns)

            print("\n改进的训练数据生成完成！")
            print(f"最终数据集: {len(train_data) + len(val_data) + len(test_data)} 样本")

            return {
                'train_data': train_data,
                'val_data': val_data,
                'test_data': test_data,
                'scaler': scaler,
                'stats': stats,
                'feature_columns': feature_columns
            }

        except Exception as e:
            print(f"数据生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    generator = ImprovedTrainingDataGenerator()
    result = generator.run()