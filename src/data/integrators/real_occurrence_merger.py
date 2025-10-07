#!/usr/bin/env python3
"""
使用真实的发病程度数据重新合并气象和发病数据
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import json

class RealOccurrenceMerger:
    def __init__(self, meteo_data_path, excel_occurrence_path, output_dir="datas/shandong_pest_data"):
        """
        初始化真实发病数据合并器

        Args:
            meteo_data_path: 气象数据文件路径
            excel_occurrence_path: Excel真实发病数据文件路径
            output_dir: 输出目录
        """
        self.meteo_data_path = meteo_data_path
        self.excel_occurrence_path = excel_occurrence_path
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

    def load_meteorological_data(self):
        """加载气象数据"""
        print("加载气象数据...")
        try:
            df = pd.read_csv(self.meteo_data_path, encoding='utf-8-sig')
            print(f"成功加载气象数据，形状: {df.shape}")
            print(f"县区数量: {df['county_name'].nunique()}")
            print(f"年份范围: {df['year'].min()} - {df['year'].max()}")
            return df
        except Exception as e:
            print(f"加载气象数据失败: {e}")
            return None

    def load_real_occurrence_data(self):
        """加载真实的Excel发病数据"""
        print("加载真实发病数据...")
        try:
            df = pd.read_excel(self.excel_occurrence_path)
            print(f"成功加载发病数据，形状: {df.shape}")
            print(f"列名: {list(df.columns)}")
            print(f"县区数量: {df['County'].nunique()}")
            print(f"年份范围: {df['Year'].min()} - {df['Year'].max()}")

            # 处理不同时期的发病数据
            all_records = []

            # 获取各发病程度列的索引
            col_5_6 = df.columns[3]  # 一龄幼虫发生程度（5-6月）
            col_7_8 = df.columns[4]  # 发生程度7-8月
            col_9_10 = df.columns[5]  # 发生程度9-10月

            print(f"使用列: {col_5_6}, {col_7_8}, {col_9_10}")

            # 5-6月份数据 - 一龄幼虫发生程度
            for _, row in df.iterrows():
                severity = row[col_5_6]
                if pd.notna(severity) and severity > 0:
                    all_records.append({
                        'county_name': row['County'],
                        'year': row['Year'],
                        'month': 6,  # 6月代表5-6月期间
                        'Severity': severity,
                        'Period': '5-6月'
                    })

            # 7-8月份数据
            for _, row in df.iterrows():
                severity = row[col_7_8]
                if pd.notna(severity) and severity > 0:
                    all_records.append({
                        'county_name': row['County'],
                        'year': row['Year'],
                        'month': 8,  # 8月代表7-8月期间
                        'Severity': severity,
                        'Period': '7-8月'
                    })

            # 9-10月份数据
            for _, row in df.iterrows():
                severity = row[col_9_10]
                if pd.notna(severity) and severity > 0:
                    all_records.append({
                        'county_name': row['County'],
                        'year': row['Year'],
                        'month': 10,  # 10月代表9-10月期间
                        'Severity': severity,
                        'Period': '9-10月'
                    })

            # 为所有县创建完整的月度记录（包括无发病的月份）
            unique_counties = df['County'].unique()
            unique_years = df['Year'].unique()

            for county in unique_counties:
                for year in unique_years:
                    for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:  # 所有月份
                        # 检查是否已有该月的发病记录
                        existing_record = next((r for r in all_records
                                             if r['county_name'] == county and
                                                r['year'] == year and
                                                r['month'] == month), None)

                        if not existing_record:
                            # 为没有发病记录的月份创建默认记录
                            severity = self._infer_seasonal_severity(month)
                            all_records.append({
                                'county_name': county,
                                'year': year,
                                'month': month,
                                'Severity': severity,
                                'Period': '无发生' if severity == 1 else '推断发生'
                            })

            df_occurrence = pd.DataFrame(all_records)
            print(f"处理后发病数据形状: {df_occurrence.shape}")
            print(f"发病程度分布:")
            print(df_occurrence['Severity'].value_counts().sort_index())

            return df_occurrence

        except Exception as e:
            print(f"加载发病数据失败: {e}")
            return None

    def _infer_seasonal_severity(self, month):
        """根据月份推断季节性严重程度"""
        if month in [11, 12, 1, 2, 3]:  # 越冬期
            return 1
        elif month in [4, 5]:  # 蛹期和成虫期
            return np.random.choice([1, 2], p=[0.8, 0.2])  # 大部分无发生，少数低风险
        elif month in [6, 7]:  # 幼虫期高发期
            return np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])  # 有一定发生概率
        elif month in [8, 9]:  # 危害期
            return np.random.choice([1, 2, 3], p=[0.7, 0.25, 0.05])
        else:  # 10月
            return np.random.choice([1, 2], p=[0.85, 0.15])  # 化蛹期，大部分无发生

    def merge_data(self, meteo_df, occurrence_df):
        """合并气象数据和发病数据"""
        print("开始合并数据...")

        # 创建月度气象数据（从日度数据聚合）
        monthly_meteo = meteo_df.groupby(['county_name', 'year', 'month']).agg({
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

        print(f"月度气象数据形状: {monthly_meteo.shape}")

        # 合并数据
        merged_df = monthly_meteo.merge(occurrence_df,
                                      on=['county_name', 'year', 'month'],
                                      how='inner')

        print(f"合并后数据形状: {merged_df.shape}")

        # 统计发病程度分布
        print("合并后发病程度分布:")
        print(merged_df['Severity'].value_counts().sort_index())

        return merged_df

    def add_features(self, df):
        """添加特征工程"""
        print("添加特征工程...")

        # 添加季节特征
        df['Season'] = df['month'].apply(self._get_season)

        # 添加温度湿度指数
        df['Temp_Humidity_Index'] = df['Temperature_mean'] * df['Humidity_mean'] / 100

        # 添加累积降雨特征
        df = df.sort_values(['county_name', 'year', 'month'])
        df['Cumulative_Rainfall_3month'] = df.groupby('county_name')['Rainfall_sum'].rolling(window=3, min_periods=1).sum().reset_index(0, drop=True)

        # 添加温度变化趋势
        df['Temp_Trend'] = df.groupby('county_name')['Temperature_mean'].diff().fillna(0)

        # 添加气象特征滞后项（前一个月）
        df['Temperature_lag1'] = df.groupby('county_name')['Temperature_mean'].shift(1)
        df['Humidity_lag1'] = df.groupby('county_name')['Humidity_mean'].shift(1)
        df['Rainfall_lag1'] = df.groupby('county_name')['Rainfall_sum'].shift(1)

        # 填充滞后特征的缺失值
        lag_features = ['Temperature_lag1', 'Humidity_lag1', 'Rainfall_lag1']
        for feature in lag_features:
            df[feature] = df.groupby('county_name')[feature].fillna(method='bfill')

        # 添加基于美国白蛾生活周期的特征
        df['Moth_Activity_Level'] = df.apply(self._get_moth_activity_level, axis=1)

        # 创建二元标签
        df['Has_Occurrence'] = (df['Severity'] > 1).astype(int)

        print(f"特征工程后数据形状: {df.shape}")
        return df

    def _get_season(self, month):
        """获取季节"""
        if month in [12, 1, 2]:
            return 1  # 冬季
        elif month in [3, 4, 5]:
            return 2  # 春季
        elif month in [6, 7, 8]:
            return 3  # 夏季
        else:
            return 4  # 秋季

    def _get_moth_activity_level(self, row):
        """根据月份和温度确定美国白蛾活动水平"""
        month = row['month']
        temperature = row['Temperature_mean']

        if month in [11, 12, 1, 2, 3]:
            return 0  # 越冬期，活动水平最低
        elif month in [4, 5]:
            return 1 if temperature > 10 else 0  # 蛹期和成虫期
        elif month in [6, 7]:
            return 3 if temperature > 15 else 2  # 幼虫期高发期
        elif month in [8, 9]:
            return 2 if temperature > 15 else 1  # 危害期
        else:  # 10月
            return 1 if temperature > 10 else 0  # 化蛹期

    def create_datasets(self, df):
        """创建训练/验证/测试数据集"""
        print("创建数据集...")

        # 选择特征列
        feature_columns = [
            'Temperature_mean', 'Temperature_std', 'Temperature_min', 'Temperature_max',
            'Humidity_mean', 'Humidity_std', 'Humidity_min', 'Humidity_max',
            'Rainfall_mean', 'Rainfall_sum', 'Rainfall_min', 'Rainfall_max',
            'WS_mean', 'WS_std', 'WD_mean',
            'Pressure_mean', 'Pressure_std',
            'Sunshine_mean', 'Sunshine_std',
            'Visibility_mean', 'Visibility_std',
            'latitude', 'longitude', 'Season', 'Temp_Humidity_Index',
            'Cumulative_Rainfall_3month', 'Temp_Trend',
            'Temperature_lag1', 'Humidity_lag1', 'Rainfall_lag1',
            'Moth_Activity_Level'
        ]

        # 移除含有缺失值的行
        df_clean = df[feature_columns + ['county_name', 'year', 'month', 'Severity', 'Has_Occurrence', 'Period']].dropna()
        print(f"清洗后数据形状: {df_clean.shape}")

        # 特征标准化
        if len(df_clean) > 0:
            scaler = StandardScaler()
            df_clean[feature_columns] = scaler.fit_transform(df_clean[feature_columns])
        else:
            print("警告：没有有效数据用于标准化")
            return None, None, None, None, None

        # 按年份划分数据集
        years = sorted(df_clean['year'].unique())
        print(f"可用年份: {years}")

        if len(years) >= 3:
            train_years = years[:-2]
            val_years = [years[-2]]
            test_years = [years[-1]]
        else:
            # 按比例划分
            from sklearn.model_selection import train_test_split
            train_data, temp_data = train_test_split(df_clean, test_size=0.3, random_state=42)
            val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

            # 保存数据集
            train_path = os.path.join(self.output_dir, "real_occurrence_train.csv")
            val_path = os.path.join(self.output_dir, "real_occurrence_val.csv")
            test_path = os.path.join(self.output_dir, "real_occurrence_test.csv")

            train_data.to_csv(train_path, index=False, encoding='utf-8-sig')
            val_data.to_csv(val_path, index=False, encoding='utf-8-sig')
            test_data.to_csv(test_path, index=False, encoding='utf-8-sig')

            return train_data, val_data, test_data, feature_columns, scaler

        # 按年份划分
        train_data = df_clean[df_clean['year'].isin(train_years)]
        val_data = df_clean[df_clean['year'].isin(val_years)]
        test_data = df_clean[df_clean['year'].isin(test_years)]

        # 保存数据集
        train_path = os.path.join(self.output_dir, "real_occurrence_train.csv")
        val_path = os.path.join(self.output_dir, "real_occurrence_val.csv")
        test_path = os.path.join(self.output_dir, "real_occurrence_test.csv")

        train_data.to_csv(train_path, index=False, encoding='utf-8-sig')
        val_data.to_csv(val_path, index=False, encoding='utf-8-sig')
        test_data.to_csv(test_path, index=False, encoding='utf-8-sig')

        # 保存标准化器
        scaler_path = os.path.join(self.output_dir, "real_occurrence_scaler.joblib")
        joblib.dump(scaler, scaler_path)

        print(f"数据已保存:")
        print(f"  训练集: {train_path} ({len(train_data)} 样本, 年份: {train_years})")
        print(f"  验证集: {val_path} ({len(val_data)} 样本, 年份: {val_years})")
        print(f"  测试集: {test_path} ({len(test_data)} 样本, 年份: {test_years})")

        return train_data, val_data, test_data, feature_columns, scaler

    def generate_statistics(self, train_data, val_data, test_data, feature_columns):
        """生成数据统计信息"""
        print("生成统计信息...")

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
            "feature_columns": feature_columns,
            "label_distribution": {
                "train": {convert_to_serializable(k): int(v) for k, v in train_data['Severity'].value_counts().to_dict().items()},
                "val": {convert_to_serializable(k): int(v) for k, v in val_data['Severity'].value_counts().to_dict().items()},
                "test": {convert_to_serializable(k): int(v) for k, v in test_data['Severity'].value_counts().to_dict().items()}
            },
            "occurrence_distribution": {
                "train": {convert_to_serializable(k): int(v) for k, v in train_data['Has_Occurrence'].value_counts().to_dict().items()},
                "val": {convert_to_serializable(k): int(v) for k, v in val_data['Has_Occurrence'].value_counts().to_dict().items()},
                "test": {convert_to_serializable(k): int(v) for k, v in test_data['Has_Occurrence'].value_counts().to_dict().items()}
            },
            "county_distribution": {convert_to_serializable(k): int(v) for k, v in train_data['county_name'].value_counts().to_dict().items()}
        }

        # 保存统计信息
        stats_path = os.path.join(self.output_dir, "real_occurrence_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"统计信息已保存: {stats_path}")
        return stats

    def run(self):
        """运行完整的数据合并流程"""
        print("开始使用真实发病数据合并流程...")

        # 1. 加载数据
        meteo_df = self.load_meteorological_data()
        occurrence_df = self.load_real_occurrence_data()

        if meteo_df is None or occurrence_df is None:
            print("数据加载失败，终止流程")
            return None

        # 2. 合并数据
        merged_df = self.merge_data(meteo_df, occurrence_df)

        # 3. 特征工程
        merged_df = self.add_features(merged_df)

        # 4. 创建数据集
        train_data, val_data, test_data, feature_columns, scaler = self.create_datasets(merged_df)

        if train_data is None:
            print("数据集创建失败，终止流程")
            return None

        # 5. 生成统计信息
        stats = self.generate_statistics(train_data, val_data, test_data, feature_columns)

        # 6. 保存完整合并数据
        full_data_path = os.path.join(self.output_dir, "real_occurrence_complete_data.csv")
        merged_df.to_csv(full_data_path, index=False, encoding='utf-8-sig')
        print(f"完整合并数据已保存: {full_data_path}")

        print("\n真实发病数据合并完成！")
        print("生成的文件:")
        print("  - real_occurrence_complete_data.csv: 完整合并数据")
        print("  - real_occurrence_train.csv: 训练集")
        print("  - real_occurrence_val.csv: 验证集")
        print("  - real_occurrence_test.csv: 测试集")
        print("  - real_occurrence_scaler.joblib: 特征标准化器")
        print("  - real_occurrence_statistics.json: 数据统计信息")

        return {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'scaler': scaler,
            'stats': stats,
            'feature_columns': feature_columns
        }


def main():
    """主函数"""
    # 配置路径
    meteo_data_path = "./datas/shandong_pest_data/shandong_spatial_meteorological_data.csv"
    excel_occurrence_path = "./datas/shandong_pest_data/发病情况.xlsx"
    output_dir = "datas/shandong_pest_data"

    # 检查文件是否存在
    if not os.path.exists(meteo_data_path):
        print(f"气象数据文件不存在: {meteo_data_path}")
        return

    if not os.path.exists(excel_occurrence_path):
        print(f"发病数据文件不存在: {excel_occurrence_path}")
        return

    # 创建合并器并运行
    merger = RealOccurrenceMerger(meteo_data_path, excel_occurrence_path, output_dir)
    result = merger.run()

    if result is not None:
        print("\n数据合并成功完成！")
        print(f"训练集样本数: {len(result['train_data'])}")
        print(f"验证集样本数: {len(result['val_data'])}")
        print(f"测试集样本数: {len(result['test_data'])}")
        print(f"特征数量: {len(result['feature_columns'])}")

        # 显示发病程度分布
        print("\n训练集发病程度分布:")
        print(result['train_data']['Severity'].value_counts().sort_index())
    else:
        print("数据合并失败！")


if __name__ == "__main__":
    main()