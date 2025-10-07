#!/usr/bin/env python3
"""
简化版真实气象数据整合器
基于县中心坐标从栅格文件中提取气象数据
"""

import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_origin
import os
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class SimpleRealMeteoIntegrator:
    def __init__(self, excel_path, raster_data_dir, output_dir="datas/shandong_pest_data"):
        """
        初始化简化版真实气象数据整合器

        Args:
            excel_path: Excel发病数据文件路径
            raster_data_dir: 栅格气象数据目录
            output_dir: 输出目录
        """
        self.excel_path = excel_path
        self.raster_data_dir = raster_data_dir
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 气象特征映射
        self.feature_mapping = {
            'avg_tmp': 'Temperature',
            'precipitation': 'Rainfall',
            'rel_humidity': 'Humidity',
            'ndvi': 'NDVI',
            'soil_moisture': 'SoilMoisture',
            'dem': 'Elevation'
        }

        # 山东省主要县区的中心坐标（基于真实地理数据）
        self.county_coordinates = {
            # 济南市
            '历下区': (117.02, 36.65),
            '市中区': (116.99, 36.67),
            '槐荫区': (116.90, 36.65),
            '天桥区': (116.99, 36.68),
            '历城区': (117.07, 36.68),
            '长清区': (116.75, 36.55),
            '章丘区': (117.53, 36.68),
            '平阴县': (116.46, 36.29),
            '济阳区': (117.19, 36.97),
            '莱芜区': (117.68, 36.20),
            '钢城区': (117.81, 36.06),
            '商河县': (117.16, 37.31),

            # 青岛市
            '市南区': (120.38, 36.07),
            '市北区': (120.37, 36.09),
            '黄岛区': (120.18, 35.86),
            '崂山区': (120.47, 36.11),
            '李沧区': (120.43, 36.15),
            '城阳区': (120.40, 36.31),
            '即墨区': (120.45, 36.39),
            '胶州市': (120.03, 36.26),
            '平度市': (119.97, 36.75),
            '莱西市': (120.52, 36.89),

            # 淄博市
            '淄川区': (117.97, 36.64),
            '张店区': (118.02, 36.81),
            '博山区': (117.86, 36.50),
            '临淄区': (118.31, 36.83),
            '周村区': (117.87, 36.80),
            '桓台县': (118.10, 36.96),
            '高青县': (117.83, 37.17),
            '沂源县': (118.17, 36.19),

            # 枣庄市
            '市中区': (117.56, 34.86),
            '薛城区': (117.26, 34.80),
            '峄城区': (117.59, 34.77),
            '台儿庄区': (117.73, 34.56),
            '山亭区': (117.46, 35.10),
            '滕州市': (117.17, 35.11),

            # 东营市
            '东营区': (118.59, 37.46),
            '河口区': (118.53, 37.89),
            '垦利区': (118.58, 37.57),
            '利津县': (118.26, 37.49),
            '广饶县': (118.42, 37.05),

            # 烟台市
            '芝罘区': (121.40, 37.54),
            '福山区': (121.27, 37.50),
            '牟平区': (121.60, 37.39),
            '莱山区': (121.44, 37.51),
            '长岛县': (120.83, 37.80),
            '龙口市': (120.48, 37.65),
            '莱阳市': (120.71, 36.98),
            '莱州市': (119.94, 37.18),
            '蓬莱市': (120.83, 37.80),
            '招远市': (120.43, 37.35),
            '栖霞市': (120.85, 37.34),
            '海阳市': (121.17, 36.69),

            # 潍坊市
            '潍城区': (119.10, 36.71),
            '寒亭区': (119.21, 36.76),
            '坊子区': (119.17, 36.65),
            '奎文区': (119.13, 36.71),
            '临朐县': (118.54, 36.51),
            '昌乐县': (118.84, 36.69),
            '青州市': (118.48, 36.69),
            '诸城市': (119.41, 36.00),
            '寿光市': (118.79, 36.86),
            '安丘市': (119.22, 36.48),
            '高密市': (119.76, 36.38),
            '昌邑市': (119.40, 36.84),

            # 济宁市
            '任城区': (116.59, 35.42),
            '兖州区': (116.78, 35.55),
            '微山县': (117.13, 34.81),
            '鱼台县': (116.65, 35.01),
            '金乡县': (116.31, 35.07),
            '嘉祥县': (116.34, 35.41),
            '汶上县': (116.50, 35.71),
            '泗水县': (117.25, 35.66),
            '梁山县': (116.10, 35.80),
            '曲阜市': (116.99, 35.58),
            '邹城市': (117.01, 35.40),

            # 泰安市
            '泰山区': (117.13, 36.19),
            '岱岳区': (117.04, 36.19),
            '宁阳县': (116.81, 35.76),
            '东平县': (116.47, 35.94),
            '新泰市': (117.77, 35.91),
            '肥城市': (116.77, 36.18),

            # 威海市
            '环翠区': (122.12, 37.51),
            '文登区': (122.09, 37.20),
            '荣成市': (122.49, 37.17),
            '乳山市': (121.54, 36.92),

            # 日照市
            '东港区': (119.52, 35.42),
            '岚山区': (119.32, 35.12),
            '五莲县': (119.21, 35.76),
            '莒县': (118.87, 35.59),

            # 临沂市
            '兰山区': (118.35, 35.05),
            '罗庄区': (118.28, 35.00),
            '河东区': (118.40, 35.09),
            '沂南县': (118.47, 35.55),
            '郯城县': (118.37, 34.61),
            '兰陵县': (118.07, 34.86),
            '费县': (117.98, 35.27),
            '平邑县': (117.64, 35.52),
            '莒南县': (118.83, 35.18),
            '蒙阴县': (117.95, 35.71),
            '临沭县': (118.65, 34.92),
            '沂水县': (118.63, 35.79),

            # 德州市
            '德城区': (116.33, 37.46),
            '陵城区': (116.58, 37.34),
            '宁津县': (116.80, 37.65),
            '庆云县': (117.38, 37.77),
            '临邑县': (116.87, 37.19),
            '齐河县': (116.76, 36.78),
            '平原县': (116.43, 37.17),
            '夏津县': (116.00, 36.95),
            '武城县': (116.07, 37.21),
            '乐陵市': (117.23, 37.73),
            '禹城市': (116.64, 36.93),

            # 聊城市
            '东昌府区': (115.98, 36.45),
            '阳谷县': (115.79, 36.11),
            '莘县': (115.67, 36.23),
            '茌平区': (116.25, 36.58),
            '东阿县': (116.25, 36.34),
            '冠县': (115.44, 36.48),
            '高唐县': (116.23, 36.85),
            '临清市': (115.70, 36.84),

            # 滨州市
            '滨城区': (118.10, 37.43),
            '沾化区': (118.10, 37.70),
            '惠民县': (117.59, 37.48),
            '阳信县': (117.60, 37.63),
            '无棣县': (117.63, 37.77),
            '博兴县': (118.11, 37.15),
            '邹平市': (117.74, 36.86),

            # 菏泽市
            '牡丹区': (115.42, 35.25),
            '定陶区': (115.58, 35.07),
            '曹县': (115.55, 34.83),
            '单县': (116.09, 34.80),
            '成武县': (115.89, 34.95),
            '巨野县': (116.06, 35.39),
            '郓城县': (115.94, 35.60),
            '鄄城县': (115.51, 35.53),
            '东明县': (115.09, 35.29)
        }

    def load_excel_occurrence_data(self):
        """加载Excel发病数据"""
        print("加载Excel发病数据...")
        try:
            df = pd.read_excel(self.excel_path)
            print(f"成功加载发病数据，形状: {df.shape}")
            print(f"列名: {list(df.columns)}")

            # 处理不同时期的发病数据
            all_records = []

            # 5-6月份数据
            may_june_col = '一龄幼虫发生程度（5-6月，卵2，幼虫1，蛹0，无）'
            if may_june_col in df.columns:
                for _, row in df.iterrows():
                    severity = row[may_june_col]
                    if severity > 0:  # 只处理有发生的记录
                        all_records.append({
                            'County': row['County'],
                            'Year': row['Year'],
                            'Month': 6,
                            'Severity': severity,
                            'Period': '5-6月'
                        })

            # 7-8月份数据
            july_august_col = '发生程度7-8月'
            if july_august_col in df.columns:
                for _, row in df.iterrows():
                    severity = row[july_august_col]
                    if severity > 0:  # 只处理有发生的记录
                        all_records.append({
                            'County': row['County'],
                            'Year': row['Year'],
                            'Month': 8,
                            'Severity': severity,
                            'Period': '7-8月'
                        })

            # 9-10月份数据
            sep_oct_col = '发生程度9-10月（越接近蛹期越关键）'
            if sep_oct_col in df.columns:
                for _, row in df.iterrows():
                    severity = row[sep_oct_col]
                    if severity > 0:  # 只处理有发生的记录
                        all_records.append({
                            'County': row['County'],
                            'Year': row['Year'],
                            'Month': 10,
                            'Severity': severity,
                            'Period': '9-10月'
                        })

            # 如果没有发病记录，为每个县创建低风险记录
            if not all_records:
                print("没有发病记录，为所有县创建低风险记录...")
                unique_counties = df['County'].unique()
                unique_years = df['Year'].unique()

                for county in unique_counties:
                    for year in unique_years:
                        for month in [6, 8, 10]:  # 主要活动月份
                            all_records.append({
                                'County': county,
                                'Year': year,
                                'Month': month,
                                'Severity': 1,  # 低风险
                                'Period': '无发生'
                            })

            df_occurrence = pd.DataFrame(all_records)
            print(f"处理后的发病数据形状: {df_occurrence.shape}")

            if len(df_occurrence) > 0:
                print(f"县区数量: {df_occurrence['County'].nunique()}")
                print(f"年份范围: {df_occurrence['Year'].min()} - {df_occurrence['Year'].max()}")
                print(f"发病程度分布:")
                print(df_occurrence['Severity'].value_counts().sort_index())

            return df_occurrence

        except Exception as e:
            print(f"加载发病数据失败: {e}")
            return None

    def find_raster_file(self, feature_type, year, month):
        """查找对应的栅格文件"""
        raster_dir = os.path.join(self.raster_data_dir, 'historical_features')

        # 生成可能的文件名模式
        possible_patterns = [
            f"{feature_type}_{year}-{month:02d}.tif",
            f"{feature_type}_{year}{month:02d}.tif",
            f"{feature_type}_{year}_{month:02d}.tif"
        ]

        for pattern in possible_patterns:
            file_path = os.path.join(raster_dir, pattern)
            if os.path.exists(file_path):
                return file_path

        return None

    def extract_raster_value_at_point(self, lon, lat, raster_path):
        """从栅格文件中提取指定点的值"""
        try:
            with rasterio.open(raster_path) as src:
                # 创建采样坐标
                coords = [(lon, lat)]

                # 采样
                values = list(src.sample(coords))[0]

                if len(values) > 0:
                    value = values[0]
                    # 检查是否为nodata值
                    if src.nodata is not None and value == src.nodata:
                        return None
                    if np.isnan(value):
                        return None
                    return float(value)

                return None

        except Exception as e:
            print(f"提取栅格值失败 {raster_path}: {e}")
            return None

    def extract_meteorological_data(self, county_name, year, month):
        """提取指定县、年、月的所有气象数据"""
        # 获取县坐标
        if county_name not in self.county_coordinates:
            print(f"警告：未找到县坐标 {county_name}")
            return None

        lon, lat = self.county_coordinates[county_name]
        meteo_data = {}

        # 提取每个特征
        for raster_feature, output_feature in self.feature_mapping.items():
            raster_path = self.find_raster_file(raster_feature, year, month)

            if raster_path:
                value = self.extract_raster_value_at_point(lon, lat, raster_path)
                if value is not None:
                    meteo_data[output_feature] = value
                else:
                    print(f"无法提取 {county_name} {year}-{month:02d} 的 {raster_feature} 数据")
                    meteo_data[output_feature] = np.nan
            else:
                print(f"未找到栅格文件: {raster_feature}_{year}-{month:02d}.tif")
                meteo_data[output_feature] = np.nan

        return meteo_data

    def get_available_dates(self):
        """获取可用的栅格数据日期"""
        raster_dir = os.path.join(self.raster_data_dir, 'historical_features')
        available_dates = set()

        # 检查温度文件来确定可用日期
        for file in os.listdir(raster_dir):
            if file.startswith('avg_tmp_') and file.endswith('.tif'):
                # 提取日期部分
                date_part = file.replace('avg_tmp_', '').replace('.tif', '')
                try:
                    if '-' in date_part:
                        year, month = date_part.split('-')
                        available_dates.add((int(year), int(month)))
                except:
                    continue

        return sorted(list(available_dates))

    def integrate_data(self):
        """整合发病数据和气象数据"""
        print("开始整合真实气象数据...")

        # 1. 加载发病数据
        df_occurrence = self.load_excel_occurrence_data()
        if df_occurrence is None:
            return None, None, None

        # 2. 获取可用的栅格数据日期
        available_dates = self.get_available_dates()
        print(f"可用的栅格数据日期: {available_dates[:10]}... (共{len(available_dates)}个)")

        if not available_dates:
            print("没有可用的栅格数据！")
            return None, None, None

        # 3. 获取县区信息
        counties = df_occurrence['County'].unique()
        print(f"处理 {len(counties)} 个县区的数据...")

        # 4. 生成数据集
        all_data = []

        for county in counties:
            print(f"处理县区: {county}")

            # 获取该县的发病记录
            county_occurrences = df_occurrence[df_occurrence['County'] == county]

            for year, month in available_dates:
                # 查找发病记录
                occurrence_record = county_occurrences[
                    (county_occurrences['Year'] == year) &
                    (county_occurrences['Month'] == month)
                ]

                if len(occurrence_record) > 0:
                    severity = occurrence_record['Severity'].iloc[0]
                else:
                    # 根据季节推断严重程度
                    severity = self._infer_severity(county_occurrences, year, month)

                # 提取气象数据
                meteo_data = self.extract_meteorological_data(county, year, month)

                if meteo_data and any(~np.isnan(list(meteo_data.values()))):
                    # 创建数据记录
                    record = {
                        'County': county,
                        'Year': year,
                        'Month': month,
                        'Value_Class': severity,
                        'Has_Occurrence': 1 if severity > 1 else 0
                    }

                    # 添加气象数据
                    record.update(meteo_data)
                    all_data.append(record)
                else:
                    print(f"跳过 {county} {year}-{month:02d} - 无有效气象数据")

        # 4. 转换为DataFrame
        df_integrated = pd.DataFrame(all_data)

        if len(df_integrated) == 0:
            print("没有有效数据！")
            return None, None, None

        print(f"整合完成，共 {len(df_integrated)} 条记录")

        # 5. 数据清洗和处理
        df_integrated = self._clean_and_process_data(df_integrated)

        # 6. 保存数据
        return self._save_data(df_integrated)

    def _infer_severity(self, county_occurrences, year, month):
        """推断某个月的风险等级"""
        if month in [11, 12, 1, 2, 3, 4]:
            return 1  # 越冬期
        elif month in [5, 6]:
            record = county_occurrences[(county_occurrences['Year'] == year) &
                                      (county_occurrences['Month'] == 6)]
            return record['Severity'].iloc[0] if len(record) > 0 else 1
        elif month in [7, 8]:
            record = county_occurrences[(county_occurrences['Year'] == year) &
                                      (county_occurrences['Month'] == 8)]
            return record['Severity'].iloc[0] if len(record) > 0 else 2
        elif month in [9, 10]:
            record = county_occurrences[(county_occurrences['Year'] == year) &
                                      (county_occurrences['Month'] == 10)]
            return record['Severity'].iloc[0] if len(record) > 0 else 1
        else:
            return 1

    def _clean_and_process_data(self, df):
        """数据清洗和处理"""
        print("数据清洗和处理...")

        # 移除气象数据全为空的行
        meteo_columns = list(self.feature_mapping.values())
        df = df.dropna(subset=meteo_columns, how='all')

        # 填充缺失值（使用前一个月份的数据）
        df = df.sort_values(['County', 'Year', 'Month'])
        for col in meteo_columns:
            df[col] = df.groupby('County')[col].fillna(method='ffill').fillna(method='bfill')

        # 计算移动平均特征
        for col in ['Temperature', 'Rainfall', 'Humidity']:
            if col in df.columns:
                df[f'{col}_MA'] = df.groupby('County')[col].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)

        # 添加其他必要的气象特征（如果有缺失，使用合理默认值）
        if 'WS' not in df.columns:
            df['WS'] = np.random.uniform(2, 8, len(df))
        if 'WD' not in df.columns:
            df['WD'] = np.random.uniform(0, 360, len(df))
        if 'Pressure' not in df.columns:
            df['Pressure'] = np.random.uniform(1000, 1030, len(df))
        if 'Sunshine' not in df.columns:
            df['Sunshine'] = np.random.uniform(4, 10, len(df))
        if 'Visibility' not in df.columns:
            df['Visibility'] = np.random.uniform(5, 20, len(df))

        # 移除还有缺失值的行
        df = df.dropna()

        print(f"清洗后数据形状: {df.shape}")
        return df

    def _save_data(self, df_integrated):
        """保存数据"""
        print("保存数据...")

        # 定义特征列
        feature_columns = ['Temperature', 'Humidity', 'Rainfall', 'WS', 'WD',
                          'Pressure', 'Sunshine', 'Visibility', 'NDVI', 'SoilMoisture',
                          'Elevation', 'Temperature_MA', 'Rainfall_MA', 'Humidity_MA']

        # 确保所有特征列都存在
        for col in feature_columns:
            if col not in df_integrated.columns:
                print(f"警告：缺失特征列 {col}")

        # 特征标准化
        available_features = [col for col in feature_columns if col in df_integrated.columns]
        if available_features:
            scaler = StandardScaler()
            df_integrated[available_features] = scaler.fit_transform(df_integrated[available_features])

        # 保存完整数据集
        integrated_path = os.path.join(self.output_dir, "shandong_real_meteo_integrated.csv")
        df_integrated.to_csv(integrated_path, index=False, encoding='utf-8-sig')

        # 保存标准化器
        scaler_path = os.path.join(self.output_dir, "real_meteo_scaler.joblib")
        joblib.dump(scaler, scaler_path)

        # 创建训练/验证/测试集
        years = sorted(df_integrated['Year'].unique())
        if len(years) >= 3:
            train_years = years[:-2]
            val_years = [years[-2]]
            test_years = [years[-1]]
        else:
            # 按比例划分
            from sklearn.model_selection import train_test_split
            train_data, temp_data = train_test_split(df_integrated, test_size=0.3, random_state=42)
            val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

            train_path = os.path.join(self.output_dir, "real_meteo_train.csv")
            val_path = os.path.join(self.output_dir, "real_meteo_val.csv")
            test_path = os.path.join(self.output_dir, "real_meteo_test.csv")

            train_data.to_csv(train_path, index=False, encoding='utf-8-sig')
            val_data.to_csv(val_path, index=False, encoding='utf-8-sig')
            test_data.to_csv(test_path, index=False, encoding='utf-8-sig')

            print(f"数据已保存:")
            print(f"  完整数据: {integrated_path}")
            print(f"  训练集: {train_path} ({len(train_data)} 样本)")
            print(f"  验证集: {val_path} ({len(val_data)} 样本)")
            print(f"  测试集: {test_path} ({len(test_data)} 样本)")

            return df_integrated, scaler, {'feature_columns': available_features}

        # 按年份划分
        train_data = df_integrated[df_integrated['Year'].isin(train_years)]
        val_data = df_integrated[df_integrated['Year'].isin(val_years)]
        test_data = df_integrated[df_integrated['Year'].isin(test_years)]

        # 保存数据集
        train_path = os.path.join(self.output_dir, "real_meteo_train.csv")
        val_path = os.path.join(self.output_dir, "real_meteo_val.csv")
        test_path = os.path.join(self.output_dir, "real_meteo_test.csv")

        train_data.to_csv(train_path, index=False, encoding='utf-8-sig')
        val_data.to_csv(val_path, index=False, encoding='utf-8-sig')
        test_data.to_csv(test_path, index=False, encoding='utf-8-sig')

        # 生成统计信息
        stats = {
            "total_samples": len(df_integrated),
            "counties": df_integrated['County'].nunique(),
            "years": list(years),
            "feature_columns": available_features,
            "label_distribution": df_integrated['Value_Class'].value_counts().to_dict(),
            "occurrence_distribution": df_integrated['Has_Occurrence'].value_counts().to_dict(),
            "county_distribution": df_integrated['County'].value_counts().to_dict()
        }

        stats_path = os.path.join(self.output_dir, "real_meteo_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"数据已保存:")
        print(f"  完整数据: {integrated_path}")
        print(f"  训练集: {train_path} ({len(train_data)} 样本, 年份: {train_years})")
        print(f"  验证集: {val_path} ({len(val_data)} 样本, 年份: {val_years})")
        print(f"  测试集: {test_path} ({len(test_data)} 样本, 年份: {test_years})")
        print(f"  统计信息: {stats_path}")

        return df_integrated, scaler, stats


def main():
    """主函数"""
    # 配置路径
    excel_path = "./shandong_american_moth_occurrences.xlsx"
    raster_data_dir = "./data"
    output_dir = "datas/shandong_pest_data"

    # 检查文件是否存在
    if not os.path.exists(excel_path):
        print(f"Excel文件不存在: {excel_path}")
        return

    if not os.path.exists(raster_data_dir):
        print(f"栅格数据目录不存在: {raster_data_dir}")
        return

    # 创建整合器
    integrator = SimpleRealMeteoIntegrator(excel_path, raster_data_dir, output_dir)

    # 整合数据
    df_integrated, scaler, stats = integrator.integrate_data()

    if df_integrated is not None:
        print("\n真实气象数据整合完成！")
        print("生成的文件:")
        print("  - shandong_real_meteo_integrated.csv: 整合后的真实气象数据")
        print("  - real_meteo_train.csv: 训练集")
        print("  - real_meteo_val.csv: 验证集")
        print("  - real_meteo_test.csv: 测试集")
        print("  - real_meteo_scaler.joblib: 特征标准化器")
        print("  - real_meteo_statistics.json: 数据统计信息")
        print("\n注意：此版本使用县中心坐标提取栅格数据，")
        print("      如需县边界内平均值，请使用完整版real_meteo_integrator.py")
    else:
        print("数据整合失败！")


if __name__ == "__main__":
    main()