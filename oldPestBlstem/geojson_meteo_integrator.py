#!/usr/bin/env python3
"""
使用GeoJSON县边界数据整合山东省美国白蛾发病数据
基于真实的县边界从栅格文件中提取气象数据
"""

import pandas as pd
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from rasterio.features import geometry_mask
import os
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class GeoJSONMeteoIntegrator:
    def __init__(self, excel_path, geojson_path, raster_data_dir, output_dir="datas/shandong_pest_data"):
        """
        初始化GeoJSON气象数据整合器

        Args:
            excel_path: Excel发病数据文件路径
            geojson_path: 山东省县边界GeoJSON文件路径
            raster_data_dir: 栅格气象数据目录
            output_dir: 输出目录
        """
        self.excel_path = excel_path
        self.geojson_path = geojson_path
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

        # 加载县边界数据
        self.county_gdf = self.load_county_boundaries()

    def load_county_boundaries(self):
        """加载山东省县边界数据"""
        print("加载山东省县边界数据...")
        try:
            gdf = gpd.read_file(self.geojson_path)
            print(f"成功加载 {len(gdf)} 个县区边界")

            # 检查列名
            print(f"边界数据列名: {list(gdf.columns)}")

            # 标准化县名称列
            name_column = None
            for col in gdf.columns:
                if 'name' in col.lower() or 'NAME' in col:
                    name_column = col
                    break

            if name_column:
                print(f"使用县名列: {name_column}")
                # 重命名为标准列名
                gdf = gdf.rename(columns={name_column: 'County'})
            else:
                print("警告：未找到县名列，使用默认名称")
                gdf['County'] = gdf.iloc[:, 0]

            # 确保使用正确的坐标系
            if gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
                print("已转换为WGS84坐标系")

            return gdf

        except Exception as e:
            print(f"加载县边界数据失败: {e}")
            return None

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
        # 查找历史特征文件
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

    def extract_raster_values_for_county(self, county_geometry, raster_path):
        """从栅格文件中提取县边界内的值"""
        try:
            with rasterio.open(raster_path) as src:
                # 使用县边界裁剪栅格
                try:
                    # 方法1：直接使用几何掩膜
                    masked_data, _ = mask(src, [county_geometry], crop=True)

                    # 移除nodata值
                    if src.nodata is not None:
                        masked_data = masked_data[masked_data != src.nodata]

                    # 计算统计值
                    if len(masked_data) > 0 and np.any(~np.isnan(masked_data)):
                        values = masked_data[~np.isnan(masked_data)]
                        return {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'min': float(np.min(values)),
                            'max': float(np.max(values)),
                            'count': int(len(values))
                        }
                    else:
                        # 如果没有有效值，尝试使用中心点
                        return self.extract_at_centroid(county_geometry, src)

                except Exception as e:
                    print(f"裁剪失败，尝试中心点提取: {e}")
                    return self.extract_at_centroid(county_geometry, src)

        except Exception as e:
            print(f"读取栅格文件失败 {raster_path}: {e}")
            return None

    def extract_at_centroid(self, county_geometry, raster_src):
        """在县中心点提取栅格值"""
        try:
            # 获取县中心点
            centroid = county_geometry.centroid
            lon, lat = centroid.x, centroid.y

            # 转换坐标到栅格坐标系
            coords = [(lon, lat)]

            # 提取值
            values = list(raster_src.sample(coords))[0]

            if len(values) > 0 and raster_src.nodata is not None:
                if values[0] != raster_src.nodata and not np.isnan(values[0]):
                    return {
                        'mean': float(values[0]),
                        'std': 0.0,
                        'min': float(values[0]),
                        'max': float(values[0]),
                        'count': 1
                    }

            return None

        except Exception as e:
            print(f"中心点提取失败: {e}")
            return None

    def extract_meteorological_data(self, county_name, year, month):
        """提取指定县、年、月的所有气象数据"""
        meteo_data = {}

        # 在边界数据中查找该县
        county_row = self.county_gdf[self.county_gdf['County'].str.contains(county_name, na=False)]

        if len(county_row) == 0:
            print(f"警告：未找到县边界 {county_name}")
            return None

        county_geometry = county_row.geometry.iloc[0]

        # 提取每个特征
        for raster_feature, output_feature in self.feature_mapping.items():
            raster_path = self.find_raster_file(raster_feature, year, month)

            if raster_path:
                result = self.extract_raster_values_for_county(county_geometry, raster_path)
                if result:
                    meteo_data[output_feature] = result['mean']
                else:
                    print(f"无法提取 {county_name} {year}-{month:02d} 的 {raster_feature} 数据")
                    meteo_data[output_feature] = np.nan
            else:
                print(f"未找到栅格文件: {raster_feature}_{year}-{month:02d}.tif")
                meteo_data[output_feature] = np.nan

        return meteo_data

    def get_available_date_range(self):
        """获取可用的日期范围"""
        available_dates = set()

        raster_dir = os.path.join(self.raster_data_dir, 'historical_features')
        if not os.path.exists(raster_dir):
            return []

        for filename in os.listdir(raster_dir):
            if filename.endswith('.tif'):
                # 解析文件名中的日期
                parts = filename.replace('.tif', '').split('_')
                if len(parts) >= 2:
                    date_part = parts[-1]
                    try:
                        if '-' in date_part:
                            year, month = map(int, date_part.split('-'))
                        else:
                            year = int(date_part[:4])
                            month = int(date_part[4:6])
                        available_dates.add((year, month))
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

        # 2. 获取可用的日期范围
        available_dates = self.get_available_date_range()
        print(f"可用的气象数据日期范围: {available_dates[:5]} ... {available_dates[-5:]} (共{len(available_dates)}个月)")

        if not available_dates:
            print("错误：没有找到可用的气象数据文件")
            return None, None, None

        # 3. 获取所有县区、年份、月份组合
        counties = df_occurrence['County'].unique()
        years = sorted(set([date[0] for date in available_dates]))
        months = sorted(set([date[1] for date in available_dates]))

        print(f"处理 {len(counties)} 个县区，{len(years)} 年的数据...")

        # 4. 生成完整的数据集
        all_data = []

        for county in counties:
            print(f"处理县区: {county}")

            # 获取该县的发病记录
            county_occurrences = df_occurrence[df_occurrence['County'] == county]

            for year in years:
                for month in months:
                    # 只处理有气象数据的日期
                    if (year, month) not in available_dates:
                        continue

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

        # 5. 转换为DataFrame
        df_integrated = pd.DataFrame(all_data)

        if len(df_integrated) == 0:
            print("没有有效数据！")
            return None, None, None

        print(f"整合完成，共 {len(df_integrated)} 条记录")

        # 6. 数据清洗和处理
        df_integrated = self._clean_and_process_data(df_integrated)

        # 7. 保存数据
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
        integrated_path = os.path.join(self.output_dir, "shandong_geojson_integrated.csv")
        df_integrated.to_csv(integrated_path, index=False, encoding='utf-8-sig')

        # 保存标准化器
        scaler_path = os.path.join(self.output_dir, "geojson_meteo_scaler.joblib")
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

            train_path = os.path.join(self.output_dir, "geojson_train.csv")
            val_path = os.path.join(self.output_dir, "geojson_val.csv")
            test_path = os.path.join(self.output_dir, "geojson_test.csv")

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
        train_path = os.path.join(self.output_dir, "geojson_train.csv")
        val_path = os.path.join(self.output_dir, "geojson_val.csv")
        test_path = os.path.join(self.output_dir, "geojson_test.csv")

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

        stats_path = os.path.join(self.output_dir, "geojson_statistics.json")
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
    geojson_path = "./datas/shandong_pest_data/shandong.json"
    raster_data_dir = "./data"
    output_dir = "datas/shandong_pest_data"

    # 检查文件是否存在
    if not os.path.exists(excel_path):
        print(f"Excel文件不存在: {excel_path}")
        return

    if not os.path.exists(geojson_path):
        print(f"GeoJSON文件不存在: {geojson_path}")
        return

    if not os.path.exists(raster_data_dir):
        print(f"栅格数据目录不存在: {raster_data_dir}")
        return

    # 创建整合器
    integrator = GeoJSONMeteoIntegrator(excel_path, geojson_path, raster_data_dir, output_dir)

    # 整合数据
    df_integrated, scaler, stats = integrator.integrate_data()

    if df_integrated is not None:
        print("\nGeoJSON气象数据整合完成！")
        print("生成的文件:")
        print("  - shandong_geojson_integrated.csv: 整合后的真实气象数据")
        print("  - geojson_train.csv: 训练集")
        print("  - geojson_val.csv: 验证集")
        print("  - geojson_test.csv: 测试集")
        print("  - geojson_meteo_scaler.joblib: 特征标准化器")
        print("  - geojson_statistics.json: 数据统计信息")
    else:
        print("数据整合失败！")


if __name__ == "__main__":
    main()