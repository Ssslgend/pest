#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成包含山东省所有县区的完整发病情况数据
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

class CompleteOccurrenceDataGenerator:
    """生成完整的发病情况数据"""

    def __init__(self):
        # 读取山东省所有县区列表
        self.df_all_counties = pd.read_csv('datas/shandong_pest_data/shandong_all_counties.csv')
        print(f"加载山东省县区列表: {len(self.df_all_counties)} 个县区")

        # 读取原始发病情况数据
        self.df_occurrence = pd.read_excel('datas/shandong_pest_data/发病情况.xlsx')
        print(f"加载原始发病数据: {len(self.df_occurrence)} 条记录")

    def extract_original_occurrence_records(self):
        """提取原始发病记录"""
        print("提取原始发病记录...")

        all_records = []

        # 获取各发病程度列的索引
        col_5_6 = self.df_occurrence.columns[3]  # 一龄幼虫发生程度（5-6月）
        col_7_8 = self.df_occurrence.columns[4]  # 发生程度7-8月
        col_9_10 = self.df_occurrence.columns[5]  # 发生程度9-10月

        print(f"使用列: {col_5_6}, {col_7_8}, {col_9_10}")

        # 5-6月份数据 - 一龄幼虫发生程度
        for _, row in self.df_occurrence.iterrows():
            severity = row[col_5_6]
            if pd.notna(severity) and severity > 0:
                all_records.append({
                    'county_name': row['County'],
                    'year': row['Year'],
                    'month': 6,  # 6月代表5-6月期间
                    'Severity': severity,
                    'Period': '5-6月',
                    'Data_Source': 'Original'
                })

        # 7-8月份数据
        for _, row in self.df_occurrence.iterrows():
            severity = row[col_7_8]
            if pd.notna(severity) and severity > 0:
                all_records.append({
                    'county_name': row['County'],
                    'year': row['Year'],
                    'month': 8,  # 8月代表7-8月期间
                    'Severity': severity,
                    'Period': '7-8月',
                    'Data_Source': 'Original'
                })

        # 9-10月份数据
        for _, row in self.df_occurrence.iterrows():
            severity = row[col_9_10]
            if pd.notna(severity) and severity > 0:
                all_records.append({
                    'county_name': row['County'],
                    'year': row['Year'],
                    'month': 10,  # 10月代表9-10月期间
                    'Severity': severity,
                    'Period': '9-10月',
                    'Data_Source': 'Original'
                })

        print(f"提取原始发病记录: {len(all_records)} 条")
        return all_records

    def fill_missing_months_as_no_occurrence(self, original_records):
        """为所有县区填充缺失的月份，设为未发生（Severity=1）"""
        print("为所有县区填充缺失月份...")

        # 获取所有县区名称
        all_counties = self.df_all_counties['name'].tolist()

        # 获取数据中的年份范围
        years = sorted(self.df_occurrence['Year'].unique())

        # 创建所有县区、所有年份、所有月份的组合
        complete_records = []

        for county in all_counties:
            for year in years:
                for month in range(1, 13):  # 1-12月
                    # 检查是否已有该月的发病记录
                    existing_record = next((r for r in original_records
                                         if r['county_name'] == county and
                                            r['year'] == year and
                                            r['month'] == month), None)

                    if existing_record:
                        # 使用原始发病记录
                        complete_records.append(existing_record)
                    else:
                        # 添加未发生记录
                        complete_records.append({
                            'county_name': county,
                            'year': year,
                            'month': month,
                            'Severity': 1,  # 未发生
                            'Period': '无发生',
                            'Data_Source': 'Filled'
                        })

        print(f"生成完整记录: {len(complete_records)} 条")
        return complete_records

    def create_complete_dataframe(self, complete_records):
        """创建完整的数据框"""
        print("创建完整数据框...")

        df_complete = pd.DataFrame(complete_records)

        # 添加地理信息
        df_counties_with_coord = self.df_all_counties[['name', 'longitude', 'latitude']].rename(
            columns={'name': 'county_name'}
        )

        df_complete = df_complete.merge(df_counties_with_coord, on='county_name', how='left')

        # 添加季节信息
        df_complete['Season'] = df_complete['month'].apply(self._get_season)

        # 添加Has_Occurrence字段
        df_complete['Has_Occurrence'] = (df_complete['Severity'] > 1).astype(int)

        print(f"最终数据框形状: {df_complete.shape}")
        print("数据分布:")
        print(f"  县区数: {df_complete['county_name'].nunique()}")
        print(f"  年份数: {df_complete['year'].nunique()}")
        print(f"  月份覆盖: {df_complete['month'].min()}-{df_complete['month'].max()}")
        print(f"  Has_Occurrence=0 (未发生): {len(df_complete[df_complete['Has_Occurrence']==0])}")
        print(f"  Has_Occurrence=1 (有发生): {len(df_complete[df_complete['Has_Occurrence']==1])}")
        print(f"  发生率: {df_complete['Has_Occurrence'].mean():.4f}")

        return df_complete

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

    def save_complete_data(self, df_complete):
        """保存完整数据"""
        print("保存完整发病数据...")

        # 创建输出目录
        output_dir = 'datas/shandong_pest_data'
        os.makedirs(output_dir, exist_ok=True)

        # 保存完整数据
        output_file = os.path.join(output_dir, 'complete_shandong_occurrence_data.csv')
        df_complete.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"完整发病数据已保存: {output_file}")

        # 生成数据统计报告
        self.generate_data_report(df_complete, output_dir)

        return output_file

    def generate_data_report(self, df_complete, output_dir):
        """生成数据统计报告"""
        print("生成数据统计报告...")

        report = {
            "data_info": {
                "total_counties": int(df_complete['county_name'].nunique()),
                "total_years": int(df_complete['year'].nunique()),
                "total_records": int(len(df_complete)),
                "year_range": f"{df_complete['year'].min()}-{df_complete['year'].max()}"
            },
            "occurrence_statistics": {
                "no_occurrence_count": int(len(df_complete[df_complete['Has_Occurrence']==0])),
                "occurrence_count": int(len(df_complete[df_complete['Has_Occurrence']==1])),
                "occurrence_rate": float(df_complete['Has_Occurrence'].mean()),
                "severity_distribution": df_complete['Severity'].value_counts().to_dict()
            },
            "data_sources": {
                "original_records": int(len(df_complete[df_complete['Data_Source']=='Original'])),
                "filled_records": int(len(df_complete[df_complete['Data_Source']=='Filled']))
            },
            "monthly_distribution": df_complete.groupby('month')['Has_Occurrence'].mean().to_dict(),
            "yearly_distribution": df_complete.groupby('year')['Has_Occurrence'].mean().to_dict()
        }

        # 保存报告
        report_file = os.path.join(output_dir, 'complete_occurrence_report.json')
        import json
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)

        print(f"数据统计报告已保存: {report_file}")

        # 打印关键统计信息
        print(f"\n=== 数据统计报告 ===")
        print(f"县区总数: {report['data_info']['total_counties']}")
        print(f"年份范围: {report['data_info']['year_range']}")
        print(f"总记录数: {report['data_info']['total_records']}")
        print(f"未发生记录: {report['occurrence_statistics']['no_occurrence_count']}")
        print(f"发生记录: {report['occurrence_statistics']['occurrence_count']}")
        print(f"总体发生率: {report['occurrence_statistics']['occurrence_rate']:.4f}")
        print(f"原始记录: {report['data_sources']['original_records']}")
        print(f"补充记录: {report['data_sources']['filled_records']}")

        return report

    def run(self):
        """运行完整数据生成流程"""
        print("开始生成完整发病数据...")

        # 1. 提取原始发病记录
        original_records = self.extract_original_occurrence_records()

        # 2. 填充缺失月份为未发生
        complete_records = self.fill_missing_months_as_no_occurrence(original_records)

        # 3. 创建完整数据框
        df_complete = self.create_complete_dataframe(complete_records)

        # 4. 保存数据
        output_file = self.save_complete_data(df_complete)

        print("\n完整发病数据生成完成！")
        return output_file, df_complete

if __name__ == "__main__":
    generator = CompleteOccurrenceDataGenerator()
    output_file, df_complete = generator.run()