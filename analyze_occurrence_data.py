#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析发病情况数据，了解为什么正样本少
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_occurrence_data():
    """分析发病情况数据"""
    print("="*60)
    print("分析发病情况数据")
    print("="*60)

    # 读取发病情况数据
    print("1. 读取原始发病情况数据...")
    try:
        df_occurrence = pd.read_excel('datas/shandong_pest_data/发病情况.xlsx')
        print(f"发病数据形状: {df_occurrence.shape}")
        print(f"列名: {df_occurrence.columns.tolist()}")
        print(f"县区数: {df_occurrence['County'].nunique()}")
        print(f"年份范围: {df_occurrence['Year'].min()}-{df_occurrence['Year'].max()}")
    except Exception as e:
        print(f"读取发病数据失败: {e}")
        return

    # 分析各时期的发病数据
    print(f"\n2. 分析各时期发病情况...")

    # 获取发病程度列
    severity_cols = df_occurrence.columns[3:6]  # 三个月的发病程度列
    print(f"发病程度列: {severity_cols.tolist()}")

    for i, col in enumerate(severity_cols):
        month_map = {0: "5-6月", 1: "7-8月", 2: "9-10月"}
        month_name = month_map[i]

        print(f"\n{month_name}发病情况:")
        severity_data = df_occurrence[col].dropna()

        # 统计各严重程度
        severity_counts = severity_data.value_counts().sort_index()
        print(f"  严重程度分布:")
        for severity, count in severity_counts.items():
            print(f"    程度{int(severity)}: {count}个县区")

        # 计算发生率 (Severity > 1)
        total_records = len(severity_data)
        occurrence_records = len(severity_data[severity_data > 1])
        occurrence_rate = occurrence_records / total_records if total_records > 0 else 0

        print(f"  总记录: {total_records}")
        print(f"  发生记录: {occurrence_records}")
        print(f"  发生率: {occurrence_rate:.2%}")

    # 分析数据覆盖范围
    print(f"\n3. 分析数据覆盖范围...")

    # 按年份统计
    year_stats = df_occurrence.groupby('Year').agg({
        'County': 'count',
        df_occurrence.columns[3]: lambda x: (x > 1).sum(),
        df_occurrence.columns[4]: lambda x: (x > 1).sum(),
        df_occurrence.columns[5]: lambda x: (x > 1).sum()
    }).round(2)

    year_stats.columns = ['总县区数', '5-6月发生数', '7-8月发生数', '9-10月发生数']
    print("按年份统计:")
    print(year_stats)

    # 分析县区覆盖情况
    print(f"\n4. 分析县区覆盖情况...")
    county_stats = df_occurrence.groupby('County').agg({
        'Year': 'count',
        df_occurrence.columns[3]: 'mean',
        df_occurrence.columns[4]: 'mean',
        df_occurrence.columns[5]: 'mean'
    }).round(2)

    county_stats.columns = ['年份数', '5-6月平均严重程度', '7-8月平均严重程度', '9-10月平均严重程度']
    print(f"覆盖县区数: {len(county_stats)}")
    print("部分县区统计:")
    print(county_stats.head(10))

    # 检查是否有其他发病数据
    print(f"\n5. 检查其他可能的发病数据...")
    import os
    data_files = []

    for root, dirs, files in os.walk('datas/shandong_pest_data'):
        for file in files:
            if any(keyword in file.lower() for keyword in ['occurrence', 'pest', 'moth', '发生', '害虫']):
                data_files.append(os.path.join(root, file))

    print("找到的相关数据文件:")
    for file in data_files:
        print(f"  - {file}")

    # 如果有其他发病数据，也进行分析
    other_occurrence_files = [
        'datas/shandong_pest_data/shandong_fall_webworm_occurrences_20250926_221822.csv',
        'datas/shandong_pest_data/shandong_american_moth_processed.csv'
    ]

    for file in other_occurrence_files:
        if os.path.exists(file):
            print(f"\n6. 分析文件: {file}")
            try:
                df = pd.read_csv(file, encoding='utf-8-sig')
                print(f"  形状: {df.shape}")
                print(f"  列名: {df.columns.tolist()[:10]}...")

                if 'Severity' in df.columns:
                    print(f"  严重程度分布:")
                    print(df['Severity'].value_counts())
                if 'Has_Occurrence' in df.columns:
                    print(f"  发生分布:")
                    print(df['Has_Occurrence'].value_counts())
                    print(f"  发生率: {df['Has_Occurrence'].mean():.2%}")

            except Exception as e:
                print(f"  读取失败: {e}")

def analyze_why_few_positive():
    """分析为什么正样本少"""
    print(f"\n" + "="*60)
    print("为什么正样本这么少？")
    print("="*60)

    print("可能的原因:")
    print("1. 美国白蛾本身就不是每年每县都发生")
    print("2. 发病数据可能只记录了严重发生的情况")
    print("3. 我们的数据处理逻辑可能过于严格")
    print("4. 可能存在其他未使用的发病数据源")
    print("5. 气象数据覆盖不全，导致很多县区月份被排除")

    print(f"\n建议的解决方案:")
    print("1. 检查是否有其他发病数据源")
    print("2. 降低Severity的阈值 (比如 Severity >= 1 就算发生)")
    print("3. 扩展气象数据的覆盖范围")
    print("4. 使用数据增强技术")
    print("5. 考虑使用更宽松的年份范围")

def suggest_improvements():
    """建议改进方案"""
    print(f"\n" + "="*60)
    print("改进建议")
    print("="*60)

    print("方案1: 降低发病阈值")
    print("- 当前: Severity > 1 才算发生")
    print("- 建议: Severity >= 1 就算发生 (即所有有记录的都算发生)")
    print("- 预期效果: 发生率可能提升到20-30%")

    print(f"\n方案2: 查找更多数据源")
    print("- 检查是否有其他病虫害监测数据")
    print("- 查找林业部门的年度报告")
    print("- 寻找农业部门的病虫害数据")

    print(f"\n方案3: 扩大时间范围")
    print("- 如果有更多年份的数据，可以增加正样本")
    print("- 考虑使用相邻年份的数据")

    print(f"\n方案4: 地理扩展")
    print("- 考虑使用周边省份的数据")
    print("- 或者专注于美国白蛾高发区域")

if __name__ == "__main__":
    analyze_occurrence_data()
    analyze_why_few_positive()
    suggest_improvements()