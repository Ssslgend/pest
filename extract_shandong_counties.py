#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取山东省所有县区列表
"""

import json
import pandas as pd

def extract_shandong_counties():
    """从shandong.json提取山东省所有县区"""
    print("提取山东省所有县区...")

    # 读取GeoJSON文件
    with open('datas/shandong_pest_data/shandong.json', 'r', encoding='utf-8') as f:
        geo_data = json.load(f)

    # 提取所有县区信息
    counties = []
    for feature in geo_data['features']:
        county_info = {
            'adcode': feature['properties']['adcode'],
            'name': feature['properties']['name'],
            'center': feature['properties']['center']
        }
        # centroid可能不存在
        if 'centroid' in feature['properties']:
            county_info['centroid'] = feature['properties']['centroid']
        counties.append(county_info)

    # 转换为DataFrame
    df_counties = pd.DataFrame(counties)
    df_counties['longitude'] = df_counties['center'].apply(lambda x: x[0])
    df_counties['latitude'] = df_counties['center'].apply(lambda x: x[1])

    print(f"山东省共有 {len(df_counties)} 个县区")
    print("前10个县区:")
    print(df_counties[['name', 'adcode', 'longitude', 'latitude']].head(10))

    # 保存县区列表
    df_counties.to_csv('datas/shandong_pest_data/shandong_all_counties.csv',
                      index=False, encoding='utf-8-sig')

    print("山东省县区列表已保存到: datas/shandong_pest_data/shandong_all_counties.csv")
    return df_counties

def check_occurrence_data_counties():
    """检查发病情况数据中包含的县区"""
    print("\n检查发病情况数据中的县区...")

    try:
        # 读取发病情况Excel文件
        df_occurrence = pd.read_excel('datas/shandong_pest_data/发病情况.xlsx')
        print(f"发病情况数据形状: {df_occurrence.shape}")
        print(f"列名: {df_occurrence.columns.tolist()}")

        # 提取县区名称
        counties_in_data = df_occurrence['County'].unique().tolist()
        print(f"发病数据包含 {len(counties_in_data)} 个县区")
        print("发病数据中的县区:")
        for county in sorted(counties_in_data):
            print(f"  - {county}")

        # 检查年份范围
        years = df_occurrence['Year'].unique().tolist()
        print(f"数据年份范围: {min(years)}-{max(years)}")

        return counties_in_data, years

    except Exception as e:
        print(f"读取发病情况数据失败: {e}")
        return [], []

if __name__ == "__main__":
    # 提取山东省所有县区
    df_all_counties = extract_shandong_counties()

    # 检查发病数据中的县区
    counties_in_data, years = check_occurrence_data_counties()

    # 对比分析
    if counties_in_data:
        all_counties = set(df_all_counties['name'].tolist())
        occurrence_counties = set(counties_in_data)

        missing_counties = all_counties - occurrence_counties
        extra_counties = occurrence_counties - all_counties

        print(f"\n对比分析:")
        print(f"山东省总县区数: {len(all_counties)}")
        print(f"发病数据县区数: {len(occurrence_counties)}")
        print(f"缺失县区数: {len(missing_counties)}")
        print(f"多余县区数: {len(extra_counties)}")

        if missing_counties:
            print(f"\n缺失的县区 (共{len(missing_counties)}个):")
            for county in sorted(missing_counties):
                print(f"  - {county}")

        if extra_counties:
            print(f"\n发病数据中多余的县区 (共{len(extra_counties)}个):")
            for county in sorted(extra_counties):
                print(f"  - {county}")