#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据质量测试脚本
"""

import pandas as pd
import numpy as np
import os

def test_data_quality():
    """测试数据质量"""
    print("测试县域数据质量...")

    # 测试县域数据
    county_file = 'datas/shandong_pest_data/county_level_firstgen_complete.csv'
    if os.path.exists(county_file):
        df = pd.read_csv(county_file, encoding='utf-8-sig')
        print(f"县域数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")

        # 检查标签分布
        if 'Has_Occurrence' in df.columns:
            print(f"Has_Occurrence 分布:")
            print(df['Has_Occurrence'].value_counts())

        if 'Severity_Level' in df.columns:
            print(f"Severity_Level 分布:")
            print(df['Severity_Level'].value_counts())

    # 测试真实发生数据
    real_file = 'datas/shandong_pest_data/real_occurrence_complete_data.csv'
    if os.path.exists(real_file):
        df = pd.read_csv(real_file, encoding='utf-8-sig')
        print(f"\n真实发生数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")

        # 检查标签分布
        if 'Has_Occurrence' in df.columns:
            print(f"Has_Occurrence 分布:")
            print(df['Has_Occurrence'].value_counts())

        if 'Severity' in df.columns:
            print(f"Severity 分布:")
            print(df['Severity'].value_counts())

if __name__ == "__main__":
    test_data_quality()