#!/usr/bin/env python3
"""
检查Excel文件结构
"""

import pandas as pd

def check_excel_structure(excel_path):
    """检查Excel文件的结构"""
    print(f"检查Excel文件: {excel_path}")

    try:
        # 读取Excel文件
        df = pd.read_excel(excel_path)

        print(f"文件形状: {df.shape}")
        print(f"列名: {list(df.columns)}")

        # 显示前几行数据
        print("\n前10行数据:")
        print(df.head(10))

        # 显示数据类型
        print("\n数据类型:")
        print(df.dtypes)

        # 检查缺失值
        print("\n缺失值统计:")
        print(df.isnull().sum())

        # 查看唯一值
        for col in df.columns:
            if df[col].dtype == 'object':
                print(f"\n{col}列的唯一值数量: {df[col].nunique()}")
                if df[col].nunique() < 20:
                    print(f"唯一值: {df[col].unique()}")

        return df

    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return None

if __name__ == "__main__":
    excel_path = "./shandong_american_moth_occurrences.xlsx"
    df = check_excel_structure(excel_path)