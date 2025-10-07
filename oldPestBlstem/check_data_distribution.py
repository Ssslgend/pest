# check_data_distribution.py
"""
检查病虫害数据的分布情况
"""

import pandas as pd
import numpy as np

def check_data_distribution():
    """检查各个数据文件的分布"""
    
    # 检查训练数据
    print("=== 检查训练数据分布 ===")
    train_data = pd.read_csv('datas/shandong_pest_data/spatial_train_data.csv')
    print(f"训练数据总记录数: {len(train_data)}")
    print(f"县数量: {train_data.county_name.nunique()}")
    print(f"时间范围: {train_data.year.min()}-{train_data.year.max()}")
    
    # Value_Class分布
    print("\nValue_Class分布:")
    vc_counts = train_data['Value_Class'].value_counts().sort_index()
    for val, count in vc_counts.items():
        print(f"  类别 {val}: {count} 条 ({count/len(train_data)*100:.1f}%)")
    
    # Has_Occurrence分布
    print("\nHas_Occurrence分布:")
    occ_counts = train_data['Has_Occurrence'].value_counts().sort_index()
    for val, count in occ_counts.items():
        status = "有病虫害" if val == 1 else "无病虫害"
        print(f"  {status}: {count} 条 ({count/len(train_data)*100:.1f}%)")
    
    # Occurrence_Intensity分布
    print("\nOccurrence_Intensity分布:")
    intensity_counts = train_data['Occurrence_Intensity'].value_counts().sort_index()
    for val, count in intensity_counts.items():
        print(f"  强度 {val}: {count} 条 ({count/len(train_data)*100:.1f}%)")
    
    # 按年份统计
    print("\n按年份统计:")
    yearly_stats = train_data.groupby('year')['Has_Occurrence'].agg(['count', 'sum', 'mean'])
    for year, stats in yearly_stats.iterrows():
        print(f"  {year}年: 总记录={stats['count']}, 发生次数={stats['sum']}, 发生率={stats['mean']*100:.1f}%")
    
    # 按县统计
    print("\n按县统计 (前10个):")
    county_stats = train_data.groupby('county_name')['Has_Occurrence'].agg(['count', 'sum', 'mean'])
    county_stats['发生率'] = county_stats['mean'] * 100
    print(county_stats.head(10))
    
    return train_data

if __name__ == "__main__":
    check_data_distribution()