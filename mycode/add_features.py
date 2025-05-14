import pandas as pd
import numpy as np

def calculate_composite_features(df):
    """计算复合特征"""
    # 1. 光照温度乘积 (LTP)
    df['LTP'] = df['SSH'] * (df['MaxT'] + df['MinT']) / 2
    
    # 2. 温度差 (TF)
    df['TF'] = df['MaxT'] - df['MinT']
    
    # 3. 降水温度比 (PTR)
    # 避免除以0的情况
    df['PTR'] = df['RF'] / ((df['MaxT'] + df['MinT']).replace(0, 0.1))
    
    # 4. 温湿度系数 (THC)
    df['THC'] = ((df['RH1'] + df['RH2']) / 2) / ((df['MaxT'] + df['MinT']) / 2).replace(0, 0.1)
    
    return df

def process_data(input_path, output_path):
    """处理数据并添加复合特征"""
    try:
        # 读取CSV文件
        df = pd.read_csv(
            input_path,
            encoding='gbk',
            na_values=['', ' ', 'NA', 'N/A', '-']
        )
        
        # 打印原始数据信息
        print("原始数据信息:")
        print(df.info())
        print("\n前5行原始数据:")
        print(df.head())
        
        # 确保所需的列存在
        required_columns = ['SSH', 'MaxT', 'MinT', 'RF', 'RH1', 'RH2']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"数据文件中缺少以下列: {missing_columns}")
        
        # 数据清理
        for col in required_columns:
            # 将非数字值替换为NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # 填充NaN值为该列的中位数
            df[col] = df[col].fillna(df[col].median())
        
        # 计算复合特征
        df = calculate_composite_features(df)
        
        # 打印新特征的信息
        print("\n新添加的特征:")
        print(df[['LTP', 'TF', 'PTR', 'THC']].describe())
        
        # 保存为新的CSV文件
        df.to_csv(output_path, index=False, encoding='gbk')
        print(f"\n数据处理完成，已保存至 {output_path}")
        
        # 打印新数据的基本信息
        print("\n新数据信息:")
        print(df.info())
        print("\n前5行新数据:")
        print(df.head())
        
    except Exception as e:
        print(f"处理文件时出错: {e}")
        print("请检查CSV文件的格式是否正确。")

if __name__ == "__main__":
    input_file = "datas/pest_rice_classified.csv"
    output_file = "datas/pest_rice_with_features.csv"
    
    process_data(input_file, output_file) 