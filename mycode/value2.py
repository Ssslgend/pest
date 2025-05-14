import pandas as pd
import numpy as np

def get_pest_type(pest_name, df=None):
    """改进的害虫类型判断函数"""
    percentage_keywords = ['percentage', 'percent', '%', '率']
    numeric_keywords = ['number', 'count', 'num']
    
    pest_name = str(pest_name).lower()
    if any(kw in pest_name for kw in percentage_keywords):
        return 'percentage'
    elif any(kw in pest_name for kw in numeric_keywords):
        return 'numeric'
    else:
        # 如果提供了数据框，使用数据判断
        if df is not None:
            sample_value = df[df['Pest/Disease Name'] == pest_name]['Pestvalue'].median()
            return 'percentage' if sample_value <= 100 else 'numeric'
        else:
            # 默认处理为数值型
            return 'numeric'

def calculate_quantiles(data):
    """计算四分位数并生成分类边界"""
    nonzero_data = data[data > 0]
    if len(nonzero_data) == 0:
        return [0, 0, 0]
    
    return [
        np.percentile(nonzero_data, 25),
        np.percentile(nonzero_data, 50),
        np.percentile(nonzero_data, 75)
    ]

def classify_value(row, pest_type, quantiles=None, custom_bins=None):
    """优化后的分类逻辑"""
    x = row['Pestvalue']
    
    if pest_type == 'percentage':
        if x == 0:
            return 0, '无风险'
        elif x < 5:
            return 1, '低风险'
        elif 5 <= x < 20:
            return 2, '中风险'
        else:
            return 3, '高风险'
    else:
        if x == 0:
            return 0, '无风险'
        
        # 优先使用自定义分箱
        if custom_bins:
            for i, threshold in enumerate(custom_bins):
                if x <= threshold:
                    return i+1, ['低风险', '中风险', '高风险'][i]
            return 3, '高风险'
        elif quantiles:
            if x <= quantiles[0]:
                return 1, '低风险'
            elif x <= quantiles[1]:
                return 2, '中风险'
            elif x <= quantiles[2]:
                return 3, '中高风险'
            else:
                return 4, '高风险'
        else:
            # 默认固定阈值
            if x < 50:
                return 1, '低风险'
            elif 50 <= x < 200:
                return 2, '中风险'
            else:
                return 3, '高风险'

def process_pest_data(input_path, output_path, numeric_method='quantile', custom_bins=None):
    """改进的主处理函数"""
    try:
        # 增强数据读取兼容性
        df = pd.read_csv(
            input_path,
            encoding='gbk',
            na_values=['', ' ', 'NA', 'N/A', '-', 'NaN'],
            dtype={'Pestvalue': 'str'}
        )
        
        # 数据清洗加强
        df['Pestvalue'] = (
            df['Pestvalue']
            .str.replace(r'[^0-9\.]', '', regex=True)
            .replace('', '0')
            .astype(float)
            .fillna(0)
        )
        
        # 分组处理
        grouped = df.groupby('Pest/Disease Name', group_keys=False)
        
        def process_group(group):
            pest_name = group.name
            pest_type = get_pest_type(pest_name, df)  # 传入df参数
            
            print(f"\n处理害虫类型: {pest_name} ({pest_type})")
            
            quantiles = None
            if pest_type == 'numeric' and numeric_method == 'quantile':
                # 计算分组数据的四分位数
                quantiles = calculate_quantiles(group['Pestvalue'])
                print(f"计算分位数: Q1={quantiles[0]:.2f}, Q2={quantiles[1]:.2f}, Q3={quantiles[2]:.2f}")
            
            # 应用分类逻辑
            group[['Value_Class', 'Risk_Level']] = group.apply(
                lambda row: classify_value(row, pest_type, quantiles, custom_bins),
                axis=1,
                result_type='expand'
            )
            
            return group
        
        # 处理分组
        df = grouped.apply(process_group)
        
        # 结果验证
        print("\n分类结果统计:")
        print("风险等级分布:")
        print(df['Risk_Level'].value_counts().sort_index())
        print("\n数值类别分布:")
        print(df['Value_Class'].value_counts().sort_index())
        
        # 保存结果
        df.to_csv(output_path, index=False, encoding='gbk')
        print(f"\n处理完成！结果已保存至: {output_path}")
        
        return df
    
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    # 配置参数
    input_file = r"H:\data_new2025\bilstm\pest_rice_with_features_2.csv"
    output_file = r"H:\data_new2025\bilstm\pest_rice_with_features_2_classified.csv"
    
    # 自定义分箱阈值示例（根据业务需求调整）
    custom_bins = [500, 1500, 3000]  # 低风险:1-500, 中风险:501-1500, 高风险:1501-3000, 极高风险:>3000
    
    processed_data = process_pest_data(
        input_file,
        output_file,
        numeric_method='quantile',  # 选项: 'quantile' 或 'custom'
        custom_bins=custom_bins
    )