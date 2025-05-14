import pandas as pd
import numpy as np

def get_pest_type(pest_name):
    """根据害虫名称判断数据类型"""
    # 检查是否包含百分比相关的关键词
    percentage_keywords = ['percentage', 'percent', '%', '率']
    if any(keyword in pest_name.lower() for keyword in percentage_keywords):
        return '百分比'
    else:
        return '数值'

def classify_percentage(x):
    """百分比型数据的分类"""
    if x == 0:                  # 无风险
        return 0, '无风险'
    elif 0 < x < 5:             # 低风险
        return 1, '低风险'
    elif 5 <= x < 20:           # 中风险
        return 2, '中风险'
    else:                       # 高风险
        return 3, '高风险'

def classify_numeric(x, bins=None, numeric_bins='quantile'):
    """数值型数据的分类"""
    if numeric_bins == 'quantile':
        # 使用四分位数自动划分
        q1 = np.percentile(x, 25)
        q2 = np.percentile(x, 50)
        q3 = np.percentile(x, 75)
        if x == 0:
            return 0, '无风险'
        elif 0 < x <= q1:
            return 1, '低风险'
        elif q1 < x <= q2:
            return 2, '中风险'
        else:
            return 3, '高风险'
    else:
        # 使用自定义阈值划分
        if x == 0:
            return 0, '无风险'
        for i, threshold in enumerate(bins):
            if x <= threshold:
                return i, ['无风险', '低风险', '中风险', '高风险'][i]
        return len(bins), '高风险'

def process_pest_data(input_path, output_path, numeric_bins='quantile', custom_bins=None):
    try:
        # 读取CSV文件
        df = pd.read_csv(
            input_path,
            encoding='gbk',
            na_values=['', ' ', 'NA', 'N/A', '-']
        )
        
        # 打印列名和数据类型以便调试
        print("CSV文件列名:", df.columns.tolist())
        print("\n数据类型:")
        print(df.dtypes)
        
        # 确保Pestvalue列存在
        if 'Pestvalue' not in df.columns:
            raise ValueError(f"数据文件中缺少Pestvalue列。现有列名: {df.columns.tolist()}")
        
        # 清理数据：移除非数字字符
        df['Pestvalue'] = df['Pestvalue'].astype(str).str.replace('[^0-9.]', '', regex=True)
        df['Pestvalue'] = df['Pestvalue'].replace('', '0')
        df['Pestvalue'] = pd.to_numeric(df['Pestvalue'], errors='coerce').fillna(0)
        
        # 按害虫类型分组处理
        results = []
        for pest_name, group in df.groupby('Pest/Disease Name'):
            pest_type = get_pest_type(pest_name)
            print(f"\n处理害虫: {pest_name}")
            print(f"类型: {pest_type}")
            
            if pest_type == '百分比':
                group[['Value_Class', 'Risk_Level']] = group['Pestvalue'].apply(
                    lambda x: pd.Series(classify_percentage(x))
                )
            else:
                group[['Value_Class', 'Risk_Level']] = group['Pestvalue'].apply(
                    lambda x: pd.Series(classify_numeric(x, bins=custom_bins, numeric_bins=numeric_bins))
                )
            
            results.append(group)
        
        # 合并所有处理后的数据
        df = pd.concat(results, ignore_index=True)
        
        # 验证分类结果
        print("\n各风险等级样本数:")
        print(df['Risk_Level'].value_counts().sort_index())
        print("\n各等级样本数:")
        print(df['Value_Class'].value_counts().sort_index())
        
        # 保存为CSV
        df.to_csv(output_path, index=False, encoding='gbk')
        print(f"\n数据处理完成，已保存至 {output_path}")
        
    except Exception as e:
        print(f"处理文件时出错: {e}")
        print("请检查CSV文件的格式是否正确。")

if __name__ == "__main__":
    input_file = r"H:\data_new2025\bilstm\pest_rice_filled.csv"
    output_file = r"H:\data_new2025\bilstm\pest_rice_classified.csv"
    
    # 数值型数据的自定义阈值（可选）
    custom_bins = [500, 1500, 3000]  # 示例：0-500, 501-1500, 1501-3000, >3000
    
    process_pest_data(
        input_file, 
        output_file,
        numeric_bins='quantile',  # 或 'custom'
        custom_bins=custom_bins
    )