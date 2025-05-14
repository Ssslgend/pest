import pandas as pd
import numpy as np

# 读取CSV文件
input_file = r"datas\pest_rice.csv"
output_file = r"datas\pest_rice_filled.csv"

try:
    # 尝试使用GBK编码读取
    df = pd.read_csv(input_file, encoding='gbk')
except UnicodeDecodeError:
    # 如果GBK失败，尝试使用ISO-8859-1
    df = pd.read_csv(input_file, encoding='ISO-8859-1')

# 将Pestvalue列中的空值替换为0
df['Pestvalue'] = df['Pestvalue'].fillna(0)

# 保存处理后的文件
df.to_csv(output_file, index=False, encoding='gbk')

print(f"处理完成！文件已保存至: {output_file}")
print(f"原始数据中的空值数量: {df['Pestvalue'].isna().sum()}")