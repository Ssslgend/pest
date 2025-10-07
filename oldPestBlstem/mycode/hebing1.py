import pandas as pd

# 输入文件路径
random_file = r"H:/data_new2025/2019_2024_sd/X_y/train/modified_random.csv"  # 请根据需要修改路径
year_file = r"H:/data_new2025/2019_2024_sd/X_y/train/modified_2024.csv"  # 请根据需要修改路径
output_file = r"H:/data_new2025/2019_2024_sd/X_y/train/combined_output2024.csv"  # 输出文件路径

# 读取CSV文件
df_random = pd.read_csv(random_file, encoding='gbk')  # 或者使用其他适合的编码
df_year = pd.read_csv(year_file, encoding='gbk')  # 或者使用其他适合的编码

# 找到相同的列名
common_columns = df_random.columns.intersection(df_year.columns)

# 创建一个新的DataFrame，包含相同列的内容
combined_df = pd.DataFrame(columns=common_columns)

# 使用pd.concat替代append
combined_df = pd.concat([df_year[common_columns], df_random[common_columns]], ignore_index=True)

# 保存结果到新的CSV文件
combined_df.to_csv(output_file, index=False)
print(f"已保存: {output_file}")