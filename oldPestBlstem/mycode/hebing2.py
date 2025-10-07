import pandas as pd
import os
import glob

# 输入文件夹路径
input_folder = r"H:/data_new2025/2019_2024_sd/X_y/train/combined"  # 请根据需要修改路径
output_file = r"H:/data_new2025/2019_2024_sd/X_y/train/combined_output.csv"  # 输出文件路径

# 获取所有CSV文件
csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

# 检查是否找到CSV文件
if not csv_files:
    print("没有找到任何CSV文件，请检查输入文件夹路径。")
else:
    print(f"找到 {len(csv_files)} 个CSV文件。")

# 初始化一个列表来存储DataFrame
dataframes = []

# 读取所有CSV文件并存储到列表中
for csv_file in csv_files:
    df = pd.read_csv(csv_file, encoding='gbk')  # 或者使用其他适合的编码
    dataframes.append(df)

# 检查是否成功读取到DataFrame
if not dataframes:
    print("没有成功读取任何DataFrame，请检查CSV文件内容。")
else:
    # 找到所有文件的共同列名
    common_columns = set(dataframes[0].columns)  # 初始化为第一个DataFrame的列名
    for df in dataframes[1:]:
        common_columns &= set(df.columns)  # 逐个与后续DataFrame的列名取交集

    # 将共同列名转换为列表
    common_columns = list(common_columns)

    # 创建一个新的DataFrame，包含相同列的内容
    combined_df = pd.DataFrame(columns=common_columns)

    # 合并所有DataFrame中相同列的数据
    for df in dataframes:
        combined_df = pd.concat([combined_df, df[common_columns]], ignore_index=True)

    # 保存结果到新的CSV文件
    combined_df.to_csv(output_file, index=False)
    print(f"已保存: {output_file}")