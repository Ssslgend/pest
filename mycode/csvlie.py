import pandas as pd
import os
import glob

# 定义列名映射
bio_mapping = {
    1: "Annual Mean Temperature",
    2: "Mean Diurnal Range (Mean of monthly (max temp - min temp))",
    3: "Isothermality (BIO2/BIO7) (×100)",
    4: "Temperature Seasonality (standard deviation ×100)",
    5: "Max Temperature of Warmest Month",
    6: "Min Temperature of Coldest Month",
    7: "Temperature Annual Range (BIO5-BIO6)",
    8: "Mean Temperature of Wettest Quarter",
    9: "Mean Temperature of Driest Quarter",
    10: "Mean Temperature of Warmest Quarter",
    11: "Mean Temperature of Coldest Quarter",
    12: "Annual Precipitation",
    13: "Precipitation of Wettest Month",
    14: "Precipitation of Driest Month",
    15: "Precipitation Seasonality (Coefficient of Variation)",
    16: "Precipitation of Wettest Quarter",
    17: "Precipitation of Driest Quarter",
    18: "Precipitation of Warmest Quarter",
    19: "Precipitation of Coldest Quarter"
}

# 输入文件夹路径
input_folder = r"H:/data_new2025/2019_2024_sd/X_y"  # 请根据需要修改路径
output_folder = r"H:/data_new2025/2019_2024_sd/X_y/train"  # 输出文件夹

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历所有CSV文件
for csv_file in glob.glob(os.path.join(input_folder, "*.csv")):  # 遍历所有CSV文件
    # 读取CSV文件
    df = pd.read_csv(csv_file, encoding='gbk')  # 或者使用其他适合的编码

    # 修改列名
    new_columns = {}
    for col in df.columns:
        if col.startswith("wc2.1_30s_bio_"):
            # 提取数字
            bio_number = int(col.split('_')[-1])
            # 检查数字是否在映射范围内
            if bio_number in bio_mapping:
                new_columns[col] = bio_mapping[bio_number]

    # 更新列名
    df.rename(columns=new_columns, inplace=True)

    # 保存结果到新的CSV文件
    output_file = os.path.join(output_folder, f"modified_{os.path.basename(csv_file)}")
    df.to_csv(output_file, index=False)
    print(f"已保存: {output_file}")

print("所有文件处理完成。")