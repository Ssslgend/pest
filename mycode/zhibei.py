# 环境安装（首次运行前执行）
# pip install rasterio pandas openpyxl

import rasterio
import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------------
# 1. 读取植被栅格（Esri Grid格式）
# ----------------------------------
# 指定Esri Grid文件夹路径（需包含hdr.adf）
veg_grid_path = Path(r"H:/data_new2025/fpr/植被类型与植被区划/veg/veg-100w")

with rasterio.open(veg_grid_path / "hdr.adf") as src:
    veg_data = src.read(1)
    crs = src.crs  # 获取坐标系
    transform = src.transform

# ----------------------------------
# 2. 读取Excel代码表并映射
# ----------------------------------
# 读取编码表（使用实际列名）
code_df = pd.read_excel(
    r"H:/data_new2025/fpr/植被类型与植被区划/veg/植被类型代码表.xlsx",
    engine='openpyxl'
)
# ------------------------- 修改第2部分代码 -------------------------
# 生成双向映射字典（数值→名称）
value_to_class = code_df.set_index("Value")["植被大类"].to_dict()

# 逆向映射字典（名称→数值，处理重复名称）
# 注意：若有重复植被大类，取第一个出现的Value值
class_to_value = {}
for value, cls in value_to_class.items():
    if cls not in class_to_value:  # 避免覆盖已有键
        class_to_value[cls] = value

# 处理原始栅格中的未定义值（设为-9999）
# 步骤1：将原始值转换为植被名称（未定义值设为"未知"）
classified_names = np.vectorize(lambda x: value_to_class.get(x, "未知"))(veg_data)

# 步骤2：将名称转回数值（未定义名称设为-9999）
classified_values = np.vectorize(lambda x: class_to_value.get(x, -9999))(classified_names)




# ----------------------------------
# 3. 保存为可移植的GeoTIFF
# ----------------------------------
output_meta = {
    "driver": "GTiff",
    "dtype": "int32",
    "nodata": -9999,
    "width": veg_data.shape[1],
    "height": veg_data.shape[0],
    "count": 1,
    "crs": crs,
    "transform": transform
}
# ------------------------- 修改第3部分元数据 -------------------------
output_meta["nodata"] = -9999  # 确保与缺失值一致
with rasterio.open(r"H:/data_new2025/fpr/zhibei.tif", "w", **output_meta) as dst:
    dst.write(classified_values.astype('int32'), 1)

print("处理完成！输出文件：zhibei.tif")