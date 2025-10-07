import pandas as pd
import rasterio
from rasterio.transform import rowcol
from rasterio.crs import CRS
import glob
import os
import numpy as np

# 配置路径
csv_path = "H:/data_new2025/2019_2024_sd/label/sd/random.csv"#输入的CSV文件路径
tif_folder = "H:/data_new2025/2019_2024_sd/tiqu"  # TIFF文件存放文件夹
output_csv = "H:/data_new2025/2019_2024_sd/X_y/random.csv"       # 输出CSV路径

# 读取CSV文件
df = pd.read_csv(csv_path, encoding='gbk')

# 确保存在经度纬度列
if '发生样点经度' not in df.columns or '发生样点纬度' not in df.columns:
    raise ValueError("CSV文件中必须包含发生样点经度和发生样点纬度列")

# 遍历所有TIFF文件
for tif_path in glob.glob(os.path.join(tif_folder, "*.tif")):
    # 获取TIFF文件名作为列名
    col_name = os.path.splitext(os.path.basename(tif_path))[0]
    print(f"正在处理: {col_name}")
    
    # 存储提取值的列表
    extracted_values = []
    
    # 打开TIFF文件
    with rasterio.open(tif_path) as src:
        # 获取坐标系和变换参数
        tif_crs = src.crs
        transform = src.transform
        
        print(f"Transform type: {type(transform)}")
        
        # 检查是否需要坐标转换（假设CSV是WGS84经纬度）
        if tif_crs != CRS.from_epsg(4326):
            # 创建坐标转换器
            from rasterio.warp import transform
            # 将经纬度转换为TIFF的坐标系
            coords = list(zip(df['发生样点经度'], df['发生样点纬度']))
            xs, ys = transform(CRS.from_epsg(4326), tif_crs, [x for x, y in coords], [y for x, y in coords])
        else:
            xs = df['发生样点经度'].values
            ys = df['发生样点纬度'].values
        
        # 遍历所有坐标点
        for x, y in zip(xs, ys):
            # 计算行列号
            try:
                row, col = rowcol(transform, x, y)
            except Exception as e:
                print(f"计算行列号时出错: {e}，坐标: ({x}, {y})")
                extracted_values.append(np.nan)
                continue
            
            # 检查是否在图像范围内
            if 0 <= row < src.height and 0 <= col < src.width:
                # 读取值（假设单波段）
                value = src.read(1)[row, col]
                # 处理nodata值
                if value == src.nodata or np.isnan(value):
                    extracted_values.append(np.nan)
                else:
                    extracted_values.append(float(value))
            else:
                print(f"坐标超出范围: ({x}, {y})")  # 打印超出范围的坐标
                extracted_values.append(np.nan)
    
    # 添加新列到DataFrame
    df[col_name] = extracted_values

# 保存结果
df.to_csv(output_csv, index=False)
print(f"处理完成，结果已保存到 {output_csv}")

# 打印经纬度的描述信息
print(df[['发生样点经度', '发生样点纬度']].describe())