import os
import glob
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import CRS

def convert_utm_to_gcs(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历所有 TIFF 文件
    for tif_path in glob.glob(os.path.join(input_folder, "*.tif")):
        # 获取文件名
        file_name = os.path.basename(tif_path)
        output_path = os.path.join(output_folder, file_name)

        print(f"正在处理: {file_name}")

        with rasterio.open(tif_path) as src:
            # 获取源坐标系
            src_crs = src.crs
            print(f"源坐标系: {src_crs}")

            # 定义目标坐标系
            dst_crs = CRS.from_epsg(4326)  # GCS_WGS_1984

            # 计算转换参数
            transform, width, height = calculate_default_transform(
                src_crs, dst_crs, src.width, src.height, *src.bounds)

            # 创建输出文件
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=src.count,
                dtype=src.dtypes[0],
                crs=dst_crs,
                transform=transform,
            ) as dst:
                # 逐波段重投影
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src_crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest,
                    )

        print(f"转换完成: {output_path}")

# 使用示例
input_folder = r"H:\data_new2025\2019_2024_sd\20192024PET_250m"  # 输入文件夹路径
output_folder = "H:/data_new2025/2019_2024_sd/tongyizuobiao/lst"  # 输出文件夹路径
convert_utm_to_gcs(input_folder, output_folder)