import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from affine import Affine # Rasterio uses the 'affine' library for geotransforms
import numpy as np # For defining NoData value if needed
import os

def force_align_raster_rasterio(input_raster_path, output_raster_path,
                                expected_crs_epsg, expected_transform_params,
                                expected_rows, expected_cols,
                                resampling_method_str="bilinear"): # 默认值仍为bilinear，但调用时会修改
    """
    使用 Rasterio 将栅格强制对齐到指定的目标属性。

    Args:
        input_raster_path (str): 输入栅格的路径 (例如, "zhibei.tif").
        output_raster_path (str): 保存新的对齐后栅格的路径。
        expected_crs_epsg (int): 目标坐标系的 EPSG 代码 (例如, 4326).
        expected_transform_params (list): 仿射变换参数列表 (6个值)
                                         [a, b, c, d, e, f] 对应 Affine(a,b,c,d,e,f)
                                         c = 左上角x, f = 左上角y, a = x方向分辨率, e = y方向分辨率 (通常为负)
                                         示例: [0.01, 0.0, 73.50, 0.0, -0.01, 53.57]
        expected_rows (int): 输出栅格的行数。
        expected_cols (int): 输出栅格的列数。
        resampling_method_str (str): 重采样方法字符串 ("nearest", "bilinear", "cubic", 等).
    """
    print(f"--- 开始处理: {input_raster_path} ---")
    print(f"--- 输出至: {output_raster_path} ---")

    if not os.path.exists(input_raster_path):
        print(f"错误：输入栅格 '{input_raster_path}' 不存在。")
        return

    # 将重采样方法字符串转换为 Rasterio Resampling 枚举
    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "cubic_spline": Resampling.cubic_spline,
        "lanczos": Resampling.lanczos,
        "average": Resampling.average,
        "mode": Resampling.mode,
        "gauss": Resampling.gauss,
    }
    if resampling_method_str.lower() not in resampling_map:
        print(f"警告: 不支持的重采样方法 '{resampling_method_str}'. 将使用 'nearest' (适用于分类数据如植被).")
        # 对于植被这类分类数据，如果指定的方法无效，默认到 nearest 更安全
        resampling_method = Resampling.nearest
    else:
        resampling_method = resampling_map[resampling_method_str.lower()]

    # 1. 定义目标属性
    dst_crs = f'EPSG:{expected_crs_epsg}'
    dst_transform = Affine(expected_transform_params[0], expected_transform_params[1], expected_transform_params[2],
                           expected_transform_params[3], expected_transform_params[4], expected_transform_params[5])
    dst_height = expected_rows
    dst_width = expected_cols

    print(f"目标属性:")
    print(f"  CRS: {dst_crs}")
    print(f"  Transform: {dst_transform}")
    print(f"  尺寸 (高x宽): {dst_height} x {dst_width}")
    print(f"  重采样方法: {resampling_method_str} (实际使用: {resampling_method})")

    try:
        with rasterio.open(input_raster_path) as src:
            dst_kwargs = src.meta.copy()

            nodata_value = src.nodata
            if nodata_value is None:
                # 对于分类数据（如植被），通常有一个特定的整数NoData值，或者没有NoData值。
                # 如果源数据没有，我们可能需要根据数据情况设定，或者不设定（如果数据保证全覆盖）。
                # 这里我们暂时不主动设置，除非源数据有。如果重采样后需要填充，reproject会使用0或目标nodata。
                # 或者，如果确定一个通用的NoData值（例如255对于8位植被分类，或0）
                if src.dtypes[0].startswith('uint') or src.dtypes[0].startswith('int'):
                     nodata_value = 0 # 假设0可以作为分类数据的NoData，或者根据实际情况修改
                     print(f"警告: 源数据 '{src.name}' 没有NoData值. 为输出设置NoData为: {nodata_value} (针对分类数据)")
                elif src.dtypes[0].startswith('float'):
                     nodata_value = -32768.0
                     print(f"警告: 源数据 '{src.name}' 没有NoData值. 为输出设置NoData为: {nodata_value} (针对浮点型数据)")


            dst_kwargs.update({
                'crs': dst_crs,
                'transform': dst_transform,
                'width': dst_width,
                'height': dst_height,
                'nodata': nodata_value  # 确保NoData已设置
            })

            print(f"输出栅格元数据: {dst_kwargs}")

            with rasterio.open(output_raster_path, 'w', **dst_kwargs) as dst:
                for i in range(1, src.count + 1): # 遍历所有波段 (通常植被是单波段)
                    print(f"  正在处理波段 {i}...")
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=resampling_method, # 使用选择的重采样方法
                        dst_nodata=nodata_value
                    )
        print(f"--- 处理完成. 已保存对齐后的栅格: {output_raster_path} ---")

        # 可选: 验证输出
        with rasterio.open(output_raster_path) as as_aligned_ds:
            print("\n--- 输出栅格属性验证 ---")
            print(f"  CRS: {as_aligned_ds.crs}")
            print(f"  Transform: {as_aligned_ds.transform}")
            print(f"  尺寸 (高x宽): {as_aligned_ds.height} x {as_aligned_ds.width}")
            print(f"  NoData: {as_aligned_ds.nodata}")


    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

# --- 配置参数 ---
if __name__ == "__main__":
    # 1. 输入zhibei栅格的路径
    input_raster_filepath = r"H:/data_new2025/fpr/dem/china_dem_poxiang.tif"  # <--- 修改为您的 zhibei.tif 路径

    # 2. 对齐后的zhibei栅格输出路径
    output_aligned_filepath = r"H:/data_new2025/fpr/dem_1/china_dem_poxiang_rasterio_aligned_001deg.tif" # <--- 修改为您希望的输出路径

    # 3. 目标属性 (这些是0.01度分辨率网格的参数)
    target_crs_epsg = 4326
    target_transform_coeffs = [0.01, 0.00, 73.50, 0.00, -0.01, 53.57]
    target_rows = 5968
    target_columns = 7392

    # 4. 对于植被 (zhibei) 这类分类数据，必须使用 "nearest" (最近邻法)
    #    以避免在重采样过程中产生不存在的类别值。
    resampling_algorithm_for_categorical = "nearest"

    # --- 运行对齐函数 ---
    output_dir = os.path.dirname(output_aligned_filepath)
    if output_dir and not os.path.exists(output_dir):
        print(f"警告: 输出目录 '{output_dir}' 不存在。正在尝试创建...")
        try:
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")
        except Exception as e:
            print(f"创建输出目录失败: {e}")
            # exit() # 如果希望在目录创建失败时退出，可以取消此行注释

    force_align_raster_rasterio(input_raster_filepath, output_aligned_filepath,
                               target_crs_epsg, target_transform_coeffs,
                               target_rows, target_columns,
                               resampling_method_str=resampling_algorithm_for_categorical) # <--- 确保这里传递正确的方法