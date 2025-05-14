import rasterio
from rasterio.transform import Affine # 用于更稳健地比较变换参数
import numpy as np
import os # 引入 os 模块

def compare_raster_properties(raster_path1, raster_path2, float_precision=15):
    """
    以高精度比较两个栅格的关键地理空间属性。

    Args:
        raster_path1 (str): 第一个栅格的路径。
        raster_path2 (str): 第二个栅格的路径。
        float_precision (int): 浮点数比较时使用的小数位数。
    """
    print(f"--- 开始比较栅格属性 ---")
    print(f"栅格 1: {raster_path1}")
    print(f"栅格 2: {raster_path2}\n")

    if not rasterio.os.path.exists(raster_path1): # rasterio.os.path 已经被 os.path 替代
        print(f"错误：栅格 1 '{raster_path1}' 未找到。")
        return
    if not rasterio.os.path.exists(raster_path2): # rasterio.os.path 已经被 os.path 替代
        print(f"错误：栅格 2 '{raster_path2}' 未找到。")
        return

    try:
        with rasterio.open(raster_path1) as src1, rasterio.open(raster_path2) as src2:
            # 1. CRS (坐标参考系统)
            crs1 = src1.crs
            crs2 = src2.crs
            # 移除了 pretty=True 参数
            print(f"CRS 1 (WKT格式): {crs1.to_wkt()}")
            print(f"CRS 2 (WKT格式): {crs2.to_wkt()}")
            if crs1 == crs2:
                print("CRS 匹配: 是\n")
            else:
                # 尝试获取EPSG代码，如果失败则留空
                epsg1_str = ""
                epsg2_str = ""
                try:
                    epsg1_str = f"EPSG1: {crs1.to_epsg()}"
                except Exception:
                    epsg1_str = "EPSG1: (无法获取)"
                try:
                    epsg2_str = f"EPSG2: {crs2.to_epsg()}"
                except Exception:
                    epsg2_str = "EPSG2: (无法获取)"
                print(f"CRS 匹配: 否 ({epsg1_str}, {epsg2_str})\n")

            # 2. Transform (变换参数，Affine 对象)
            transform1 = src1.transform
            transform2 = src2.transform
            print(f"变换参数 1 (元素): {tuple(transform1)}") # 默认精度
            print(f"变换参数 2 (元素): {tuple(transform2)}")
            print(f"变换参数 1 (高精度): (a={transform1.a:.{float_precision}f}, b={transform1.b:.{float_precision}f}, c={transform1.c:.{float_precision}f}, d={transform1.d:.{float_precision}f}, e={transform1.e:.{float_precision}f}, f={transform1.f:.{float_precision}f})")
            print(f"变换参数 2 (高精度): (a={transform2.a:.{float_precision}f}, b={transform2.b:.{float_precision}f}, c={transform2.c:.{float_precision}f}, d={transform2.d:.{float_precision}f}, e={transform2.e:.{float_precision}f}, f={transform2.f:.{float_precision}f})")

            # 使用 Affine.almost_equals 进行稳健比较
            # 您可以调整 almost_equals 的精度
            if transform1.almost_equals(transform2, precision=1e-9): # 根据需要调整精度
                print("变换参数匹配 (almost_equals, 1e-9 精度): 是\n")
            else:
                print("变换参数匹配 (almost_equals, 1e-9 精度): 否\n")
                # 逐元素比较以进行调试
                elements = ['a (x方向分辨率)', 'b (x方向扭曲)', 'c (左上角x坐标)', 'd (y方向扭曲)', 'e (y方向分辨率)', 'f (左上角y坐标)']
                for i in range(6):
                    if not np.isclose(transform1[i], transform2[i], atol=1e-9):
                        print(f"  变换参数元素 {elements[i]} 不匹配: {transform1[i]:.{float_precision}f} vs {transform2[i]:.{float_precision}f}")


            # 3. Shape (形状 - 高度, 宽度)
            shape1 = (src1.height, src1.width)
            shape2 = (src2.height, src2.width)
            print(f"形状 1 (高度, 宽度): {shape1}")
            print(f"形状 2 (高度, 宽度): {shape2}")
            if shape1 == shape2:
                print("形状匹配: 是\n")
            else:
                print("形状匹配: 否\n")

            # 4. Bounds (范围 - 从变换参数和形状派生)
            bounds1 = src1.bounds
            bounds2 = src2.bounds
            print(f"范围 1 (左,下,右,上 高精度): ({bounds1.left:.{float_precision}f}, {bounds1.bottom:.{float_precision}f}, {bounds1.right:.{float_precision}f}, {bounds1.top:.{float_precision}f})")
            print(f"范围 2 (左,下,右,上 高精度): ({bounds2.left:.{float_precision}f}, {bounds2.bottom:.{float_precision}f}, {bounds2.right:.{float_precision}f}, {bounds2.top:.{float_precision}f})")
            if (np.isclose(bounds1.left, bounds2.left, atol=1e-9) and
                np.isclose(bounds1.bottom, bounds2.bottom, atol=1e-9) and
                np.isclose(bounds1.right, bounds2.right, atol=1e-9) and
                np.isclose(bounds1.top, bounds2.top, atol=1e-9)):
                print("范围匹配 (isclose, 1e-9 精度): 是\n")
            else:
                print("范围匹配 (isclose, 1e-9 精度): 否\n")


            # 5. Data Types (数据类型)
            dtypes1 = src1.dtypes
            dtypes2 = src2.dtypes
            print(f"数据类型 1: {dtypes1}")
            print(f"数据类型 2: {dtypes2}")
            if dtypes1 == dtypes2:
                print("数据类型匹配: 是\n")
            else:
                print("数据类型匹配: 否\n")


            # 6. NoData Values (NoData值)
            nodata1 = src1.nodatavals # 这是每个波段NoData值的元组
            nodata2 = src2.nodatavals
            print(f"NoData值 1: {nodata1}")
            print(f"NoData值 2: {nodata2}")
            # 小心比较浮点型的NoData值
            nodata_match = True
            if len(nodata1) == len(nodata2):
                for n1, n2 in zip(nodata1, nodata2):
                    if n1 is None and n2 is None:
                        continue
                    if (n1 is None and n2 is not None) or \
                       (n1 is not None and n2 is None) or \
                       (isinstance(n1, float) and not np.isclose(n1, n2, equal_nan=True)) or \
                       (not isinstance(n1, float) and n1 != n2) :
                        nodata_match = False
                        break
            else:
                nodata_match = False

            if nodata_match:
                print("NoData值匹配: 是\n")
            else:
                print("NoData值匹配: 否\n")


            # 7. Band Count (波段数量)
            count1 = src1.count
            count2 = src2.count
            print(f"波段数量 1: {count1}")
            print(f"波段数量 2: {count2}")
            if count1 == count2:
                print("波段数量匹配: 是\n")
            else:
                print("波段数量匹配: 否\n")

    except Exception as e:
        print(f"比较过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

# --- 配置 ---
# !!! 重要: 请将这些路径替换为您的实际文件路径 !!!
path_to_your_aligned_dem = r"H:/data_new2025/fpr/dem_1/china_dem_rasterio_aligned.tif" # 由前一个 Rasterio 脚本生成的文件
path_to_ACTUAL_reference_raster = r"H:/data_new2025/19_china/wc2.1_30s_bio_1.tif" # 您的“检查脚本”实际用作参考的*那个*文件

if __name__ == "__main__":
    if not os.path.exists(path_to_your_aligned_dem) or \
       not os.path.exists(path_to_ACTUAL_reference_raster) :
        print("错误：请确保上面配置的两个栅格文件路径 (path_to_your_aligned_dem 和 path_to_ACTUAL_reference_raster) 是正确的且文件存在。")
    else:
        compare_raster_properties(path_to_your_aligned_dem, path_to_ACTUAL_reference_raster)