import os
import re
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject
from osgeo import gdal, osr

# ------------------------- 参数配置 ----------------------------
input_root = r"H:\data_new2025\fpr\NASA SRTM1 v3.0 30米精度 全国分省"
output_national = r"H:\data_new2025\fpr\dem\China_SRTM1000m1.tif"
target_epsg = 4326
target_res = 0.0083333333  # 1000米分辨率（约0.008333度）
compress_method = "LZW"
overview_levels = [2, 4, 8]
valid_sizes = {1201, 3601, 1801}  # 合法HGT文件尺寸

# ------------------------- 工具函数 ----------------------------
def get_province_name(folder):
    """增强省份名称提取，支持多种分隔符"""
    # 匹配格式: "SRTM1[vV]3.0[...]省份名" 或 "NASA SRTM1 v3.0 [...]省份名"
    match = re.search(
        r"(?:NASA\s+)?SRTM1\s*[vV]?3\.0[\s_\-]+([\u4e00-\u9fa5]+)", 
        folder, 
        flags=re.IGNORECASE
    )
    return match.group(1).strip() if match else folder

def validate_hgt_size(data, file_path):
    """保持原有验证逻辑不变"""
    size = int(np.sqrt(data.size))
    if size not in valid_sizes:
        raise ValueError(f"文件 {os.path.basename(file_path)} 尺寸异常 ({size}x{size})")
    return size

def read_hgt(file_path):
    """读取并验证HGT文件"""
    try:
        with open(file_path, 'rb') as f:
            data = np.fromfile(f, dtype=np.int16).astype(np.float32)
            data[data == -32768] = np.nan
            
            # 验证文件尺寸
            size = validate_hgt_size(data, file_path)
            return data.reshape(size, size), size
    except Exception as e:
        print(f"读取文件 {os.path.basename(file_path)} 失败: {str(e)}")
        return None, None

def resample_to_low_res(data, coord, output_path, target_res):
    """增强的重采样函数（处理内存数据）"""
    lat, lon = coord
    size = data.shape[0]
    
    # 创建原始数据的transform
    res = 1.0 / (size - 1)
    src_transform = rasterio.transform.from_origin(
        west=lon, 
        north=lat + 1,
        xsize=res,
        ysize=res
    )
    
    # 计算目标transform
    dst_transform, width, height = calculate_default_transform(
        CRS.from_epsg(4326),
        CRS.from_epsg(4326),
        data.shape[1],
        data.shape[0],
        *rasterio.transform.array_bounds(data.shape[0], data.shape[1], src_transform),
        resolution=target_res
    )
    
    # 执行重采样
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='float32',
        crs=CRS.from_epsg(4326),
        transform=dst_transform,
        compress=compress_method,
        nodata=np.nan,
        tiled=True
    ) as dst:
        dst.write(data, 1)

# ------------------------- 主流程 ----------------------------
def process_province(province_dir):
    """处理单个省份到1000米分辨率"""
    province_name = get_province_name(os.path.basename(province_dir))
    print(f"\n处理省份: {province_name}")
    
    # 收集HGT文件（兼容大小写）
    hgt_files = []
    for root, _, files in os.walk(province_dir):
        for f in files:
            if re.match(r"^[NSns]\d{2}[EeWw]\d{3}\.[Hh][Gg][Tt]$", f):
                full_path = os.path.join(root, f)
                hgt_files.append(full_path)
    
    if not hgt_files:
        print(f"跳过无数据省份: {province_name}")
        return None
    
    # 处理有效文件
    valid_tifs = []
    for hgt_path in hgt_files:
        print(f"处理文件: {os.path.basename(hgt_path)}")
        
        # 读取并验证数据
        data, size = read_hgt(hgt_path)
        if data is None:
            continue
        
        try:
            # 解析地理坐标
            filename = os.path.basename(hgt_path).upper().replace(".HGT", "")
            lat_sign = -1 if filename[0] == 'S' else 1
            lon_sign = -1 if filename[3] == 'W' else 1
            lat = lat_sign * int(filename[1:3])
            lon = lon_sign * int(filename[4:7])
            
            # 生成低分辨率版本
            output_path = os.path.join(province_dir, f"lowres_{os.path.basename(hgt_path)}.tif")
            resample_to_low_res(data, (lat, lon), output_path, target_res)
            valid_tifs.append(output_path)
            
        except Exception as e:
            print(f"处理失败: {str(e)}")
            continue

    # 合并省级DEM
    if not valid_tifs:
        print(f"省份 {province_name} 无可合并数据")
        return None
    
    province_output = os.path.join(province_dir, f"{province_name}_1000m.tif")
    try:
        # 使用VRT合并
        gdal.BuildVRT(vrt_path := os.path.join(province_dir, "temp.vrt"), valid_tifs)
        gdal.Translate(
            province_output, 
            vrt_path,
            creationOptions=[
                f"COMPRESS={compress_method}",
                "BIGTIFF=YES",
                "TILED=YES"
            ]
        )
    finally:
        if os.path.exists(vrt_path):
            os.remove(vrt_path)
        # 清理临时文件
        for f in valid_tifs:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except PermissionError:
                    print(f"警告: 无法删除临时文件 {f}")

    return province_output

def main():
    os.makedirs(os.path.dirname(output_national), exist_ok=True)
    
    # 处理所有省份（增强文件夹匹配）
    provincial_results = []
    for folder in os.listdir(input_root):
        province_dir = os.path.join(input_root, folder)
        # 正则匹配所有有效省份文件夹
        if os.path.isdir(province_dir) and re.search(
            r"^(NASA\s+)?SRTM1\s*[vV]?3\.0", 
            folder, 
            flags=re.IGNORECASE
        ):
            print(f"\n开始处理省份目录: {folder}")
            if result := process_province(province_dir):
                provincial_results.append(result)
    
    # 后续流程保持不变
    if not provincial_results:
        raise ValueError("没有找到可合并的省级DEM文件")
    
    print("\n合成全国DEM...")
    try:
        gdal.BuildVRT(vrt_path := output_national.replace(".tif", ".vrt"), provincial_results)
        gdal.Translate(
            output_national,
            vrt_path,
            creationOptions=[
                f"COMPRESS={compress_method}",
                "BIGTIFF=YES",
                "TILED=YES"
            ]
        )
    finally:
        if os.path.exists(vrt_path):
            os.remove(vrt_path)
    
    print("构建金字塔...")
    ds = gdal.Open(output_national, gdal.GA_Update)
    ds.BuildOverviews("AVERAGE", overview_levels)
    band = ds.GetRasterBand(1)
    band.ComputeStatistics(False)
    ds = None
    
    print(f"全国1000米DEM生成完成: {output_national}")

if __name__ == "__main__":
    main()