#!/usr/bin/env python
# sd_raster_prediction/spatial_smoothing.py
import os
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
import matplotlib.pyplot as plt
from rasterio.plot import show
import time

def apply_gaussian_smoothing(raster_array, nodata_value, sigma=1.0):
    """
    应用高斯平滑滤波器处理栅格数据，保持NoData值不变
    
    参数:
        raster_array: 输入栅格数组
        nodata_value: NoData值（不参与平滑计算）
        sigma: 高斯滤波器的标准差（值越大，平滑效果越强）
        
    返回:
        平滑后的栅格数组
    """
    # 创建掩膜，标记有效数据区域
    valid_mask = (raster_array != nodata_value)
    
    # 仅对有效数据应用高斯平滑
    result = np.copy(raster_array)
    valid_data = raster_array[valid_mask]
    
    # 应用高斯滤波
    smoothed_data = np.copy(valid_data)
    temp_array = np.copy(raster_array)
    temp_array[~valid_mask] = 0  # 临时将NoData设为0，便于滤波操作
    
    # 使用高斯滤波处理整个数组
    smoothed_array = gaussian_filter(temp_array, sigma=sigma)
    
    # 创建权重数组，对NoData区域进行高斯权重计算
    weight_array = np.ones_like(raster_array, dtype=np.float32)
    weight_array[~valid_mask] = 0
    weight_smoothed = gaussian_filter(weight_array, sigma=sigma)
    
    # 避免除以零
    weight_smoothed[weight_smoothed == 0] = 1
    
    # 通过权重归一化，修正NoData区域的影响
    final_smoothed = smoothed_array / weight_smoothed
    
    # 替换有效区域的值为平滑后的值
    result[valid_mask] = final_smoothed[valid_mask]
    
    return result

def apply_median_smoothing(raster_array, nodata_value, size=3):
    """
    应用中值滤波平滑栅格数据，保持NoData值不变
    
    参数:
        raster_array: 输入栅格数组
        nodata_value: NoData值（不参与平滑计算）
        size: 滤波器窗口大小，必须是奇数（如3、5、7等）
        
    返回:
        平滑后的栅格数组
    """
    # 创建掩膜，标记有效数据区域
    valid_mask = (raster_array != nodata_value)
    
    # 仅对有效数据应用中值滤波
    result = np.copy(raster_array)
    
    # 复制原始数组，将NoData值临时替换为可计算的极值
    temp_array = np.copy(raster_array)
    if np.isnan(nodata_value):
        # 如果NoData是NaN，用最小值替代
        temp_array[~valid_mask] = np.nanmin(temp_array[valid_mask]) - 1
    else:
        # 否则用一个极值
        temp_array[~valid_mask] = -9999
    
    # 应用中值滤波
    smoothed_array = median_filter(temp_array, size=size)
    
    # 仅更新有效区域的值
    result[valid_mask] = smoothed_array[valid_mask]
    
    return result

def apply_mean_smoothing(raster_array, nodata_value, size=3):
    """
    应用均值滤波平滑栅格数据，保持NoData值不变
    
    参数:
        raster_array: 输入栅格数组
        nodata_value: NoData值（不参与平滑计算）
        size: 滤波器窗口大小
        
    返回:
        平滑后的栅格数组
    """
    # 创建掩膜，标记有效数据区域
    valid_mask = (raster_array != nodata_value)
    
    # 仅对有效数据应用均值滤波
    result = np.copy(raster_array)
    
    # 使用与高斯滤波相似的权重方法处理均值滤波
    temp_array = np.copy(raster_array)
    temp_array[~valid_mask] = 0  # 临时将NoData设为0
    
    # 使用均值滤波处理整个数组
    smoothed_array = uniform_filter(temp_array, size=size)
    
    # 创建权重数组，计算每个像素周围的有效数据比例
    weight_array = np.ones_like(raster_array, dtype=np.float32)
    weight_array[~valid_mask] = 0
    weight_smoothed = uniform_filter(weight_array, size=size)
    
    # 避免除以零
    weight_smoothed[weight_smoothed == 0] = 1
    
    # 通过权重归一化
    final_smoothed = smoothed_array / weight_smoothed
    
    # 替换有效区域的值为平滑后的值
    result[valid_mask] = final_smoothed[valid_mask]
    
    return result

def apply_localized_smoothing(raster_array, nodata_value, method='gaussian', **params):
    """
    应用自适应局部平滑处理，根据局部区域的方差动态调整平滑强度
    
    参数:
        raster_array: 输入栅格数组
        nodata_value: NoData值
        method: 平滑方法，'gaussian'、'median'或'mean'
        params: 额外参数，如高斯平滑的sigma
        
    返回:
        平滑后的栅格数组
    """
    # 计算局部方差（使用3x3窗口）
    valid_mask = (raster_array != nodata_value)
    local_variance = np.zeros_like(raster_array, dtype=np.float32)
    
    # 复制原始数组，将NoData临时替换
    temp_array = np.copy(raster_array)
    temp_array[~valid_mask] = np.nanmean(raster_array[valid_mask])
    
    # 计算局部方差
    for i in range(1, raster_array.shape[0]-1):
        for j in range(1, raster_array.shape[1]-1):
            if valid_mask[i, j]:
                neighborhood = temp_array[i-1:i+2, j-1:j+2]
                local_variance[i, j] = np.var(neighborhood)
    
    # 根据方差调整平滑参数
    # 方差大的区域减弱平滑，方差小的区域增强平滑
    if method == 'gaussian':
        base_sigma = params.get('sigma', 1.0)
        max_sigma = base_sigma * 2.0
        
        # 归一化方差到0-1范围
        var_min = np.min(local_variance[valid_mask])
        var_max = np.max(local_variance[valid_mask])
        if var_max > var_min:
            norm_variance = (local_variance - var_min) / (var_max - var_min)
        else:
            norm_variance = np.zeros_like(local_variance)
        
        # 方差大的区域使用较小的sigma，方差小的区域使用较大的sigma
        adaptive_sigma = max_sigma - norm_variance * (max_sigma - base_sigma/2)
        
        result = np.copy(raster_array)
        for sigma in np.unique(adaptive_sigma[valid_mask]):
            mask = (adaptive_sigma == sigma) & valid_mask
            if np.any(mask):
                temp = np.copy(raster_array)
                temp[~mask] = nodata_value
                smoothed = apply_gaussian_smoothing(temp, nodata_value, sigma)
                result[mask] = smoothed[mask]
        
        return result
    
    elif method == 'median':
        base_size = params.get('size', 3)
        return apply_median_smoothing(raster_array, nodata_value, base_size)
    
    elif method == 'mean':
        base_size = params.get('size', 3)
        return apply_mean_smoothing(raster_array, nodata_value, base_size)
    
    else:
        print(f"未知平滑方法: {method}，使用默认高斯平滑")
        return apply_gaussian_smoothing(raster_array, nodata_value)

def smooth_raster_file(input_path, output_path, method='gaussian', visualize=False, **params):
    """
    平滑处理栅格文件并保存结果
    
    参数:
        input_path: 输入栅格文件路径
        output_path: 输出栅格文件路径
        method: 平滑方法，'gaussian'、'median'、'mean'或'adaptive'
        visualize: 是否可视化平滑前后的对比
        params: 额外参数，如高斯平滑的sigma
    """
    print(f"正在对 {os.path.basename(input_path)} 应用 {method} 平滑处理...")
    start_time = time.time()
    
    # 读取栅格文件
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        nodata_value = src.nodata
        input_array = src.read(1)
        
        # 对数据应用平滑处理
        if method == 'gaussian':
            sigma = params.get('sigma', 1.0)
            print(f"应用高斯平滑，sigma={sigma}")
            output_array = apply_gaussian_smoothing(input_array, nodata_value, sigma)
        elif method == 'median':
            size = params.get('size', 3)
            print(f"应用中值平滑，窗口大小={size}")
            output_array = apply_median_smoothing(input_array, nodata_value, size)
        elif method == 'mean':
            size = params.get('size', 3)
            print(f"应用均值平滑，窗口大小={size}")
            output_array = apply_mean_smoothing(input_array, nodata_value, size)
        elif method == 'adaptive':
            adapt_method = params.get('adapt_method', 'gaussian')
            print(f"应用自适应局部平滑，基础方法={adapt_method}")
            output_array = apply_localized_smoothing(input_array, nodata_value, adapt_method, **params)
        else:
            print(f"未知平滑方法: {method}，使用默认高斯平滑")
            output_array = apply_gaussian_smoothing(input_array, nodata_value)
    
    # 保存平滑后的栅格
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(output_array, 1)
    
    end_time = time.time()
    print(f"平滑处理完成，耗时 {end_time - start_time:.2f} 秒")
    print(f"结果已保存到: {output_path}")
    
    # 可视化对比（如果需要）
    if visualize:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        show(input_array, title="原始栅格")
        
        plt.subplot(1, 2, 2)
        show(output_array, title="平滑后栅格")
        
        # 保存图像到相同目录
        vis_path = os.path.join(os.path.dirname(output_path), 
                               f"smooth_comparison_{os.path.basename(output_path).split('.')[0]}.png")
        plt.tight_layout()
        plt.savefig(vis_path, dpi=300)
        plt.close()
        print(f"对比图已保存到: {vis_path}")

def process_batch_smoothing(input_dir, output_dir, method='gaussian', pattern='*.tif', **params):
    """
    批量处理目录中的栅格文件
    
    参数:
        input_dir: 输入栅格文件目录
        output_dir: 输出栅格文件目录
        method: 平滑方法
        pattern: 文件匹配模式
        params: 额外参数
    """
    import glob
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找匹配的文件
    files = glob.glob(os.path.join(input_dir, pattern))
    print(f"找到 {len(files)} 个匹配的栅格文件")
    
    for file_path in files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, f"smoothed_{filename}")
        
        try:
            smooth_raster_file(file_path, output_path, method, **params)
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")

if __name__ == "__main__":
    # 示例用法
    print("空间平滑处理模块。请从主程序调用相关函数。") 