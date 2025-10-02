
# run_smoothed_visualize.py - 可视化平滑处理前后的风险地图对比

import os
import sys
import traceback
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import show
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import pandas as pd
from sd_raster_prediction.config_raster_new import get_config

def print_header(text):
    """打印带有格式的标题"""
    line = "=" * 80
    print(f"\n{line}")
    print(f"{text.center(80)}")
    print(f"{line}\n")

def create_custom_colormap(name, colors):
    """创建自定义颜色映射"""
    return LinearSegmentedColormap.from_list(name, colors)

def create_comparison_visualization(original_path, smoothed_path, output_path, title, is_categorical=False):
    """
    创建原始和平滑后的栅格对比可视化
    
    参数:
        original_path: 原始栅格文件路径
        smoothed_path: 平滑后的栅格文件路径
        output_path: 输出图像路径
        title: 图像标题
        is_categorical: 是否为分类数据（如风险等级）
    """
    # 读取原始和平滑后的栅格
    with rasterio.open(original_path) as src1:
        original_data = src1.read(1)
        nodata_value = src1.nodata
        if nodata_value is not None:
            original_data = np.ma.masked_equal(original_data, nodata_value)
            
    with rasterio.open(smoothed_path) as src2:
        smoothed_data = src2.read(1)
        if nodata_value is not None:
            smoothed_data = np.ma.masked_equal(smoothed_data, nodata_value)
    
    # 创建图像
    plt.figure(figsize=(15, 8))
    
    # 设置颜色映射
    if is_categorical:
        # 风险等级的颜色映射
        colors = [
            '#0000FF',  # 蓝色 - 无风险 (0)
            '#00FFFF',  # 青色 - 低风险 (1)
            '#FFFF00',  # 黄色 - 中风险 (2)
            '#FF7F00',  # 橙色 - 高风险 (3)
            '#FF0000'   # 红色 - 极高风险 (4)
        ]
        cmap = create_custom_colormap('risk_levels', colors)
        vmin, vmax = 0, 4
        tick_labels = ['无风险', '低风险', '中风险', '高风险', '极高风险']
        ticks = [0, 1, 2, 3, 4]
    else:
        # 概率值的颜色映射
        cmap = create_custom_colormap('probability', [
            '#0000FF',  # 蓝色 - 低概率
            '#00FFFF',  # 青色
            '#00FF00',  # 绿色
            '#FFFF00',  # 黄色
            '#FF7F00',  # 橙色
            '#FF0000'   # 红色 - 高概率
        ])
        vmin, vmax = 0, 1
        tick_labels = None
        ticks = None
    
    # 绘制原始数据
    plt.subplot(1, 2, 1)
    img1 = plt.imshow(original_data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title("原始数据", fontsize=14)
    plt.axis('off')
    
    # 添加颜色条
    if is_categorical:
        cbar1 = plt.colorbar(img1, ticks=ticks, orientation='vertical', shrink=0.7)
        if tick_labels:
            cbar1.set_ticklabels(tick_labels)
    else:
        plt.colorbar(img1, orientation='vertical', shrink=0.7)
    
    # 绘制平滑后的数据
    plt.subplot(1, 2, 2)
    img2 = plt.imshow(smoothed_data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title("平滑后数据", fontsize=14)
    plt.axis('off')
    
    # 添加颜色条
    if is_categorical:
        cbar2 = plt.colorbar(img2, ticks=ticks, orientation='vertical', shrink=0.7)
        if tick_labels:
            cbar2.set_ticklabels(tick_labels)
    else:
        plt.colorbar(img2, orientation='vertical', shrink=0.7)
    
    # 设置总标题
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存到: {output_path}")

def create_difference_visualization(original_path, smoothed_path, output_path, title):
    """
    创建平滑前后数据差异可视化
    
    参数:
        original_path: 原始栅格文件路径
        smoothed_path: 平滑后的栅格文件路径
        output_path: 输出图像路径
        title: 图像标题
    """
    # 读取原始和平滑后的栅格
    with rasterio.open(original_path) as src1:
        original_data = src1.read(1)
        nodata_value = src1.nodata
        if nodata_value is not None:
            original_data = np.ma.masked_equal(original_data, nodata_value)
            
    with rasterio.open(smoothed_path) as src2:
        smoothed_data = src2.read(1)
        if nodata_value is not None:
            smoothed_data = np.ma.masked_equal(smoothed_data, nodata_value)
    
    # 计算差异
    difference = smoothed_data - original_data
    
    # 创建图像
    plt.figure(figsize=(10, 8))
    
    # 为差异创建发散色彩映射
    diff_cmap = create_custom_colormap('diff', [
        '#0000FF',  # 蓝色 - 负差异（平滑后值减小）
        '#FFFFFF',  # 白色 - 无变化
        '#FF0000'   # 红色 - 正差异（平滑后值增大）
    ])
    
    # 计算差异范围，使用对称的范围
    max_diff = max(abs(np.nanmin(difference)), abs(np.nanmax(difference)))
    vmin, vmax = -max_diff, max_diff
    
    # 绘制差异图
    img = plt.imshow(difference, cmap=diff_cmap, vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=14)
    plt.axis('off')
    
    # 添加颜色条
    cbar = plt.colorbar(img, orientation='vertical', shrink=0.7)
    cbar.set_label('差异值 (平滑后 - 原始)')
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"差异图已保存到: {output_path}")

def create_histogram_comparison(original_path, smoothed_path, output_path, title, is_categorical=False):
    """
    创建平滑前后值分布直方图对比
    
    参数:
        original_path: 原始栅格文件路径
        smoothed_path: 平滑后的栅格文件路径
        output_path: 输出图像路径
        title: 图像标题
        is_categorical: 是否为分类数据
    """
    # 读取原始和平滑后的栅格
    with rasterio.open(original_path) as src1:
        original_data = src1.read(1)
        nodata_value = src1.nodata
        if nodata_value is not None:
            original_data = np.ma.masked_equal(original_data, nodata_value)
            
    with rasterio.open(smoothed_path) as src2:
        smoothed_data = src2.read(1)
        if nodata_value is not None:
            smoothed_data = np.ma.masked_equal(smoothed_data, nodata_value)
    
    # 创建图像
    plt.figure(figsize=(12, 6))
    
    if is_categorical:
        # 分类数据的柱状图比较
        categories = np.arange(5)  # 假设有5个风险类别
        category_labels = ['无风险', '低风险', '中风险', '高风险', '极高风险']
        
        # 计算每个类别的像素数量
        original_counts = []
        smoothed_counts = []
        for i in categories:
            original_counts.append(np.sum(original_data == i))
            smoothed_counts.append(np.sum(smoothed_data == i))
        
        # 计算百分比
        total_original = sum(original_counts)
        total_smoothed = sum(smoothed_counts)
        original_pct = [100 * c / total_original for c in original_counts]
        smoothed_pct = [100 * c / total_smoothed for c in smoothed_counts]
        
        # 设置柱状图的位置
        x = np.arange(len(categories))
        width = 0.35
        
        # 绘制柱状图
        plt.bar(x - width/2, original_pct, width, label='原始数据', color='lightblue')
        plt.bar(x + width/2, smoothed_pct, width, label='平滑后数据', color='salmon')
        
        plt.ylabel('百分比 (%)')
        plt.title(title)
        plt.xticks(x, category_labels, rotation=45)
        plt.legend()
        
    else:
        # 连续数据的直方图比较
        # 扁平化数据并移除掩码值
        original_flat = original_data.compressed()
        smoothed_flat = smoothed_data.compressed()
        
        # 计算直方图的公共范围
        min_val = min(np.min(original_flat), np.min(smoothed_flat))
        max_val = max(np.max(original_flat), np.max(smoothed_flat))
        
        # 计算直方图
        bins = np.linspace(min_val, max_val, 30)
        
        plt.hist(original_flat, bins=bins, alpha=0.5, label='原始数据', color='blue')
        plt.hist(smoothed_flat, bins=bins, alpha=0.5, label='平滑后数据', color='red')
        
        plt.xlabel('值')
        plt.ylabel('像素数量')
        plt.title(title)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"直方图对比已保存到: {output_path}")

def create_detail_zoom_comparison(original_path, smoothed_path, output_path, title, bbox=None):
    """
    创建局部区域放大的对比可视化
    
    参数:
        original_path: 原始栅格文件路径
        smoothed_path: 平滑后的栅格文件路径
        output_path: 输出图像路径
        title: 图像标题
        bbox: 放大区域的边界框 (行开始, 行结束, 列开始, 列结束)
    """
    # 读取原始和平滑后的栅格
    with rasterio.open(original_path) as src1:
        original_data = src1.read(1)
        nodata_value = src1.nodata
        if nodata_value is not None:
            original_data = np.ma.masked_equal(original_data, nodata_value)
        
        # 如果没有提供边界框，选择中心区域的一小部分
        if bbox is None:
            h, w = original_data.shape
            h_center, w_center = h // 2, w // 2
            h_range, w_range = h // 8, w // 8
            bbox = (
                max(0, h_center - h_range),
                min(h, h_center + h_range),
                max(0, w_center - w_range),
                min(w, w_center + w_range)
            )
            
    with rasterio.open(smoothed_path) as src2:
        smoothed_data = src2.read(1)
        if nodata_value is not None:
            smoothed_data = np.ma.masked_equal(smoothed_data, nodata_value)
    
    # 截取局部区域
    row_start, row_end, col_start, col_end = bbox
    original_zoom = original_data[row_start:row_end, col_start:col_end]
    smoothed_zoom = smoothed_data[row_start:row_end, col_start:col_end]
    
    # 确定是否为分类数据
    is_categorical = 'risk_class' in original_path
    
    # 设置颜色映射
    if is_categorical:
        # 风险等级的颜色映射
        colors = [
            '#0000FF',  # 蓝色 - 无风险 (0)
            '#00FFFF',  # 青色 - 低风险 (1)
            '#FFFF00',  # 黄色 - 中风险 (2)
            '#FF7F00',  # 橙色 - 高风险 (3)
            '#FF0000'   # 红色 - 极高风险 (4)
        ]
        cmap = create_custom_colormap('risk_levels', colors)
        vmin, vmax = 0, 4
    else:
        # 概率值的颜色映射
        cmap = create_custom_colormap('probability', [
            '#0000FF',  # 蓝色 - 低概率
            '#00FFFF',  # 青色
            '#00FF00',  # 绿色
            '#FFFF00',  # 黄色
            '#FF7F00',  # 橙色
            '#FF0000'   # 红色 - 高概率
        ])
        vmin, vmax = np.min(original_zoom), np.max(original_zoom)
    
    # 创建图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 绘制原始数据
    im1 = ax1.imshow(original_zoom, cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title("原始数据 (局部)", fontsize=12)
    ax1.axis('off')
    
    # 绘制平滑后的数据
    im2 = ax2.imshow(smoothed_zoom, cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_title("平滑后数据 (局部)", fontsize=12)
    ax2.axis('off')
    
    # 添加颜色条
    cbar = fig.colorbar(im2, ax=[ax1, ax2], orientation='horizontal', shrink=0.8, pad=0.05)
    
    # 设置总标题
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"局部区域对比图已保存到: {output_path}")

def main():
    """可视化平滑处理前后的风险地图对比"""
    print_header("空间平滑效果可视化工具")
    
    print("这个脚本用于比较空间平滑处理前后的风险预测地图差异。")
    print("它生成多种可视化图形，包括直接对比、差异图和分布直方图。\n")
    
    # 获取配置
    try:
        config = get_config()
        prediction_dir = config['prediction_output_dir']
        smoothed_dir = os.path.join(prediction_dir, 'smoothed')
        
        # 创建可视化输出目录
        output_dir = os.path.join(smoothed_dir, 'visualizations')
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"读取配置时出错: {e}")
        traceback.print_exc()
        return
    
    # 检查平滑处理的文件是否存在
    if not os.path.exists(smoothed_dir):
        print(f"错误: 找不到平滑结果目录 {smoothed_dir}")
        print("请先运行 run_smooth.py 进行平滑处理")
        return
    
    # 检查需要比较的文件对
    file_pairs = []
    
    # 概率文件对比
    prob_original = os.path.join(prediction_dir, 'sd_probability.tif')
    prob_smoothed = os.path.join(smoothed_dir, 'smoothed_sd_probability.tif')
    prob_adaptive = os.path.join(smoothed_dir, 'adaptive_sd_probability.tif')
    
    if os.path.exists(prob_original) and os.path.exists(prob_smoothed):
        file_pairs.append(('概率数据 (高斯平滑)', prob_original, prob_smoothed, False))
    
    if os.path.exists(prob_original) and os.path.exists(prob_adaptive):
        file_pairs.append(('概率数据 (自适应平滑)', prob_original, prob_adaptive, False))
    
    # 风险等级文件对比
    risk_original = os.path.join(prediction_dir, 'sd_risk_class.tif')
    risk_smoothed = os.path.join(smoothed_dir, 'smoothed_sd_risk_class.tif')
    
    if os.path.exists(risk_original) and os.path.exists(risk_smoothed):
        file_pairs.append(('风险等级数据 (中值平滑)', risk_original, risk_smoothed, True))
    
    if not file_pairs:
        print("错误: 找不到可比较的文件对")
        return
    
    # 为每对文件创建可视化
    for desc, original, smoothed, is_categorical in file_pairs:
        print(f"\n正在为{desc}创建可视化...")
        
        # 1. 直接对比可视化
        comparison_path = os.path.join(output_dir, f"comparison_{os.path.basename(smoothed)}.png")
        create_comparison_visualization(original, smoothed, comparison_path, 
                                       f"平滑前后对比: {desc}", is_categorical)
        
        # 2. 差异可视化 (仅适用于连续数据)
        if not is_categorical:
            diff_path = os.path.join(output_dir, f"difference_{os.path.basename(smoothed)}.png")
            create_difference_visualization(original, smoothed, diff_path, 
                                          f"平滑前后差异: {desc}")
        
        # 3. 直方图对比可视化
        hist_path = os.path.join(output_dir, f"histogram_{os.path.basename(smoothed)}.png")
        create_histogram_comparison(original, smoothed, hist_path, 
                                   f"平滑前后分布对比: {desc}", is_categorical)
        
        # 4. 局部区域放大对比
        zoom_path = os.path.join(output_dir, f"zoom_{os.path.basename(smoothed)}.png")
        create_detail_zoom_comparison(original, smoothed, zoom_path, 
                                     f"局部区域平滑效果: {desc}")
    
    print("\n所有可视化已完成！")
    print(f"可视化结果已保存到: {output_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        traceback.print_exc()
        print("程序异常终止。") 