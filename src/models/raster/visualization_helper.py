#!/usr/bin/env python
# sd_raster_prediction/visualization_helper.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用黑体，其次微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像负号'-'显示为方块的问题
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from sd_raster_prediction.config_raster_new import get_config

def create_custom_colormap(name, colors):
    """创建自定义颜色映射"""
    return LinearSegmentedColormap.from_list(name, colors)

def setup_chinese_font():
    """
    检查系统上可用的中文字体并设置matplotlib
    """
    # 常见中文字体列表
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi', 
                     'STSong', 'STKaiti', 'STHeiti', 'STFangsong', 'Arial Unicode MS', 
                     'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei']
    
    # 检查系统上哪些字体可用
    available_fonts = []
    font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    
    for font_path in font_paths:
        try:
            font = fm.FontProperties(fname=font_path)
            font_name = font.get_name()
            if any(chinese in font_name for chinese in chinese_fonts):
                available_fonts.append(font_name)
        except:
            continue
    
    # 如果找到可用的中文字体，设置为默认字体
    if available_fonts:
        print(f"Found available font: {available_fonts[0]}")
        plt.rcParams['font.sans-serif'] = [available_fonts[0]] + plt.rcParams['font.sans-serif']
    else:
        # 尝试直接加载Windows系统字体
        try:
            windows_font_path = 'C:/Windows/Fonts/simhei.ttf'  # 黑体
            if os.path.exists(windows_font_path):
                print(f"Loading Windows SimHei font: {windows_font_path}")
                font_prop = fm.FontProperties(fname=windows_font_path)
                plt.rcParams['font.family'] = font_prop.get_family()
                fm.fontManager.addfont(windows_font_path)
            else:
                # 尝试微软雅黑
                windows_font_path = 'C:/Windows/Fonts/msyh.ttc'  # 微软雅黑
                if os.path.exists(windows_font_path):
                    print(f"Loading Windows YaHei font: {windows_font_path}")
                    font_prop = fm.FontProperties(fname=windows_font_path)
                    plt.rcParams['font.family'] = font_prop.get_family()
                    fm.fontManager.addfont(windows_font_path)
                else:
                    print("Warning: No suitable font found. Text may not display correctly.")
        except Exception as e:
            print(f"Failed to load Windows system font: {e}")
            print("Warning: No suitable font found. Text may not display correctly.")

def visualize_probability_map(probability_path, boundary_path, output_path, title="Pest Occurrence Probability", with_colorbar=True):
    """
    可视化概率栅格，并使用边界Shape文件裁剪
    
    参数:
        probability_path: 概率栅格文件路径
        boundary_path: 边界Shape文件路径
        output_path: 输出图像路径
        title: 图像标题
        with_colorbar: 是否显示颜色条
    """
    # 读取边界Shape文件
    if os.path.exists(boundary_path):
        boundary = gpd.read_file(boundary_path)
        has_boundary = True
    else:
        print(f"Warning: Boundary file {boundary_path} not found, showing full raster")
        has_boundary = False
    
    # 读取概率栅格
    with rasterio.open(probability_path) as src:
        # 获取掩膜区域
        if has_boundary:
            try:
                shapes = [geometry for geometry in boundary.geometry]
                masked_data, masked_transform = mask(src, shapes, crop=True, nodata=src.nodata)
                probability = masked_data[0]
                transform = masked_transform
            except Exception as e:
                print(f"Masking error: {e}")
                probability = src.read(1)
                transform = src.transform
        else:
            probability = src.read(1)
            transform = src.transform
            
        # 处理NoData值
        if src.nodata is not None:
            probability = np.ma.masked_equal(probability, src.nodata)
    
    # 创建颜色映射
    cmap = create_custom_colormap('risk', [
        '#0000FF',  # Blue - Low risk
        '#00FFFF',  # Cyan
        '#00FF00',  # Green
        '#FFFF00',  # Yellow
        '#FF7F00',  # Orange
        '#FF0000'   # Red - High risk
    ])
    
    # 创建图像
    plt.figure(figsize=(12, 10))
    img = plt.imshow(probability, cmap=cmap, vmin=0, vmax=1)
    
    if with_colorbar:
        cbar = plt.colorbar(img, orientation='vertical', shrink=0.7)
        cbar.set_label('Probability')
    
    plt.title(title, fontsize=14)
    plt.axis('off')
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Probability map saved to: {output_path}")

def visualize_risk_map(risk_path, boundary_path, output_path, title="Pest Risk Levels", with_colorbar=True):
    """
    可视化风险等级栅格，并使用边界Shape文件裁剪
    
    参数:
        risk_path: 风险等级栅格文件路径
        boundary_path: 边界Shape文件路径
        output_path: 输出图像路径
        title: 图像标题
        with_colorbar: 是否显示颜色条
    """
    # 读取边界Shape文件
    if os.path.exists(boundary_path):
        boundary = gpd.read_file(boundary_path)
        has_boundary = True
    else:
        print(f"Warning: Boundary file {boundary_path} not found, showing full raster")
        has_boundary = False
    
    # 读取风险等级栅格
    with rasterio.open(risk_path) as src:
        # 获取掩膜区域
        if has_boundary:
            try:
                shapes = [geometry for geometry in boundary.geometry]
                masked_data, masked_transform = mask(src, shapes, crop=True, nodata=255)
                risk = masked_data[0]
                transform = masked_transform
            except Exception as e:
                print(f"Masking error: {e}")
                risk = src.read(1)
                transform = src.transform
        else:
            risk = src.read(1)
            transform = src.transform
            
        # 处理NoData值
        risk = np.ma.masked_equal(risk, 255)
    
    # 创建颜色映射 - 五个风险等级
    colors = [
        '#0000FF',  # Blue - No risk (0)
        '#00FFFF',  # Cyan - Low risk (1)
        '#FFFF00',  # Yellow - Medium risk (2)
        '#FF7F00',  # Orange - High risk (3)
        '#FF0000'   # Red - Extreme risk (4)
    ]
    cmap = create_custom_colormap('risk_levels', colors)
    
    # 创建图像
    plt.figure(figsize=(12, 10))
    img = plt.imshow(risk, cmap=cmap, vmin=0, vmax=4)
    
    if with_colorbar:
        cbar = plt.colorbar(img, orientation='vertical', shrink=0.7, ticks=[0, 1, 2, 3, 4])
        cbar.set_ticklabels(['No Risk', 'Low Risk', 'Medium Risk', 'High Risk', 'Extreme Risk'])
    
    plt.title(title, fontsize=14)
    plt.axis('off')
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Risk level map saved to: {output_path}")

def create_probability_histogram(probability_path, output_path, bins=20, title="Probability Distribution Histogram"):
    """
    创建概率值直方图
    
    参数:
        probability_path: 概率栅格文件路径
        output_path: 输出图像路径
        bins: 直方图分组数量
        title: 图像标题
    """
    # 读取概率栅格
    with rasterio.open(probability_path) as src:
        probability = src.read(1)
        
        # 处理NoData值
        if src.nodata is not None:
            probability = np.ma.masked_equal(probability, src.nodata)
    
    # 转换为一维数组并过滤掉无效值
    valid_values = probability.compressed()
    
    # 创建直方图
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(valid_values, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    
    plt.title(title, fontsize=14)
    plt.xlabel('Probability Value', fontsize=12)
    plt.ylabel('Pixel Count', fontsize=12)
    plt.grid(axis='y', alpha=0.75, linestyle='--')
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Probability histogram saved to: {output_path}")

def create_risk_pie_chart(risk_path, output_path, title="Risk Level Distribution"):
    """
    创建风险等级分布饼图
    
    参数:
        risk_path: 风险等级栅格文件路径
        output_path: 输出图像路径
        title: 图像标题
    """
    # 读取风险等级栅格
    with rasterio.open(risk_path) as src:
        risk = src.read(1)
        
        # 处理NoData值
        risk = np.ma.masked_equal(risk, 255)
    
    # 统计各风险等级的像素数量
    risk_levels = {
        0: 'No Risk',
        1: 'Low Risk',
        2: 'Medium Risk',
        3: 'High Risk', 
        4: 'Extreme Risk'
    }
    
    counts = {}
    for level, name in risk_levels.items():
        counts[name] = np.sum(risk == level)
    
    # 过滤掉计数为0的风险等级
    filtered_counts = {k: v for k, v in counts.items() if v > 0}
    
    if not filtered_counts:
        print("Warning: No valid risk level data found")
        return
    
    # 设置颜色
    colors = ['#0000FF', '#00FFFF', '#FFFF00', '#FF7F00', '#FF0000']
    
    # 创建饼图
    plt.figure(figsize=(10, 8))
    labels = filtered_counts.keys()
    sizes = filtered_counts.values()
    
    plt.pie(sizes, labels=labels, colors=colors[:len(filtered_counts)], 
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')  # 确保饼图为正圆形
    plt.title(title, fontsize=14)
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Risk distribution pie chart saved to: {output_path}")

def create_all_visualizations(config=None):
    """
    生成所有可视化效果
    
    参数:
        config: 可选的配置字典，如果为None则使用默认配置
    """
    # 设置中文字体
    setup_chinese_font()
    
    # 获取配置
    if config is None:
        config = get_config()
    
    # 输入文件路径
    probability_path = config['prediction_tif_path']
    risk_path = config['prediction_risk_class_tif_path']
    boundary_path = config['boundary_shp_path']
    output_dir = config['prediction_output_dir']
    
    # 获取年份（如果有）添加到标题中
    year_str = ""
    if 'prediction_year' in config:
        year_str = f" ({config['prediction_year']}年)"
    
    # 输出文件路径
    prob_img_path = os.path.join(output_dir, 'probability_map.png')
    risk_img_path = os.path.join(output_dir, 'risk_map.png')
    hist_img_path = os.path.join(output_dir, 'probability_histogram.png')
    pie_img_path = os.path.join(output_dir, 'risk_distribution_pie.png')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查输入文件是否存在
    if os.path.exists(probability_path):
        print(f"生成概率分布图...")
        visualize_probability_map(
            probability_path, 
            boundary_path, 
            prob_img_path,
            title=f"山东美国白蛾病虫害发生概率分布{year_str}"
        )
        create_probability_histogram(
            probability_path, 
            hist_img_path,
            title=f"山东美国白蛾病虫害概率分布直方图{year_str}"
        )
    else:
        print(f"错误: 概率栅格文件 {probability_path} 不存在")
    
    if os.path.exists(risk_path):
        print(f"生成风险等级分布图...")
        visualize_risk_map(
            risk_path, 
            boundary_path, 
            risk_img_path,
            title=f"山东美国白蛾病虫害风险等级分布{year_str}"
        )
        create_risk_pie_chart(
            risk_path, 
            pie_img_path,
            title=f"山东美国白蛾病虫害风险等级比例{year_str}"
        )
    else:
        print(f"错误: 风险等级栅格文件 {risk_path} 不存在")

if __name__ == "__main__":
    print("Starting visualization generation...")
    create_all_visualizations()
    print("Visualization completed.") 