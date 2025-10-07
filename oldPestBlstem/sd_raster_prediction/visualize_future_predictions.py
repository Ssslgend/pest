# sd_raster_prediction/visualize_future_predictions.py
import os
import sys
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import pandas as pd
from datetime import datetime
import argparse
from tqdm import tqdm

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入自定义模块
from config_future import get_future_config
from sd_raster_prediction.predict_raster import calculate_risk_distribution

def get_risk_colors():
    """获取风险等级对应的颜色"""
    return {
        0: '#2ca02c',  # 绿色 - 无风险
        1: '#1f77b4',  # 蓝色 - 低风险
        2: '#ff7f0e',  # 橙色 - 中风险
        3: '#d62728',  # 红色 - 高风险
        4: '#9467bd',  # 紫色 - 极高风险
        255: '#f0f0f0'  # 灰色 - NoData
    }

def get_risk_labels():
    """获取风险等级对应的标签"""
    return {
        0: '无风险',
        1: '低风险',
        2: '中风险',
        3: '高风险',
        4: '极高风险',
        255: 'NoData'
    }

def visualize_single_period(risk_tif_path, output_path, title=None, add_timestamp=True, with_basemap=False, boundary_shp=None):
    """
    可视化单个时期的风险分类图
    
    参数:
        risk_tif_path: 风险分类TIF文件路径
        output_path: 输出图像路径
        title: 图像标题，如果为None则使用默认标题
        add_timestamp: 是否添加时间戳
        with_basemap: 是否添加底图
        boundary_shp: 边界shapefile文件路径，用于掩码
    """
    try:
        # 读取风险分类栅格
        with rasterio.open(risk_tif_path) as src:
            risk_data = src.read(1)
            transform = src.transform
            crs = src.crs
            
            # 获取坐标范围
            bounds = src.bounds
            extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
            
            # 设置NoData值为255
            nodata_mask = (risk_data == src.nodata) if src.nodata is not None else (risk_data == 255)
            risk_data = np.ma.masked_array(risk_data, mask=nodata_mask)
        
        # 准备颜色映射
        risk_colors = get_risk_colors()
        risk_labels = get_risk_labels()
        cmap = mcolors.ListedColormap([risk_colors[0], risk_colors[1], risk_colors[2], risk_colors[3], risk_colors[4]])
        bounds = [0, 1, 2, 3, 4, 5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # 创建图形
        plt.figure(figsize=(12, 10))
        
        # 如果需要底图
        if with_basemap:
            try:
                import cartopy.crs as ccrs
                import cartopy.feature as cfeature
                
                # 创建地图投影
                ax = plt.axes(projection=ccrs.PlateCarree())
                
                # 添加自然地球特征
                ax.add_feature(cfeature.LAND, facecolor='#f2f2f2')
                ax.add_feature(cfeature.OCEAN, facecolor='#c8ebfa')
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linewidth=0.5)
                ax.add_feature(cfeature.RIVERS, linewidth=0.5)
                
                # 添加网格线
                gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                
                # 设置地图范围
                ax.set_extent([extent[0], extent[1], extent[2], extent[3]], crs=ccrs.PlateCarree())
                
                # 添加风险分类数据
                img = ax.imshow(
                    risk_data, 
                    cmap=cmap, 
                    norm=norm, 
                    extent=extent, 
                    transform=ccrs.PlateCarree(),
                    alpha=0.8
                )
            except ImportError:
                print("警告: 未安装cartopy，使用基本绘图")
                with_basemap = False
        
        # 如果不使用底图或加载底图失败
        if not with_basemap:
            img = plt.imshow(risk_data, cmap=cmap, norm=norm, extent=extent)
            plt.axis('off')
        
        # 添加边界
        if boundary_shp is not None:
            try:
                import geopandas as gpd
                boundary = gpd.read_file(boundary_shp)
                if with_basemap:
                    boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
                else:
                    ax = plt.gca()
                    boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
            except Exception as e:
                print(f"警告: 加载边界文件出错: {e}")
        
        # 添加颜色条和图例
        legend_elements = [
            Patch(facecolor=risk_colors[0], label=risk_labels[0]),
            Patch(facecolor=risk_colors[1], label=risk_labels[1]),
            Patch(facecolor=risk_colors[2], label=risk_labels[2]),
            Patch(facecolor=risk_colors[3], label=risk_labels[3]),
            Patch(facecolor=risk_colors[4], label=risk_labels[4])
        ]
        plt.legend(handles=legend_elements, title='风险等级', loc='lower right')
        
        # 添加标题
        if title:
            plt.title(title, fontsize=16)
        else:
            plt.title('病虫害风险分类图', fontsize=16)
        
        # 添加时间戳
        if add_timestamp:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            plt.figtext(0.99, 0.01, f'生成时间: {timestamp}', ha='right', fontsize=8)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"风险分类图已保存到: {output_path}")
        return True
    
    except Exception as e:
        print(f"可视化时出错: {e}")
        return False

def visualize_difference(base_tif_path, compare_tif_path, output_path, title=None):
    """
    可视化两个时期之间的风险变化
    
    参数:
        base_tif_path: 基准时期风险分类TIF文件路径
        compare_tif_path: 比较时期风险分类TIF文件路径
        output_path: 输出图像路径
        title: 图像标题，如果为None则使用默认标题
    """
    try:
        # 读取基准时期栅格
        with rasterio.open(base_tif_path) as src:
            base_data = src.read(1)
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
            base_nodata = src.nodata if src.nodata is not None else 255
        
        # 读取比较时期栅格
        with rasterio.open(compare_tif_path) as src:
            compare_data = src.read(1)
            compare_nodata = src.nodata if src.nodata is not None else 255
        
        # 创建掩码（两者都有有效数据的区域）
        valid_mask = (base_data != base_nodata) & (compare_data != compare_nodata)
        
        # 计算差异（比较时期 - 基准时期）
        diff_data = np.zeros_like(base_data)
        diff_data[valid_mask] = compare_data[valid_mask] - base_data[valid_mask]
        
        # 差异图的值范围：-4 到 4
        # -4: 从极高风险变为无风险
        # -3, -2, -1: 风险等级下降
        # 0: 风险等级不变
        # 1, 2, 3: 风险等级上升
        # 4: 从无风险变为极高风险
        
        # 设置颜色映射
        colors = [
            '#67001f',  # 深红 - 风险显著下降 (-4)
            '#d6604d',  # 红色 - 风险中度下降 (-3)
            '#f4a582',  # 粉红 - 风险轻度下降 (-2, -1)
            '#f7f7f7',  # 白色 - 风险不变 (0)
            '#92c5de',  # 浅蓝 - 风险轻度上升 (1, 2)
            '#4393c3',  # 蓝色 - 风险中度上升 (3)
            '#053061',  # 深蓝 - 风险显著上升 (4)
        ]
        
        # 创建自定义颜色映射
        cmap = mcolors.LinearSegmentedColormap.from_list('diff_cmap', colors, N=9)
        bounds = [-4.5, -3.5, -2.5, -0.5, 0.5, 2.5, 3.5, 4.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # 创建图形
        plt.figure(figsize=(12, 10))
        img = plt.imshow(diff_data, cmap=cmap, norm=norm, extent=extent)
        plt.axis('off')
        
        # 添加颜色条
        cbar = plt.colorbar(img, ticks=[-4, -3, -2, 0, 2, 3, 4])
        cbar.set_label('风险等级变化')
        cbar.ax.set_yticklabels(['显著下降(-4)', '中度下降(-3)', '轻度下降(-2,-1)', '不变(0)', 
                                '轻度上升(1,2)', '中度上升(3)', '显著上升(4)'])
        
        # 添加标题
        if title:
            plt.title(title, fontsize=16)
        else:
            plt.title('病虫害风险变化图', fontsize=16)
        
        # 添加时间戳
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        plt.figtext(0.99, 0.01, f'生成时间: {timestamp}', ha='right', fontsize=8)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"风险变化图已保存到: {output_path}")
        return True
    
    except Exception as e:
        print(f"计算风险变化时出错: {e}")
        return False

def visualize_all_periods(future_output_dir, config=None):
    """
    可视化所有时期的风险分类图
    
    参数:
        future_output_dir: 未来预测输出目录
        config: 配置字典，如果为None则从配置文件加载
    """
    print("\n" + "="*80)
    print("           未来时期风险预测结果可视化工具           ")
    print("="*80 + "\n")
    
    # --- 1. 加载配置 ---
    if config is None:
        CONFIG = get_future_config()
    else:
        CONFIG = config
    
    # 提取未来预测配置
    FUTURE_CONFIG = CONFIG['future']
    PERIOD_NAMES = FUTURE_CONFIG['period_names']
    BASE_PERIOD_NAME = FUTURE_CONFIG['visualization'].get('base_period_name', 'Current')
    SHOW_DIFFERENCE = FUTURE_CONFIG['visualization'].get('show_difference', True)
    
    # 获取边界文件路径
    BOUNDARY_SHP_PATH = CONFIG.get('boundary_shp_path', None)
    
    # --- 2. 准备输出目录 ---
    viz_output_dir = os.path.join(future_output_dir, 'visualizations')
    os.makedirs(viz_output_dir, exist_ok=True)
    
    # --- 3. 可视化当前时期（基准期）---
    print("\n--- 可视化当前时期（基准期）---")
    base_period_dir = os.path.join(future_output_dir, BASE_PERIOD_NAME)
    if os.path.exists(base_period_dir):
        base_risk_tif = os.path.join(base_period_dir, 'risk_classification.tif')
        if os.path.exists(base_risk_tif):
            base_output_path = os.path.join(viz_output_dir, f'{BASE_PERIOD_NAME}_risk_classification.png')
            visualize_single_period(
                base_risk_tif, 
                base_output_path, 
                title=f'当前时期病虫害风险分类图',
                with_basemap=False,
                boundary_shp=BOUNDARY_SHP_PATH
            )
        else:
            print(f"警告: 未找到基准期风险分类文件: {base_risk_tif}")
            base_risk_tif = None
    else:
        print(f"警告: 未找到基准期目录: {base_period_dir}")
        base_risk_tif = None
    
    # --- 4. 可视化未来各时期 ---
    print("\n--- 可视化未来各时期 ---")
    for period_name in PERIOD_NAMES:
        period_dir = os.path.join(future_output_dir, period_name)
        if os.path.exists(period_dir):
            risk_tif = os.path.join(period_dir, 'risk_classification.tif')
            if os.path.exists(risk_tif):
                # 可视化单个时期
                output_path = os.path.join(viz_output_dir, f'{period_name}_risk_classification.png')
                visualize_single_period(
                    risk_tif, 
                    output_path, 
                    title=f'{period_name} 病虫害风险分类图',
                    with_basemap=False,
                    boundary_shp=BOUNDARY_SHP_PATH
                )
                
                # 如果需要可视化差异且存在基准期数据
                if SHOW_DIFFERENCE and base_risk_tif is not None:
                    diff_output_path = os.path.join(viz_output_dir, f'{period_name}_vs_{BASE_PERIOD_NAME}_difference.png')
                    visualize_difference(
                        base_risk_tif,
                        risk_tif,
                        diff_output_path,
                        title=f'{period_name} vs {BASE_PERIOD_NAME} 风险变化图'
                    )
            else:
                print(f"警告: 未找到时期 {period_name} 的风险分类文件: {risk_tif}")
        else:
            print(f"警告: 未找到时期 {period_name} 的目录: {period_dir}")
    
    # --- 5. 创建风险变化动画 (GIF) ---
    print("\n--- 创建风险变化动画 ---")
    try:
        import imageio
        
        # 准备所有图像文件路径
        image_paths = [os.path.join(viz_output_dir, f'{BASE_PERIOD_NAME}_risk_classification.png')]
        for period_name in PERIOD_NAMES:
            image_path = os.path.join(viz_output_dir, f'{period_name}_risk_classification.png')
            if os.path.exists(image_path):
                image_paths.append(image_path)
        
        if len(image_paths) > 1:
            gif_path = os.path.join(viz_output_dir, 'risk_classification_animation.gif')
            
            # 读取所有图像
            images = []
            for image_path in image_paths:
                images.append(imageio.imread(image_path))
            
            # 创建GIF，每帧停留2秒
            imageio.mimsave(gif_path, images, duration=2, loop=0)
            print(f"风险变化动画已保存到: {gif_path}")
        else:
            print("警告: 没有足够的图像创建动画")
    
    except ImportError:
        print("警告: 未安装imageio，无法创建风险变化动画")
    except Exception as e:
        print(f"创建风险变化动画时出错: {e}")
    
    print(f"\n所有可视化结果已保存到: {viz_output_dir}")

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='未来时期风险预测结果可视化工具')
    parser.add_argument('--future_dir', type=str, help='未来预测输出目录', default=None)
    parser.add_argument('--with_basemap', action='store_true', help='是否添加底图')
    args = parser.parse_args()
    
    # 加载配置
    CONFIG = get_future_config()
    
    # 确定未来预测输出目录
    if args.future_dir:
        future_output_dir = args.future_dir
    else:
        future_output_dir = CONFIG['future']['future_output_dir']
    
    # 运行可视化
    visualize_all_periods(future_output_dir, CONFIG)

if __name__ == '__main__':
    main() 