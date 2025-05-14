# visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import rasterio
from rasterio.plot import show
import pandas as pd
from datetime import datetime
import logging

# 设置日志
logger = logging.getLogger(__name__)

def get_risk_colors():
    """获取风险等级对应的颜色"""
    return {
        0: '#2ca02c',  # 绿色 - 极低风险
        1: '#1f77b4',  # 蓝色 - 低风险
        2: '#ff7f0e',  # 橙色 - 中风险
        3: '#d62728',  # 红色 - 高风险
        4: '#9467bd',  # 紫色 - 极高风险
        255: '#f0f0f0'  # 灰色 - NoData
    }

def get_risk_labels(config=None):
    """获取风险等级对应的标签"""
    if config and 'future' in config and 'visualization' in config['future'] and 'risk_labels' in config['future']['visualization']:
        labels = config['future']['visualization']['risk_labels']
        if len(labels) >= 5:
            return {
                0: labels[0],
                1: labels[1],
                2: labels[2],
                3: labels[3],
                4: labels[4],
                255: 'NoData'
            }
    
    # 默认标签
    return {
        0: '极低风险',
        1: '低风险',
        2: '中风险',
        3: '高风险',
        4: '极高风险',
        255: 'NoData'
    }

def visualize_prediction(prediction, output_path, title=None, add_timestamp=True, config=None):
    """
    可视化预测结果并保存为图像
    
    Args:
        prediction: 预测结果数组
        output_path: 输出图像路径
        title: 图像标题，如果为None则使用默认标题
        add_timestamp: 是否添加时间戳
        config: 配置信息
    
    Returns:
        bool: 是否成功保存图像
    """
    try:
        # 获取配置参数
        cmap_name = 'RdYlGn_r'  # 默认颜色映射
        dpi = 300  # 默认DPI
        title_fontsize = 14  # 默认标题字体大小
        
        if config and 'future' in config and 'visualization' in config['future']:
            viz_config = config['future']['visualization']
            if 'cmap' in viz_config:
                cmap_name = viz_config['cmap']
            if 'dpi' in viz_config:
                dpi = viz_config['dpi']
            if 'title_fontsize' in viz_config:
                title_fontsize = viz_config['title_fontsize']
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 如果预测值介于0-1之间，则使用连续颜色映射
        if np.nanmin(prediction) >= 0 and np.nanmax(prediction) <= 1:
            # 概率图
            im = plt.imshow(prediction, cmap=cmap_name, vmin=0, vmax=1)
            plt.colorbar(im, label='风险概率')
        else:
            # 风险分类图
            risk_colors = get_risk_colors()
            risk_labels = get_risk_labels(config)
            
            # 创建自定义颜色映射
            cmap = mcolors.ListedColormap(
                [risk_colors[0], risk_colors[1], risk_colors[2], 
                 risk_colors[3], risk_colors[4]]
            )
            bounds = [0, 1, 2, 3, 4, 5]
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            
            # 绘制风险分类图
            im = plt.imshow(prediction, cmap=cmap, norm=norm)
            
            # 添加颜色条和图例
            cbar = plt.colorbar(im, ticks=[0.5, 1.5, 2.5, 3.5, 4.5])
            cbar.set_ticklabels([risk_labels[0], risk_labels[1], risk_labels[2], 
                                risk_labels[3], risk_labels[4]])
        
        # 添加标题
        if title:
            plt.title(title, fontsize=title_fontsize)
        else:
            plt.title('害虫风险预测', fontsize=title_fontsize)
        
        # 关闭坐标轴
        plt.axis('off')
        
        # 添加时间戳
        if add_timestamp:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            plt.figtext(0.99, 0.01, f'生成时间: {timestamp}', ha='right', fontsize=8)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"可视化结果已保存到: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"可视化预测结果时出错: {e}")
        return False

def visualize_comparison(predictions, output_path, title=None, config=None):
    """
    可视化多个时期的预测结果比较
    
    Args:
        predictions: 字典，键为时期名称，值为预测结果数组
        output_path: 输出图像路径
        title: 图像标题，如果为None则使用默认标题
        config: 配置信息
    
    Returns:
        bool: 是否成功保存图像
    """
    try:
        # 获取配置参数
        cmap_name = 'RdYlGn_r'  # 默认颜色映射
        dpi = 300  # 默认DPI
        
        if config and 'future' in config and 'visualization' in config['future']:
            viz_config = config['future']['visualization']
            if 'cmap' in viz_config:
                cmap_name = viz_config['cmap']
            if 'dpi' in viz_config:
                dpi = viz_config['dpi']
        
        # 确定子图布局
        n_periods = len(predictions)
        if n_periods <= 3:
            n_cols = n_periods
            n_rows = 1
        elif n_periods <= 6:
            n_cols = 3
            n_rows = 2
        else:
            n_cols = 4
            n_rows = (n_periods + 3) // 4
        
        # 创建图形
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        
        # 绘制每个时期的预测结果
        for i, (period_name, prediction) in enumerate(predictions.items()):
            if i < len(axes):
                ax = axes[i]
                im = ax.imshow(prediction, cmap=cmap_name)
                ax.set_title(period_name)
                ax.axis('off')
        
        # 隐藏多余的子图
        for i in range(len(predictions), len(axes)):
            axes[i].axis('off')
        
        # 添加共享颜色条
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='风险水平')
        
        # 添加总标题
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle('多时期害虫风险预测比较', fontsize=16)
        
        # 保存图像
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"比较可视化结果已保存到: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"可视化比较结果时出错: {e}")
        return False

def create_risk_distribution_plot(risk_distributions, output_path, title=None, config=None):
    """
    创建风险分布趋势图
    
    Args:
        risk_distributions: 字典，键为时期名称，值为风险分布字典
        output_path: 输出图像路径
        title: 图像标题，如果为None则使用默认标题
        config: 配置信息
    
    Returns:
        bool: 是否成功保存图像
    """
    try:
        # 获取配置参数
        dpi = 300  # 默认DPI
        
        if config and 'future' in config and 'visualization' in config['future']:
            viz_config = config['future']['visualization']
            if 'dpi' in viz_config:
                dpi = viz_config['dpi']
        
        # 准备数据
        periods = list(risk_distributions.keys())
        risk_levels = get_risk_labels(config)
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 为每个风险等级绘制一条线
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
        markers = ['o', 's', '^', 'd', 'p']
        
        for i, risk_level in enumerate(range(5)):  # 0-4风险等级
            risk_label = risk_levels[risk_level]
            values = []
            
            for period in periods:
                if period in risk_distributions:
                    period_data = risk_distributions[period]
                    risk_value = period_data.get(risk_level, 0)
                    values.append(risk_value)
                else:
                    values.append(0)
            
            plt.plot(
                periods, 
                values, 
                marker=markers[i], 
                label=risk_label, 
                linewidth=2, 
                color=colors[i]
            )
        
        # 添加标题和标签
        if title:
            plt.title(title, fontsize=16)
        else:
            plt.title('多时期风险等级分布趋势', fontsize=16)
        
        plt.xlabel('预测时期')
        plt.ylabel('比例 (%)')
        plt.legend(title='风险等级')
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"风险分布趋势图已保存到: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"创建风险分布趋势图时出错: {e}")
        return False

def visualize_sensitivity(sensitivity_results, output_path, title=None, config=None):
    """
    可视化敏感性分析结果
    
    Args:
        sensitivity_results: 字典，键为特征名称，值为敏感性值（如变化率）
        output_path: 输出图像路径
        title: 图像标题，如果为None则使用默认标题
        config: 配置信息
    
    Returns:
        bool: 是否成功保存图像
    """
    try:
        # 获取配置参数
        dpi = 300  # 默认DPI
        
        if config and 'future' in config and 'visualization' in config['future']:
            viz_config = config['future']['visualization']
            if 'dpi' in viz_config:
                dpi = viz_config['dpi']
        
        # 准备数据
        features = list(sensitivity_results.keys())
        values = list(sensitivity_results.values())
        
        # 对特征按敏感性值排序
        sorted_indices = np.argsort(values)
        features = [features[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 绘制水平条形图
        bars = plt.barh(features, values, color='skyblue')
        
        # 为正负值使用不同颜色
        for i, v in enumerate(values):
            if v < 0:
                bars[i].set_color('salmon')
        
        # 添加标题和标签
        if title:
            plt.title(title, fontsize=16)
        else:
            plt.title('特征敏感性分析', fontsize=16)
        
        plt.xlabel('敏感性值（风险变化率）')
        plt.ylabel('特征')
        plt.grid(True, axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(values):
            plt.text(v + (0.01 if v >= 0 else -0.01), 
                    i, 
                    f'{v:.4f}', 
                    va='center', 
                    ha='left' if v >= 0 else 'right')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"敏感性分析结果已保存到: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"可视化敏感性分析结果时出错: {e}")
        return False 