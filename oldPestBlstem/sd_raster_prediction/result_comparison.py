# result_comparison.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import logging
import glob
from datetime import datetime
import sys
from tqdm import tqdm

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入自定义模块
from sd_raster_prediction.config_future import get_future_config
from sd_raster_prediction.visualization import visualize_comparison, create_risk_distribution_plot

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_prediction_files(directory):
    """
    查找目录中的预测结果文件
    
    Args:
        directory: 要搜索的目录路径
        
    Returns:
        字典，键为时期名称，值为预测文件路径
    """
    prediction_files = {}
    
    # 查找 prediction_*.tif 文件
    for file_path in glob.glob(os.path.join(directory, "prediction_*.tif")):
        filename = os.path.basename(file_path)
        # 提取时期名称
        parts = filename.replace("prediction_", "").replace(".tif", "").split("_")
        if len(parts) > 0:
            period_name = "_".join(parts)
            prediction_files[period_name] = file_path
    
    return prediction_files

def load_predictions(directory):
    """
    加载目录中的所有预测结果
    
    Args:
        directory: 预测结果目录
        
    Returns:
        dict: 预测结果字典，键为时期名称，值为预测数组
        dict: 元数据字典，键为时期名称，值为rasterio元数据
    """
    predictions = {}
    metadata = {}
    
    prediction_files = find_prediction_files(directory)
    if not prediction_files:
        logger.warning(f"在目录 {directory} 中未找到预测结果文件")
        return predictions, metadata
    
    for period_name, file_path in prediction_files.items():
        try:
            with rasterio.open(file_path) as src:
                predictions[period_name] = src.read(1)
                metadata[period_name] = src.meta.copy()
                logger.info(f"已加载时期 {period_name} 的预测结果")
        except Exception as e:
            logger.error(f"加载预测文件 {file_path} 时出错: {e}")
    
    return predictions, metadata

def calculate_risk_distribution(prediction, risk_levels=5):
    """
    计算预测结果的风险分布
    
    Args:
        prediction: 预测结果数组
        risk_levels: 风险等级数量
        
    Returns:
        dict: 风险分布字典，键为风险等级，值为占比
    """
    # 排除无效数据
    valid_mask = ~np.isnan(prediction)
    valid_data = prediction[valid_mask]
    
    if len(valid_data) == 0:
        return {i: 0 for i in range(risk_levels)}
    
    # 如果数据是连续值（0-1之间），则按阈值离散化
    if np.min(valid_data) >= 0 and np.max(valid_data) <= 1:
        # 计算风险等级阈值
        thresholds = np.linspace(0, 1, risk_levels + 1)
        
        # 计算每个风险等级的像素数量和比例
        distribution = {}
        for i in range(risk_levels):
            if i == 0:
                mask = (valid_data >= thresholds[i]) & (valid_data <= thresholds[i+1])
            else:
                mask = (valid_data > thresholds[i]) & (valid_data <= thresholds[i+1])
            
            pixel_count = np.sum(mask)
            percentage = (pixel_count / len(valid_data)) * 100
            distribution[i] = percentage
    
    # 如果数据已经是离散值
    else:
        # 计算每个风险等级的像素数量和比例
        distribution = {}
        for i in range(risk_levels):
            mask = (valid_data == i)
            pixel_count = np.sum(mask)
            percentage = (pixel_count / len(valid_data)) * 100
            distribution[i] = percentage
    
    return distribution

def compare_predictions(predictions, output_dir=None, config=None):
    """
    比较多个时期的预测结果
    
    Args:
        predictions: 预测结果字典，键为时期名称，值为预测数组
        output_dir: 输出目录，如果为None则创建新目录
        config: 配置信息，如果为None则从配置文件加载
        
    Returns:
        dict: 比较结果
    """
    # 加载配置
    if config is None:
        config = get_future_config()
    
    # 确定输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(config['future']['future_output_dir'], f'comparison_{timestamp}')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志文件
    log_file = os.path.join(output_dir, 'comparison.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("开始预测结果比较")
    logger.info(f"输出目录: {output_dir}")
    
    if not predictions:
        logger.warning("没有提供预测结果")
        return None
    
    # 创建比较可视化
    comparison_image_path = os.path.join(output_dir, 'prediction_comparison.png')
    visualize_comparison(predictions, comparison_image_path, config=config)
    
    # 计算每个时期的风险分布
    risk_distributions = {}
    risk_levels = config['future']['visualization'].get('risk_levels', 5)
    
    for period_name, prediction in predictions.items():
        risk_distributions[period_name] = calculate_risk_distribution(prediction, risk_levels)
    
    # 创建风险分布趋势图
    trend_image_path = os.path.join(output_dir, 'risk_distribution_trend.png')
    create_risk_distribution_plot(risk_distributions, trend_image_path, config=config)
    
    # 保存风险分布数据
    risk_data = []
    for period_name, distribution in risk_distributions.items():
        for risk_level, percentage in distribution.items():
            risk_data.append({
                'Period': period_name,
                'Risk_Level': risk_level,
                'Percentage': percentage
            })
    
    risk_df = pd.DataFrame(risk_data)
    risk_csv_path = os.path.join(output_dir, 'risk_distributions.csv')
    risk_df.to_csv(risk_csv_path, index=False, encoding='utf-8-sig')
    
    # 计算时期间的差异
    period_names = list(predictions.keys())
    if len(period_names) > 1:
        logger.info("计算时期间的差异")
        
        for i in range(len(period_names) - 1):
            base_period = period_names[i]
            compare_period = period_names[i + 1]
            
            base_prediction = predictions[base_period]
            compare_prediction = predictions[compare_period]
            
            # 确保两个数组形状相同
            if base_prediction.shape != compare_prediction.shape:
                logger.warning(f"时期 {base_period} 和 {compare_period} 的预测形状不同，无法计算差异")
                continue
            
            # 计算差异
            diff = compare_prediction - base_prediction
            
            # 可视化差异
            plt.figure(figsize=(10, 8))
            im = plt.imshow(diff, cmap='RdBu')
            plt.colorbar(im, label='风险变化')
            plt.title(f"{compare_period} vs {base_period} 风险变化")
            plt.axis('off')
            
            diff_image_path = os.path.join(output_dir, f'diff_{compare_period}_vs_{base_period}.png')
            plt.savefig(diff_image_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"已保存时期 {compare_period} 和 {base_period} 的差异图像")
    
    logger.info("预测结果比较完成")
    return {
        'output_dir': output_dir,
        'risk_distributions': risk_distributions
    }

def compare_prediction_directories(dir1, dir2, output_dir=None, config=None):
    """
    比较两个预测结果目录
    
    Args:
        dir1: 第一个预测结果目录
        dir2: 第二个预测结果目录
        output_dir: 输出目录，如果为None则创建新目录
        config: 配置信息，如果为None则从配置文件加载
        
    Returns:
        dict: 比较结果
    """
    # 加载配置
    if config is None:
        config = get_future_config()
    
    # 确定输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(config['future']['future_output_dir'], f'comparison_dirs_{timestamp}')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志文件
    log_file = os.path.join(output_dir, 'comparison.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("开始预测结果目录比较")
    logger.info(f"第一个目录: {dir1}")
    logger.info(f"第二个目录: {dir2}")
    logger.info(f"输出目录: {output_dir}")
    
    # 加载两个目录的预测结果
    predictions1, meta1 = load_predictions(dir1)
    predictions2, meta2 = load_predictions(dir2)
    
    if not predictions1:
        logger.warning(f"目录 {dir1} 中没有找到预测结果")
        return None
    
    if not predictions2:
        logger.warning(f"目录 {dir2} 中没有找到预测结果")
        return None
    
    # 查找两个目录中的共同时期
    common_periods = set(predictions1.keys()).intersection(set(predictions2.keys()))
    
    if not common_periods:
        logger.warning("两个目录中没有找到共同的预测时期")
        return None
    
    logger.info(f"找到 {len(common_periods)} 个共同的预测时期: {common_periods}")
    
    # 对每个共同时期进行比较
    for period in common_periods:
        # 创建时期输出目录
        period_dir = os.path.join(output_dir, period)
        os.makedirs(period_dir, exist_ok=True)
        
        pred1 = predictions1[period]
        pred2 = predictions2[period]
        
        # 确保两个数组形状相同
        if pred1.shape != pred2.shape:
            logger.warning(f"时期 {period} 的两个预测形状不同，无法比较")
            continue
        
        # 计算差异
        diff = pred2 - pred1
        
        # 可视化两个预测和差异
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 第一个预测
        im1 = axes[0].imshow(pred1, cmap='RdYlGn_r')
        axes[0].set_title(f"目录1: {period}")
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # 第二个预测
        im2 = axes[1].imshow(pred2, cmap='RdYlGn_r')
        axes[1].set_title(f"目录2: {period}")
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 差异
        im3 = axes[2].imshow(diff, cmap='RdBu')
        axes[2].set_title(f"差异: (目录2 - 目录1)")
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # 保存比较图像
        comparison_path = os.path.join(period_dir, 'comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"已保存时期 {period} 的比较图像")
        
        # 计算统计信息
        diff_stats = {
            'Mean': float(np.mean(diff)),
            'Std': float(np.std(diff)),
            'Min': float(np.min(diff)),
            'Max': float(np.max(diff)),
            'Abs_Mean': float(np.mean(np.abs(diff)))
        }
        
        # 保存统计信息
        stats_df = pd.DataFrame([diff_stats])
        stats_path = os.path.join(period_dir, 'diff_stats.csv')
        stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
        
        # 计算风险分布
        dist1 = calculate_risk_distribution(pred1)
        dist2 = calculate_risk_distribution(pred2)
        
        # 可视化风险分布比较
        plt.figure(figsize=(10, 6))
        
        levels = range(len(dist1))
        width = 0.35
        x = np.arange(len(levels))
        
        bars1 = plt.bar(x - width/2, [dist1.get(i, 0) for i in levels], width, label='目录1')
        bars2 = plt.bar(x + width/2, [dist2.get(i, 0) for i in levels], width, label='目录2')
        
        plt.xlabel('风险等级')
        plt.ylabel('占比 (%)')
        plt.title(f"时期 {period} 风险分布比较")
        plt.xticks(x, [f"等级{i}" for i in levels])
        plt.legend()
        
        # 保存风险分布比较图像
        dist_path = os.path.join(period_dir, 'risk_dist_comparison.png')
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"已保存时期 {period} 的风险分布比较图像")
    
    logger.info("预测结果目录比较完成")
    return {
        'output_dir': output_dir,
        'common_periods': list(common_periods)
    }

def run_comparison_tool(predictions_dir=None, config=None):
    """
    运行比较工具的主函数
    
    Args:
        predictions_dir: 预测结果目录，如果为None则从配置中获取
        config: 配置信息，如果为None则从配置文件加载
        
    Returns:
        dict: 比较结果
    """
    # 加载配置
    if config is None:
        config = get_future_config()
    
    # 确定预测目录
    if predictions_dir is None:
        predictions_dir = config['future']['future_output_dir']
    
    # 检查目录是否存在
    if not os.path.exists(predictions_dir):
        logger.error(f"预测目录 {predictions_dir} 不存在")
        return None
    
    # 加载预测结果
    predictions, metadata = load_predictions(predictions_dir)
    
    if not predictions:
        logger.error(f"在目录 {predictions_dir} 中没有找到预测结果")
        return None
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(config['future']['future_output_dir'], f'comparison_{timestamp}')
    
    # 运行比较
    return compare_predictions(predictions, output_dir=output_dir, config=config)

if __name__ == "__main__":
    # 测试比较工具
    config = get_future_config()
    result = run_comparison_tool(config=config)
    if result:
        print(f"比较结果已保存到: {result['output_dir']}") 