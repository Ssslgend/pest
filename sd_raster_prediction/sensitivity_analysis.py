# sensitivity_analysis.py
import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import rasterio
from tqdm import tqdm
from datetime import datetime
import sys
import joblib

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入自定义模块
from sd_raster_prediction.config_future import get_future_config
from sd_raster_prediction.future_prediction import run_future_prediction, load_model, load_scaler
from sd_raster_prediction.visualization import visualize_sensitivity

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_mean_risk(prediction):
    """
    计算预测风险的平均值
    
    Args:
        prediction: 预测结果数组
    
    Returns:
        float: 风险平均值
    """
    # 排除NoData值
    valid_mask = ~np.isnan(prediction)
    if np.sum(valid_mask) == 0:
        return 0
    return np.mean(prediction[valid_mask])

def run_sensitivity_analysis(config=None, output_dir=None, feature_name=None, change_rates=None, periods=1):
    """
    针对单个特征进行敏感性分析
    
    Args:
        config: 配置信息，如果为None则从配置文件加载
        output_dir: 输出目录，如果为None则使用配置中的路径
        feature_name: 要分析的特征名称，如果为None则分析所有特征
        change_rates: 特征变化率列表，如果为None则使用默认值
        periods: 预测的未来时期数
        
    Returns:
        dict: 敏感性分析结果
    """
    # 加载配置
    if config is None:
        config = get_future_config()
    
    # 确定输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(config['future']['future_output_dir'], f'sensitivity_analysis_{timestamp}')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志文件
    log_file = os.path.join(output_dir, 'sensitivity_analysis.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("开始敏感性分析")
    logger.info(f"输出目录: {output_dir}")
    
    # 确定要分析的特征
    features_to_analyze = []
    if feature_name is not None:
        if feature_name in config['features']['prediction_features']:
            features_to_analyze = [feature_name]
        else:
            logger.warning(f"特征 {feature_name} 不在预测特征列表中")
            return None
    else:
        features_to_analyze = config['features']['prediction_features']
    
    logger.info(f"要分析的特征: {features_to_analyze}")
    
    # 设置特征变化率
    if change_rates is None:
        change_rates = [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2]
    
    logger.info(f"特征变化率: {change_rates}")
    
    # 创建结果目录结构
    baseline_dir = os.path.join(output_dir, 'baseline')
    os.makedirs(baseline_dir, exist_ok=True)
    
    # 运行基准预测（不改变特征）
    logger.info("运行基准预测...")
    baseline_predictions, _ = run_future_prediction(periods=periods, output_dir=baseline_dir)
    
    # 计算基准风险平均值
    baseline_risks = {}
    for period_name, prediction in baseline_predictions.items():
        baseline_risks[period_name] = calculate_mean_risk(prediction)
    
    logger.info(f"基准风险值: {baseline_risks}")
    
    # 对每个特征进行敏感性分析
    sensitivity_results = {}
    for feature in features_to_analyze:
        logger.info(f"分析特征: {feature}")
        feature_dir = os.path.join(output_dir, feature)
        os.makedirs(feature_dir, exist_ok=True)
        
        feature_results = {}
        for rate in change_rates:
            if rate == 0:  # 跳过变化率为0的情况（与基准相同）
                continue
                
            logger.info(f"特征 {feature} 变化率: {rate}")
            
            # 创建变化率目录
            rate_dir = os.path.join(feature_dir, f'change_{rate}')
            os.makedirs(rate_dir, exist_ok=True)
            
            # 修改特征数据
            # 注意：实际实现中需要根据特征类型和数据格式进行不同的处理
            # 这里使用简化的方法，假设所有特征都是通过相同方式加载和处理的
            
            # 修改配置以反映特征变化
            modified_config = config.copy()
            if 'sensitivity_analysis' not in modified_config:
                modified_config['sensitivity_analysis'] = {}
            
            modified_config['sensitivity_analysis']['feature'] = feature
            modified_config['sensitivity_analysis']['change_rate'] = rate
            
            # 运行修改后的预测
            logger.info(f"运行修改后的预测...")
            modified_predictions, _ = run_future_prediction(
                periods=periods, 
                output_dir=rate_dir,
                config=modified_config
            )
            
            # 计算修改后的风险平均值
            modified_risks = {}
            for period_name, prediction in modified_predictions.items():
                modified_risks[period_name] = calculate_mean_risk(prediction)
            
            # 计算风险变化率
            risk_changes = {}
            for period_name in baseline_risks.keys():
                if period_name in modified_risks:
                    baseline_risk = baseline_risks[period_name]
                    modified_risk = modified_risks[period_name]
                    
                    # 避免除以零
                    if baseline_risk != 0:
                        change = (modified_risk - baseline_risk) / baseline_risk
                    else:
                        change = modified_risk - baseline_risk
                    
                    risk_changes[period_name] = change
            
            # 计算平均风险变化率（所有时期）
            avg_risk_change = np.mean(list(risk_changes.values()))
            feature_results[rate] = avg_risk_change
            
            logger.info(f"特征 {feature} 变化率 {rate} 导致风险变化率: {avg_risk_change:.4f}")
        
        # 计算特征敏感性（使用斜率）
        if len(feature_results) >= 2:
            rates = np.array(list(feature_results.keys()))
            changes = np.array(list(feature_results.values()))
            
            # 线性回归计算斜率
            slope, _ = np.polyfit(rates, changes, 1)
            sensitivity_results[feature] = slope
            
            logger.info(f"特征 {feature} 敏感性系数: {slope:.4f}")
            
            # 绘制敏感性曲线
            plt.figure(figsize=(10, 6))
            plt.plot(rates, changes, 'o-', linewidth=2)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
            plt.title(f"特征 {feature} 敏感性分析")
            plt.xlabel("特征变化率")
            plt.ylabel("风险变化率")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            curve_path = os.path.join(feature_dir, 'sensitivity_curve.png')
            plt.savefig(curve_path, dpi=300)
            plt.close()
            
            logger.info(f"特征 {feature} 敏感性曲线已保存到: {curve_path}")
    
    # 保存所有特征的敏感性结果
    if sensitivity_results:
        # 将结果保存为CSV
        results_df = pd.DataFrame({
            'Feature': list(sensitivity_results.keys()),
            'Sensitivity': list(sensitivity_results.values())
        })
        
        csv_path = os.path.join(output_dir, 'sensitivity_results.csv')
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"敏感性分析结果已保存到: {csv_path}")
        
        # 可视化所有特征的敏感性
        vis_path = os.path.join(output_dir, 'sensitivity_results.png')
        visualize_sensitivity(sensitivity_results, vis_path, title='特征敏感性分析结果', config=config)
    
    logger.info("敏感性分析完成")
    return sensitivity_results

def analyze_feature_importance(output_dir=None, config=None):
    """
    分析特征重要性并生成报告
    
    Args:
        output_dir: 输出目录，如果为None则生成新目录
        config: 配置信息，如果为None则从配置文件加载
        
    Returns:
        dict: 特征重要性结果
    """
    # 加载配置
    if config is None:
        config = get_future_config()
    
    # 确定输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(config['future']['future_output_dir'], f'feature_importance_{timestamp}')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志文件
    log_file = os.path.join(output_dir, 'feature_importance.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("开始特征重要性分析")
    
    # 加载模型
    model_path = os.path.join(config['common']['model_dir'], config['model']['model_checkpoint'])
    model = load_model(model_path)
    
    # 尝试使用模型的特征重要性（如果可用）
    importance_results = {}
    
    try:
        # 对于基于树的模型，可以直接获取特征重要性
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            features = config['features']['prediction_features']
            
            for i, feature in enumerate(features):
                if i < len(importance):
                    importance_results[feature] = importance[i]
        # 对于深度学习模型，可以使用排列重要性等方法
        else:
            # 运行敏感性分析作为替代
            logger.info("模型不支持直接获取特征重要性，运行敏感性分析作为替代...")
            importance_results = run_sensitivity_analysis(
                config=config,
                output_dir=output_dir,
                periods=1
            )
    except Exception as e:
        logger.error(f"获取特征重要性时出错: {e}")
        # 回退到敏感性分析
        logger.info("回退到敏感性分析...")
        importance_results = run_sensitivity_analysis(
            config=config,
            output_dir=output_dir,
            periods=1
        )
    
    # 保存特征重要性结果
    if importance_results:
        # 将结果保存为CSV
        results_df = pd.DataFrame({
            'Feature': list(importance_results.keys()),
            'Importance': list(importance_results.values())
        })
        
        # 按重要性排序
        results_df = results_df.sort_values(by='Importance', ascending=False)
        
        csv_path = os.path.join(output_dir, 'feature_importance.csv')
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"特征重要性结果已保存到: {csv_path}")
        
        # 可视化特征重要性
        plt.figure(figsize=(10, 8))
        bars = plt.barh(results_df['Feature'], results_df['Importance'], color='skyblue')
        plt.title('特征重要性分析')
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        
        vis_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(vis_path, dpi=300)
        plt.close()
        
        logger.info(f"特征重要性可视化已保存到: {vis_path}")
    
    logger.info("特征重要性分析完成")
    return importance_results

if __name__ == "__main__":
    # 测试敏感性分析
    config = get_future_config()
    feature_to_test = config['features']['prediction_features'][0]  # 第一个特征
    results = run_sensitivity_analysis(feature_name=feature_to_test, periods=2)
    print(f"敏感性分析结果: {results}")
    
    # 测试特征重要性分析
    importance = analyze_feature_importance()
    print(f"特征重要性结果: {importance}") 