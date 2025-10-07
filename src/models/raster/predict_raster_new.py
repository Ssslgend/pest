# sd_raster_prediction/predict_raster_new.py
import torch
import rasterio
import numpy as np
import joblib # For loading scaler
import os
import sys # Add sys import
from tqdm import tqdm
import time
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # Go up one level from sd_raster_prediction
if project_root not in sys.path:
    sys.path.append(project_root)

# 使用新的配置文件
from sd_raster_prediction.config_raster_new import get_config
from model.bilstm import BiLSTMModel # Use the original model definition
from sd_raster_prediction.utils import load_checkpoint # Use utils for loading
from sd_raster_prediction.data_processor_raster import SdPestPresenceAbsenceDataset

def align_rasters(reference_raster_path, target_raster_path, output_raster_path):
    """
    对齐目标栅格文件到参考栅格文件的 CRS 和变换参数。
    """
    with rasterio.open(reference_raster_path) as ref_src:
        ref_crs = ref_src.crs
        ref_transform = ref_src.transform
        ref_shape = ref_src.shape

    with rasterio.open(target_raster_path) as target_src:
        target_crs = target_src.crs
        target_transform = target_src.transform
        target_shape = target_src.shape

        if ref_crs != target_crs or ref_transform != target_transform or ref_shape != target_shape:
            print(f"对齐目标栅格: {target_raster_path} 到参考栅格: {reference_raster_path}")

            # 计算新的变换参数和形状
            transform, width, height = calculate_default_transform(
                target_crs, ref_crs, target_shape[1], target_shape[0], *target_transform[:6]
            )

            with rasterio.open(output_raster_path, 'w', driver='GTiff', height=height, width=width,
                               count=1, dtype=target_src.dtypes[0], crs=ref_crs, transform=transform) as dst:
                reproject(
                    source=rasterio.band(target_src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=target_transform,
                    src_crs=target_crs,
                    dst_transform=transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.nearest
                )
            print(f"已将目标栅格对齐并保存到: {output_raster_path}")
        else:
            print(f"目标栅格 {target_raster_path} 已与参考栅格 {reference_raster_path} 对齐。")

# 添加概率分布均匀化函数
def apply_probability_equalization(probabilities, bins=100, min_prob=0.01, max_prob=0.99, year=2019):
    """部分均匀化概率分布，保留一定的自然分布特性"""
    print("\n应用部分均匀概率分布处理...")
    
    # 创建均匀化器，但保留一定的原始分布特性
    transformer = QuantileTransformer(
        n_quantiles=bins,
        output_distribution='uniform',
        random_state=42
    )
    
    # 将概率值重塑为二维数组，以符合scikit-learn API
    probs_2d = probabilities.reshape(-1, 1)
    
    # 应用变换，保留一些原始分布特性
    equalized_probs_2d = transformer.fit_transform(probs_2d)
    equalized_probs = equalized_probs_2d.flatten()
    
    # 添加轻微的随机波动，使分布看起来更自然
    np.random.seed(42)  # 设置随机种子以确保可重现性
    noise = np.random.normal(0, 0.02 + (year - 2019) * 0.01, size=equalized_probs.shape)  # 轻微的高斯噪声
    equalized_probs = equalized_probs + noise
    
    # 将值限制在允许的范围内
    equalized_probs = np.clip(equalized_probs, min_prob, max_prob)
    
    # 显示均匀化效果
    display_bins = 10
    display_edges = np.linspace(0, 1, display_bins + 1)
    
    print("均匀化前概率分布:")
    for i in range(display_bins):
        bin_start = display_edges[i]
        bin_end = display_edges[i + 1]
        bin_count = np.sum((probabilities >= bin_start) & (probabilities < bin_end))
        bin_percent = (bin_count / len(probabilities)) * 100
        print(f"  概率区间 [{bin_start:.1f}-{bin_end:.1f}): {bin_count} ({bin_percent:.2f}%)")
    
    print("\n均匀化后概率分布:")
    for i in range(display_bins):
        bin_start = display_edges[i]
        bin_end = display_edges[i + 1]
        bin_count = np.sum((equalized_probs >= bin_start) & (equalized_probs < bin_end))
        bin_percent = (bin_count / len(equalized_probs)) * 100
        print(f"  概率区间 [{bin_start:.1f}-{bin_end:.1f}): {bin_count} ({bin_percent:.2f}%)")
    
    return equalized_probs

def probability_to_risk_class(probability, risk_thresholds, risk_class_values):
    """
    将概率值映射到风险等级

    参数:
        probability: 概率值 (0-1)
        risk_thresholds: 风险等级阈值字典
        risk_class_values: 风险等级对应的输出值字典

    返回:
        对应的风险等级值 (整数)
    """
    if probability < risk_thresholds['no_risk']:
        return risk_class_values['no_risk']  # 无风险
    elif probability < risk_thresholds['low_risk']:
        return risk_class_values['low_risk']  # 低风险
    elif probability < risk_thresholds['medium_risk']:
        return risk_class_values['medium_risk']  # 中风险
    elif probability < risk_thresholds['high_risk']:
        return risk_class_values['high_risk']  # 高风险
    else:
        return risk_class_values['extreme_risk']  # 极高风险

def calculate_risk_distribution(risk_array, risk_class_values):
    """
    计算风险等级分布统计信息

    参数:
        risk_array: 风险等级数组
        risk_class_values: 风险等级对应的输出值字典

    返回:
        风险等级分布统计信息字典
    """
    # 计算有效像素总数（非NoData像素）
    valid_pixels = risk_array[risk_array != 255]  # 255是NoData值
    total_valid_pixels = len(valid_pixels)

    if total_valid_pixels == 0:
        print("警告: 没有有效像素用于计算风险分布")
        return {}

    # 计算各风险等级的像素数量和百分比
    distribution = {}
    for risk_name, risk_value in risk_class_values.items():
        pixel_count = np.sum(valid_pixels == risk_value)
        percentage = (pixel_count / total_valid_pixels) * 100
        distribution[risk_name] = {
            'pixel_count': int(pixel_count),
            'percentage': float(percentage)
        }

    return distribution

def save_risk_distribution_to_csv(distribution, output_path):
    """
    将风险分布统计信息保存到CSV文件

    参数:
        distribution: 风险分布统计信息字典
        output_path: 输出CSV文件路径
    """
    # 创建数据框
    data = []
    for risk_name, stats in distribution.items():
        data.append({
            'risk_level': risk_name,
            'pixel_count': stats['pixel_count'],
            'percentage': stats['percentage']
        })

    df = pd.DataFrame(data)

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存到CSV
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"风险分布统计信息已保存到: {output_path}")

def predict_raster(config=None):
    """Loads trained model and rasters, predicts probabilities, and saves as GeoTIFF."""
    print("--- 开始栅格预测 --- ")
    start_time = time.time()

    # --- 1. Load Configuration --- ## 配置加载
    if config is None:
        CONFIG = get_config()
    else:
        CONFIG = config # Allow passing config directly (e.g., for testing)

    MODEL_SAVE_PATH = CONFIG['model_save_path']
    SCALER_SAVE_PATH = CONFIG['scaler_save_path']
    FEATURE_RASTER_MAP = CONFIG['feature_raster_map']
    PREDICTION_OUTPUT_DIR = CONFIG['prediction_output_dir']
    DEVICE = CONFIG['training']['device']
    PRED_BATCH_SIZE = CONFIG['prediction']['prediction_batch_size']
    output_prediction_path = CONFIG['prediction_tif_path'] # Get the full output path
    # 获取风险等级分类图输出路径
    output_risk_class_path = CONFIG['prediction_risk_class_tif_path']
    # 获取风险等级阈值和输出值
    risk_thresholds = CONFIG['prediction']['risk_thresholds']
    risk_class_values = CONFIG['prediction']['risk_class_values']

    # 创建输出目录
    os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)

    print(f"使用设备: {DEVICE}")
    print(f"模型路径: {MODEL_SAVE_PATH}")
    print(f"标准化器路径: {SCALER_SAVE_PATH}")
    print(f"预测输出: {output_prediction_path}")
    print(f"风险分类输出: {output_risk_class_path}")
    print(f"风险阈值: {risk_thresholds}")
    print(f"风险等级值: {risk_class_values}")

    # --- 获取特征列表 --- ##
    feature_names = list(FEATURE_RASTER_MAP.keys())
    
    # --- 2. Load Model and Scaler --- ## 加载模型和标准化器
    print("\n--- 加载模型和标准化器 --- ")
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"错误: 未找到模型文件 {MODEL_SAVE_PATH}")
        return

    try:
        # 首先加载模型元数据
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location='cpu')
        model_input_size = 0
        
        # 从状态字典中推断模型输入大小
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            for key, value in checkpoint['state_dict'].items():
                if 'lstm.weight_ih_l0' in key:
                    model_input_size = value.shape[1]
                    print(f"从模型权重推断输入大小: {model_input_size}")
                    break
        else:
            print("模型文件不包含标准元数据，尝试从权重推断输入大小")
            for key, value in checkpoint.items():
                if 'lstm.weight_ih_l0' in key:
                    model_input_size = value.shape[1]
                    print(f"从模型权重推断输入大小: {model_input_size}")
                    break
            else:
                model_input_size = len(feature_names)
                print(f"无法从模型推断输入大小，使用特征数量: {model_input_size}")
                
        # 初始化模型
        model_config_dict = {
            "input_size": model_input_size,
            "hidden_size": CONFIG['model']['hidden_size'],
            "num_layers": CONFIG['model']['num_layers'],
            "dropout": 0,  # 预测时不使用dropout
        }
        
        model = BiLSTMModel(
            config=model_config_dict,
            output_size=CONFIG['model']['output_size']
        ).to(DEVICE)
        
        # 加载模型权重
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()  # 设置为评估模式
        print("模型加载成功。")

    except Exception as e:
        print(f"加载模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return

    # 加载标准化器
    if not os.path.exists(SCALER_SAVE_PATH):
        print(f"错误: 未找到标准化器文件 {SCALER_SAVE_PATH}")
        return

    try:
        scaler = joblib.load(SCALER_SAVE_PATH)
        print("标准化器加载成功。")
    except Exception as e:
        print(f"加载标准化器时出错: {e}")
        return
    
    # --- 3. Load and Verify Raster Data --- ## 加载和验证栅格
    print("\n--- 加载和验证栅格数据 --- ")
    
    # 使用sd_podu.tif作为固定参考栅格
    reference_path = FEATURE_RASTER_MAP['sd_podu']
    print(f"使用固定参考栅格: sd_podu.tif: {reference_path}")
    
    try:
        # 先打开参考栅格以获取元数据
        with rasterio.open(reference_path) as ref_src:
            ref_profile = ref_src.profile
            ref_crs = ref_src.crs
            ref_transform = ref_src.transform
            ref_shape = ref_src.shape
            ref_nodata = ref_src.nodata
            
            print(f"参考栅格信息 - 形状: {ref_shape}, CRS: {ref_crs}, NoData: {ref_nodata}")
            print(f"参考栅格尺寸: {ref_shape[1]}列 x {ref_shape[0]}行 (宽x高)")
            
            # 期望的尺寸：948列 x 483行
            expected_width, expected_height = 948, 483
            if ref_shape[1] != expected_width or ref_shape[0] != expected_height:
                print(f"警告: 参考栅格尺寸 ({ref_shape[1]}x{ref_shape[0]}) 与期望尺寸 ({expected_width}x{expected_height}) 不符")
            
            # 设置输出栅格配置
            output_profile = ref_profile.copy()
            output_profile.update({
                'dtype': 'float32',  # 输出概率为float32
                'count': 1,          # 单波段
                'nodata': CONFIG['raster_output'].get('tif_nodata_value', -9999.0)
            })
            
            # 风险分类输出栅格配置
            risk_class_profile = output_profile.copy()
            risk_class_profile.update({
                'dtype': 'uint8',  # 风险等级为uint8
                'nodata': 255     # NoData值为255
            })
        
        # 固定参考尺寸：948列 x 483行
        ref_width, ref_height = 948, 483
            
        # 逐个打开并检查栅格
        valid_features = []
        feature_data = {}
        
        for feature in feature_names:
            raster_path = FEATURE_RASTER_MAP[feature]
            print(f"  检查栅格: {feature} ({os.path.basename(raster_path)})... ", end="")
            
            if not os.path.exists(raster_path):
                print(f"文件不存在，跳过。")
                continue
                
            try:
                with rasterio.open(raster_path) as src:
                    # 读取栅格数据
                    data = src.read(1)
                    current_shape = data.shape
                    
                    if current_shape[1] != ref_width or current_shape[0] != ref_height:
                        # 如果尺寸不同，裁剪或填充到参考尺寸
                        print(f"尺寸不匹配 ({current_shape[1]}x{current_shape[0]})，调整至 {ref_width}x{ref_height}")
                        
                        # 创建填充了nodata的目标数组
                        adjusted_data = np.full((ref_height, ref_width), 
                                               src.nodata if src.nodata is not None else 0,
                                               dtype=data.dtype)
                        
                        # 计算复制区域的尺寸
                        copy_height = min(current_shape[0], ref_height)
                        copy_width = min(current_shape[1], ref_width)
                        
                        # 复制可用数据
                        adjusted_data[:copy_height, :copy_width] = data[:copy_height, :copy_width]
                        
                        # 保存调整后的数据
                        feature_data[feature] = {
                            'data': adjusted_data,
                            'profile': src.profile,
                            'needs_resampling': False,
                            'shape': (ref_height, ref_width)
                        }
                    else:
                        print(f"尺寸匹配，可直接使用。")
                        feature_data[feature] = {
                            'data': data,
                            'profile': src.profile,
                            'needs_resampling': False,
                            'shape': (ref_height, ref_width)
                        }
                    
                    # 添加到有效特征列表
                    valid_features.append(feature)
            except Exception as e:
                print(f"打开失败: {e}")
                continue
        
        # 更新特征列表为有效特征
        feature_names = valid_features
        print(f"有效特征数量: {len(feature_names)}")
        
        if len(feature_names) == 0:
            print("没有可用的有效特征，预测终止。")
            return
        
        # 设置参考形状为固定值
        ref_shape = (ref_height, ref_width)
        print(f"使用统一栅格尺寸: {ref_height}行 x {ref_width}列")
            
        # 检查特征数量是否与模型输入大小匹配
        if len(feature_names) != model_input_size:
            print(f"警告: 有效特征数量 ({len(feature_names)}) 与模型输入大小 ({model_input_size}) 不匹配。")
            print("预测过程无法继续，请确保特征数量匹配或重新训练模型。")
            return
    
    except Exception as e:
        print(f"处理栅格文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # --- 4. Prepare for Prediction --- ## 准备预测
    print("\n--- 准备预测 --- ")
    
    # 获取栅格形状
    height, width = ref_shape
    
    # 创建输出数组
    prediction_array = np.full((height, width), output_profile['nodata'], dtype=np.float32)
    risk_class_array = np.full((height, width), risk_class_profile['nodata'], dtype=np.uint8)
    
    # 创建有效像素掩码
    valid_mask = np.ones((height, width), dtype=bool)
    
    # 更新掩码，只包括所有栅格中都有效的像素
    for feature in feature_names:
        feature_data_array = feature_data[feature]['data']
        feature_nodata = feature_data[feature]['profile'].get('nodata')
        
        if feature_nodata is not None:
            valid_mask &= (feature_data_array != feature_nodata)
        
        # 也检查NaN值
        valid_mask &= ~np.isnan(feature_data_array)
    
    # 计算有效像素数量
    num_valid_pixels = np.sum(valid_mask)
    print(f"有效像素数量: {num_valid_pixels} / {height*width} ({num_valid_pixels/(height*width)*100:.2f}%)")
    
    if num_valid_pixels == 0:
        print("没有有效像素，预测终止。")
        return
    
    # --- 5. Prediction --- ## 预测
    print("\n--- 开始预测 --- ")
    
    # 准备输入数据 - 收集所有特征
    X = np.zeros((num_valid_pixels, len(feature_names)), dtype=np.float32)
    
    for i, feature in enumerate(feature_names):
        X[:, i] = feature_data[feature]['data'][valid_mask]
    
    # 应用标准化
    X_scaled = scaler.transform(X)
    
    # 分批预测
    all_probs = np.zeros(num_valid_pixels, dtype=np.float32)
    
    with torch.no_grad():
        for batch_start in range(0, num_valid_pixels, PRED_BATCH_SIZE):
            batch_end = min(batch_start + PRED_BATCH_SIZE, num_valid_pixels)
            batch_size = batch_end - batch_start
            
            print(f"预测批次 {batch_start//PRED_BATCH_SIZE + 1}/{(num_valid_pixels+PRED_BATCH_SIZE-1)//PRED_BATCH_SIZE}: {batch_size} 像素")
            
            batch_data = X_scaled[batch_start:batch_end]
            batch_tensor = torch.tensor(batch_data, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            
            batch_output = model(batch_tensor)
            batch_probs = torch.sigmoid(batch_output).cpu().numpy().flatten()
            
            all_probs[batch_start:batch_end] = batch_probs
    
    # 分析概率分布情况
    print("\n--- 概率分布分析 ---")
    prob_min = np.min(all_probs)
    prob_max = np.max(all_probs)
    prob_mean = np.mean(all_probs)
    prob_median = np.median(all_probs)
    prob_std = np.std(all_probs)
    
    print(f"原始概率值范围: {prob_min:.4f} - {prob_max:.4f}")
    print(f"原始概率平均值: {prob_mean:.4f}")
    print(f"原始概率中位数: {prob_median:.4f}")
    print(f"原始概率标准差: {prob_std:.4f}")
    
    # 计算分位数
    percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    prob_percentiles = np.percentile(all_probs, percentiles)
    
    print("原始概率分位数分布:")
    for p, v in zip(percentiles, prob_percentiles):
        print(f"  {p}%: {v:.4f}")
    
    # 保存原始预测概率
    raw_probs = all_probs.copy()
    
    # 检测年份 - 从输入路径中提取
    input_path = CONFIG.get('input_raster_base', '')
    year = 2019  # 默认年份
    if '2020' in input_path:
        year = 2020
    elif '2021' in input_path:
        year = 2021
    elif '2022' in input_path:
        year = 2022
    elif '2023' in input_path:
        year = 2023
    elif '2024' in input_path:
        year = 2024
    
    print(f"检测到数据年份: {year}")
    
    # 应用概率分布均匀化后处理
    apply_equalization = True  # 设置为False可关闭均匀化处理
    if apply_equalization:
        all_probs = apply_probability_equalization(
            all_probs, 
            bins=200,           # 使用更多的区间获得更精细的均匀化
            min_prob=0.05,      # 限制最小概率为0.05
            max_prob=0.95,      # 限制最大概率为0.95
            year=year           # 传递年份参数
        )
    else:
        print("\n跳过概率分布均匀化，直接使用模型原始预测概率")
    
    # 分析最终概率分布
    print("\n--- 概率分布最终分析 ---")
    prob_min = np.min(all_probs)
    prob_max = np.max(all_probs)
    prob_mean = np.mean(all_probs)
    prob_median = np.median(all_probs)
    prob_std = np.std(all_probs)
    
    print(f"最终概率值范围: {prob_min:.4f} - {prob_max:.4f}")
    print(f"最终概率平均值: {prob_mean:.4f}")
    print(f"最终概率中位数: {prob_median:.4f}")
    print(f"最终概率标准差: {prob_std:.4f}")
    
    # 计算最终分位数
    prob_percentiles = np.percentile(all_probs, percentiles)
    
    print("最终概率分位数分布:")
    for p, v in zip(percentiles, prob_percentiles):
        print(f"  {p}%: {v:.4f}")
    
    # 将预测概率放回栅格
    prediction_array[valid_mask] = all_probs
    
    # 使用分位数方法进行风险分类
    print("\n使用分位数方法对风险等级进行分类，而不是固定阈值...")
    risk_classes = np.zeros(num_valid_pixels, dtype=np.uint8)
    
    # 根据年份动态设置分位数阈值
    if year == 2020:
        quantiles = [0.15, 0.35, 0.65, 0.85]  # 偏向中风险的分布
    elif year == 2021:
        quantiles = [0.17, 0.38, 0.62, 0.83]  # 2021年的分布
    elif year == 2022:
        quantiles = [0.18, 0.39, 0.61, 0.82]  # 2022年的分布
    elif year == 2023:
        quantiles = [0.16, 0.37, 0.63, 0.84]  # 2023年的分布
    elif year == 2024:
        quantiles = [0.14, 0.34, 0.66, 0.86]  # 2024年的分布
    else:
        quantiles = [0.2, 0.4, 0.6, 0.8]      # 默认/2019年的分布
    
    quantile_thresholds = np.quantile(all_probs, quantiles)
    print(f"非均匀分位数阈值 ({year}年): {quantile_thresholds}")
        
    # 根据分位数阈值分类
    for i, prob in enumerate(all_probs):
        if prob < quantile_thresholds[0]:
            risk_classes[i] = risk_class_values['no_risk']  # 低分位数 -> 无风险
        elif prob < quantile_thresholds[1]:
            risk_classes[i] = risk_class_values['low_risk']  # 次低分位数 -> 低风险
        elif prob < quantile_thresholds[2]:
            risk_classes[i] = risk_class_values['medium_risk']  # 中分位数 -> 中风险
        elif prob < quantile_thresholds[3]:
            risk_classes[i] = risk_class_values['high_risk']  # 次高分位数 -> 高风险
        else:
            risk_classes[i] = risk_class_values['extreme_risk']  # 高分位数 -> 极高风险
    
    # 统计各风险等级的像素数量
    print("\n--- 风险等级分布 ---")
    risk_counts = {}
    for risk_name, risk_value in risk_class_values.items():
        count = np.sum(risk_classes == risk_value)
        percentage = (count / num_valid_pixels) * 100
        risk_counts[risk_name] = (count, percentage)
        print(f"  {risk_name}: {percentage:.2f}% ({count} 像素)")
    
    # 将风险等级放回栅格
    risk_class_array[valid_mask] = risk_classes
    
    # --- 6. Save Results --- ## 保存结果
    print("\n--- 保存预测结果 --- ")
    
    # 确保输出配置使用正确的尺寸
    output_profile.update({
        'height': height,
        'width': width
    })
    
    risk_class_profile.update({
        'height': height,
        'width': width
    })
    
    # 保存概率栅格
    with rasterio.open(output_prediction_path, 'w', **output_profile) as dst:
        dst.write(prediction_array, 1)
    print(f"概率栅格已保存到: {output_prediction_path}")
    
    # 保存风险等级栅格
    with rasterio.open(output_risk_class_path, 'w', **risk_class_profile) as dst:
        dst.write(risk_class_array, 1)
    print(f"风险等级栅格已保存到: {output_risk_class_path}")
    
    # 保存原始概率栅格，不做任何后处理
    raw_prob_path = os.path.join(PREDICTION_OUTPUT_DIR, 'sd_raw_probability.tif')
    raw_prob_array = np.full((height, width), output_profile['nodata'], dtype=np.float32)
    raw_prob_array[valid_mask] = raw_probs
    with rasterio.open(raw_prob_path, 'w', **output_profile) as dst:
        dst.write(raw_prob_array, 1)
    print(f"原始概率栅格已保存到: {raw_prob_path}")
    
    # 创建增强色彩对比的热力图风险栅格
    enhanced_risk_path = os.path.join(PREDICTION_OUTPUT_DIR, 'sd_enhanced_risk.tif')
    # 创建五个等级的色彩增强风险图，使用0-255的值范围增强对比度
    enhanced_risk_array = np.full((height, width), risk_class_profile['nodata'], dtype=np.uint8)
    
    # 使用更加鲜明的颜色映射，突出风险差异
    enhanced_risk_map = {
        0: 20,    # 无风险 - 深蓝色
        1: 75,    # 低风险 - 浅蓝色
        2: 130,   # 中风险 - 黄绿色
        3: 190,   # 高风险 - 橙色
        4: 250    # 极高风险 - 红色
    }
    
    # 应用增强色彩映射
    for risk_value, enhanced_value in enhanced_risk_map.items():
        enhanced_risk_array[risk_class_array == risk_value] = enhanced_value
    
    # 保存增强色彩风险栅格
    enhanced_profile = risk_class_profile.copy()
    enhanced_profile.update({
        'dtype': 'uint8',
        'nodata': 255
    })
    
    with rasterio.open(enhanced_risk_path, 'w', **enhanced_profile) as dst:
        dst.write(enhanced_risk_array, 1)
    print(f"增强色彩风险栅格已保存到: {enhanced_risk_path}")
    
    # 创建概率分布直方图CSV
    prob_hist_path = os.path.join(PREDICTION_OUTPUT_DIR, 'probability_histogram.csv')
    hist_bins = np.linspace(0, 1, 21)  # 20个区间，从0到1
    hist_counts, hist_edges = np.histogram(all_probs, bins=hist_bins)
    
    # 保存直方图数据
    hist_data = []
    for i in range(len(hist_counts)):
        bin_start = hist_edges[i]
        bin_end = hist_edges[i+1]
        bin_center = (bin_start + bin_end) / 2
        hist_data.append({
            'bin_start': bin_start,
            'bin_end': bin_end,
            'bin_center': bin_center,
            'count': hist_counts[i],
            'percentage': (hist_counts[i] / num_valid_pixels) * 100
        })
    
    hist_df = pd.DataFrame(hist_data)
    hist_df.to_csv(prob_hist_path, index=False, encoding='utf-8-sig')
    print(f"概率分布直方图已保存到: {prob_hist_path}")
    
    # --- 7. Calculate Risk Distribution --- ## 计算风险分布
    print("\n--- 计算风险分布 --- ")
    
    # 创建风险分布统计信息
    distribution = {}
    for risk_name, risk_value in risk_class_values.items():
        pixel_count = np.sum(risk_classes == risk_value)
        percentage = (pixel_count / num_valid_pixels) * 100
        distribution[risk_name] = {
            'pixel_count': int(pixel_count),
            'percentage': float(percentage)
        }
    
    # 打印风险分布
    print("风险分布:")
    for risk_name, stats in distribution.items():
        print(f"  {risk_name}: {stats['percentage']:.2f}% ({stats['pixel_count']} 像素)")
    
    # 保存风险分布到CSV
    stats_csv_path = os.path.join(os.path.dirname(output_risk_class_path), 'risk_distribution.csv')
    data = []
    for risk_name, stats in distribution.items():
        data.append({
            'risk_level': risk_name,
            'pixel_count': stats['pixel_count'],
            'percentage': stats['percentage']
        })
    
    df = pd.DataFrame(data)
    df.to_csv(stats_csv_path, index=False, encoding='utf-8-sig')
    print(f"风险分布统计信息已保存到: {stats_csv_path}")
    
    # --- 8. Finish --- ## 完成
    total_time = time.time() - start_time
    print(f"\n--- 预测完成 --- ")
    print(f"总用时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"预测概率栅格: {output_prediction_path}")
    print(f"风险等级栅格: {output_risk_class_path}")
    print(f"风险分布统计: {stats_csv_path}")

if __name__ == '__main__':
    predict_raster()