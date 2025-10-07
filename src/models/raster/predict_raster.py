# sd_raster_prediction/predict_raster.py
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

# Assume these files are in the same directory or properly installed/pathed
from config_raster import get_config
from model.bilstm import BiLSTMModel # Use the original model definition
from utils import load_checkpoint # Use utils for loading

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
    print("--- Starting Raster Prediction (Diagnostic: Raw Probabilities) --- ")
    start_time = time.time()

    # --- 1. Load Configuration --- ## 配置加载
    if config is None:
        CONFIG = get_config()
    else:
        CONFIG = config # Allow passing config directly (e.g., for testing)

    MODEL_SAVE_PATH = CONFIG['model_save_path']
    SCALER_SAVE_PATH = CONFIG['scaler_save_path']
    # FEATURE_RASTER_MAP now points to ALIGNED rasters from config_raster.py
    FEATURE_RASTER_MAP = CONFIG['feature_raster_map']
    PREDICTION_OUTPUT_DIR = CONFIG['prediction_output_dir']
    DEVICE = CONFIG['training']['device']
    PRED_BATCH_SIZE = CONFIG['prediction']['prediction_batch_size']
    output_prediction_path = CONFIG['prediction_tif_path'] # Get the full output path
    # 获取风险等级分类图输出路径
    output_risk_class_path = CONFIG['prediction_risk_class_tif_path']
    output_raw_prob_path = CONFIG['prediction_raw_prob_tif_path'] # Get new path
    # 获取风险等级阈值和输出值
    risk_thresholds = CONFIG['prediction']['risk_thresholds']
    risk_class_values = CONFIG['prediction']['risk_class_values']

    os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Model path: {MODEL_SAVE_PATH}")
    print(f"Scaler path: {SCALER_SAVE_PATH}")
    print(f"Prediction output: {output_prediction_path}")
    print(f"Risk classification output: {output_risk_class_path}")
    print(f"Risk thresholds: {risk_thresholds}")
    print(f"Risk class values: {risk_class_values}")
    print(f"Outputting RAW probabilities to: {output_raw_prob_path}") # Log new path

    # 加载配置
    FEATURE_RASTER_MAP = CONFIG['feature_raster_map']
    reference_raster_path = list(FEATURE_RASTER_MAP.values())[0]  # 假设第一个栅格作为参考

    # 对齐所有栅格文件
    for feature_name, target_raster_path in FEATURE_RASTER_MAP.items():
        output_raster_path = os.path.join(CONFIG['prediction_output_dir'], f"aligned_{feature_name}.tif")
        align_rasters(reference_raster_path, target_raster_path, output_raster_path)

    # --- 2. Load Model and Scaler --- ## 加载模型和标准化器
    print("\n--- Loading Model and Scaler --- ")
    # Load model checkpoint to get input_size and feature_names
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Model checkpoint not found at {MODEL_SAVE_PATH}")
        return

    # Load checkpoint just to get metadata first
    try:
        # For diagnosing this specific pth file, reverting to weights_only=False as user owns the file.
        # Ideally, for future models, save checkpoints such that they are loadable with weights_only=True.
        checkpoint_meta = torch.load(MODEL_SAVE_PATH, map_location='cpu', weights_only=False) 
        input_size = checkpoint_meta.get('input_size')
        feature_names_train = checkpoint_meta.get('feature_names')
        if input_size is None or feature_names_train is None:
            raise ValueError("Checkpoint missing 'input_size' or 'feature_names'.")
        print(f"Model trained with input size: {input_size}")
        print(f"Model trained with features: {feature_names_train}")
    except Exception as e:
        print(f"Error loading model metadata from checkpoint: {e}")
        return

    # Initialize model structure based on saved metadata
    model_config_dict = {
        "input_size": input_size,
        "hidden_size": CONFIG['model']['hidden_size'],
        "num_layers": CONFIG['model']['num_layers'],
        "dropout": 0, # No dropout during prediction
    }
    model = BiLSTMModel(
        config=model_config_dict, # Pass the dict
        output_size=CONFIG['model']['output_size'] # Should be 1
    )

    # Load model weights using the utility function
    # Assuming load_checkpoint internally handles or also needs weights_only=False for this specific pth file
    checkpoint, _ = load_checkpoint(MODEL_SAVE_PATH, model, device=DEVICE) 
    if checkpoint is None:
        print("Failed to load model weights.")
        return
    model.to(DEVICE)
    model.eval() # Set model to evaluation mode

    # Load scaler
    if not os.path.exists(SCALER_SAVE_PATH):
        print(f"Error: Scaler file not found at {SCALER_SAVE_PATH}")
        return
    try:
        scaler = joblib.load(SCALER_SAVE_PATH)
        print("Scaler loaded successfully.")
        if not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
             print("Warning: Loaded object might not be a fitted StandardScaler.")
        elif scaler.n_features_in_ != len(feature_names_train):
            print(f"Warning: Scaler was fit on {scaler.n_features_in_} features, but model expects {len(feature_names_train)}.")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return

    # --- 3. Load and Verify Raster Data --- ## 加载和验证栅格
    print("\n--- Loading and Verifying Raster Data (Expecting ALIGNED Rasters) --- ")
    raster_srcs = {} # Keep raster sources open for reading blocks later
    profile = None # To store metadata from the first raster
    nodata_value = None # Store the detected nodata value

    # Check if the features defined in the config match the ones used for training
    config_features = list(FEATURE_RASTER_MAP.keys())
    if set(config_features) != set(feature_names_train):
        print("Error: Features in config_raster.py do not match features used for training!")
        print(f"  Config Features:  {sorted(config_features)}")
        print(f"  Training Features:{sorted(feature_names_train)}")
        # Clean up opened files before returning
        for src in raster_srcs.values(): src.close()
        return

    # Ensure rasters are loaded in the same order as feature_names_train
    raster_files_ordered = [FEATURE_RASTER_MAP[fname] for fname in feature_names_train]

    try:
        for i, (feature_name, raster_path) in enumerate(zip(feature_names_train, raster_files_ordered)):
            print(f"  Opening raster: {feature_name} ({os.path.basename(raster_path)})... ", end="")
            if not os.path.exists(raster_path):
                print(f"\nError: Raster file not found: {raster_path}")
                raise FileNotFoundError(raster_path) # Raise specific error

            src = rasterio.open(raster_path)
            raster_srcs[feature_name] = src # Store the opened rasterio dataset object
            current_profile = src.profile
            current_nodata = src.nodata

            if i == 0:
                profile = current_profile
                nodata_value = current_nodata # Store the nodata value from the first raster
                if nodata_value is None:
                    print("\nWarning: First raster has no defined NoData value. Prediction may be inaccurate for edge/fill areas.")
                    # We need a value to fill the output array later
                    nodata_value_out = CONFIG['raster_output'].get('tif_nodata_value', -9999.0) # Use config or default
                else:
                    nodata_value_out = nodata_value # Use the raster's nodata if available

                profile['dtype'] = 'float32' # Output probabilities are float32
                profile['count'] = 1 # Output is single band probability
                profile['nodata'] = nodata_value_out # Set a common NoData for output
                print(f"Profile set from first raster. Shape: {src.shape}, Output NoData: {profile['nodata']}")
            else:
                # Verify alignment (optional but recommended)
                if (current_profile['crs'] != profile['crs'] or
                    current_profile['transform'] != profile['transform'] or
                    current_profile['width'] != profile['width'] or
                    current_profile['height'] != profile['height']):
                    print(f"\nError: Raster {feature_name} ({os.path.basename(raster_path)}) is STILL not aligned with the first raster!")
                    print(f"  Expected CRS: {profile['crs']}, Got: {current_profile['crs']}")
                    print(f"  Expected Transform: {profile['transform']}, Got: {current_profile['transform']}")
                    print(f"  Expected Shape: ({profile['height']}, {profile['width']}), Got: ({current_profile['height']}, {current_profile['width']})")
                    # Clean up opened files
                    for src_obj in raster_srcs.values(): src_obj.close()
                    return
                print(f"Aligned. Shape: {src.shape}")

    except Exception as e:
        print(f"\nError opening or reading raster: {e}")
        # Clean up opened files
        for src_obj in raster_srcs.values(): src_obj.close()
        return

    height, width = profile['height'], profile['width']
    print(f"Raster verification complete. Dimensions: (Height: {height}, Width: {width})")

    # --- 4. Prepare Data for Prediction (Block Processing) --- ## 准备预测数据
    print("\n--- Preparing Data and Predicting Block by Block --- ")

    # Create the output raster files
    try:
        with rasterio.open(output_prediction_path, 'w', **profile) as dst_prob, \
             rasterio.open(output_risk_class_path, 'w', **profile) as dst_risk, \
             rasterio.open(output_raw_prob_path, 'w', **profile) as dst_raw_prob: # Open new file for raw probs
            
            # 创建风险等级分类输出栅格文件（复制概率栅格的配置，但将数据类型改为uint8）
            risk_class_profile = profile.copy()
            risk_class_profile['dtype'] = 'uint8'  # 使用8位无符号整数存储风险等级
            risk_class_profile['nodata'] = 255  # 为uint8类型设置合适的NoData值

            # Define block size (can be configured)
            block_size = CONFIG['prediction'].get('processing_block_size', 1024) # Default 1024 if not in config
            block_shape = (block_size, block_size)

            # Calculate number of blocks
            n_blocks_y = (height + block_shape[0] - 1) // block_shape[0]
            n_blocks_x = (width + block_shape[1] - 1) // block_shape[1]

            with tqdm(total=n_blocks_y * n_blocks_x, desc="Processing Blocks") as pbar:
                for j in range(n_blocks_y):
                    for i in range(n_blocks_x):
                        # Define the window for the current block
                        row_off = j * block_shape[0]
                        col_off = i * block_shape[1]
                        # Adjust height and width for edge blocks
                        current_height = min(block_shape[0], height - row_off)
                        current_width = min(block_shape[1], width - col_off)
                        window = rasterio.windows.Window(col_off, row_off, current_width, current_height)

                        # Read data for all features within the window
                        block_data = np.zeros((len(feature_names_train), current_height, current_width), dtype=np.float32)
                        valid_mask = np.ones((current_height, current_width), dtype=bool) # Start assuming all valid

                        for band_idx, feature_name in enumerate(feature_names_train):
                            src = raster_srcs[feature_name]
                            band_data = src.read(1, window=window)

                            # Update mask based on this band's nodata value (if defined)
                            if src.nodata is not None:
                                valid_mask &= (band_data != src.nodata)
                            # Also check for NaNs, especially if nodata is not defined
                            valid_mask &= ~np.isnan(band_data)

                            block_data[band_idx, :, :] = band_data.astype(np.float32)

                        # Prepare data for scaling and prediction
                        # Reshape to (num_pixels, num_features)
                        block_data_reshaped = block_data.reshape(len(feature_names_train), -1).T
                        valid_mask_flat = valid_mask.flatten()

                        # Select only valid pixels for scaling and prediction
                        valid_pixels_data = block_data_reshaped[valid_mask_flat, :]

                        # Initialize output blocks with NoData
                        output_block_prob_final = np.full((current_height, current_width), profile['nodata'], dtype=np.float32)
                        output_block_risk = np.full((current_height, current_width), risk_class_profile['nodata'], dtype=np.uint8)
                        output_block_raw_prob = np.full((current_height, current_width), profile['nodata'], dtype=np.float32) # For raw probabilities

                        if valid_pixels_data.shape[0] > 0: # If there are valid pixels
                            # --- 5. Scale Data --- ## 标准化
                            try:
                                valid_pixels_scaled = scaler.transform(valid_pixels_data)
                            except Exception as e:
                                print(f"\nError applying scaler in block ({j},{i}): {e}")
                                # Decide how to handle: skip block? fill with nodata?
                                # Currently, output_block remains NoData, so skipping prediction for this block.
                                pbar.update(1)
                                continue # Skip to next block

                            # --- 6. Predict Probabilities (in batches) --- ## 预测
                            # For raw probabilities, we'll store them before calibration/transformation
                            raw_pixel_probs = np.zeros(valid_pixels_scaled.shape[0], dtype=np.float32)

                            with torch.no_grad():
                                for batch_start in range(0, valid_pixels_scaled.shape[0], PRED_BATCH_SIZE):
                                    batch_end = min(batch_start + PRED_BATCH_SIZE, valid_pixels_scaled.shape[0]) # Corrected num_pixels_in_block to valid_pixels_scaled.shape[0]
                                    batch_data_scaled = valid_pixels_scaled[batch_start:batch_end] 
                                    batch_tensor = torch.tensor(batch_data_scaled, dtype=torch.float32).unsqueeze(1).to(DEVICE)
                                    batch_output = model(batch_tensor)
                                    current_raw_probs = torch.sigmoid(batch_output).cpu().numpy().flatten()
                                    raw_pixel_probs[batch_start:batch_end] = current_raw_probs
                            
                            output_block_raw_prob[valid_mask] = raw_pixel_probs # Store raw probabilities

                            # --- DIAGNOSTIC: For this run, predicted_probs will be raw_probs --- 
                            predicted_probs_for_risk = raw_pixel_probs.copy() # Use raw for risk classification for now
                            # Note: Original script had calibration and quantile transform here. We are bypassing them.

                            # If you want to re-enable calibration for the main output_block_prob_final (but not for raw_prob output):
                            # prob_calibration = CONFIG.get('prediction', {}).get('probability_calibration', {})
                            # prob_scale = prob_calibration.get('scale_factor', 1.0) 
                            # prob_shift = prob_calibration.get('shift_factor', 0.0)
                            # if j == 0 and i == 0: print(f"Using prob calibration for final prob output: scale={prob_scale}, shift={prob_shift}")
                            # calibrated_pixel_probs = raw_pixel_probs * prob_scale + prob_shift
                            # calibrated_pixel_probs = np.clip(calibrated_pixel_probs, 0.0, 1.0)
                            # output_block_prob_final[valid_mask] = calibrated_pixel_probs 
                            # predicted_probs_for_risk = calibrated_pixel_probs # If using calibration for risk
                            # ELSE, if final prob output should also be raw for this test:
                            output_block_prob_final[valid_mask] = raw_pixel_probs

                            # 对概率应用阈值，映射到风险等级 (using predicted_probs_for_risk which is currently raw_probs)
                            predicted_risk_classes = np.zeros(predicted_probs_for_risk.shape[0], dtype=np.uint8)
                            for idx, prob_val in enumerate(predicted_probs_for_risk):
                                predicted_risk_classes[idx] = probability_to_risk_class(prob_val, risk_thresholds, risk_class_values)
                            output_block_risk[valid_mask] = predicted_risk_classes

                        # --- 7. Write Output Blocks --- ## 保存结果块
                        dst_prob.write(output_block_prob_final, 1, window=window) # This will be raw prob for now
                        dst_risk.write(output_block_risk, 1, window=window)
                        dst_raw_prob.write(output_block_raw_prob, 1, window=window) # Write the dedicated raw prob output
                        pbar.update(1)

    except Exception as e:
        print(f"\nError during block processing or writing output raster: {e}")
        # Clean up opened files is crucial here
    finally:
        # --- Crucial Cleanup ---
        print("\n--- Closing all raster files ---")
        for src in raster_srcs.values():
            src.close()

    # 读取风险等级分类图，计算风险分布统计信息
    try:
        with rasterio.open(output_risk_class_path, 'r') as src_risk_array: # Renamed src to src_risk_array
            risk_array = src_risk_array.read(1)
            # 计算风险分布
            risk_distribution = calculate_risk_distribution(risk_array, risk_class_values)

            # 打印风险分布统计信息
            print("\n--- 风险分布统计信息 (based on raw probabilities directly mapped to risk) ---")
            for risk_name, stats in risk_distribution.items():
                print(f"{risk_name}: {stats['percentage']:.2f}% ({stats['pixel_count']} 像素)")

            # 保存风险分布统计信息到CSV文件
            stats_csv_path = os.path.join(os.path.dirname(output_risk_class_path), 'risk_distribution_statistics_raw_probs.csv') # new name for stats csv
            save_risk_distribution_to_csv(risk_distribution, stats_csv_path)
    except Exception as e:
        print(f"\n警告: 无法计算风险分布统计信息: {e}")

    total_duration = time.time() - start_time
    print(f"\n--- Prediction Finished --- ")
    print(f"MAIN prediction probability raster (currently RAW) saved to: {output_prediction_path}")
    print(f"DEDICATED RAW prediction probability raster saved to: {output_raw_prob_path}")
    print(f"Risk classification raster saved to: {output_risk_class_path}")
    print(f"Total duration: {total_duration:.2f} seconds")


if __name__ == '__main__':
    predict_raster()
