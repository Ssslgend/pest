# sd_raster_prediction/config_raster_new.py
import torch
import os

# --- 添加预测年份支持 ---
# 默认年份为2023，可通过get_config函数参数修改
DEFAULT_PREDICTION_YEAR = 2024

# --- Paths --- (Adjust base paths if needed)
BASE_OUTPUT_DIR = 'E:/code/0424/pestBIstm/pestBIstm/results'  # 恢复原始绝对路径
DATA_DIR = 'E:/code/0424/pestBIstm/pestBIstm/datas'  # 恢复原始绝对路径

# --- File Paths --- ## 文件的路径
CSV_DATA_PATH = os.path.join(DATA_DIR, 'train.csv') # Source CSV for training points
MODEL_SAVE_PATH = os.path.join(BASE_OUTPUT_DIR, 'trained_model/sd_bilstm_presence_pseudo.pth')
SCALER_SAVE_PATH = os.path.join(BASE_OUTPUT_DIR, 'trained_model/scaler_presence_pseudo.joblib')
BOUNDARY_SHP_PATH = 'H:/yanyi/Stydy_data/shandong.shp' # Optional boundary for masking
PREDICTION_SHP_PATH = 'H:/yanyi/Stydy_data/shandong.shp'

# --- Data Processing Parameters --- ## 数据参数
DATA_PROCESSING_CONFIG = {
    'pseudo_absence_ratio': 1.0,     # Ratio of pseudo-absence points to generate
    'test_size': 0.25,               # 测试集比例
    'val_size': 0.2,                 # 验证集比例
    'random_state': 42,
    'coordinate_columns': ['发生样点纬度', '发生样点经度'], # CSV中的坐标列名
    'label_column': 'label',         # CSV中的标签列名
    'excluded_cols_from_features': ['year'] # 排除年份列不作为特征
}

# --- Model Hyperparameters (BiLSTM specific) --- ## 模型超参数
MODEL_CONFIG = {
    # input_size will be determined by data processor
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.5,  # 增加dropout比例从0.4到0.5
    'output_size': 1 # Fixed to 1 for probability prediction
}

# --- Training Parameters --- ## 训练参数
TRAIN_PARAMS = {
    'batch_size': 32,
    'learning_rate': 0.0002,  # 降低学习率
    'weight_decay': 1e-4,     # 增加权重衰减以增加正则化
    'num_epochs': 150,        # 增加最大训练轮次
    'patience': 25,           # 增加早停耐心值
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# --- Prediction Parameters --- ## 预测参数
PREDICT_PARAMS = {
    'prediction_batch_size': 512 * 512, # Pixels per prediction batch
    'processing_block_size': 1024,    # Rasterio window read size
    'save_shapefile': True,
    'mask_with_boundary': True, # Set to False if boundary shapefile is not available or not needed

    # 概率校准参数 - 直接使用原始概率，不进行校准
    'probability_calibration': {
        'scale_factor': 1.0,  # 中性缩放因子，不改变概率分布
        'shift_factor': 0.0,  # 中性偏移因子，不改变概率分布
    },
    
    # 使用分位数方法进行风险分类
    'use_quantile_classification': True,  # 启用分位数分类方法
    'quantiles': [0.2, 0.4, 0.6, 0.8],    # 5等分风险等级的分位数
    
    # 基于模型实际输出分布调整的阈值（仅当不使用分位数方法时使用）
    'risk_thresholds': {
        'no_risk': 0.45,      # 小于0.45的为无风险
        'low_risk': 0.50,     # 0.45-0.50为低风险
        'medium_risk': 0.55,  # 0.50-0.55为中风险
        'high_risk': 0.60     # 0.55-0.60为高风险，大于0.60为极高风险
    },

    # 风险等级对应的栅格输出值
    'risk_class_values': {
        'no_risk': 0,      # 无风险值为0
        'low_risk': 1,     # 低风险值为1
        'medium_risk': 2,  # 中风险值为2
        'high_risk': 3,    # 高风险值为3
        'extreme_risk': 4  # 极高风险值为4
    }
}

# --- Raster Prediction Input Map --- ## 输入的栅格数据
# Maps feature names (MUST match training order) to their corresponding raster file paths
# User MUST verify these keys and paths!
def get_feature_raster_map(prediction_year):
    """根据预测年份返回对应的特征栅格映射"""
    input_base = f'H:/data_new2025/2019_2024_sd/prediction_year/{prediction_year}'
    
    return {
        'Mean Diurnal Range (Mean of monthly (max temp - min temp))': os.path.join(input_base, 'wc2.1_30s_bio_2.tif'),
        'Precipitation of Coldest Quarter': os.path.join(input_base, 'wc2.1_30s_bio_19.tif'),
        'Precipitation of Wettest Month': os.path.join(input_base, 'wc2.1_30s_bio_13.tif'),
        'Precipitation of Driest Quarter': os.path.join(input_base, 'wc2.1_30s_bio_17.tif'),
        'Annual Precipitation': os.path.join(input_base, 'wc2.1_30s_bio_12.tif'),
        'Annual Mean Temperature': os.path.join(input_base, 'wc2.1_30s_bio_1.tif'),
        'Mean Temperature of Driest Quarter': os.path.join(input_base, 'wc2.1_30s_bio_9.tif'),
        'Precipitation of Driest Month': os.path.join(input_base, 'wc2.1_30s_bio_14.tif'),
        'Temperature Annual Range (BIO5-BIO6)': os.path.join(input_base, 'wc2.1_30s_bio_7.tif'),
        'Min Temperature of Coldest Month': os.path.join(input_base, 'wc2.1_30s_bio_6.tif'),
        'Precipitation Seasonality (Coefficient of Variation)': os.path.join(input_base, 'wc2.1_30s_bio_15.tif'),
        'Precipitation of Wettest Quarter': os.path.join(input_base, 'wc2.1_30s_bio_16.tif'),
        'Temperature Seasonality (standard deviation ×100)': os.path.join(input_base, 'wc2.1_30s_bio_4.tif'),
        'Max Temperature of Warmest Month': os.path.join(input_base, 'wc2.1_30s_bio_5.tif'),
        'Mean Temperature of Warmest Quarter (BIO8)': os.path.join(input_base, 'wc2.1_30s_bio_8.tif'),
        'Isothermality (BIO2/BIO7) (×100)': os.path.join(input_base, 'wc2.1_30s_bio_3.tif'),
        'Mean Temperature of Warmest Quarter': os.path.join(input_base, 'wc2.1_30s_bio_10.tif'),
        'Precipitation of Warmest Quarter': os.path.join(input_base, 'wc2.1_30s_bio_18.tif'),
        'Mean Temperature of Coldest Quarter': os.path.join(input_base, 'wc2.1_30s_bio_11.tif'),
        'NDVI': os.path.join(input_base, 'ndvi.tif'),
        'EVI': os.path.join(input_base, 'evi.tif'),
        'lai': os.path.join(input_base, 'lai.tif'),
        'lst': os.path.join(input_base, 'lst.tif'),
        'PET': os.path.join(input_base, 'pet.tif'),
        'sd_dem': os.path.join(input_base, 'sd_dem.tif'),
        'sd_podu': os.path.join(input_base, 'sd_podu.tif'),
        'sd_poxiang': os.path.join(input_base, 'sd_poxiang.tif'),
    }

# --- GeoTIFF Output Parameters --- ## 地理数据参数
RASTER_OUTPUT_CONFIG = {
    'tif_resolution': 0.01,
    'tif_interpolation_method': 'cubic', # Note: This is for the older interpolation script, not used here directly
    'tif_nodata_value': -9999.0,
    'crs': 'EPSG:4326' # Assumed CRS for output raster and shapefile
}

def get_config(prediction_year=DEFAULT_PREDICTION_YEAR):
    """
    获取配置，支持指定预测年份
    
    参数:
        prediction_year: int, 预测年份，默认为 DEFAULT_PREDICTION_YEAR
    
    返回:
        配置字典
    """
    # 根据年份动态设置输入和输出路径
    input_raster_base = f'H:/data_new2025/2019_2024_sd/prediction_year/{prediction_year}'
    prediction_output_dir = f'H:/data_new2025/2019_2024_sd/prediction_year/results/{prediction_year}'
    
    # 创建特定年份的输出路径
    prediction_tif_path = os.path.join(prediction_output_dir, 'sd_predicted_probability.tif')
    prediction_masked_tif_path = os.path.join(prediction_output_dir, 'sd_masked_predicted_probability.tif')
    prediction_risk_class_tif_path = os.path.join(prediction_output_dir, 'sd_risk_classification.tif')
    prediction_raw_prob_tif_path = os.path.join(prediction_output_dir, 'sd_raw_probability.tif')

    # Ensure output directories exist
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(prediction_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(FEATURE_IMPORTANCE_PLOT_PATH), exist_ok=True)

    # 获取特定年份的特征栅格映射
    feature_raster_map = get_feature_raster_map(prediction_year)

    return {
        'csv_data_path': CSV_DATA_PATH,
        'model_save_path': MODEL_SAVE_PATH,
        'scaler_save_path': SCALER_SAVE_PATH,
        'boundary_shp_path': BOUNDARY_SHP_PATH,
        'prediction_output_dir': prediction_output_dir,
        'prediction_tif_path': prediction_tif_path,
        'prediction_masked_tif_path': prediction_masked_tif_path,
        'prediction_shp_path': PREDICTION_SHP_PATH,
        'prediction_risk_class_tif_path': prediction_risk_class_tif_path,
        'prediction_raw_prob_tif_path': prediction_raw_prob_tif_path,
        'feature_importance_plot_path': FEATURE_IMPORTANCE_PLOT_PATH,
        'feature_importance_csv_path': FEATURE_IMPORTANCE_CSV_PATH,
        'training_history_plot_path': TRAINING_HISTORY_PLOT_PATH,
        'data_processing': DATA_PROCESSING_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAIN_PARAMS,
        'prediction': PREDICT_PARAMS,
        'feature_raster_map': feature_raster_map,
        'raster_output': RASTER_OUTPUT_CONFIG,
        'input_raster_base': input_raster_base,  # 添加输入路径到配置中
        'prediction_year': prediction_year  # 添加预测年份到配置中
    }

# 定义和get_config函数同级的常量
FEATURE_IMPORTANCE_PLOT_PATH = os.path.join(BASE_OUTPUT_DIR, 'analysis/feature_importance.png')
FEATURE_IMPORTANCE_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'analysis/feature_importance.csv')
TRAINING_HISTORY_PLOT_PATH = os.path.join(BASE_OUTPUT_DIR, 'analysis/training_history.png')

if __name__ == '__main__':
    # Print the config to verify paths when run directly
    config = get_config()
    import json
    print(json.dumps(config, indent=4)) 