# sd_raster_prediction/config_raster.py
import torch
import os

# --- Paths --- (Adjust base paths if needed)
BASE_OUTPUT_DIR = 'E:/code/0424/pestBIstm/pestBIstm/sd_raster_prediction/results'
DATA_DIR = 'E:/code/0424/pestBIstm/pestBIstm/datas'
INPUT_RASTER_BASE = 'H:/data_new2025/bilstm/X_pest_touying' # Base path for input rasters

# --- File Paths --- ## 文件的路径
#TRAIN_CONFIG ={
CSV_DATA_PATH = os.path.join(DATA_DIR, 'china_19_1.csv') # Source CSV for training points
MODEL_SAVE_PATH = os.path.join(BASE_OUTPUT_DIR, 'trained_model/sd_bilstm_presence_pseudo_anti_overfit.pth')
SCALER_SAVE_PATH = os.path.join(BASE_OUTPUT_DIR, 'trained_model/scaler_presence_pseudo_anti_overfit.joblib')
BOUNDARY_SHP_PATH = os.path.join('H:/yanyi/Stydy_data/shandong.shp') # Optional boundary for masking

# 添加新的输出目录
# 默认输出目录
#PREDICTION_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'predictions')
# 新的输出目录，与参考内容一致
PREDICTION_OUTPUT_DIR = 'E:/code/0424/pestBIstm/outcome_data/prediction'

PREDICTION_TIF_PATH = os.path.join(PREDICTION_OUTPUT_DIR, 'china_predicted_probability.tif')
PREDICTION_MASKED_TIF_PATH = os.path.join(PREDICTION_OUTPUT_DIR, 'china_masked_predicted_probability.tif')
PREDICTION_SHP_PATH = os.path.join(PREDICTION_OUTPUT_DIR, 'china_predicted_points.shp')
# 添加风险等级分类图的路径
PREDICTION_RISK_CLASS_TIF_PATH = os.path.join(PREDICTION_OUTPUT_DIR, 'china_risk_classification.tif')
PREDICTION_RAW_PROB_TIF_PATH = os.path.join(PREDICTION_OUTPUT_DIR, 'china_raw_predicted_probability.tif')
FEATURE_IMPORTANCE_PLOT_PATH = os.path.join(BASE_OUTPUT_DIR, 'analysis/feature_importance.png')
FEATURE_IMPORTANCE_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'analysis/feature_importance.csv')
TRAINING_HISTORY_PLOT_PATH = os.path.join(BASE_OUTPUT_DIR, 'analysis/training_history.png')

# --- Data Processing Parameters --- ## 数据参数
DATA_PROCESSING_CONFIG = {
    'pseudo_absence_ratio': 1.0, # Ratio of pseudo-absence points to generate
    'test_size': 0.25,           # 增加测试集比例，从0.2提高到0.25
    'val_size': 0.2,             # 增加验证集比例，从0.15提高到0.2
    'random_state': 42,
    'coordinate_columns': ['X', 'Y'], # Column names for coordinates in CSV
    'label_column': '白蛾',       # CORRECTED: Column name for presence label in CSV
    'excluded_cols_from_features': ['id'] # CORRECTED: Other columns in CSV to exclude from features
}

# --- Model Hyperparameters (BiLSTM specific) --- ## 模型超参数
MODEL_CONFIG = {
    # input_size will be determined by data processor
    'hidden_size': 48,           # 提高隐藏层大小，从32增加到48，但仍小于原始的64
    'num_layers': 1,             # 保持BiLSTM层数为1
    'dropout': 0.5,              # 适当降低dropout率，从0.8降到0.5，保持适度正则化
    'output_size': 1 # Fixed to 1 for probability prediction
}

# --- Training Parameters --- ## 训练参数
TRAIN_PARAMS = {
    'batch_size': 12,            # 适当增加batch_size，从8增加到12
    'learning_rate': 0.0002,     # 增加学习率，从0.0001增加到0.0002
    'weight_decay': 2e-4,        # 适当降低权重衰减，从5e-4降到2e-4
    'num_epochs': 100,           # 增加最大训练轮次，确保模型有足够时间收敛
    'patience': 10,              # 增加早停耐心值，从5增加到10
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# --- Prediction Parameters --- ## 预测参数
PREDICT_PARAMS = {
    'prediction_batch_size': 512 * 512, # Pixels per prediction batch
    'processing_block_size': 1024,    # Rasterio window read size
    'save_shapefile': True,
    'mask_with_boundary': True, # Set to False if boundary shapefile is not available or not needed

    # 概率校准参数 - 用于调整模型输出的概率分布
    'probability_calibration': {
        'scale_factor': 0.8,  # 缩放因子，将概率范围压缩
        'shift_factor': -0.1, # 偏移因子，将概率整体向下偏移
    },

    # 风险等级划分阈值 - 用于将连续概率转换为风险等级
    # 使用分位数方法后，我们可以使用均匀的阈值
    'risk_thresholds': {
        'no_risk': 0.25,     # 小于0.25的为无风险
        'low_risk': 0.5,     # 0.25-0.5为低风险
        'medium_risk': 0.75, # 0.5-0.75为中风险
        'high_risk': 0.95    # 0.75-0.95为高风险，大于0.95为极高风险
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
FEATURE_RASTER_MAP = {
    '平均日较差': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_2.tif'),
    '年均温': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_1.tif'),
    '年降水量': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_12.tif'),
    '最冷季度平均气温': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_11.tif'),
    '最冷季度（或最寒季度）总降水量': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_19.tif'),
    '最冷月（或最寒月）最低温': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_6.tif'),
    '最干季度平均气温': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_9.tif'),
    '最干季度（或最干燥季度）总降水量': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_17.tif'),
    '最干月（或最干燥月）降水量': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_14.tif'),
    '最暖季度平均气温': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_10.tif'),
    '最暖季度（或最热季度）总降水量': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_18.tif'),
    '最暖月（或最热月）最高温': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_5.tif'),
    '最湿季度平均气温': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_8.tif'),
    '最湿季度（或最湿润季度）总降水量': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_16.tif'),
    '最湿月（或最湿润月）降水量': os.path.join(INPUT_RASTER_BASE, '13.tif'), #  *请务必确认此文件名 wc2.1_30s_bio_13.tif 是否正确!*
    '气温季节性': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_4.tif'),
    '气温年较差': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_7.tif'),
    '等温性': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_3.tif'),
    '降水季节性': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_15.tif'),
    'dem': os.path.join(INPUT_RASTER_BASE, 'sd_dem1.tif'),
    'podu': os.path.join(INPUT_RASTER_BASE, 'sd_podu.tif'),
    'poxiang1': os.path.join(INPUT_RASTER_BASE, 'sd_poxiang.tif'),
    'zhibei': os.path.join(INPUT_RASTER_BASE, 'zhibei_sd.tif'),
}

# --- GeoTIFF Output Parameters --- ## 地理数据参数
RASTER_OUTPUT_CONFIG = {
    'tif_resolution': 0.01,
    'tif_interpolation_method': 'cubic', # Note: This is for the older interpolation script, not used here directly
    'tif_nodata_value': -9999.0,
    'crs': 'EPSG:4326' # Assumed CRS for output raster and shapefile
}

def get_config():
    # Ensure output directories exist
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(FEATURE_IMPORTANCE_PLOT_PATH), exist_ok=True)

    return {
        'csv_data_path': CSV_DATA_PATH,
        'model_save_path': MODEL_SAVE_PATH,
        'scaler_save_path': SCALER_SAVE_PATH,
        'boundary_shp_path': BOUNDARY_SHP_PATH,
        'prediction_output_dir': PREDICTION_OUTPUT_DIR,
        'prediction_tif_path': PREDICTION_TIF_PATH,
        'prediction_masked_tif_path': PREDICTION_MASKED_TIF_PATH,
        'prediction_shp_path': PREDICTION_SHP_PATH,
        'prediction_risk_class_tif_path': PREDICTION_RISK_CLASS_TIF_PATH,
        'prediction_raw_prob_tif_path': PREDICTION_RAW_PROB_TIF_PATH,
        'feature_importance_plot_path': FEATURE_IMPORTANCE_PLOT_PATH,
        'feature_importance_csv_path': FEATURE_IMPORTANCE_CSV_PATH,
        'training_history_plot_path': TRAINING_HISTORY_PLOT_PATH,
        'data_processing': DATA_PROCESSING_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAIN_PARAMS,
        'prediction': PREDICT_PARAMS,
        'feature_raster_map': FEATURE_RASTER_MAP,
        'raster_output': RASTER_OUTPUT_CONFIG
    }

if __name__ == '__main__':
    # Print the config to verify paths when run directly
    config = get_config()
    import json
    print(json.dumps(config, indent=4))