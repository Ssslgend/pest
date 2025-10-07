# sd_raster_prediction/config_raster.py
import torch
import os

# --- Paths --- (Adjust base paths if needed)
BASE_OUTPUT_DIR = 'E:/0424/pestBIstm/pestBIstm/results'
DATA_DIR = 'E:/0424/pestBIstm/pestBIstm/datas'
INPUT_RASTER_BASE = 'H:/data_new2025/bilstm/X_pest_touying' # Base path for input rasters

# --- File Paths --- ## 文件的路径
#TRAIN_CONFIG ={
CSV_DATA_PATH = os.path.join(DATA_DIR, 'train.csv') # Source CSV for training points
MODEL_SAVE_PATH = os.path.join(BASE_OUTPUT_DIR, 'trained_model/sd_bilstm_presence_pseudo.pth')
SCALER_SAVE_PATH = os.path.join(BASE_OUTPUT_DIR, 'trained_model/scaler_presence_pseudo.joblib')
BOUNDARY_SHP_PATH = os.path.join(DATA_DIR, 'shandong_boundary.shp') # Optional boundary for masking

# 添加新的输出目录
# 默认输出目录
#PREDICTION_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'predictions')
# 新的输出目录，与参考内容一致
PREDICTION_OUTPUT_DIR = 'E:/0424/pestBIstm/outcome_data/prediction'

PREDICTION_TIF_PATH = os.path.join(PREDICTION_OUTPUT_DIR, 'sd_predicted_probability.tif')
PREDICTION_MASKED_TIF_PATH = os.path.join(PREDICTION_OUTPUT_DIR, 'sd_masked_predicted_probability.tif')
PREDICTION_SHP_PATH = os.path.join(PREDICTION_OUTPUT_DIR, 'sd_predicted_points.shp')
# 添加风险等级分类图的路径
PREDICTION_RISK_CLASS_TIF_PATH = os.path.join(PREDICTION_OUTPUT_DIR, 'sd_risk_classification.tif')
FEATURE_IMPORTANCE_PLOT_PATH = os.path.join(BASE_OUTPUT_DIR, 'analysis/feature_importance.png')
FEATURE_IMPORTANCE_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'analysis/feature_importance.csv')
TRAINING_HISTORY_PLOT_PATH = os.path.join(BASE_OUTPUT_DIR, 'analysis/training_history.png')

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
    'dropout': 0.4,
    'output_size': 1 # Fixed to 1 for probability prediction
}

# --- Training Parameters --- ## 训练参数
TRAIN_PARAMS = {
    'batch_size': 32,
    'learning_rate': 0.0005,
    'weight_decay': 1e-5,
    'num_epochs': 100,
    'patience': 15, # Early stopping patience
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
    'Mean Diurnal Range (Mean of monthly (max temp - min temp))': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_2.tif'),
    'Precipitation of Coldest Quarter': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_19.tif'),
    'Precipitation of Wettest Month': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_13.tif'),
    'Precipitation of Driest Quarter': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_17.tif'),
    'Annual Precipitation': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_12.tif'),
    'Annual Mean Temperature': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_1.tif'),
    'Mean Temperature of Driest Quarter': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_9.tif'),
    'Precipitation of Driest Month': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_14.tif'),
    'Temperature Annual Range (BIO5-BIO6)': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_7.tif'),
    'Min Temperature of Coldest Month': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_6.tif'),
    'Precipitation Seasonality (Coefficient of Variation)': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_15.tif'),
    'Precipitation of Wettest Quarter': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_16.tif'),
    'Temperature Seasonality (standard deviation ×100)': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_4.tif'),
    'Max Temperature of Warmest Month': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_5.tif'),
    'sd_poxiang': os.path.join(INPUT_RASTER_BASE, 'sd_poxiang.tif'),
    'Isothermality (BIO2/BIO7) (×100)': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_3.tif'),
    'Mean Temperature of Warmest Quarter': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_10.tif'),
    'Precipitation of Warmest Quarter': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_18.tif'),
    'Mean Temperature of Coldest Quarter': os.path.join(INPUT_RASTER_BASE, 'wc2.1_30s_bio_11.tif'),
    'NDVI': os.path.join(INPUT_RASTER_BASE, 'ndvi.tif'),
    'EVI': os.path.join(INPUT_RASTER_BASE, 'evi.tif'),
    'lai': os.path.join(INPUT_RASTER_BASE, 'lai.tif'),
    'lst': os.path.join(INPUT_RASTER_BASE, 'lst.tif'),
    'PET': os.path.join(INPUT_RASTER_BASE, 'pet.tif'),
    'sd_dem': os.path.join(INPUT_RASTER_BASE, 'sd_dem.tif'),
    'sd_podu': os.path.join(INPUT_RASTER_BASE, 'sd_podu.tif'),
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