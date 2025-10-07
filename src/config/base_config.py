"""
基础配置文件
包含项目的全局配置参数
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 数据目录配置
DATA_DIRS = {
    'raw': PROJECT_ROOT / 'data' / 'raw',
    'processed': PROJECT_ROOT / 'data' / 'processed',
    'external': PROJECT_ROOT / 'data' / 'external',
    'training': PROJECT_ROOT / 'data' / 'processed' / 'training',
    'validation': PROJECT_ROOT / 'data' / 'processed' / 'validation',
    'test': PROJECT_ROOT / 'data' / 'processed' / 'test'
}

# 结果目录配置
RESULT_DIRS = {
    'models': PROJECT_ROOT / 'results' / 'models',
    'predictions': PROJECT_ROOT / 'results' / 'predictions',
    'evaluations': PROJECT_ROOT / 'results' / 'evaluations',
    'visualizations': PROJECT_ROOT / 'results' / 'visualizations',
    'reports': PROJECT_ROOT / 'results' / 'reports'
}

# 源数据文件路径
SOURCE_FILES = {
    'occurrence_excel': PROJECT_ROOT / 'datas' / 'shandong_pest_data' / '发病情况.xlsx',
    'county_boundaries': PROJECT_ROOT / 'datas' / 'shandong_pest_data' / 'shandong.json',
    'meteorological_data': PROJECT_ROOT / 'datas' / 'shandong_pest_data' / 'shandong_spatial_meteorological_data.csv',
    'training_data': PROJECT_ROOT / 'datas' / 'shandong_pest_data' / 'real_occurrence_train.csv',
    'validation_data': PROJECT_ROOT / 'datas' / 'shandong_pest_data' / 'real_occurrence_val.csv',
    'test_data': PROJECT_ROOT / 'datas' / 'shandong_pest_data' / 'real_occurrence_test.csv',
    'scaler': PROJECT_ROOT / 'datas' / 'shandong_pest_data' / 'real_occurrence_scaler.joblib'
}

# 模型配置
MODEL_CONFIG = {
    'bilstm': {
        'input_size': 31,
        'hidden_size': 256,
        'num_layers': 4,
        'dropout': 0.3,
        'bidirectional': True,
        'attention': True
    },
    'sequence_length': 8,
    'num_classes': 3,  # 发病程度: 1, 2, 3
    'binary_classification': True  # 是否进行二分类(Has_Occurrence)
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'early_stopping_patience': 10,
    'validation_split': 0.2,
    'random_seed': 42
}

# 数据处理配置
DATA_CONFIG = {
    'feature_columns': [
        'Temperature_mean', 'Temperature_std', 'Temperature_min', 'Temperature_max',
        'Humidity_mean', 'Humidity_std', 'Humidity_min', 'Humidity_max',
        'Rainfall_mean', 'Rainfall_sum', 'Rainfall_min', 'Rainfall_max',
        'WS_mean', 'WS_std', 'WD_mean',
        'Pressure_mean', 'Pressure_std',
        'Sunshine_mean', 'Sunshine_std',
        'Visibility_mean', 'Visibility_std',
        'latitude', 'longitude', 'Season', 'Temp_Humidity_Index',
        'Cumulative_Rainfall_3month', 'Temp_Trend',
        'Temperature_lag1', 'Humidity_lag1', 'Rainfall_lag1',
        'Moth_Activity_Level'
    ],
    'target_columns': ['Severity', 'Has_Occurrence'],
    'categorical_columns': ['Season', 'Moth_Activity_Level'],
    'numerical_columns': [col for col in MODEL_CONFIG['bilstm']['input_size'] * [''] if col != 'Season' and col != 'Moth_Activity_Level']
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_path': PROJECT_ROOT / 'logs' / 'pest_prediction.log'
}

# 确保目录存在
for dir_dict in [DATA_DIRS, RESULT_DIRS]:
    for dir_path in dir_dict.values():
        dir_path.mkdir(parents=True, exist_ok=True)

# 日志目录
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)