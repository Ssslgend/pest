#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
山东省县域美国白蛾第一代发病情况预测模型配置文件
"""

import os

class CountyLevelConfig:
    """县域级别模型配置"""

    # 数据路径配置
    DATA_DIR = "datas/shandong_pest_data"
    TRAIN_DATA_PATH = os.path.join(DATA_DIR, "county_level_firstgen_train.csv")
    VAL_DATA_PATH = os.path.join(DATA_DIR, "county_level_firstgen_val.csv")
    TEST_DATA_PATH = os.path.join(DATA_DIR, "county_level_firstgen_test.csv")
    COMPLETE_DATA_PATH = os.path.join(DATA_DIR, "county_level_firstgen_complete.csv")

    # 模型保存路径
    MODEL_DIR = "models/county_level"
    RESULTS_DIR = "results/county_level"

    # 特征列配置
    METEOROLOGICAL_FEATURES = [
        'Temperature_mean', 'Temperature_std', 'Temperature_min', 'Temperature_max', 'Temperature_median',
        'Humidity_mean', 'Humidity_std', 'Humidity_min', 'Humidity_max', 'Humidity_median',
        'Rainfall_mean', 'Rainfall_std', 'Rainfall_min', 'Rainfall_max', 'Rainfall_median',
        'Pressure_mean', 'Pressure_std', 'Pressure_min', 'Pressure_max', 'Pressure_median',
        'Temp_Humidity_Index_mean', 'Temp_Humidity_Index_std', 'Temp_Humidity_Index_min',
        'Temp_Humidity_Index_max', 'Temp_Humidity_Index_median'
    ]

    SPATIAL_FEATURES = ['Latitude', 'Longitude']

    # 目标变量
    TARGET_CLASSIFICATION = 'Has_Occurrence'  # 二分类目标
    TARGET_REGRESSION = 'Severity_Level'      # 回归目标（发病程度）

    # 所有特征
    ALL_FEATURES = METEOROLOGICAL_FEATURES + SPATIAL_FEATURES

    # 模型参数
    CLASSIFICATION_PARAMS = {
        'model_type': 'classification',
        'test_size': 0.2,
        'random_state': 42,
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    }

    REGRESSION_PARAMS = {
        'model_type': 'regression',
        'test_size': 0.2,
        'random_state': 42,
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    }

    # 训练参数
    TRAINING_PARAMS = {
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'early_stopping_patience': 10,
        'validation_split': 0.2
    }

    # 数据预处理参数
    PREPROCESSING_PARAMS = {
        'normalize_features': True,
        'handle_missing': 'mean',
        'outlier_detection': True,
        'feature_selection': True
    }

    # 评估指标
    CLASSIFICATION_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'confusion_matrix']
    REGRESSION_METRICS = ['mse', 'rmse', 'mae', 'r2']

    # 交叉验证参数
    CV_PARAMS = {
        'cv_folds': 5,
        'scoring': 'f1_weighted',
        'n_jobs': -1
    }

    # 数据集信息
    DATASET_INFO = {
        'description': '山东省县域美国白蛾第一代（5-6月）发病情况预测数据集',
        'features_count': len(ALL_FEATURES),
        'target_classes': 2,  # 0: 无发病, 1: 有发病
        'severity_levels': 3,  # 1: 轻度, 2: 中度, 3: 重度
        'temporal_range': '2019-2023',
        'spatial_coverage': '山东省110个县',
        'prediction_target': '第一代美国白蛾（5-6月）发病情况'
    }

    @classmethod
    def ensure_directories(cls):
        """确保必要的目录存在"""
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)

    @classmethod
    def get_feature_info(cls):
        """获取特征信息"""
        return {
            'meteorological_features': cls.METEOROLOGICAL_FEATURES,
            'spatial_features': cls.SPATIAL_FEATURES,
            'total_features': len(cls.ALL_FEATURES),
            'target_classification': cls.TARGET_CLASSIFICATION,
            'target_regression': cls.TARGET_REGRESSION
        }

if __name__ == "__main__":
    # 测试配置
    print("=== 县域级别美国白蛾预测模型配置 ===")
    print(f"数据目录: {CountyLevelConfig.DATA_DIR}")
    print(f"特征数量: {len(CountyLevelConfig.ALL_FEATURES)}")
    print(f"气象特征: {len(CountyLevelConfig.METEOROLOGICAL_FEATURES)}")
    print(f"空间特征: {len(CountyLevelConfig.SPATIAL_FEATURES)}")
    print(f"目标变量（分类）: {CountyLevelConfig.TARGET_CLASSIFICATION}")
    print(f"目标变量（回归）: {CountyLevelConfig.TARGET_REGRESSION}")

    # 确保目录存在
    CountyLevelConfig.ensure_directories()
    print("配置目录创建完成")