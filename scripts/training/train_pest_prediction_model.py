#!/usr/bin/env python3
"""
美国白蛾预测模型主训练脚本
使用重构后的模块化代码结构
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import logging

from src.config.base_config import *
from src.data.loaders.excel_loader import ExcelDataLoader
from src.data.processors.occurrence_processor import OccurrenceDataProcessor
from src.data.integrators.real_occurrence_merger import RealOccurrenceMerger
from src.models.base.bilstm import BiLSTMWithAttention
from src.training.trainers.base_trainer import BaseTrainer
from src.utils.logging_utils import setup_logging

def main():
    """主训练函数"""
    # 设置日志
    setup_logging(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)

    logger.info("开始美国白蛾预测模型训练...")

    # 1. 加载数据
    logger.info("加载训练数据...")
    train_data = pd.read_csv(SOURCE_FILES['training_data'])
    val_data = pd.read_csv(SOURCE_FILES['validation_data'])
    test_data = pd.read_csv(SOURCE_FILES['test_data'])

    logger.info(f"训练数据形状: {train_data.shape}")
    logger.info(f"验证数据形状: {val_data.shape}")
    logger.info(f"测试数据形状: {test_data.shape}")

    # 2. 准备特征和标签
    feature_columns = DATA_CONFIG['feature_columns']
    target_column = 'Severity'  # 或 'Has_Occurrence' 用于二分类

    X_train = train_data[feature_columns].values
    y_train = train_data[target_column].values - 1  # 将1-3转换为0-2

    X_val = val_data[feature_columns].values
    y_val = val_data[target_column].values - 1

    X_test = test_data[feature_columns].values
    y_test = test_data[target_column].values - 1

    # 3. 数据标准化
    logger.info("数据标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 保存标准化器
    scaler_path = RESULT_DIRS['models'] / 'standard_scaler.joblib'
    joblib.dump(scaler, scaler_path)
    logger.info(f"标准化器已保存: {scaler_path}")

    # 4. 创建数据加载器
    logger.info("创建数据加载器...")

    # 这里需要创建PyTorch Dataset类
    # 暂时使用numpy数据进行演示
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.LongTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.LongTensor(y_val)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test_scaled),
        torch.LongTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False)

    # 5. 创建模型
    logger.info("创建BiLSTM模型...")
    model_config = MODEL_CONFIG['bilstm']
    model = BiLSTMWithAttention(
        input_size=len(feature_columns),
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_classes=MODEL_CONFIG['num_classes'],
        dropout=model_config['dropout'],
        bidirectional=model_config['bidirectional']
    )

    # 6. 设置训练组件
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # 7. 创建训练器
    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=TRAINING_CONFIG,
        save_dir=RESULT_DIRS['models']
    )

    # 8. 开始训练
    logger.info("开始模型训练...")
    trainer.train()

    # 9. 评估模型
    logger.info("评估模型性能...")
    test_loss, test_accuracy = trainer.evaluate(test_loader)
    logger.info(f"测试损失: {test_loss:.4f}, 测试准确率: {test_accuracy:.4f}")

    # 10. 保存最终模型
    final_model_path = RESULT_DIRS['models'] / 'pest_prediction_model_final.pth'
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"最终模型已保存: {final_model_path}")

    logger.info("训练完成！")

if __name__ == "__main__":
    main()