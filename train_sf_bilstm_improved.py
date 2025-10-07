#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的SF-BiLSTM训练脚本 - 解决类别不平衡问题
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
import time
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils import class_weight
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# 导入SF-BiLSTM模型组件
import sys
sys.path.append('oldPestBlstem/ablation_study')
from bilstm_variants import BiLSTMComplete, BiLSTMNoAttention

class ImprovedDataProcessor:
    """改进的数据处理器，解决类别不平衡问题"""

    def __init__(self):
        self.scaler = StandardScaler()

    def load_and_preprocess_data(self, csv_path):
        """加载和预处理数据"""
        print(f"加载数据: {csv_path}")
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"数据形状: {df.shape}")

        # 检查标签分布
        print(f"原始标签分布:")
        print(df['Has_Occurrence'].value_counts())

        # 选择数值型特征
        excluded_cols = ['Has_Occurrence', 'county_name', 'year', 'month', 'Severity', 'Period', 'Season']
        feature_cols = [col for col in df.columns if col not in excluded_cols and df[col].dtype in ['int64', 'float64', 'int32', 'float32']]

        print(f"使用特征数量: {len(feature_cols)}")

        # 提取特征和标签
        X = df[feature_cols].values
        y = df['Has_Occurrence'].values

        # 处理缺失值
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        return X, y, feature_cols

    def balance_data(self, X, y, method='oversample'):
        """平衡数据集"""
        print(f"\n使用 {method} 方法平衡数据...")

        if method == 'oversample':
            # 对少数类进行过采样
            df = pd.DataFrame(X)
            df['label'] = y

            # 分离多数类和少数类
            df_majority = df[df['label'] == 0]
            df_minority = df[df['label'] == 1]

            # 过采样少数类
            df_minority_oversampled = resample(df_minority,
                                            replace=True,     # 有放回采样
                                            n_samples=len(df_majority),  # 与多数类数量相同
                                            random_state=42)

            # 合并数据
            df_balanced = pd.concat([df_majority, df_minority_oversampled])

            X_balanced = df_balanced.drop('label', axis=1).values
            y_balanced = df_balanced['label'].values

        elif method == 'undersample':
            # 对多数类进行欠采样
            df = pd.DataFrame(X)
            df['label'] = y

            df_majority = df[df['label'] == 0]
            df_minority = df[df['label'] == 1]

            # 欠采样多数类
            df_majority_undersampled = resample(df_majority,
                                             replace=False,    # 无放回采样
                                             n_samples=len(df_minority) * 2,  # 少数类的2倍
                                             random_state=42)

            df_balanced = pd.concat([df_majority_undersampled, df_minority])

            X_balanced = df_balanced.drop('label', axis=1).values
            y_balanced = df_balanced['label'].values

        else:
            X_balanced, y_balanced = X, y

        print(f"平衡后标签分布: {np.bincount(y_balanced)}")

        return X_balanced, y_balanced

    def create_data_loaders(self, X, y, test_size=0.2, val_size=0.2, batch_size=32, balance_method='oversample'):
        """创建数据加载器"""

        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        # 数据分割
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )

        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )

        print(f"数据分割:")
        print(f"训练集: {X_train.shape}, 标签分布: {np.bincount(y_train)}")
        print(f"验证集: {X_val.shape}, 标签分布: {np.bincount(y_val)}")
        print(f"测试集: {X_test.shape}, 标签分布: {np.bincount(y_test)}")

        # 平衡训练数据
        if balance_method:
            X_train_balanced, y_train_balanced = self.balance_data(X_train, y_train, balance_method)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train

        # 计算类别权重
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_balanced),
            y=y_train_balanced
        )
        class_weights = torch.FloatTensor(class_weights).to('cpu')

        # 创建数据加载器
        X_train_tensor = torch.FloatTensor(X_train_balanced.reshape(X_train_balanced.shape[0], 1, X_train_balanced.shape[1]))
        y_train_tensor = torch.FloatTensor(y_train_balanced.reshape(-1, 1))

        X_val_tensor = torch.FloatTensor(X_val.reshape(X_val.shape[0], 1, X_val.shape[1]))
        y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1))

        X_test_tensor = torch.FloatTensor(X_test.reshape(X_test.shape[0], 1, X_test.shape[1]))
        y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1))

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, class_weights, (X_test, y_test)

class WeightedBCELoss(nn.Module):
    """加权二元交叉熵损失"""
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(inputs.device)
        else:
            pos_weight = 1.0

        loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight)
        return loss

def train_improved_model():
    """训练改进的SF-BiLSTM模型"""
    print("开始改进的SF-BiLSTM训练...")

    # 配置
    config = {
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'epochs': 30,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'batch_size': 32
    }

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    data_file = 'datas/shandong_pest_data/real_occurrence_complete_data.csv'
    processor = ImprovedDataProcessor()

    X, y, feature_cols = processor.load_and_preprocess_data(data_file)

    # 测试不同的平衡方法
    balance_methods = ['oversample', 'undersample', None]
    results = {}

    for balance_method in balance_methods:
        method_name = balance_method if balance_method else 'original'
        print(f"\n{'='*60}")
        print(f"测试平衡方法: {method_name}")
        print(f"{'='*60}")

        # 创建数据加载器
        train_loader, val_loader, test_loader, class_weights, test_data = processor.create_data_loaders(
            X, y, batch_size=config['batch_size'], balance_method=balance_method
        )

        # 创建模型
        model_config = {
            "input_size": len(feature_cols),
            "hidden_size": config['hidden_size'],
            "num_layers": config['num_layers'],
            "dropout": config['dropout'],
            "num_classes": 1
        }

        model = BiLSTMComplete(model_config).to(device)

        # 设置优化器和损失函数
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

        # 使用加权损失函数
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]]) if len(class_weights) > 1 else torch.tensor([1.0])
        criterion = WeightedBCELoss(pos_weight=pos_weight)

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        # 训练模型
        best_val_auc = -np.inf
        best_model_state = None

        print(f"开始训练 {method_name}...")
        for epoch in range(config['epochs']):
            # 训练阶段
            model.train()
            train_loss = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 验证阶段
            model.eval()
            val_loss = 0
            all_val_preds = []
            all_val_targets = []

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()

                    pred_probs = torch.sigmoid(output)
                    all_val_preds.extend(pred_probs.cpu().numpy())
                    all_val_targets.extend(target.cpu().numpy())

            val_loss /= len(val_loader)

            # 计算验证AUC
            val_preds_np = np.array(all_val_preds).flatten()
            val_targets_np = np.array(all_val_targets).flatten()

            try:
                val_auc = roc_auc_score(val_targets_np, val_preds_np)
            except:
                val_auc = 0.5

            scheduler.step(val_auc)

            # 保存最佳模型
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()

            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch+1}/{config["epochs"]} | Train Loss: {train_loss:.6f} | Val AUC: {val_auc:.4f}')

        # 加载最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # 评估模型
        model.eval()
        all_test_preds = []
        all_test_targets = []
        all_test_probs = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                pred_probs = torch.sigmoid(output)
                pred = (pred_probs > 0.5).float()

                all_test_preds.append(pred.cpu().numpy())
                all_test_targets.append(target.cpu().numpy())
                all_test_probs.append(pred_probs.cpu().numpy())

        # 合并结果
        test_preds = np.vstack(all_test_preds).flatten()
        test_targets = np.vstack(all_test_targets).flatten()
        test_probs = np.vstack(all_test_probs).flatten()

        # 计算测试指标
        accuracy = accuracy_score(test_targets, test_preds)
        precision = precision_score(test_targets, test_preds, zero_division=0)
        recall = recall_score(test_targets, test_preds, zero_division=0)
        f1 = f1_score(test_targets, test_preds, zero_division=0)

        try:
            auc = roc_auc_score(test_targets, test_probs)
        except:
            auc = 0.5

        # 计算特异度
        conf_matrix = confusion_matrix(test_targets, test_preds)
        if len(conf_matrix) > 1:
            tn, fp, fn, tp = conf_matrix.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            specificity = 0

        results[method_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'specificity': specificity,
            'confusion_matrix': conf_matrix.tolist(),
            'best_val_auc': best_val_auc
        }

        print(f"{method_name} 测试结果:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Specificity: {specificity:.4f}")

    # 保存结果
    os.makedirs('results/sf_bilstm_improved', exist_ok=True)
    with open('results/sf_bilstm_improved/improved_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # 打印对比结果
    print(f"\n{'='*80}")
    print("改进方法对比结果")
    print(f"{'='*80}")

    print(f"{'方法':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
    print("-" * 80)

    for method_name, metrics in results.items():
        print(f"{method_name:<15} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {metrics['auc']:<12.4f}")

    print(f"\n最佳结果:")
    best_method = max(results.keys(), key=lambda x: results[x]['f1'])
    print(f"方法: {best_method}")
    print(f"F1分数: {results[best_method]['f1']:.4f}")
    print(f"AUC: {results[best_method]['auc']:.4f}")

if __name__ == "__main__":
    train_improved_model()