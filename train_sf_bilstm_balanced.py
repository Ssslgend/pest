#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用平衡数据训练SF-BiLSTM模型
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import time
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')

# 导入SF-BiLSTM模型
import sys
sys.path.append('oldPestBlstem/ablation_study')
from bilstm_variants import BiLSTMComplete

class BalancedDataProcessor:
    """平衡数据处理器"""

    def __init__(self):
        self.scaler = None

    def load_balanced_datasets(self):
        """加载平衡的数据集"""
        print("加载平衡数据集...")

        base_path = 'datas/shandong_pest_data'

        # 加载数据
        train_data = pd.read_csv(os.path.join(base_path, 'balanced_train.csv'))
        val_data = pd.read_csv(os.path.join(base_path, 'balanced_val.csv'))
        test_data = pd.read_csv(os.path.join(base_path, 'balanced_test.csv'))

        print(f"训练集: {train_data.shape}")
        print(f"验证集: {val_data.shape}")
        print(f"测试集: {test_data.shape}")

        # 检查标签分布
        print(f"\n标签分布:")
        print(f"训练集 - 发生率: {train_data['Has_Occurrence'].mean():.4f} ({train_data['Has_Occurrence'].sum()}正样本)")
        print(f"验证集 - 发生率: {val_data['Has_Occurrence'].mean():.4f} ({val_data['Has_Occurrence'].sum()}正样本)")
        print(f"测试集 - 发生率: {test_data['Has_Occurrence'].mean():.4f} ({test_data['Has_Occurrence'].sum()}正样本)")

        return train_data, val_data, test_data

    def extract_features_and_labels(self, train_data, val_data, test_data):
        """提取特征和标签"""
        print("提取特征和标签...")

        # 排除非特征列
        excluded_cols = [
            'Has_Occurrence', 'Severity', 'county_name', 'year', 'month',
            'Period', 'Data_Source'
        ]

        # 选择特征列（数值型）
        feature_cols = []
        for col in train_data.columns:
            if col not in excluded_cols and train_data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                feature_cols.append(col)

        print(f"使用特征数量: {len(feature_cols)}")
        print(f"特征列: {feature_cols[:10]}...")

        # 提取特征和标签
        X_train = train_data[feature_cols].values
        y_train = train_data['Has_Occurrence'].values

        X_val = val_data[feature_cols].values
        y_val = val_data['Has_Occurrence'].values

        X_test = test_data[feature_cols].values
        y_test = test_data['Has_Occurrence'].values

        print(f"特征矩阵形状: 训练={X_train.shape}, 验证={X_val.shape}, 测试={X_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

    def create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
        """创建数据加载器"""
        print("创建数据加载器...")

        # 转换为PyTorch张量，为LSTM添加序列维度
        X_train_tensor = torch.FloatTensor(X_train.reshape(X_train.shape[0], 1, X_train.shape[1]))
        y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))

        X_val_tensor = torch.FloatTensor(X_val.reshape(X_val.shape[0], 1, X_val.shape[1]))
        y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1))

        X_test_tensor = torch.FloatTensor(X_test.reshape(X_test.shape[0], 1, X_test.shape[1]))
        y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1))

        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"数据加载器创建完成，批次大小: {batch_size}")
        return train_loader, val_loader, test_loader

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

        loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight)
        return loss

def train_sf_bilstm_balanced():
    """使用平衡数据训练SF-BiLSTM模型"""
    print("=" * 60)
    print("使用平衡数据训练SF-BiLSTM模型")
    print("=" * 60)

    # 配置参数
    config = {
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'epochs': 60,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'batch_size': 32,
        'early_stopping_patience': 15
    }

    print(f"模型配置: {config}")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据处理
    processor = BalancedDataProcessor()

    # 加载数据
    train_data, val_data, test_data = processor.load_balanced_datasets()

    # 提取特征和标签
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = processor.extract_features_and_labels(
        train_data, val_data, test_data
    )

    # 创建数据加载器
    train_loader, val_loader, test_loader = processor.create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=config['batch_size']
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
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 计算类别权重（平衡数据中权重应该更合理）
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]])
    print(f"正样本权重: {pos_weight.item():.4f}")
    print(f"类别权重: {dict(zip(np.unique(y_train), class_weights))}")

    # 设置优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = WeightedBCELoss(pos_weight=pos_weight)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)

    # 训练循环
    best_val_f1 = -np.inf
    best_model_state = None
    patience_counter = 0
    train_history = []

    print(f"\n开始训练 ({config['epochs']} 个epoch)...")
    print("-" * 60)

    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 计算训练准确率
            pred_probs = torch.sigmoid(output)
            pred = (pred_probs > 0.5).float()
            train_correct += (pred == target).sum().item()
            train_total += target.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

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
                pred = (pred_probs > 0.5).float()
                all_val_preds.extend(pred.cpu().numpy())
                all_val_targets.extend(target.cpu().numpy())

        val_loss /= len(val_loader)

        # 计算验证指标
        val_preds_np = np.array(all_val_preds).flatten()
        val_targets_np = np.array(all_val_targets).flatten()

        try:
            val_acc = accuracy_score(val_targets_np, val_preds_np)
            val_precision = precision_score(val_targets_np, val_preds_np, zero_division=0)
            val_recall = recall_score(val_targets_np, val_preds_np, zero_division=0)
            val_f1 = f1_score(val_targets_np, val_preds_np, zero_division=0)
            val_auc = roc_auc_score(val_targets_np, torch.sigmoid(model(next(iter(val_loader))[0].to(device))).cpu().numpy().flatten())
        except:
            val_acc = val_precision = val_recall = val_f1 = 0.5
            val_auc = 0.5

        scheduler.step(val_f1)

        # 保存训练历史
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_auc': val_auc
        })

        # 早停检查（基于F1分数）
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # 打印进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch {epoch+1:3d} | '
                  f'Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.4f} | '
                  f'Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}')

        # 早停
        if patience_counter >= config['early_stopping_patience']:
            print(f"\n早停触发，在第 {epoch+1} 轮停止训练")
            break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"加载最佳模型，验证F1: {best_val_f1:.4f}")

    # 在测试集上评估
    print(f"\n测试集评估:")
    print("-" * 40)
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

    # 打印结果
    print(f"测试结果:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  混淆矩阵:")
    print(f"    TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # 保存结果
    results = {
        'config': config,
        'test_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'specificity': specificity,
            'confusion_matrix': conf_matrix.tolist(),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        },
        'best_val_f1': best_val_f1,
        'train_history': train_history,
        'feature_count': len(feature_cols),
        'data_info': {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'train_positive_rate': float(np.mean(y_train)),
            'train_imbalance_ratio': float((len(y_train) - np.sum(y_train)) / np.sum(y_train))
        }
    }

    # 保存结果
    os.makedirs('results/sf_bilstm_balanced', exist_ok=True)
    with open('results/sf_bilstm_balanced/training_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # 保存模型
    torch.save(model.state_dict(), 'results/sf_bilstm_balanced/best_model.pth')

    # 保存训练历史
    history_df = pd.DataFrame(train_history)
    history_df.to_csv('results/sf_bilstm_balanced/training_history.csv', index=False)

    print(f"\n训练完成！结果已保存到: results/sf_bilstm_balanced/")

    # 与之前结果对比
    print(f"\n与原始数据对比:")
    print(f"原始数据 - F1: 0.1131, AUC: 0.9069, 发生率: 4.05%")
    print(f"平衡数据 - F1: {f1:.4f}, AUC: {auc:.4f}, 发生率: {np.mean(y_train):.1%}")

    improvement_f1 = f1 - 0.1131
    print(f"F1改善: {improvement_f1:+.4f} ({improvement_f1/0.1131*100:+.1f}%)")

    return results, model

if __name__ == "__main__":
    results, model = train_sf_bilstm_balanced()