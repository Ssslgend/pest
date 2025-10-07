#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SF-BiLSTM模型真实数据训练脚本
适配新的县域气象遥感数据，确保使用真实数据进行训练和测试
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import time
import json
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')

# 导入SF-BiLSTM模型组件
import sys
sys.path.append('oldPestBlstem/ablation_study')
from bilstm_variants import (
    BiLSTMComplete,
    BiLSTMNoAttention,
    BiLSTMNoResidual,
    BiLSTMNoCalibration,
    BiLSTMNoExperts,
    UnidirectionalLSTM
)

# 设置中文字体
matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
matplotlib.rcParams['axes.unicode_minus'] = False

class RealDataProcessor:
    """真实数据处理器，专门处理县域气象遥感数据"""

    def __init__(self, data_type="county_level"):
        self.data_type = data_type
        self.scaler = StandardScaler()

    def load_county_level_data(self, csv_path):
        """加载县域级别数据"""
        print(f"加载县域级别数据: {csv_path}")
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='gbk')

        print(f"数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")

        # 检查数据质量
        print(f"\n数据质量检查:")
        print(f"缺失值数量: {df.isnull().sum().sum()}")
        print(f"标签分布 (Severity_Level):")
        print(df['Severity_Level'].value_counts() if 'Severity_Level' in df.columns else "N/A")
        print(f"发生分布 (Has_Occurrence):")
        print(df['Has_Occurrence'].value_counts() if 'Has_Occurrence' in df.columns else "N/A")

        return df

    def load_real_occurrence_data(self, csv_path):
        """加载真实发生数据"""
        print(f"加载真实发生数据: {csv_path}")
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='gbk')

        print(f"数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")

        # 检查数据质量
        print(f"\n数据质量检查:")
        print(f"缺失值数量: {df.isnull().sum().sum()}")
        print(f"严重程度分布 (Severity):")
        print(df['Severity'].value_counts() if 'Severity' in df.columns else "N/A")
        print(f"发生分布 (Has_Occurrence):")
        print(df['Has_Occurrence'].value_counts() if 'Has_Occurrence' in df.columns else "N/A")

        return df

    def preprocess_county_level_data(self, df):
        """预处理县域级别数据"""
        print("\n预处理县域级别数据...")

        # 确定目标变量 - 优先使用Severity_Level，因为Has_Occurrence全部为1
        if 'Severity_Level' in df.columns:
            target_col = 'Severity_Level'
            print(f"使用目标变量: {target_col}")
            # 将Severity_Level转换为二分类 (1,2,3 -> 1，但我们需要创建一些负样本)
            # 将Severity_Level=1作为负样本(0)，2和3作为正样本(1)
            df['label'] = (df[target_col] > 1).astype(int)
        elif 'Has_Occurrence' in df.columns:
            target_col = 'Has_Occurrence'
            print(f"使用目标变量: {target_col}")
            df['label'] = df[target_col].astype(int)
        else:
            raise ValueError("没有找到合适的目标变量")

        # 排除非特征列
        excluded_cols = ['label', target_col, 'Year', 'County', 'City', 'county_name', 'year', 'month']
        non_feature_cols = ['Latitude', 'latitude', 'Longitude', 'longitude', 'Period', 'Season', 'Moth_Activity_Level']

        for col in non_feature_cols:
            if col in df.columns:
                excluded_cols.append(col)

        # 选择特征列
        feature_cols = [col for col in df.columns if col not in excluded_cols]
        print(f"使用特征数量: {len(feature_cols)}")
        print(f"特征列: {feature_cols[:10]}...")  # 显示前10个特征

        # 提取特征和标签
        X = df[feature_cols].values
        y = df['label'].values

        # 处理缺失值和异常值
        if np.isnan(X).any() or np.isinf(X).any():
            print("处理缺失值和异常值...")
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        if np.isnan(y).any() or np.isinf(y).any():
            print("处理标签异常值...")
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)

        print(f"特征矩阵形状: {X.shape}")
        print(f"标签分布: {np.bincount(y.astype(int))}")

        return X, y, feature_cols

    def preprocess_real_occurrence_data(self, df):
        """预处理真实发生数据"""
        print("\n预处理真实发生数据...")

        # 确定目标变量
        target_col = 'Has_Occurrence'
        if target_col not in df.columns:
            raise ValueError(f"目标列 '{target_col}' 未找到")

        df['label'] = df[target_col].astype(int)

        # 排除非特征列
        excluded_cols = ['label', target_col, 'county_name', 'year', 'month', 'Severity', 'Period', 'Season']
        non_feature_cols = ['latitude', 'longitude', 'Moth_Activity_Level']

        for col in non_feature_cols:
            if col in df.columns:
                excluded_cols.append(col)

        # 选择特征列 - 排除分类变量
        feature_cols = []
        for col in df.columns:
            if col not in excluded_cols:
                # 只选择数值型特征
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    feature_cols.append(col)

        print(f"使用特征数量: {len(feature_cols)}")
        print(f"特征列: {feature_cols[:10]}...")  # 显示前10个特征

        # 提取特征和标签
        X = df[feature_cols].values
        y = df['label'].values

        # 处理缺失值和异常值
        if np.isnan(X).any() or np.isinf(X).any():
            print("处理缺失值和异常值...")
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        if np.isnan(y).any() or np.isinf(y).any():
            print("处理标签异常值...")
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)

        print(f"特征矩阵形状: {X.shape}")
        print(f"标签分布: {np.bincount(y.astype(int))}")

        return X, y, feature_cols

    def create_data_loaders(self, X, y, test_size=0.2, val_size=0.2, batch_size=32, random_state=42):
        """创建数据加载器"""
        print(f"\n创建数据加载器...")
        print(f"测试集比例: {test_size}, 验证集比例: {val_size}")

        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        # 数据分割
        # 首先分离测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 然后分离训练集和验证集
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )

        print(f"训练集: {X_train.shape}, 标签分布: {np.bincount(y_train.astype(int))}")
        print(f"验证集: {X_val.shape}, 标签分布: {np.bincount(y_val.astype(int))}")
        print(f"测试集: {X_test.shape}, 标签分布: {np.bincount(y_test.astype(int))}")

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

        return train_loader, val_loader, test_loader, (X_train, y_train), (X_val, y_val), (X_test, y_test)

class SF_BiLSTM_Trainer:
    """SF-BiLSTM模型训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 创建输出目录
        self.output_dir = config.get('output_dir', 'results/sf_bilstm_real_data')
        os.makedirs(self.output_dir, exist_ok=True)

        # 设置随机种子
        self.set_seed(config.get('random_seed', 42))

    def set_seed(self, seed=42):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def create_model(self, model_class, input_size):
        """创建模型"""
        model_config = {
            "input_size": input_size,
            "hidden_size": self.config.get('hidden_size', 128),
            "num_layers": self.config.get('num_layers', 2),
            "dropout": self.config.get('dropout', 0.3),
            "num_classes": 1
        }

        model = model_class(model_config).to(self.device)
        return model, model_config

    def train_model(self, model, train_loader, val_loader, model_name, class_weights=None):
        """训练单个模型"""
        print(f"\n开始训练 {model_name}...")

        # 设置优化器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )

        # 设置学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )

        # 设置损失函数
        if self.config.get('use_focal_loss', False):
            criterion = self.FocalLoss(alpha=0.75, gamma=2.0)
            print("使用Focal Loss")
        else:
            criterion = nn.BCEWithLogitsLoss()
            print("使用BCEWithLogitsLoss")

        # 训练历史
        history = {
            'train_losses': [],
            'val_losses': [],
            'val_accuracies': [],
            'val_aucs': [],
            'training_time': 0
        }

        best_val_auc = -np.inf
        best_model_state = None
        start_time = time.time()

        for epoch in range(self.config.get('epochs', 100)):
            # 训练阶段
            model.train()
            train_loss = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

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
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)

                    val_loss += criterion(output, target).item()

                    pred_probs = torch.sigmoid(output)
                    all_val_preds.extend(pred_probs.cpu().numpy())
                    all_val_targets.extend(target.cpu().numpy())

            val_loss /= len(val_loader)

            # 计算验证指标
            val_preds_np = np.array(all_val_preds).flatten()
            val_targets_np = np.array(all_val_targets).flatten()

            val_accuracy = accuracy_score(val_targets_np, (val_preds_np > 0.5).astype(int))
            try:
                val_auc = roc_auc_score(val_targets_np, val_preds_np)
            except:
                val_auc = 0.5

            # 记录历史
            history['train_losses'].append(train_loss)
            history['val_losses'].append(val_loss)
            history['val_accuracies'].append(val_accuracy)
            history['val_aucs'].append(val_auc)

            # 学习率调度
            scheduler.step(val_auc)

            # 保存最佳模型
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()

            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{self.config.get("epochs", 100)} | '
                      f'Train Loss: {train_loss:.6f} | '
                      f'Val Loss: {val_loss:.6f} | '
                      f'Val Acc: {val_accuracy:.4f} | '
                      f'Val AUC: {val_auc:.4f}')

        # 加载最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        training_time = time.time() - start_time
        history['training_time'] = training_time

        print(f'{model_name} 训练完成，用时: {training_time:.2f}秒，最佳验证AUC: {best_val_auc:.4f}')

        return model, history

    def evaluate_model(self, model, test_loader):
        """评估模型"""
        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)

                pred_probs = torch.sigmoid(output)
                pred = (pred_probs > 0.5).float()

                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                all_probs.append(pred_probs.cpu().numpy())

        # 合并结果
        all_preds = np.vstack(all_preds).flatten()
        all_targets = np.vstack(all_targets).flatten()
        all_probs = np.vstack(all_probs).flatten()

        # 计算指标
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)

        try:
            auc = roc_auc_score(all_targets, all_probs)
        except:
            auc = 0.5

        # 计算特异度
        conf_matrix = confusion_matrix(all_targets, all_preds)
        if len(conf_matrix) > 1:
            tn, fp, fn, tp = conf_matrix.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            specificity = 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'specificity': specificity,
            'confusion_matrix': conf_matrix.tolist()
        }

    class FocalLoss(nn.Module):
        """Focal Loss for handling class imbalance"""
        def __init__(self, alpha=0.75, gamma=2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-BCE_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
            return focal_loss.mean()

    def run_ablation_study(self, train_loader, val_loader, test_loader, input_size):
        """运行消融实验"""
        print("\n" + "="*60)
        print("开始SF-BiLSTM消融实验")
        print("="*60)

        # 模型变体
        model_variants = {
            "完整SF-BiLSTM": BiLSTMComplete,
            "无注意力机制": BiLSTMNoAttention,
            "无残差连接": BiLSTMNoResidual,
            "无概率校准": BiLSTMNoCalibration,
            "无混合专家": BiLSTMNoExperts,
            "单向LSTM": UnidirectionalLSTM
        }

        all_results = {}
        all_histories = {}

        for model_name, model_class in model_variants.items():
            print(f"\n{'-'*50}")
            print(f"训练模型: {model_name}")

            # 创建模型
            model, model_config = self.create_model(model_class, input_size)

            # 训练模型
            model, history = self.train_model(
                model, train_loader, val_loader, model_name
            )

            # 评估模型
            results = self.evaluate_model(model, test_loader)

            print(f"{model_name} 测试结果:")
            for metric, value in results.items():
                if metric != 'confusion_matrix':
                    print(f"  {metric}: {value:.4f}")

            all_results[model_name] = results
            all_histories[model_name] = history

        # 保存结果
        results_path = os.path.join(self.output_dir, 'ablation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

        histories_path = os.path.join(self.output_dir, 'training_histories.json')
        with open(histories_path, 'w', encoding='utf-8') as f:
            json.dump(all_histories, f, indent=4)

        # 生成可视化
        self.plot_results(all_results, all_histories)

        return all_results, all_histories

    def plot_results(self, results, histories):
        """生成结果可视化"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        # 1. 性能对比柱状图
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity']
        model_names = list(results.keys())

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in model_names]

            bars = axes[i].bar(model_names, values, color='skyblue', alpha=0.7)
            axes[i].set_title(f'{metric.upper()}', fontsize=14, fontweight='bold')
            axes[i].set_ylabel('分数', fontsize=12)
            axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis='x', rotation=45)

            # 添加数值标签
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 训练历史曲线
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 训练损失
        for model_name, history in histories.items():
            axes[0, 0].plot(history['train_losses'], label=model_name, linewidth=2)
        axes[0, 0].set_title('训练损失', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('轮次')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 验证损失
        for model_name, history in histories.items():
            axes[0, 1].plot(history['val_losses'], label=model_name, linewidth=2)
        axes[0, 1].set_title('验证损失', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('轮次')
        axes[0, 1].set_ylabel('损失')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 验证准确率
        for model_name, history in histories.items():
            axes[1, 0].plot(history['val_accuracies'], label=model_name, linewidth=2)
        axes[1, 0].set_title('验证准确率', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('轮次')
        axes[1, 0].set_ylabel('准确率')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 验证AUC
        for model_name, history in histories.items():
            axes[1, 1].plot(history['val_aucs'], label=model_name, linewidth=2)
        axes[1, 1].set_title('验证AUC', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('轮次')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 性能热力图
        metrics_data = []
        for model_name in model_names:
            for metric in metrics:
                metrics_data.append({
                    'Model': model_name,
                    'Metric': metric.upper(),
                    'Score': results[model_name][metric]
                })

        df_metrics = pd.DataFrame(metrics_data)
        pivot_df = df_metrics.pivot(index='Model', columns='Metric', values='Score')

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.3f',
                   cbar_kws={'label': '分数'})
        plt.title('SF-BiLSTM消融实验性能热力图', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('评估指标', fontsize=14)
        plt.ylabel('模型变体', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_heatmap.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n可视化结果已保存到: {self.output_dir}")

def main():
    """主函数"""
    print("SF-BiLSTM真实数据训练和消融实验")
    print("="*60)

    # 配置
    config = {
        'output_dir': 'results/sf_bilstm_real_data',
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'batch_size': 32,
        'test_size': 0.2,
        'val_size': 0.2,
        'use_focal_loss': True,
        'random_seed': 42
    }

    # 创建训练器
    trainer = SF_BiLSTM_Trainer(config)

    # 测试两种数据集 - 优先使用真实发生数据
    datasets = [
        {
            'name': '真实发生数据',
            'type': 'real_occurrence',
            'path': 'datas/shandong_pest_data/real_occurrence_complete_data.csv'
        },
        {
            'name': '县域级别数据',
            'type': 'county_level',
            'path': 'datas/shandong_pest_data/county_level_firstgen_complete.csv'
        }
    ]

    for dataset_info in datasets:
        print(f"\n{'='*80}")
        print(f"处理数据集: {dataset_info['name']}")
        print(f"{'='*80}")

        # 检查数据文件是否存在
        if not os.path.exists(dataset_info['path']):
            print(f"数据文件不存在: {dataset_info['path']}")
            continue

        # 创建数据处理器
        processor = RealDataProcessor(dataset_info['type'])

        # 加载数据
        df = processor.load_county_level_data(dataset_info['path']) if dataset_info['type'] == 'county_level' else processor.load_real_occurrence_data(dataset_info['path'])

        # 预处理数据
        if dataset_info['type'] == 'county_level':
            X, y, feature_cols = processor.preprocess_county_level_data(df)
        else:
            X, y, feature_cols = processor.preprocess_real_occurrence_data(df)

        # 创建数据加载器
        train_loader, val_loader, test_loader, train_data, val_data, test_data = processor.create_data_loaders(
            X, y,
            test_size=config['test_size'],
            val_size=config['val_size'],
            batch_size=config['batch_size']
        )

        # 更新输出目录
        dataset_output_dir = os.path.join(config['output_dir'], dataset_info['type'])
        os.makedirs(dataset_output_dir, exist_ok=True)
        trainer.output_dir = dataset_output_dir

        # 保存数据信息
        data_info = {
            'dataset_name': dataset_info['name'],
            'data_type': dataset_info['type'],
            'total_samples': len(X),
            'feature_count': len(feature_cols),
            'feature_names': feature_cols,
            'train_samples': len(train_data[0]),
            'val_samples': len(val_data[0]),
            'test_samples': len(test_data[0]),
            'label_distribution': {
                'total': np.bincount(y.astype(int)).tolist(),
                'train': np.bincount(train_data[1].astype(int)).tolist(),
                'val': np.bincount(val_data[1].astype(int)).tolist(),
                'test': np.bincount(test_data[1].astype(int)).tolist()
            }
        }

        with open(os.path.join(dataset_output_dir, 'data_info.json'), 'w', encoding='utf-8') as f:
            json.dump(data_info, f, ensure_ascii=False, indent=4)

        # 运行消融实验
        results, histories = trainer.run_ablation_study(
            train_loader, val_loader, test_loader, len(feature_cols)
        )

        # 打印最终结果
        print(f"\n{'='*60}")
        print(f"{dataset_info['name']} - 最终性能排名 (按F1分数)")
        print(f"{'='*60}")

        sorted_models = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
        for i, (model_name, metrics) in enumerate(sorted_models, 1):
            print(f"{i}. {model_name}")
            print(f"   F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}, "
                  f"Acc: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, "
                  f"Recall: {metrics['recall']:.4f}")

        # 保存最终结果
        final_results = {
            'dataset_info': data_info,
            'model_performance': results,
            'ranking': {f"rank_{i}": model_name for i, (model_name, _) in enumerate(sorted_models, 1)},
            'best_model': sorted_models[0][0],
            'best_performance': sorted_models[0][1]
        }

        with open(os.path.join(dataset_output_dir, 'final_results.json'), 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)

        print(f"\n{dataset_info['name']} 实验完成!")
        print(f"结果保存到: {dataset_output_dir}")

if __name__ == "__main__":
    main()