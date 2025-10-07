#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据均衡优化的分类实验
通过过采样、欠采样和加权损失等方法改善不平衡数据分类
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from county_level_config import CountyLevelConfig
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class BalancedDataset(Dataset):
    """支持数据均衡的数据集"""

    def __init__(self, data, balance_method='weighted_loss', sequence_length=2):
        self.data = data
        self.balance_method = balance_method
        self.sequence_length = sequence_length

        # 创建样本
        self.samples = self._create_samples()

        # 数据均衡处理
        if balance_method == 'oversample':
            self.samples = self._oversample_minority()
        elif balance_method == 'undersample':
            self.samples = self._undersample_majority()

        # 标准化特征
        if len(self.samples) > 0:
            all_features = np.concatenate([sample['features'] for sample in self.samples])
            self.scaler = StandardScaler()
            self.scaler.fit(all_features)

        # 计算类别权重
        self.class_weights = self._calculate_class_weights()

    def _create_samples(self):
        """创建时间序列样本"""
        samples = []
        counties = self.data['County'].unique()

        for county in counties:
            county_data = self.data[self.data['County'] == county].sort_values('Year')

            if len(county_data) >= self.sequence_length:
                for i in range(len(county_data) - self.sequence_length + 1):
                    sequence_data = county_data.iloc[i:i+self.sequence_length]

                    features = sequence_data[list(CountyLevelConfig().ALL_FEATURES)].values
                    target = 0 if sequence_data.iloc[-1]['Severity_Level'] == 1 else 1  # 二分类：1级 vs 2+级

                    samples.append({
                        'county': county,
                        'year': sequence_data.iloc[-1]['Year'],
                        'features': features,
                        'target': target
                    })

        return samples

    def _calculate_class_weights(self):
        """计算类别权重"""
        if not self.samples:
            return torch.tensor([1.0, 1.0])

        targets = [sample['target'] for sample in self.samples]
        class_counts = np.bincount(targets)
        total_samples = len(targets)

        # 反比权重
        weights = total_samples / (len(class_counts) * class_counts)
        weights = weights / weights.sum() * len(class_counts)  # 归一化

        return torch.FloatTensor(weights)

    def _oversample_minority(self):
        """过采样少数类"""
        df_samples = pd.DataFrame(self.samples)

        # 分离多数类和少数类
        majority_class = df_samples[df_samples['target'] == 0]
        minority_class = df_samples[df_samples['target'] == 1]

        if len(minority_class) == 0:
            return self.samples

        # 过采样少数类到多数类的80%
        target_minority_size = int(len(majority_class) * 0.8)
        minority_oversampled = resample(
            minority_class,
            replace=True,
            n_samples=target_minority_size,
            random_state=42
        )

        # 合并数据
        balanced_df = pd.concat([majority_class, minority_oversampled])

        return balanced_df.to_dict('records')

    def _undersample_majority(self):
        """欠采样多数类"""
        df_samples = pd.DataFrame(self.samples)

        # 分离多数类和少数类
        majority_class = df_samples[df_samples['target'] == 0]
        minority_class = df_samples[df_samples['target'] == 1]

        if len(minority_class) == 0:
            return self.samples

        # 欠采样多数类到少数类的2倍
        target_majority_size = len(minority_class) * 2
        majority_undersampled = resample(
            majority_class,
            replace=False,
            n_samples=min(target_majority_size, len(majority_class)),
            random_state=42
        )

        # 合并数据
        balanced_df = pd.concat([majority_undersampled, minority_class])

        return balanced_df.to_dict('records')

    def get_class_distribution(self):
        """获取类别分布"""
        if not self.samples:
            return {}
        targets = [sample['target'] for sample in self.samples]
        class_counts = np.bincount(targets)
        return {0: class_counts[0], 1: class_counts[1] if len(class_counts) > 1 else 0}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features_scaled = self.scaler.transform(sample['features'])

        return {
            'county': sample['county'],
            'year': int(sample['year']),
            'sequence': torch.FloatTensor(features_scaled),
            'target': torch.LongTensor([sample['target']])
        }

class ImprovedClassifier(nn.Module):
    """改进的分类器模型"""

    def __init__(self, input_size, hidden_size=32, num_classes=2, dropout=0.3):
        super(ImprovedClassifier, self).__init__()

        # 特征提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.LayerNorm(input_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # 特征提取
        batch_size, seq_len, features = x.shape
        x = x.view(-1, features)
        x = self.feature_extractor(x)
        x = x.view(batch_size, seq_len, features)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # 注意力机制
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)

        # 加权平均
        attended_features = torch.sum(lstm_out * attention_weights, dim=1)

        # 分类
        output = self.classifier(attended_features)

        return output

class BalancedClassificationExperiment:
    """数据均衡分类实验"""

    def __init__(self):
        self.config = CountyLevelConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载数据
        self.load_data()

    def load_data(self):
        """加载数据"""
        print("Loading data for balanced classification...")
        data = pd.read_csv(self.config.COMPLETE_DATA_PATH)

        # 创建二分类标签：1级=0，2级及以上=1
        data['binary_target'] = data['Severity_Level'].apply(lambda x: 0 if x == 1 else 1)

        print(f"原始数据: {len(data)} 样本")
        print("二分类分布:")
        binary_dist = data['binary_target'].value_counts().sort_index()
        for level, count in binary_dist.items():
            level_name = '1级(轻度)' if level == 0 else '2级+(中重度)'
            print(f"  {level_name}: {count} 样本 ({count/len(data)*100:.1f}%)")

        # 分割数据
        self.train_data = data[data['Year'] <= 2020]
        self.test_data = data[data['Year'] > 2020]

        print(f"\n训练数据: {len(self.train_data)} 样本")
        print(f"测试数据: {len(self.test_data)} 样本")

    def train_with_balance_strategy(self, balance_method, num_epochs=40):
        """使用特定均衡策略训练模型"""
        print(f"\n=== {balance_method} 均衡策略 ===")

        # 创建数据集
        train_dataset = BalancedDataset(self.train_data, balance_method)
        test_dataset = BalancedDataset(self.test_data, 'none')  # 测试集不均衡

        print(f"训练集分布: {train_dataset.get_class_distribution()}")
        print(f"测试集分布: {test_dataset.get_class_distribution()}")

        # 创建数据加载器
        if balance_method == 'weighted_sampler':
            # 使用加权采样器
            targets = [sample['target'] for sample in train_dataset.samples]
            class_weights = 1. / torch.bincount(torch.tensor(targets), minlength=2).float()
            sample_weights = [class_weights[t] for t in targets]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 初始化模型
        model = ImprovedClassifier(
            input_size=len(self.config.ALL_FEATURES),
            hidden_size=32,
            num_classes=2,
            dropout=0.3
        ).to(self.device)

        # 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

        # 损失函数
        if balance_method == 'weighted_loss':
            # 使用加权损失
            class_weights = train_dataset.class_weights.to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"类别权重: {class_weights}")
        else:
            criterion = nn.CrossEntropyLoss()

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.7
        )

        # 训练
        model.train()
        train_losses = []
        best_f1 = 0.0

        for epoch in range(num_epochs):
            total_loss = 0.0
            model.train()

            for batch in train_loader:
                sequences = batch['sequence'].to(self.device)
                targets = batch['target'].squeeze(-1).to(self.device)

                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)

            # 每5个epoch评估一次
            if (epoch + 1) % 5 == 0:
                # 简单验证
                model.eval()
                val_preds = []
                val_targets = []

                with torch.no_grad():
                    for batch in test_loader:
                        sequences = batch['sequence'].to(self.device)
                        targets = batch['target'].squeeze(-1).to(self.device)

                        outputs = model(sequences)
                        preds = torch.argmax(outputs, dim=1)

                        val_preds.extend(preds.cpu().numpy())
                        val_targets.extend(targets.cpu().numpy())

                val_f1 = f1_score(val_targets, val_preds, average='weighted')
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val F1: {val_f1:.4f}')

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    # 保存最佳模型
                    torch.save(model.state_dict(), f'models/balanced_{balance_method}_best.pth')

            scheduler.step(avg_loss)

        print(f'最佳验证F1: {best_f1:.4f}')

        # 最终测试
        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                sequences = batch['sequence'].to(self.device)
                targets = batch['target'].squeeze(-1).to(self.device)

                outputs = model(sequences)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # 计算指标
        accuracy = accuracy_score(all_targets, all_predictions)
        f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
        f1_macro = f1_score(all_targets, all_predictions, average='macro')

        print(f"最终测试准确率: {accuracy:.4f}")
        print(f"F1-加权: {f1_weighted:.4f}")
        print(f"F1-宏平均: {f1_macro:.4f}")

        # 分类报告
        class_names = ['1级(轻度)', '2级+(中重度)']
        print("\n分类报告:")
        print(classification_report(all_targets, all_predictions, target_names=class_names))

        return {
            'balance_method': balance_method,
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'train_distribution': train_dataset.get_class_distribution(),
            'test_distribution': test_dataset.get_class_distribution(),
            'train_losses': train_losses,
            'best_f1': best_f1
        }

    def run_all_balance_experiments(self):
        """运行所有均衡策略实验"""
        print("=== 开始数据均衡分类实验 ===")

        # 确保模型目录存在
        import os
        os.makedirs('models', exist_ok=True)
        os.makedirs('results/balanced_classification', exist_ok=True)

        results = []

        # 1. 无均衡（基线）
        result1 = self.train_with_balance_strategy('none')
        results.append(result1)

        # 2. 加权损失
        result2 = self.train_with_balance_strategy('weighted_loss')
        results.append(result2)

        # 3. 过采样
        result3 = self.train_with_balance_strategy('oversample')
        results.append(result3)

        # 4. 欠采样
        result4 = self.train_with_balance_strategy('undersample')
        results.append(result4)

        # 5. 加权采样器
        result5 = self.train_with_balance_strategy('weighted_sampler')
        results.append(result5)

        # 创建对比可视化
        self.create_balance_comparison_visualization(results)

        # 生成对比报告
        self.generate_balance_comparison_report(results)

        return results

    def create_balance_comparison_visualization(self, results):
        """创建均衡策略对比可视化"""
        print("\n创建均衡策略对比可视化...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        strategy_names = [r['balance_method'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        f1_weighted = [r['f1_weighted'] for r in results]
        f1_macro = [r['f1_macro'] for r in results]

        # 1. 性能对比
        x = np.arange(len(strategy_names))
        width = 0.25

        bars1 = axes[0, 0].bar(x - width, accuracies, width, label='准确率', color='skyblue')
        bars2 = axes[0, 0].bar(x, f1_weighted, width, label='F1-加权', color='lightgreen')
        bars3 = axes[0, 0].bar(x + width, f1_macro, width, label='F1-宏平均', color='salmon')

        axes[0, 0].set_title('不同均衡策略的性能对比')
        axes[0, 0].set_ylabel('性能指标')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(strategy_names, rotation=15)
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1)

        # 添加数值标签
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        # 2. 训练损失曲线对比
        for i, result in enumerate(results):
            axes[0, 1].plot(result['train_losses'], label=strategy_names[i], linewidth=2)

        axes[0, 1].set_title('训练损失曲线对比')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('损失')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 数据分布对比
        train_class_0 = [r['train_distribution'].get(0, 0) for r in results]
        train_class_1 = [r['train_distribution'].get(1, 0) for r in results]

        x = np.arange(len(strategy_names))
        width = 0.35

        bars4 = axes[1, 0].bar(x - width/2, train_class_0, width, label='1级(轻度)', color='lightblue')
        bars5 = axes[1, 0].bar(x + width/2, train_class_1, width, label='2级+(中重度)', color='lightcoral')

        axes[1, 0].set_title('训练集类别分布对比')
        axes[1, 0].set_ylabel('样本数')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(strategy_names, rotation=15)
        axes[1, 0].legend()

        # 4. 最佳F1分数对比
        best_f1s = [r['best_f1'] for r in results]
        colors = ['skyblue', 'lightgreen', 'salmon', 'orange', 'purple']

        bars6 = axes[1, 1].bar(strategy_names, best_f1s, color=colors)
        axes[1, 1].set_title('最佳验证F1分数对比')
        axes[1, 1].set_ylabel('F1分数')
        axes[1, 1].tick_params(axis='x', rotation=15)
        axes[1, 1].set_ylim(0, max(best_f1s) * 1.1)

        for i, bar in enumerate(bars6):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('results/balanced_classification/balance_strategy_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print("均衡策略对比可视化保存到 results/balanced_classification/")

    def generate_balance_comparison_report(self, results):
        """生成均衡策略对比报告"""
        print("\n生成均衡策略对比报告...")

        report = {
            'experiment_date': pd.Timestamp.now().isoformat(),
            'dataset_info': {
                'total_train_samples': len(self.train_data),
                'total_test_samples': len(self.test_data),
                'original_train_distribution': {
                    '1级': len(self.train_data[self.train_data['binary_target'] == 0]),
                    '2级+': len(self.train_data[self.train_data['binary_target'] == 1])
                },
                'original_test_distribution': {
                    '1级': len(self.test_data[self.test_data['binary_target'] == 0]),
                    '2级+': len(self.test_data[self.test_data['binary_target'] == 1])
                }
            },
            'balance_strategies': {}
        }

        for result in results:
            report['balance_strategies'][result['balance_method']] = {
                'accuracy': float(result['accuracy']),
                'f1_weighted': float(result['f1_weighted']),
                'f1_macro': float(result['f1_macro']),
                'best_validation_f1': float(result['best_f1']),
                'train_distribution': result['train_distribution']
            }

        # 找出最佳策略
        best_f1_idx = np.argmax([r['f1_weighted'] for r in results])
        best_acc_idx = np.argmax([r['accuracy'] for r in results])

        report['best_strategies'] = {
            'best_f1': {
                'strategy': results[best_f1_idx]['balance_method'],
                'score': float(results[best_f1_idx]['f1_weighted'])
            },
            'best_accuracy': {
                'strategy': results[best_acc_idx]['balance_method'],
                'score': float(results[best_acc_idx]['accuracy'])
            }
        }

        # 保存报告
        import json
        with open('results/balanced_classification/balance_comparison_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 打印摘要
        self.print_balance_comparison_summary(results, report)

        print(f"\n均衡策略对比报告保存到 results/balanced_classification/balance_comparison_report.json")

    def print_balance_comparison_summary(self, results, report):
        """打印均衡策略对比摘要"""
        print("\n" + "="*80)
        print("数据均衡策略对比实验摘要")
        print("="*80)

        print(f"\n原始数据分布:")
        print(f"  训练集 - 1级: {report['dataset_info']['original_train_distribution']['1级']}, "
              f"2级+: {report['dataset_info']['original_train_distribution']['2级+']}")
        print(f"  测试集 - 1级: {report['dataset_info']['original_test_distribution']['1级']}, "
              f"2级+: {report['dataset_info']['original_test_distribution']['2级+']}")

        print(f"\n均衡策略性能对比:")
        print(f"{'策略':<20} {'准确率':<8} {'F1-加权':<8} {'F1-宏平均':<8} {'最佳验证F1':<10}")
        print("-" * 65)

        for result in results:
            print(f"{result['balance_method']:<20} {result['accuracy']:<8.4f} "
                  f"{result['f1_weighted']:<8.4f} {result['f1_macro']:<8.4f} {result['best_f1']:<10.4f}")

        best_f1 = report['best_strategies']['best_f1']
        best_acc = report['best_strategies']['best_accuracy']

        print(f"\n最佳策略:")
        print(f"  最高F1分数: {best_f1['strategy']} ({best_f1['score']:.4f})")
        print(f"  最高准确率: {best_acc['strategy']} ({best_acc['score']:.4f})")

        print("\n" + "="*80)

def main():
    """主函数"""
    experimenter = BalancedClassificationExperiment()
    results = experimenter.run_all_balance_experiments()
    return results

if __name__ == "__main__":
    results = main()