#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分类策略对比实验
测试不同分类方法的性能：
1. 原始4分类（0,1,2,3级）
2. 实际2分类（1 vs 2级，因为数据中只有1,2,3级且3级极少）
3. 重新定义的2分类（轻度vs中度：1级合并为轻度，2-3级合并为中度）
4. 数据均衡优化
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from county_level_config import CountyLevelConfig
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ComparisonDataset(Dataset):
    """用于对比实验的数据集"""

    def __init__(self, data, classification_type='binary', sequence_length=2):
        self.data = data
        self.classification_type = classification_type
        self.sequence_length = sequence_length

        # 创建样本
        self.samples = self._create_samples()

        # 标准化特征
        if len(self.samples) > 0:
            all_features = np.concatenate([sample['features'] for sample in self.samples])
            self.scaler = StandardScaler()
            self.scaler.fit(all_features)

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
                    original_target = sequence_data.iloc[-1]['Severity_Level']

                    # 根据分类类型转换目标
                    if self.classification_type == 'binary_12':
                        # 二分类：1级 vs 2级
                        target = 0 if original_target == 1 else 1
                    elif self.classification_type == 'binary_mild_severe':
                        # 二分类：轻度(1级) vs 中重度(2-3级)
                        target = 0 if original_target == 1 else 1
                    elif self.classification_type == 'multiclass_3':
                        # 三分类：1级, 2级, 3级
                        target = original_target - 1  # 转换为0,1,2
                    else:  # original 4-class
                        target = original_target  # 保持0,1,2,3

                    samples.append({
                        'county': county,
                        'year': sequence_data.iloc[-1]['Year'],
                        'features': features,
                        'target': target,
                        'original_target': original_target
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features_scaled = self.scaler.transform(sample['features'])

        return {
            'county': sample['county'],
            'year': int(sample['year']),
            'sequence': torch.FloatTensor(features_scaled),
            'target': torch.LongTensor([sample['target']]),
            'original_target': torch.LongTensor([sample['original_target']])
        }

class SimpleClassifier(nn.Module):
    """简单分类器模型"""

    def __init__(self, input_size, hidden_size=32, num_classes=2, dropout=0.3):
        super(SimpleClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        output = self.classifier(last_output)

        return output

class ClassificationComparison:
    """分类策略对比实验"""

    def __init__(self):
        self.config = CountyLevelConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载数据
        self.load_data()

    def load_data(self):
        """加载数据"""
        print("Loading data for comparison...")
        data = pd.read_csv(self.config.COMPLETE_DATA_PATH)

        print(f"原始数据: {len(data)} 样本")
        print("发病程度分布:")
        severity_dist = data['Severity_Level'].value_counts().sort_index()
        for level, count in severity_dist.items():
            print(f"  {level}级: {count} 样本 ({count/len(data)*100:.1f}%)")

        # 分割训练和测试数据
        self.train_data = data[data['Year'] <= 2020]
        self.test_data = data[data['Year'] > 2020]

        print(f"\n训练数据: {len(self.train_data)} 样本")
        print(f"测试数据: {len(self.test_data)} 样本")

    def train_and_evaluate(self, classification_type, num_epochs=30):
        """训练和评估特定分类类型"""
        print(f"\n=== {classification_type} 分类实验 ===")

        # 创建数据集
        train_dataset = ComparisonDataset(self.train_data, classification_type)
        test_dataset = ComparisonDataset(self.test_data, classification_type)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 确定类别数
        if classification_type == 'binary_12' or classification_type == 'binary_mild_severe':
            num_classes = 2
            class_names = ['类别0', '类别1']
        elif classification_type == 'multiclass_3':
            num_classes = 3
            class_names = ['1级(轻度)', '2级(中度)', '3级(重度)']
        else:  # original
            num_classes = 4
            class_names = ['0级(健康)', '1级(轻度)', '2级(中度)', '3级(重度)']

        print(f"训练样本: {len(train_dataset)}, 测试样本: {len(test_dataset)}")
        print(f"类别数: {num_classes}")

        # 初始化模型
        model = SimpleClassifier(
            input_size=len(self.config.ALL_FEATURES),
            hidden_size=32,
            num_classes=num_classes,
            dropout=0.3
        ).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # 训练
        model.train()
        train_losses = []

        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in train_loader:
                sequences = batch['sequence'].to(self.device)
                targets = batch['target'].squeeze(-1).to(self.device)

                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # 测试
        model.eval()
        all_predictions = []
        all_targets = []
        all_original_targets = []

        with torch.no_grad():
            for batch in test_loader:
                sequences = batch['sequence'].to(self.device)
                targets = batch['target'].squeeze(-1).to(self.device)
                original_targets = batch['original_target'].squeeze(-1)

                outputs = model(sequences)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_original_targets.extend(original_targets.cpu().numpy())

        # 计算指标
        accuracy = accuracy_score(all_targets, all_predictions)
        f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
        f1_macro = f1_score(all_targets, all_predictions, average='macro')

        print(f"测试准确率: {accuracy:.4f}")
        print(f"F1-加权: {f1_weighted:.4f}")
        print(f"F1-宏平均: {f1_macro:.4f}")

        # 详细报告
        unique_labels = sorted(list(set(all_targets + all_predictions)))
        valid_class_names = [class_names[i] for i in unique_labels if i < len(class_names)]

        print("\n分类报告:")
        print(classification_report(all_targets, all_predictions,
                                  labels=unique_labels,
                                  target_names=valid_class_names))

        # 返回结果
        return {
            'classification_type': classification_type,
            'num_classes': num_classes,
            'class_names': class_names,
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset),
            'predictions': all_predictions,
            'targets': all_targets,
            'original_targets': all_original_targets,
            'train_losses': train_losses
        }

    def run_all_comparisons(self):
        """运行所有分类策略对比"""
        print("=== 开始分类策略对比实验 ===")

        results = []

        # 1. 原始4分类（实际上只有1,2,3级）
        print("\n" + "="*60)
        result1 = self.train_and_evaluate('original_4class')
        results.append(result1)

        # 2. 二分类：1级 vs 2级
        print("\n" + "="*60)
        result2 = self.train_and_evaluate('binary_12')
        results.append(result2)

        # 3. 二分类：轻度 vs 中重度
        print("\n" + "="*60)
        result3 = self.train_and_evaluate('binary_mild_severe')
        results.append(result3)

        # 4. 三分类：1,2,3级
        print("\n" + "="*60)
        result4 = self.train_and_evaluate('multiclass_3')
        results.append(result4)

        # 创建对比可视化
        self.create_comparison_visualizations(results)

        # 生成对比报告
        self.generate_comparison_report(results)

        return results

    def create_comparison_visualizations(self, results):
        """创建对比可视化"""
        print("\n创建对比可视化图表...")

        os.makedirs('results/classification_comparison', exist_ok=True)

        # 1. 性能对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        strategy_names = [r['classification_type'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        f1_weighted = [r['f1_weighted'] for r in results]
        f1_macro = [r['f1_macro'] for r in results]
        num_classes = [r['num_classes'] for r in results]

        # 准确率对比
        bars1 = axes[0, 0].bar(strategy_names, accuracies, color=['skyblue', 'lightgreen', 'salmon', 'orange'])
        axes[0, 0].set_title('不同分类策略的准确率对比')
        axes[0, 0].set_ylabel('准确率')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=15)
        for i, bar in enumerate(bars1):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                           f'{accuracies[i]:.3f}', ha='center', va='bottom')

        # F1分数对比
        x = np.arange(len(strategy_names))
        width = 0.35

        bars2a = axes[0, 1].bar(x - width/2, f1_weighted, width, label='F1-Weighted', color='lightblue')
        bars2b = axes[0, 1].bar(x + width/2, f1_macro, width, label='F1-Macro', color='lightcoral')

        axes[0, 1].set_title('F1分数对比')
        axes[0, 1].set_ylabel('F1分数')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(strategy_names, rotation=15)
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1)

        # 添加数值标签
        for bars in [bars2a, bars2b]:
            for bar in bars:
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        # 类别数 vs 性能散点图
        axes[1, 0].scatter(num_classes, accuracies, s=100, c='red', alpha=0.7, label='准确率')
        axes[1, 0].scatter(num_classes, f1_weighted, s=100, c='blue', alpha=0.7, label='F1-Weighted')
        axes[1, 0].set_xlabel('类别数')
        axes[1, 0].set_ylabel('性能指标')
        axes[1, 0].set_title('类别数 vs 性能关系')
        axes[1, 0].legend()
        axes[1, 0].set_xticks([2, 3, 4])
        axes[1, 0].grid(True, alpha=0.3)

        # 训练损失曲线对比
        for i, result in enumerate(results):
            axes[1, 1].plot(result['train_losses'], label=strategy_names[i], linewidth=2)

        axes[1, 1].set_title('训练损失曲线对比')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('损失')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/classification_comparison/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 混淆矩阵对比
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        for i, result in enumerate(results):
            row, col = i // 2, i % 2
            if row >= 2 or col >= 2:
                break

            cm = confusion_matrix(result['targets'], result['predictions'])

            # 使用动态标签
            class_names = result['class_names']
            if len(class_names) > len(cm):
                class_names = class_names[:len(cm)]

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[row, col])
            axes[row, col].set_title(f'{result["classification_type"]} 混淆矩阵')
            axes[row, col].set_xlabel('预测')
            axes[row, col].set_ylabel('实际')

        plt.tight_layout()
        plt.savefig('results/classification_comparison/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("可视化图表保存到 results/classification_comparison/")

    def generate_comparison_report(self, results):
        """生成对比报告"""
        print("\n生成对比报告...")

        report = {
            'comparison_date': pd.Timestamp.now().isoformat(),
            'dataset_info': {
                'total_samples': len(self.train_data) + len(self.test_data),
                'train_samples': len(self.train_data),
                'test_samples': len(self.test_data),
                'features': len(self.config.ALL_FEATURES)
            },
            'strategies': {}
        }

        for result in results:
            report['strategies'][result['classification_type']] = {
                'num_classes': result['num_classes'],
                'accuracy': float(result['accuracy']),
                'f1_weighted': float(result['f1_weighted']),
                'f1_macro': float(result['f1_macro']),
                'train_samples': result['train_samples'],
                'test_samples': result['test_samples']
            }

        # 找出最佳策略
        best_accuracy_idx = np.argmax([r['accuracy'] for r in results])
        best_f1_idx = np.argmax([r['f1_weighted'] for r in results])

        report['best_strategies'] = {
            'best_accuracy': {
                'strategy': results[best_accuracy_idx]['classification_type'],
                'score': float(results[best_accuracy_idx]['accuracy']),
                'num_classes': results[best_accuracy_idx]['num_classes']
            },
            'best_f1': {
                'strategy': results[best_f1_idx]['classification_type'],
                'score': float(results[best_f1_idx]['f1_weighted']),
                'num_classes': results[best_f1_idx]['num_classes']
            }
        }

        # 保存报告
        with open('results/classification_comparison/comparison_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 打印摘要
        self.print_comparison_summary(results, report)

        print(f"\n对比报告保存到 results/classification_comparison/comparison_report.json")

    def print_comparison_summary(self, results, report):
        """打印对比摘要"""
        print("\n" + "="*80)
        print("分类策略对比实验摘要")
        print("="*80)

        print(f"\n数据集信息:")
        print(f"  总样本数: {report['dataset_info']['total_samples']}")
        print(f"  训练样本: {report['dataset_info']['train_samples']}")
        print(f"  测试样本: {report['dataset_info']['test_samples']}")

        print(f"\n各策略性能对比:")
        print(f"{'策略':<25} {'类别数':<8} {'准确率':<8} {'F1-加权':<8} {'F1-宏平均':<10}")
        print("-" * 70)

        for result in results:
            print(f"{result['classification_type']:<25} {result['num_classes']:<8} "
                  f"{result['accuracy']:<8.4f} {result['f1_weighted']:<8.4f} {result['f1_macro']:<10.4f}")

        best_acc = report['best_strategies']['best_accuracy']
        best_f1 = report['best_strategies']['best_f1']

        print(f"\n最佳策略:")
        print(f"  最高准确率: {best_acc['strategy']} ({best_acc['score']:.4f}, {best_acc['num_classes']}类)")
        print(f"  最高F1分数: {best_f1['strategy']} ({best_f1['score']:.4f}, {best_f1['num_classes']}类)")

        print("\n" + "="*80)

def main():
    """主函数"""
    comparator = ClassificationComparison()
    results = comparator.run_all_comparisons()
    return results

if __name__ == "__main__":
    import os
    import json
    results = main()