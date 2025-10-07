#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于训练权重的改进深度学习模型测试评估
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from improved_deep_learning_models import (
    ImprovedBiLSTMModel, ImprovedGCNModel, AdaptiveFusionModel,
    ImprovedPestDataset, CrossCountySpatialDataset
)
from county_level_config import CountyLevelConfig
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ImprovedModelTester:
    """改进的深度学习模型测试器"""

    def __init__(self):
        self.config = CountyLevelConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载数据
        self.load_data()

        # 准备测试数据集
        self.prepare_test_datasets()

    def load_data(self):
        """加载完整数据"""
        print("Loading data for testing...")
        self.data = pd.read_csv(self.config.COMPLETE_DATA_PATH)

        # 数据分割
        train_data = self.data[self.data['Year'] <= 2020]
        test_data = self.data[self.data['Year'] > 2020]

        print(f"训练数据: {len(train_data)} 样本")
        print(f"测试数据: {len(test_data)} 样本")

        self.train_data = train_data
        self.test_data = test_data

    def prepare_test_datasets(self):
        """准备测试数据集"""
        print("Preparing test datasets...")

        # 1. 时间序列测试数据集
        self.test_time_dataset = ImprovedPestDataset(
            self.test_data,
            feature_cols=self.config.ALL_FEATURES,
            target_col='Severity_Level',
            sequence_length=2,
            augment=False,  # 测试时不进行数据增强
            cross_county_augment=False
        )

        # 2. 空间测试数据集
        self.test_spatial_dataset = CrossCountySpatialDataset(
            self.test_data,
            feature_cols=self.config.ALL_FEATURES,
            target_col='Severity_Level'
        )

        # 3. 创建测试图数据
        test_years = sorted(self.test_data['Year'].unique())
        self.test_graphs = []

        for year in test_years:
            graph_data = self.test_spatial_dataset.create_enhanced_graph_data(year)
            if graph_data is not None:
                self.test_graphs.append(graph_data)

        print(f"时间序列测试样本: {len(self.test_time_dataset)}")
        print(f"空间图测试数据: {len(self.test_graphs)} 年份")

        # 创建数据加载器
        self.test_time_loader = DataLoader(
            self.test_time_dataset,
            batch_size=32,
            shuffle=False
        )

    def load_improved_bilstm_model(self):
        """加载改进的BiLSTM模型"""
        print("Loading Improved BiLSTM model...")

        model = ImprovedBiLSTMModel(
            input_size=len(self.config.ALL_FEATURES),
            hidden_size=32,
            num_layers=1,
            num_classes=3,
            dropout=0.3
        ).to(self.device)

        # 加载权重
        model_path = 'models/county_level/improved_bilstm_best.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"√ Improved BiLSTM模型权重加载成功: {model_path}")
        else:
            print(f"× 模型文件不存在: {model_path}")
            return None

        model.eval()
        return model

    def load_improved_gcn_model(self):
        """加载改进的GCN模型"""
        print("Loading Improved GCN model...")

        model = ImprovedGCNModel(
            input_size=len(self.config.ALL_FEATURES),
            hidden_size=32,
            num_classes=3,
            num_layers=2,
            dropout=0.3
        ).to(self.device)

        # 加载权重
        model_path = 'models/county_level/improved_gcn_best.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"√ Improved GCN模型权重加载成功: {model_path}")
        else:
            print(f"× 模型文件不存在: {model_path}")
            return None

        model.eval()
        return model

    def load_adaptive_fusion_model(self):
        """加载自适应融合模型"""
        print("Loading Adaptive Fusion model...")

        model = AdaptiveFusionModel(
            input_size=len(self.config.ALL_FEATURES),
            hidden_size=32,
            num_classes=3,
            dropout=0.3
        ).to(self.device)

        # 加载权重
        model_path = 'models/county_level/adaptive_fusion_best.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"√ Adaptive Fusion模型权重加载成功: {model_path}")
        else:
            print(f"× 模型文件不存在: {model_path}")
            return None

        model.eval()
        return model

    def test_bilstm_model(self, model):
        """测试BiLSTM模型"""
        print("\n=== Testing Improved BiLSTM Model ===")

        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for batch in self.test_time_loader:
                sequences = batch['sequence'].to(self.device)
                targets = batch['target'].squeeze(-1).to(self.device)

                outputs = model(sequences)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # 转换回原始标签 (0,1,2 -> 1,2,3)
        all_predictions = np.array(all_predictions) + 1
        all_targets = np.array(all_targets) + 1
        all_probabilities = np.array(all_probabilities)

        # 计算指标
        accuracy = accuracy_score(all_targets, all_predictions)
        f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
        f1_macro = f1_score(all_targets, all_predictions, average='macro')

        print(f"BiLSTM Test Accuracy: {accuracy:.4f}")
        print(f"BiLSTM Test F1-Weighted: {f1_weighted:.4f}")
        print(f"BiLSTM Test F1-Macro: {f1_macro:.4f}")

        # 详细分类报告
        print("\nBiLSTM Classification Report:")
        unique_labels = np.unique(np.concatenate([all_targets, all_predictions]))
        target_names = ['1级(轻度)', '2级(中度)', '3级(重度)']
        valid_target_names = [target_names[i-1] for i in unique_labels]  # 标签从1开始

        print(classification_report(all_targets, all_predictions,
                                  labels=unique_labels,
                                  target_names=valid_target_names))

        return {
            'model_name': 'Improved BiLSTM',
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities,
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'classification_report': classification_report(all_targets, all_predictions,
                                                          target_names=['1级(轻度)', '2级(中度)', '3级(重度)'],
                                                          output_dict=True)
        }

    def test_gcn_model(self, model):
        """测试GCN模型"""
        print("\n=== Testing Improved GCN Model ===")

        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_counties = []

        with torch.no_grad():
            for graph_data in self.test_graphs:
                graph_data = graph_data.to(self.device)

                outputs = model(graph_data)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(graph_data.y.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_counties.extend(graph_data.county_names)

        # 转换回原始标签
        all_predictions = np.array(all_predictions) + 1
        all_targets = np.array(all_targets) + 1
        all_probabilities = np.array(all_probabilities)

        # 计算指标
        accuracy = accuracy_score(all_targets, all_predictions)
        f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
        f1_macro = f1_score(all_targets, all_predictions, average='macro')

        print(f"GCN Test Accuracy: {accuracy:.4f}")
        print(f"GCN Test F1-Weighted: {f1_weighted:.4f}")
        print(f"GCN Test F1-Macro: {f1_macro:.4f}")

        # 详细分类报告
        print("\nGCN Classification Report:")
        unique_labels = np.unique(np.concatenate([all_targets, all_predictions]))
        target_names = ['1级(轻度)', '2级(中度)', '3级(重度)']
        valid_target_names = [target_names[i-1] for i in unique_labels]

        print(classification_report(all_targets, all_predictions,
                                  labels=unique_labels,
                                  target_names=valid_target_names))

        return {
            'model_name': 'Improved GCN',
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities,
            'counties': all_counties,
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'classification_report': classification_report(all_targets, all_predictions,
                                                          target_names=['1级(轻度)', '2级(中度)', '3级(重度)'],
                                                          output_dict=True)
        }

    def test_fusion_model(self, model):
        """测试融合模型"""
        print("\n=== Testing Adaptive Fusion Model ===")

        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for graph_data in self.test_graphs:
                graph_data = graph_data.to(self.device)

                # 获取时间序列数据（使用第一个batch）
                time_batch = next(iter(self.test_time_loader))
                sequences = time_batch['sequence'].to(self.device)

                outputs = model(sequences, graph_data)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(graph_data.y.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # 转换回原始标签
        all_predictions = np.array(all_predictions) + 1
        all_targets = np.array(all_targets) + 1
        all_probabilities = np.array(all_probabilities)

        # 计算指标
        accuracy = accuracy_score(all_targets, all_predictions)
        f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
        f1_macro = f1_score(all_targets, all_predictions, average='macro')

        print(f"Fusion Test Accuracy: {accuracy:.4f}")
        print(f"Fusion Test F1-Weighted: {f1_weighted:.4f}")
        print(f"Fusion Test F1-Macro: {f1_macro:.4f}")

        # 详细分类报告
        print("\nFusion Classification Report:")
        unique_labels = np.unique(np.concatenate([all_targets, all_predictions]))
        target_names = ['1级(轻度)', '2级(中度)', '3级(重度)']
        valid_target_names = [target_names[i-1] for i in unique_labels]

        print(classification_report(all_targets, all_predictions,
                                  labels=unique_labels,
                                  target_names=valid_target_names))

        return {
            'model_name': 'Adaptive Fusion',
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities,
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'classification_report': classification_report(all_targets, all_predictions,
                                                          target_names=['1级(轻度)', '2级(中度)', '3级(重度)'],
                                                          output_dict=True)
        }

    def create_visualizations(self, results):
        """创建可视化图表"""
        print("\nCreating visualizations...")

        os.makedirs('results/improved_model_test', exist_ok=True)

        # 1. 性能对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        model_names = [result['model_name'] for result in results]
        accuracies = [result['accuracy'] for result in results]
        f1_weighted = [result['f1_weighted'] for result in results]
        f1_macro = [result['f1_macro'] for result in results]

        # 准确率对比
        bars1 = axes[0, 0].bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        for i, bar in enumerate(bars1):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                           f'{accuracies[i]:.3f}', ha='center', va='bottom')

        # F1-Weighted对比
        bars2 = axes[0, 1].bar(model_names, f1_weighted, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0, 1].set_title('Test F1-Weighted Comparison')
        axes[0, 1].set_ylabel('F1-Weighted')
        axes[0, 1].set_ylim(0, 1)
        for i, bar in enumerate(bars2):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                           f'{f1_weighted[i]:.3f}', ha='center', va='bottom')

        # F1-Macro对比
        bars3 = axes[1, 0].bar(model_names, f1_macro, color=['skyblue', 'lightgreen', 'salmon'])
        axes[1, 0].set_title('Test F1-Macro Comparison')
        axes[1, 0].set_ylabel('F1-Macro')
        axes[1, 0].set_ylim(0, 1)
        for i, bar in enumerate(bars3):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                           f'{f1_macro[i]:.3f}', ha='center', va='bottom')

        # 综合性能雷达图
        categories = ['Accuracy', 'F1-Weighted', 'F1-Macro']

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        ax_radar = plt.subplot(2, 2, 4, projection='polar')

        colors = ['skyblue', 'lightgreen', 'salmon']
        for i, result in enumerate(results):
            values = [result['accuracy'], result['f1_weighted'], result['f1_macro']]
            values += values[:1]  # 闭合图形

            ax_radar.plot(angles, values, 'o-', linewidth=2, label=result['model_name'], color=colors[i])
            ax_radar.fill(angles, values, alpha=0.25, color=colors[i])

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Model Performance Radar Chart')
        ax_radar.legend()

        plt.tight_layout()
        plt.savefig('results/improved_model_test/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 混淆矩阵
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        class_names = ['1级(轻度)', '2级(中度)', '3级(重度)']

        for i, result in enumerate(results):
            cm = confusion_matrix(result['targets'], result['predictions'])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[i])
            axes[i].set_title(f'{result["model_name"]} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')

        plt.tight_layout()
        plt.savefig('results/improved_model_test/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Visualizations saved to results/improved_model_test/")

    def generate_test_report(self, results):
        """生成测试报告"""
        print("\nGenerating test report...")

        report = {
            'test_date': pd.Timestamp.now().isoformat(),
            'test_dataset_info': {
                'total_samples': len(self.test_data),
                'test_years': sorted(self.test_data['Year'].unique()),
                'time_series_samples': len(self.test_time_dataset),
                'spatial_graphs': len(self.test_graphs),
                'features': len(self.config.ALL_FEATURES),
                'num_classes': 3
            },
            'model_results': {}
        }

        # 添加每个模型的结果
        for result in results:
            report['model_results'][result['model_name']] = {
                'accuracy': float(result['accuracy']),
                'f1_weighted': float(result['f1_weighted']),
                'f1_macro': float(result['f1_macro']),
                'classification_report': result['classification_report']
            }

        # 找出最佳模型
        best_accuracy_idx = np.argmax([r['accuracy'] for r in results])
        best_f1_idx = np.argmax([r['f1_weighted'] for r in results])

        report['best_models'] = {
            'best_accuracy': {
                'model': results[best_accuracy_idx]['model_name'],
                'score': float(results[best_accuracy_idx]['accuracy'])
            },
            'best_f1': {
                'model': results[best_f1_idx]['model_name'],
                'score': float(results[best_f1_idx]['f1_weighted'])
            }
        }

        # 性能对比分析
        accuracies = [r['accuracy'] for r in results]
        f1_scores = [r['f1_weighted'] for r in results]

        report['performance_analysis'] = {
            'accuracy_range': {
                'min': float(np.min(accuracies)),
                'max': float(np.max(accuracies)),
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies))
            },
            'f1_range': {
                'min': float(np.min(f1_scores)),
                'max': float(np.max(f1_scores)),
                'mean': float(np.mean(f1_scores)),
                'std': float(np.std(f1_scores))
            }
        }

        # 保存报告
        with open('results/improved_model_test/test_results.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 打印摘要
        self.print_test_summary(report)

        print(f"\nTest report saved to results/improved_model_test/test_results.json")

    def print_test_summary(self, report):
        """打印测试摘要"""
        print("\n" + "="*80)
        print("IMPROVED DEEP LEARNING MODELS TEST SUMMARY")
        print("="*80)

        dataset_info = report['test_dataset_info']
        print(f"\n测试数据集信息:")
        print(f"  总样本数: {dataset_info['total_samples']}")
        print(f"  测试年份: {dataset_info['test_years']}")
        print(f"  时间序列样本: {dataset_info['time_series_samples']}")
        print(f"  空间图数据: {dataset_info['spatial_graphs']} 年份")

        print(f"\n模型性能对比:")
        for model_name, result in report['model_results'].items():
            print(f"  {model_name}:")
            print(f"    准确率: {result['accuracy']:.4f}")
            print(f"    F1-加权: {result['f1_weighted']:.4f}")
            print(f"    F1-宏平均: {result['f1_macro']:.4f}")

        best_models = report['best_models']
        print(f"\n最佳模型:")
        print(f"  最高准确率: {best_models['best_accuracy']['model']} ({best_models['best_accuracy']['score']:.4f})")
        print(f"  最高F1分数: {best_models['best_f1']['model']} ({best_models['best_f1']['score']:.4f})")

        analysis = report['performance_analysis']
        print(f"\n性能统计分析:")
        print(f"  准确率范围: {analysis['accuracy_range']['min']:.4f} - {analysis['accuracy_range']['max']:.4f}")
        print(f"  准确率均值: {analysis['accuracy_range']['mean']:.4f} ± {analysis['accuracy_range']['std']:.4f}")
        print(f"  F1分数范围: {analysis['f1_range']['min']:.4f} - {analysis['f1_range']['max']:.4f}")
        print(f"  F1分数均值: {analysis['f1_range']['mean']:.4f} ± {analysis['f1_range']['std']:.4f}")

        print("\n" + "="*80)

    def run_comprehensive_test(self):
        """运行全面的模型测试"""
        print("=== Starting Comprehensive Model Testing ===")

        results = []

        # 1. 测试改进的BiLSTM模型
        bilstm_model = self.load_improved_bilstm_model()
        if bilstm_model is not None:
            bilstm_result = self.test_bilstm_model(bilstm_model)
            results.append(bilstm_result)

        # 2. 测试改进的GCN模型
        gcn_model = self.load_improved_gcn_model()
        if gcn_model is not None:
            gcn_result = self.test_gcn_model(gcn_model)
            results.append(gcn_result)

        # 3. 测试自适应融合模型
        fusion_model = self.load_adaptive_fusion_model()
        if fusion_model is not None:
            fusion_result = self.test_fusion_model(fusion_model)
            results.append(fusion_result)

        if results:
            # 创建可视化
            self.create_visualizations(results)

            # 生成测试报告
            self.generate_test_report(results)

            print(f"\nTesting completed! {len(results)} models evaluated.")
            print("Results saved to results/improved_model_test/")
        else:
            print("No models were successfully loaded for testing.")

        return results

def main():
    """主函数"""
    tester = ImprovedModelTester()
    results = tester.run_comprehensive_test()
    return results

if __name__ == "__main__":
    results = main()