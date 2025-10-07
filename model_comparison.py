#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传统机器学习模型与深度学习模型性能对比
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from county_level_config import CountyLevelConfig
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ModelComparison:
    """模型性能对比分析"""

    def __init__(self):
        self.config = CountyLevelConfig()
        self.load_results()

    def load_results(self):
        """加载所有模型的结果"""
        print("Loading model results for comparison...")

        # 加载传统机器学习模型结果
        with open('results/county_level/evaluation_results.json', 'r', encoding='utf-8') as f:
            self.ml_results = json.load(f)

        # 加载深度学习模型结果
        try:
            with open('results/deep_learning/deep_learning_results.json', 'r', encoding='utf-8') as f:
                self.dl_results = json.load(f)
        except FileNotFoundError:
            print("Deep learning results not found")
            self.dl_results = {}

        print(f"Loaded {len(self.ml_results)} ML models and {len(self.dl_results)} DL models")

    def extract_classification_results(self):
        """提取分类模型结果"""
        results = {}

        # 传统ML模型
        for model_name, metrics in self.ml_results.items():
            if 'classification' in model_name:
                clean_name = model_name.replace('classification_', '')
                results[clean_name] = {
                    'type': 'Traditional ML',
                    'accuracy': metrics['validation']['accuracy'],
                    'f1_weighted': metrics['validation']['f1_weighted'],
                    'precision_macro': metrics['validation'].get('precision_macro', 0),
                    'recall_macro': metrics['validation'].get('recall_macro', 0),
                    'f1_macro': metrics['validation'].get('f1_macro', 0)
                }

        # 深度学习模型
        for model_name, metrics in self.dl_results.items():
            if model_name == 'BiLSTM':
                results[model_name] = {
                    'type': 'Deep Learning (Time Series)',
                    'accuracy': metrics['final_accuracy'],
                    'f1_weighted': metrics['final_f1'],
                    'precision_macro': 0,  # 深度学习模型没有计算这些指标
                    'recall_macro': 0,
                    'f1_macro': 0
                }
            elif model_name == 'GCN':
                results[model_name] = {
                    'type': 'Deep Learning (Graph)',
                    'accuracy': metrics['final_accuracy'],
                    'f1_weighted': metrics['final_f1'],
                    'precision_macro': 0,
                    'recall_macro': 0,
                    'f1_macro': 0
                }

        return pd.DataFrame(results).T

    def extract_regression_results(self):
        """提取回归模型结果"""
        results = {}

        # 传统ML模型
        for model_name, metrics in self.ml_results.items():
            if 'regression' in model_name:
                clean_name = model_name.replace('regression_', '')
                results[clean_name] = {
                    'type': 'Traditional ML',
                    'r2': metrics['validation']['r2'],
                    'rmse': metrics['validation']['rmse'],
                    'mae': metrics['validation'].get('mae', 0),
                    'mse': metrics['validation'].get('mse', 0)
                }

        return pd.DataFrame(results).T

    def create_comparison_visualizations(self):
        """创建对比可视化图表"""
        print("Creating comparison visualizations...")

        # 创建结果目录
        import os
        os.makedirs('results/comparison', exist_ok=True)

        # 1. 分类模型性能对比
        self._plot_classification_comparison()

        # 2. 回归模型性能对比
        self._plot_regression_comparison()

        # 3. 模型类型对比
        self._plot_model_type_comparison()

    def _plot_classification_comparison(self):
        """绘制分类模型对比图"""
        df_class = self.extract_classification_results()

        if df_class.empty:
            print("No classification results found")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 准确率对比
        df_class_sorted = df_class.sort_values('accuracy', ascending=True)
        bars1 = axes[0, 0].barh(df_class_sorted.index, df_class_sorted['accuracy'])
        axes[0, 0].set_title('Classification Models - Accuracy')
        axes[0, 0].set_xlabel('Accuracy')

        # 添加数值标签
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            axes[0, 0].text(width + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{width:.3f}', ha='left', va='center')

        # F1分数对比
        df_class_sorted_f1 = df_class.sort_values('f1_weighted', ascending=True)
        bars2 = axes[0, 1].barh(df_class_sorted_f1.index, df_class_sorted_f1['f1_weighted'])
        axes[0, 1].set_title('Classification Models - F1-Score (Weighted)')
        axes[0, 1].set_xlabel('F1-Score')

        # 添加数值标签
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            axes[0, 1].text(width + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{width:.3f}', ha='left', va='center')

        # 模型类型分布
        type_counts = df_class['type'].value_counts()
        axes[1, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Model Types Distribution')

        # 性能vs类型散点图
        colors = {'Traditional ML': 'blue', 'Deep Learning (Time Series)': 'red',
                 'Deep Learning (Graph)': 'green'}
        for model_type in df_class['type'].unique():
            subset = df_class[df_class['type'] == model_type]
            axes[1, 1].scatter(subset['accuracy'], subset['f1_weighted'],
                              c=colors[model_type], label=model_type, s=100, alpha=0.7)

        axes[1, 1].set_xlabel('Accuracy')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].set_title('Accuracy vs F1-Score by Model Type')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/comparison/classification_models_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_regression_comparison(self):
        """绘制回归模型对比图"""
        df_reg = self.extract_regression_results()

        if df_reg.empty:
            print("No regression results found")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # R²对比
        df_reg_sorted_r2 = df_reg.sort_values('r2', ascending=True)
        bars1 = axes[0, 0].barh(df_reg_sorted_r2.index, df_reg_sorted_r2['r2'])
        axes[0, 0].set_title('Regression Models - R² Score')
        axes[0, 0].set_xlabel('R²')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)

        # 添加数值标签
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            axes[0, 0].text(width + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{width:.3f}', ha='left', va='center')

        # RMSE对比
        df_reg_sorted_rmse = df_reg.sort_values('rmse', ascending=True)
        bars2 = axes[0, 1].barh(df_reg_sorted_rmse.index, df_reg_sorted_rmse['rmse'])
        axes[0, 1].set_title('Regression Models - RMSE')
        axes[0, 1].set_xlabel('RMSE')

        # 添加数值标签
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            axes[0, 1].text(width + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{width:.3f}', ha='left', va='center')

        # R² vs RMSE散点图
        axes[1, 0].scatter(df_reg['r2'], df_reg['rmse'], s=100, alpha=0.7)
        axes[1, 0].set_xlabel('R²')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].set_title('R² vs RMSE')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)

        # 添加模型标签
        for i, model in enumerate(df_reg.index):
            axes[1, 0].annotate(model, (df_reg['r2'].iloc[i], df_reg['rmse'].iloc[i]),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)

        # 性能排名表
        performance_ranking = df_reg.copy()
        performance_ranking['R2_Rank'] = performance_ranking['r2'].rank(ascending=False)
        performance_ranking['RMSE_Rank'] = performance_ranking['rmse'].rank(ascending=True)
        performance_ranking['Average_Rank'] = (performance_ranking['R2_Rank'] +
                                               performance_ranking['RMSE_Rank']) / 2
        performance_ranking = performance_ranking.sort_values('Average_Rank')

        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table_data = performance_ranking[['r2', 'rmse', 'Average_Rank']].round(4)
        table = axes[1, 1].table(cellText=table_data.values,
                                 rowLabels=table_data.index,
                                 colLabels=['R²', 'RMSE', 'Avg Rank'],
                                 cellLoc='center',
                                 loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Regression Models Performance Ranking', pad=20)

        plt.tight_layout()
        plt.savefig('results/comparison/regression_models_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_model_type_comparison(self):
        """绘制模型类型对比"""
        df_class = self.extract_classification_results()

        if df_class.empty:
            print("No classification results found for type comparison")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 按模型类型分组对比准确率
        type_avg = df_class.groupby('type').agg({
            'accuracy': ['mean', 'std'],
            'f1_weighted': ['mean', 'std']
        }).round(4)

        # 准确率对比
        means = type_avg[('accuracy', 'mean')]
        stds = type_avg[('accuracy', 'std')]
        bars = axes[0].bar(type_avg.index, means, yerr=stds, capsize=5, alpha=0.7)
        axes[0].set_title('Average Accuracy by Model Type')
        axes[0].set_ylabel('Accuracy')
        axes[0].tick_params(axis='x', rotation=15)

        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + stds[i] + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')

        # F1分数对比
        means_f1 = type_avg[('f1_weighted', 'mean')]
        stds_f1 = type_avg[('f1_weighted', 'std')]
        bars_f1 = axes[1].bar(type_avg.index, means_f1, yerr=stds_f1, capsize=5, alpha=0.7)
        axes[1].set_title('Average F1-Score by Model Type')
        axes[1].set_ylabel('F1-Score')
        axes[1].tick_params(axis='x', rotation=15)

        # 添加数值标签
        for i, bar in enumerate(bars_f1):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + stds_f1[i] + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('results/comparison/model_type_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def generate_comparison_report(self):
        """生成对比报告"""
        print("Generating comparison report...")

        df_class = self.extract_classification_results()
        df_reg = self.extract_regression_results()

        report = {
            'comparison_date': pd.Timestamp.now().isoformat(),
            'dataset_info': {
                'total_samples': 565,
                'train_samples': 452,
                'val_samples': 113,
                'features': 27,
                'num_classes': 3
            },
            'classification_results': {},
            'regression_results': {},
            'key_findings': []
        }

        # 分类模型结果
        if not df_class.empty:
            best_class_acc = df_class['accuracy'].idxmax()
            best_class_f1 = df_class['f1_weighted'].idxmax()

            report['classification_results'] = {
                'total_models': len(df_class),
                'best_accuracy_model': best_class_acc,
                'best_accuracy_score': float(df_class.loc[best_class_acc, 'accuracy']),
                'best_f1_model': best_class_f1,
                'best_f1_score': float(df_class.loc[best_class_f1, 'f1_weighted']),
                'average_accuracy': float(df_class['accuracy'].mean()),
                'average_f1': float(df_class['f1_weighted'].mean())
            }

        # 回归模型结果
        if not df_reg.empty:
            best_reg_r2 = df_reg['r2'].idxmax()
            best_reg_rmse = df_reg['rmse'].idxmin()

            report['regression_results'] = {
                'total_models': len(df_reg),
                'best_r2_model': best_reg_r2,
                'best_r2_score': float(df_reg.loc[best_reg_r2, 'r2']),
                'best_rmse_model': best_reg_rmse,
                'best_rmse_score': float(df_reg.loc[best_reg_rmse, 'rmse']),
                'average_r2': float(df_reg['r2'].mean()),
                'average_rmse': float(df_reg['rmse'].mean())
            }

        # 关键发现
        if not df_class.empty:
            if 'GCN' in df_class.index:
                gcn_acc = df_class.loc['GCN', 'accuracy']
                if gcn_acc > df_class['accuracy'].mean():
                    report['key_findings'].append(
                        f"GCN模型表现优异，准确率{gcn_acc:.3f}超过平均水平"
                    )

            if 'SVM' in df_class.index:
                svm_acc = df_class.loc['SVM', 'accuracy']
                if svm_acc > 0.7:
                    report['key_findings'].append(
                        f"传统SVM模型表现良好，准确率达到{svm_acc:.3f}"
                    )

        # 保存报告
        with open('results/comparison/model_comparison_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("Comparison report saved to results/comparison/model_comparison_report.json")

        # 打印摘要
        self._print_summary(report)

    def _print_summary(self, report):
        """打印对比摘要"""
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)

        if 'classification_results' in report and report['classification_results']:
            class_results = report['classification_results']
            print(f"\n分类模型对比:")
            print(f"  模型总数: {class_results['total_models']}")
            print(f"  最佳准确率模型: {class_results['best_accuracy_model']} ({class_results['best_accuracy_score']:.3f})")
            print(f"  最佳F1分数模型: {class_results['best_f1_model']} ({class_results['best_f1_score']:.3f})")
            print(f"  平均准确率: {class_results['average_accuracy']:.3f}")
            print(f"  平均F1分数: {class_results['average_f1']:.3f}")

        if 'regression_results' in report and report['regression_results']:
            reg_results = report['regression_results']
            print(f"\n回归模型对比:")
            print(f"  模型总数: {reg_results['total_models']}")
            print(f"  最佳R2模型: {reg_results['best_r2_model']} ({reg_results['best_r2_score']:.3f})")
            print(f"  最佳RMSE模型: {reg_results['best_rmse_model']} ({reg_results['best_rmse_score']:.3f})")
            print(f"  平均R2: {reg_results['average_r2']:.3f}")
            print(f"  平均RMSE: {reg_results['average_rmse']:.3f}")

        if report['key_findings']:
            print(f"\n关键发现:")
            for i, finding in enumerate(report['key_findings'], 1):
                print(f"  {i}. {finding}")

        print("\n" + "="*60)

    def run_comparison(self):
        """运行完整对比分析"""
        print("=== Starting Model Comparison Analysis ===")

        # 创建可视化
        self.create_comparison_visualizations()

        # 生成报告
        self.generate_comparison_report()

        print("Model comparison completed!")
        print("Results saved to results/comparison/")

def main():
    """主函数"""
    comparator = ModelComparison()
    comparator.run_comparison()

if __name__ == "__main__":
    main()