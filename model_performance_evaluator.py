#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型性能综合评估系统
输出详细的模型评估指标和可视化结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import torch
import torch.nn.functional as F
import json
import os
from datetime import datetime

from enhanced_county_config import EnhancedCountyLevelConfig
from enhanced_model_trainer import EnhancedModelTrainer, EnhancedDataset

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ModelPerformanceEvaluator:
    """模型性能评估器"""

    def __init__(self):
        self.config = EnhancedCountyLevelConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

    def load_trained_model(self):
        """加载训练好的模型"""
        print("=== 加载训练好的模型 ===")

        # 创建模型
        from enhanced_model_trainer import EnhancedBiLSTMGCNModel
        model = EnhancedBiLSTMGCNModel(
            input_size=self.config.NUM_FEATURES,
            hidden_size=64,
            num_classes=self.config.NUM_CLASSES,
            dropout=0.3
        ).to(self.device)

        # 加载权重
        model_path = os.path.join(self.config.MODEL_SAVE_DIR, 'enhanced_bilstm_gcn_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            print(f"模型权重加载成功: {model_path}")
        else:
            print(f"模型权重文件不存在: {model_path}")
            return None

        # 加载模型信息
        info_path = os.path.join(self.config.MODEL_SAVE_DIR, 'enhanced_model_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            print(f"模型信息: {model_info['model_type']}")
            print(f"输入特征: {model_info['input_size']}")
            print(f"训练日期: {model_info['training_date']}")

        self.model = model
        return model

    def load_test_data(self):
        """加载测试数据"""
        print("\n=== 加载测试数据 ===")

        # 加载完整数据
        data = pd.read_csv(self.config.ENHANCED_COMPLETE_DATA_PATH)
        test_data = data[data['Year'].isin(self.config.TEST_YEARS)]

        print(f"测试数据: {len(test_data)} 样本")
        print(f"覆盖县数: {test_data['County'].nunique()}")

        # 创建数据集
        test_dataset = EnhancedDataset(test_data)
        print(f"测试序列样本: {len(test_dataset)}")

        # 获取数据加载器
        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return test_loader, test_dataset, test_data

    def generate_predictions(self, test_loader, test_dataset):
        """生成模型预测结果"""
        print("\n=== 生成预测结果 ===")

        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_targets = []
        all_counties = []
        all_years = []

        with torch.no_grad():
            for batch in test_loader:
                sequences = batch['sequence'].to(self.device)
                targets = batch['target'].squeeze(-1).to(self.device)

                outputs = self.model(sequences)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_counties.extend(batch['county'])
                all_years.extend(batch['year'].numpy())

        # 转换为numpy数组
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_targets = np.array(all_targets)

        print(f"预测完成: {len(all_predictions)} 个样本")
        return all_predictions, all_probabilities, all_targets, all_counties, all_years

    def calculate_comprehensive_metrics(self, y_true, y_pred, y_prob):
        """计算综合评估指标"""
        print("\n=== 计算综合评估指标 ===")

        metrics = {}

        # 基础分类指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # 各类别详细指标
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        metrics['per_class_metrics'] = {}
        for i, class_name in enumerate(self.config.CLASS_NAMES):
            if i < len(precision_per_class):
                metrics['per_class_metrics'][class_name] = {
                    'precision': float(precision_per_class[i]) if i < len(precision_per_class) else 0.0,
                    'recall': float(recall_per_class[i]) if i < len(recall_per_class) else 0.0,
                    'f1': float(f1_per_class[i]) if i < len(f1_per_class) else 0.0
                }

        # 尝试计算AUC（对于多分类）
        try:
            if len(np.unique(y_true)) > 1:  # 确保有多个类别
                metrics['auc_macro'] = roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')
                metrics['auc_weighted'] = roc_auc_score(y_true, y_prob, average='weighted', multi_class='ovr')
            else:
                metrics['auc_macro'] = 0.0
                metrics['auc_weighted'] = 0.0
        except:
            metrics['auc_macro'] = 0.0
            metrics['auc_weighted'] = 0.0

        return metrics

    def print_detailed_metrics(self, metrics):
        """打印详细指标"""
        print("\n" + "="*80)
        print("模型性能详细评估报告")
        print("="*80)

        print(f"\n📊 整体性能指标:")
        print(f"  准确率 (Accuracy):           {metrics['accuracy']:.4f}")
        print(f"  精确率-宏平均 (Precision):    {metrics['precision_macro']:.4f}")
        print(f"  精确率-加权 (Precision):      {metrics['precision_weighted']:.4f}")
        print(f"  召回率-宏平均 (Recall):       {metrics['recall_macro']:.4f}")
        print(f"  召回率-加权 (Recall):         {metrics['recall_weighted']:.4f}")
        print(f"  F1分数-宏平均 (F1-Score):     {metrics['f1_macro']:.4f}")
        print(f"  F1分数-加权 (F1-Score):       {metrics['f1_weighted']:.4f}")

        if 'auc_macro' in metrics:
            print(f"  AUC-宏平均:                  {metrics['auc_macro']:.4f}")
            print(f"  AUC-加权:                    {metrics['auc_weighted']:.4f}")

        print(f"\n📈 各类别详细指标:")
        print(f"{'类别':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8}")
        print("-" * 35)
        for class_name, class_metrics in metrics['per_class_metrics'].items():
            print(f"{class_name:<8} {class_metrics['precision']:<8.4f} {class_metrics['recall']:<8.4f} {class_metrics['f1']:<8.4f}")

    def create_comprehensive_visualizations(self, y_true, y_pred, y_prob, counties, years):
        """创建综合可视化"""
        print("\n=== 创建综合可视化 ===")

        os.makedirs('results/enhanced_visualizations', exist_ok=True)

        # 1. 混淆矩阵
        self.create_enhanced_confusion_matrix(y_true, y_pred)

        # 2. ROC曲线（多分类）
        if len(np.unique(y_true)) > 1:
            self.create_multiclass_roc_curve(y_true, y_prob)

        # 3. 精确率-召回率曲线
        if len(np.unique(y_true)) > 1:
            self.create_precision_recall_curve(y_true, y_prob)

        # 4. 预测置信度分析
        self.create_prediction_confidence_analysis(y_true, y_pred, y_prob)

        # 5. 县级预测结果分析
        self.create_county_prediction_analysis(counties, years, y_true, y_pred)

        # 6. 类别分布对比
        self.create_class_distribution_comparison(y_true, y_pred)

        print("可视化图表保存完成")

    def create_enhanced_confusion_matrix(self, y_true, y_pred):
        """创建增强版混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))

        # 计算归一化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 绝对数量混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.config.CLASS_NAMES[:len(cm)],
                   yticklabels=self.config.CLASS_NAMES[:len(cm)],
                   ax=ax1, cbar_kws={'label': '样本数'})
        ax1.set_title('混淆矩阵 (绝对数量)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('预测类别')
        ax1.set_ylabel('真实类别')

        # 归一化混淆矩阵
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.config.CLASS_NAMES[:len(cm_normalized)],
                   yticklabels=self.config.CLASS_NAMES[:len(cm_normalized)],
                   ax=ax2, cbar_kws={'label': '比例'})
        ax2.set_title('混淆矩阵 (归一化)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('预测类别')
        ax2.set_ylabel('真实类别')

        plt.tight_layout()
        plt.savefig('results/enhanced_visualizations/enhanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_multiclass_roc_curve(self, y_true, y_prob):
        """创建多分类ROC曲线"""
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc

        # 二值化标签
        y_true_bin = label_binarize(y_true, classes=range(self.config.NUM_CLASSES))

        plt.figure(figsize=(12, 8))

        # 为每个类别绘制ROC曲线
        for i in range(self.config.NUM_CLASSES):
            if i < y_prob.shape[1] and i < y_true_bin.shape[1]:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, linewidth=2,
                        label=f'{self.config.CLASS_NAMES[i]} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机分类器')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (FPR)', fontsize=12)
        plt.ylabel('真正率 (TPR)', fontsize=12)
        plt.title('多分类ROC曲线', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/enhanced_visualizations/multiclass_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_precision_recall_curve(self, y_true, y_prob):
        """创建精确率-召回率曲线"""
        from sklearn.preprocessing import label_binarize

        y_true_bin = label_binarize(y_true, classes=range(self.config.NUM_CLASSES))

        plt.figure(figsize=(12, 8))

        for i in range(self.config.NUM_CLASSES):
            if i < y_prob.shape[1] and i < y_true_bin.shape[1]:
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                avg_precision = average_precision_score(y_true_bin[:, i], y_prob[:, i])
                plt.plot(recall, precision, linewidth=2,
                        label=f'{self.config.CLASS_NAMES[i]} (AP = {avg_precision:.3f})')

        plt.xlabel('召回率 (Recall)', fontsize=12)
        plt.ylabel('精确率 (Precision)', fontsize=12)
        plt.title('精确率-召回率曲线', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/enhanced_visualizations/precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_prediction_confidence_analysis(self, y_true, y_pred, y_prob):
        """创建预测置信度分析"""
        plt.figure(figsize=(15, 10))

        # 1. 预测置信度分布
        plt.subplot(2, 3, 1)
        max_probs = np.max(y_prob, axis=1)
        plt.hist(max_probs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('最大预测概率')
        plt.ylabel('频次')
        plt.title('预测置信度分布')
        plt.grid(True, alpha=0.3)

        # 2. 正确预测 vs 错误预测的置信度
        plt.subplot(2, 3, 2)
        correct_mask = (y_true == y_pred)
        correct_probs = max_probs[correct_mask]
        incorrect_probs = max_probs[~correct_mask]

        if len(incorrect_probs) > 0:
            plt.hist([correct_probs, incorrect_probs], bins=15, alpha=0.7,
                    label=['正确预测', '错误预测'], color=['green', 'red'])
            plt.legend()
        else:
            plt.hist(correct_probs, bins=15, alpha=0.7, color='green', label='正确预测')
            plt.legend()
        plt.xlabel('预测置信度')
        plt.ylabel('频次')
        plt.title('预测结果置信度对比')
        plt.grid(True, alpha=0.3)

        # 3. 各类别预测概率分布
        plt.subplot(2, 3, 3)
        for i in range(min(self.config.NUM_CLASSES, len(self.config.CLASS_NAMES))):
            if i < y_prob.shape[1]:
                class_probs = y_prob[y_true == i, i] if np.any(y_true == i) else []
                if len(class_probs) > 0:
                    plt.hist(class_probs, bins=10, alpha=0.5,
                            label=f'{self.config.CLASS_NAMES[i]}')
        plt.xlabel('预测概率')
        plt.ylabel('频次')
        plt.title('各类别预测概率分布')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. 预测概率热力图
        plt.subplot(2, 3, 4)
        if len(y_prob) > 0:
            # 取前20个样本的热力图
            sample_probs = y_prob[:min(20, len(y_prob))]
            im = plt.imshow(sample_probs.T, cmap='YlOrRd', aspect='auto')
            plt.colorbar(im, label='预测概率')
            plt.xlabel('样本序号')
            plt.ylabel('预测类别')
            plt.title('预测概率热力图 (前20个样本)')
            plt.yticks(range(self.config.NUM_CLASSES), self.config.CLASS_NAMES)

        # 5. 置信度vs准确性
        plt.subplot(2, 3, 5)
        confidence_bins = np.linspace(0.5, 1.0, 10)
        accuracy_by_confidence = []

        for i in range(len(confidence_bins) - 1):
            mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i+1])
            if np.sum(mask) > 0:
                accuracy = np.mean(y_true[mask] == y_pred[mask])
                accuracy_by_confidence.append(accuracy)
            else:
                accuracy_by_confidence.append(0)

        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        plt.plot(bin_centers, accuracy_by_confidence, 'o-', linewidth=2, markersize=6)
        plt.xlabel('置信度区间')
        plt.ylabel('准确率')
        plt.title('置信度 vs 准确率关系')
        plt.grid(True, alpha=0.3)

        # 6. 预测类别概率分布
        plt.subplot(2, 3, 6)
        class_predictions = np.argmax(y_prob, axis=1)
        class_counts = np.bincount(class_predictions, minlength=self.config.NUM_CLASSES)
        colors = self.config.CLASS_COLORS[:len(class_counts)]

        bars = plt.bar(range(len(class_counts)), class_counts, color=colors, alpha=0.7)
        plt.xlabel('预测类别')
        plt.ylabel('样本数')
        plt.title('预测类别分布')
        plt.xticks(range(len(self.config.CLASS_NAMES)), self.config.CLASS_NAMES)
        plt.grid(True, alpha=0.3)

        # 添加数值标签
        for bar, count in zip(bars, class_counts):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{count}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('results/enhanced_visualizations/prediction_confidence_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_county_prediction_analysis(self, counties, years, y_true, y_pred):
        """创建县级预测结果分析"""
        plt.figure(figsize=(15, 10))

        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'County': counties,
            'Year': years,
            'Actual': y_true,
            'Predicted': y_pred,
            'Correct': (y_true == y_pred).astype(int)
        })

        # 1. 各县预测准确性
        plt.subplot(2, 3, 1)
        county_accuracy = results_df.groupby('County')['Correct'].mean()
        plt.hist(county_accuracy, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('县级预测准确率')
        plt.ylabel('县数')
        plt.title('各县预测准确率分布')
        plt.grid(True, alpha=0.3)

        # 2. 年度预测准确性
        plt.subplot(2, 3, 2)
        year_accuracy = results_df.groupby('Year')['Correct'].mean()
        plt.bar(year_accuracy.index, year_accuracy.values, color='lightblue', alpha=0.7)
        plt.xlabel('年份')
        plt.ylabel('预测准确率')
        plt.title('年度预测准确率')
        plt.xticks(year_accuracy.index)
        plt.grid(True, alpha=0.3)

        # 3. 实际vs预测类别分布对比
        plt.subplot(2, 3, 3)
        actual_dist = results_df['Actual'].value_counts().sort_index()
        pred_dist = results_df['Predicted'].value_counts().sort_index()

        x = np.arange(len(actual_dist))
        width = 0.35

        plt.bar(x - width/2, actual_dist.values, width, label='实际', alpha=0.7, color='blue')
        plt.bar(x + width/2, pred_dist.values, width, label='预测', alpha=0.7, color='red')

        plt.xlabel('发病等级')
        plt.ylabel('样本数')
        plt.title('实际vs预测分布对比')
        plt.xticks(x, [self.config.CLASS_NAMES[i] for i in actual_dist.index])
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. 错误预测分析
        plt.subplot(2, 3, 4)
        incorrect_results = results_df[results_df['Correct'] == 0]
        if len(incorrect_results) > 0:
            error_matrix = pd.crosstab(incorrect_results['Actual'],
                                     incorrect_results['Predicted'],
                                     normalize='index')
            sns.heatmap(error_matrix, annot=True, fmt='.2f', cmap='Reds',
                       xticklabels=[self.config.CLASS_NAMES[i] for i in error_matrix.columns],
                       yticklabels=[self.config.CLASS_NAMES[i] for i in error_matrix.index])
            plt.title('错误预测模式')
            plt.xlabel('预测类别')
            plt.ylabel('实际类别')
        else:
            plt.text(0.5, 0.5, '无错误预测', ha='center', va='center',
                    transform=plt.gca().transAxes, fontsize=14)
            plt.title('错误预测模式')

        # 5. 各县样本数量分布
        plt.subplot(2, 3, 5)
        county_samples = results_df.groupby('County').size()
        plt.hist(county_samples, bins=15, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('每县样本数')
        plt.ylabel('县数')
        plt.title('各县样本数量分布')
        plt.grid(True, alpha=0.3)

        # 6. 预测结果时间序列
        plt.subplot(2, 3, 6)
        time_accuracy = results_df.groupby(['Year', 'County'])['Correct'].mean().groupby('Year').mean()
        plt.plot(time_accuracy.index, time_accuracy.values, 'o-', linewidth=2, markersize=8)
        plt.xlabel('年份')
        plt.ylabel('平均准确率')
        plt.title('年度平均预测准确率趋势')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/enhanced_visualizations/county_prediction_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_class_distribution_comparison(self, y_true, y_pred):
        """创建类别分布对比"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 实际vs预测分布 - 柱状图
        ax1 = axes[0, 0]
        actual_counts = np.bincount(y_true, minlength=self.config.NUM_CLASSES)
        pred_counts = np.bincount(y_pred, minlength=self.config.NUM_CLASSES)

        x = np.arange(self.config.NUM_CLASSES)
        width = 0.35

        bars1 = ax1.bar(x - width/2, actual_counts, width, label='实际', alpha=0.7, color='blue')
        bars2 = ax1.bar(x + width/2, pred_counts, width, label='预测', alpha=0.7, color='red')

        ax1.set_xlabel('发病等级')
        ax1.set_ylabel('样本数')
        ax1.set_title('实际vs预测样本分布')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.config.CLASS_NAMES)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')

        # 2. 分布比例对比 - 饼图
        ax2 = axes[0, 1]
        actual_labels = [f'{self.config.CLASS_NAMES[i]}\n({actual_counts[i]})'
                        for i in range(self.config.NUM_CLASSES) if actual_counts[i] > 0]
        actual_sizes = [actual_counts[i] for i in range(self.config.NUM_CLASSES) if actual_counts[i] > 0]

        ax2.pie(actual_sizes, labels=actual_labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title('实际分布比例')

        # 3. 预测准确性矩阵
        ax3 = axes[1, 0]
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens',
                   xticklabels=self.config.CLASS_NAMES[:len(cm_normalized)],
                   yticklabels=self.config.CLASS_NAMES[:len(cm_normalized)],
                   ax=ax3, cbar_kws={'label': '准确率'})
        ax3.set_title('各类别预测准确性')
        ax3.set_xlabel('预测类别')
        ax3.set_ylabel('实际类别')

        # 4. 性能指标雷达图
        ax4 = axes[1, 1]
        categories = ['精确率', '召回率', 'F1分数']

        # 计算各类别的指标
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        # 只显示有样本的类别
        valid_classes = [i for i in range(self.config.NUM_CLASSES)
                        if i < len(precision_per_class) and actual_counts[i] > 0]

        if len(valid_classes) > 0:
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]  # 闭合

            valid_class_names = [self.config.CLASS_NAMES[i] for i in valid_classes]
            colors = plt.cm.Set3(np.linspace(0, 1, len(valid_classes)))

            for i, class_idx in enumerate(valid_classes):
                values = [
                    precision_per_class[class_idx],
                    recall_per_class[class_idx],
                    f1_per_class[class_idx]
                ]
                values += values[:1]  # 闭合

                ax4.plot(angles, values, 'o-', linewidth=2, label=valid_class_names[i], color=colors[i])
                ax4.fill(angles, values, alpha=0.1, color=colors[i])

            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(categories)
            ax4.set_ylim(0, 1)
            ax4.set_title('各类别性能雷达图')
            ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, '数据不足', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=14)
            ax4.set_title('各类别性能雷达图')

        plt.tight_layout()
        plt.savefig('results/enhanced_visualizations/class_distribution_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def generate_comprehensive_report(self, metrics, test_data):
        """生成综合评估报告"""
        print("\n=== 生成综合评估报告 ===")

        report = {
            'evaluation_date': datetime.now().isoformat(),
            'model_info': {
                'type': 'EnhancedBiLSTMGCNModel',
                'input_features': self.config.NUM_FEATURES,
                'num_classes': self.config.NUM_CLASSES,
                'feature_categories': self.config.get_feature_categories()
            },
            'dataset_info': {
                'test_samples': len(test_data),
                'test_counties': test_data['County'].nunique(),
                'test_years': self.config.TEST_YEARS,
                'class_distribution': {
                    str(level): int(test_data[test_data['Severity_Level'] == level].shape[0])
                    for level in range(self.config.NUM_CLASSES)
                }
            },
            'performance_metrics': {
                'overall': {
                    'accuracy': float(metrics['accuracy']),
                    'precision_macro': float(metrics['precision_macro']),
                    'precision_weighted': float(metrics['precision_weighted']),
                    'recall_macro': float(metrics['recall_macro']),
                    'recall_weighted': float(metrics['recall_weighted']),
                    'f1_macro': float(metrics['f1_macro']),
                    'f1_weighted': float(metrics['f1_weighted'])
                },
                'per_class': metrics['per_class_metrics']
            },
            'data_enhancement_impact': {
                'total_counties_covered': 135,
                'healthy_counties_added': 25,
                'remote_sensing_features': len([f for f in self.config.ALL_FEATURES
                                              if any(x in f for x in ['NDVI', 'EVI', 'LST', 'TRMM', 'Soil'])]),
                'geographical_features': len([f for f in self.config.ALL_FEATURES
                                            if any(x in f for x in ['Coastal', 'Forest', 'Influence'])])
            },
            'visualization_files': [
                'enhanced_confusion_matrix.png',
                'multiclass_roc_curve.png',
                'precision_recall_curve.png',
                'prediction_confidence_analysis.png',
                'county_prediction_analysis.png',
                'class_distribution_comparison.png'
            ]
        }

        # 保存报告
        report_path = os.path.join(self.config.RESULTS_DIR, 'comprehensive_performance_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"综合评估报告保存到: {report_path}")
        return report

def main():
    """主函数"""
    print("=== 模型性能综合评估系统 ===")
    print("开始时间:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # 创建评估器
    evaluator = ModelPerformanceEvaluator()

    # 加载模型和数据
    model = evaluator.load_trained_model()
    if model is None:
        print("模型加载失败，请先训练模型")
        return

    test_loader, test_dataset, test_data = evaluator.load_test_data()

    # 生成预测
    predictions, probabilities, targets, counties, years = evaluator.generate_predictions(test_loader, test_dataset)

    # 计算指标
    metrics = evaluator.calculate_comprehensive_metrics(targets, predictions, probabilities)

    # 打印详细指标
    evaluator.print_detailed_metrics(metrics)

    # 创建可视化
    evaluator.create_comprehensive_visualizations(targets, predictions, probabilities, counties, years)

    # 生成报告
    report = evaluator.generate_comprehensive_report(metrics, test_data)

    print(f"\n=== 评估完成 ===")
    print(f"所有评估结果和可视化图表已保存到 results/enhanced_visualizations/ 目录")
    print(f"综合报告保存到: results/enhanced_predictions/comprehensive_performance_report.json")
    print(f"结束时间:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    return evaluator, report

if __name__ == "__main__":
    evaluator, report = main()