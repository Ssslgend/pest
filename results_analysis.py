#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SF-BiLSTM实验结果分析脚本
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_results():
    """分析实验结果"""
    print("分析SF-BiLSTM实验结果...")

    # 检查结果文件
    result_file = 'results/sf_bilstm_test/quick_test_results.json'

    if not os.path.exists(result_file):
        print(f"结果文件不存在: {result_file}")
        return

    # 加载结果
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print(f"找到 {len(results)} 个模型的结果")

    # 创建性能对比表格
    print("\n" + "="*80)
    print("SF-BiLSTM模型性能对比")
    print("="*80)

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity']

    # 打印表头
    print(f"{'模型':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12} {'Specificity':<12}")
    print("-" * 100)

    # 打印每个模型的结果
    for model_name, model_results in results.items():
        row = f"{model_name:<20}"
        for metric in metrics:
            value = model_results.get(metric, 0)
            row += f" {value:<12.4f}"
        print(row)

    # 分析问题
    print("\n" + "="*80)
    print("结果分析")
    print("="*80)

    print("1. 数据不平衡问题:")
    print("   - 真实发生数据中，Has_Occurrence=0: 5817个样本 (86.4%)")
    print("   - Has_Occurrence=1: 918个样本 (13.6%)")
    print("   - 存在严重的类别不平衡")

    print("\n2. 模型性能分析:")
    for model_name, model_results in results.items():
        print(f"\n   {model_name}:")
        print(f"   - AUC: {model_results['auc']:.4f} (模型区分能力)")
        print(f"   - Accuracy: {model_results['accuracy']:.4f} (总体准确率)")
        print(f"   - Precision: {model_results['precision']:.4f} (正样本预测精度)")
        print(f"   - Recall: {model_results['recall']:.4f} (正样本召回率)")
        print(f"   - F1: {model_results['f1']:.4f} (综合指标)")
        print(f"   - Specificity: {model_results['specificity']:.4f} (负样本正确率)")

        # 分析问题
        if model_results['precision'] == 0 and model_results['recall'] == 0:
            print("   - 问题: 模型可能预测全部为负样本，无法检测正样本")
        elif model_results['precision'] < 0.3:
            print("   - 问题: 正样本预测精度较低，存在较多误报")
        elif model_results['recall'] < 0.3:
            print("   - 问题: 正样本召回率较低，漏检较多")

    print("\n3. 改进建议:")
    print("   a) 处理数据不平衡:")
    print("      - 使用过采样(SMOTE)增加正样本")
    print("      - 使用欠采样减少负样本")
    print("      - 调整类别权重")
    print("      - 使用不同的损失函数(如Weighted BCE)")

    print("\n   b) 模型调优:")
    print("      - 调整学习率")
    print("      - 增加训练轮次")
    print("      - 调整模型复杂度")
    print("      - 使用不同的正则化方法")

    print("\n   c) 特征工程:")
    print("      - 特征选择和降维")
    print("      - 创建新的复合特征")
    print("      - 特征标准化和归一化")

    print("\n   d) 验证策略:")
    print("      - 使用交叉验证")
    print("      - 分层采样确保类别平衡")
    print("      - 使用不同的评估指标")

    # 保存分析报告
    report = {
        'experiment_summary': {
            'dataset': 'real_occurrence_data',
            'total_samples': 6735,
            'positive_samples': 918,
            'negative_samples': 5817,
            'imbalance_ratio': 5817/918
        },
        'model_results': results,
        'analysis': {
            'main_issue': 'severe_class_imbalance',
            'recommendations': [
                'use_class_balancing_techniques',
                'adjust_loss_function_weights',
                'try_different_sampling_strategies',
                'increase_model_complexity'
            ]
        }
    }

    with open('results/sf_bilstm_test/analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    print(f"\n详细分析报告已保存到: results/sf_bilstm_test/analysis_report.json")

if __name__ == "__main__":
    analyze_results()