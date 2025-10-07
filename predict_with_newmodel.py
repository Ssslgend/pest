#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用新训练的模型进行预测
"""

import torch
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import sys
sys.path.append('oldPestBlstem/ablation_study')
from bilstm_variants import BiLSTMComplete

def load_trained_model():
    """加载训练好的模型"""
    print("加载训练好的模型...")

    # 加载模型配置和结果
    with open('results/sf_bilstm_newdata/training_results.json', 'r', encoding='utf-8') as f:
        results = json.load(f)

    config = results['config']
    feature_count = results['feature_count']

    print(f"模型配置: {config}")
    print(f"特征数量: {feature_count}")

    # 创建模型
    model_config = {
        "input_size": feature_count,
        "hidden_size": config['hidden_size'],
        "num_layers": config['num_layers'],
        "dropout": config['dropout'],
        "num_classes": 1
    }

    model = BiLSTMComplete(model_config)

    # 加载模型权重
    model.load_state_dict(torch.load('results/sf_bilstm_newdata/best_model.pth', map_location='cpu'))
    model.eval()

    print("模型加载完成")
    return model, config, results

def load_test_data():
    """加载测试数据"""
    print("加载测试数据...")

    test_data = pd.read_csv('datas/shandong_pest_data/improved_test.csv')
    print(f"测试数据形状: {test_data.shape}")

    # 提取特征和标签
    excluded_cols = [
        'Has_Occurrence', 'Severity', 'county_name', 'year', 'month',
        'Period', 'Data_Source'
    ]

    feature_cols = []
    for col in test_data.columns:
        if col not in excluded_cols and test_data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            feature_cols.append(col)

    X_test = test_data[feature_cols].values
    y_test = test_data['Has_Occurrence'].values

    print(f"特征数量: {len(feature_cols)}")
    print(f"测试集发生率: {np.mean(y_test):.4f}")

    return X_test, y_test, test_data, feature_cols

def make_predictions(model, X_test):
    """进行预测"""
    print("进行预测...")

    # 转换为PyTorch张量
    X_test_tensor = torch.FloatTensor(X_test.reshape(X_test.shape[0], 1, X_test.shape[1]))

    # 预测
    with torch.no_grad():
        model.eval()
        outputs = model(X_test_tensor)
        probabilities = torch.sigmoid(outputs).numpy().flatten()
        predictions = (probabilities > 0.5).astype(int)

    print(f"预测完成")
    return predictions, probabilities

def analyze_predictions(y_true, y_pred, probabilities):
    """分析预测结果"""
    print("\n预测结果分析:")
    print("=" * 50)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    # 基本指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_true, probabilities)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc_score:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n混淆矩阵:")
    print(f"              预测")
    print(f"真实    负样本    正样本")
    print(f"负样本   {tn:6d}    {fp:6d}")
    print(f"正样本   {fn:6d}    {tp:6d}")

    print(f"\n具体指标:")
    print(f"真阴性(TN): {tn}")
    print(f"假阳性(FP): {fp}")
    print(f"假阴性(FN): {fn}")
    print(f"真阳性(TP): {tp}")
    print(f"特异度(Specificity): {specificity:.4f}")

    # 分类报告
    print(f"\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=['未发生', '发生']))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'specificity': specificity,
        'confusion_matrix': cm.tolist(),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }

def create_visualization(y_true, probabilities, metrics):
    """创建可视化图表"""
    print("创建可视化图表...")

    # 创建ROC曲线
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12, 4))

    # ROC曲线
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    # 概率分布
    plt.subplot(1, 2, 2)
    plt.hist(probabilities[y_true==0], bins=20, alpha=0.5, label='未发生', color='blue')
    plt.hist(probabilities[y_true==1], bins=20, alpha=0.5, label='发生', color='red')
    plt.xlabel('预测概率')
    plt.ylabel('频数')
    plt.title('预测概率分布')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/sf_bilstm_newdata/prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("可视化图表已保存: results/sf_bilstm_newdata/prediction_analysis.png")

def save_predictions(test_data, predictions, probabilities):
    """保存预测结果"""
    print("保存预测结果...")

    # 创建结果数据框
    results_df = test_data.copy()
    results_df['prediction'] = predictions
    results_df['probability'] = probabilities

    # 保存结果
    results_df.to_csv('results/sf_bilstm_newdata/test_predictions.csv', index=False, encoding='utf-8-sig')
    print("预测结果已保存: results/sf_bilstm_newdata/test_predictions.csv")

    # 保存预测统计
    pred_stats = {
        'total_samples': len(predictions),
        'predicted_positive': int(np.sum(predictions)),
        'predicted_negative': int(len(predictions) - np.sum(predictions)),
        'mean_probability': float(np.mean(probabilities)),
        'positive_samples': int(np.sum(test_data['Has_Occurrence'])),
        'negative_samples': int(len(test_data) - np.sum(test_data['Has_Occurrence']))
    }

    with open('results/sf_bilstm_newdata/prediction_stats.json', 'w', encoding='utf-8') as f:
        json.dump(pred_stats, f, ensure_ascii=False, indent=4)

    print(f"预测统计:")
    print(f"  总样本数: {pred_stats['total_samples']}")
    print(f"  预测为发生: {pred_stats['predicted_positive']}")
    print(f"  预测为未发生: {pred_stats['predicted_negative']}")
    print(f"  平均预测概率: {pred_stats['mean_probability']:.4f}")

def run_prediction():
    """运行完整的预测流程"""
    print("=" * 60)
    print("SF-BiLSTM模型预测")
    print("=" * 60)

    try:
        # 1. 加载模型
        model, config, training_results = load_trained_model()

        # 2. 加载测试数据
        X_test, y_test, test_data, feature_cols = load_test_data()

        # 3. 进行预测
        predictions, probabilities = make_predictions(model, X_test)

        # 4. 分析结果
        metrics = analyze_predictions(y_test, predictions, probabilities)

        # 5. 创建可视化
        create_visualization(y_test, probabilities, metrics)

        # 6. 保存结果
        save_predictions(test_data, predictions, probabilities)

        # 保存完整预测报告
        full_results = {
            'training_results': training_results,
            'prediction_metrics': metrics,
            'model_config': config,
            'feature_count': len(feature_cols)
        }

        with open('results/sf_bilstm_newdata/complete_prediction_report.json', 'w', encoding='utf-8') as f:
            json.dump(full_results, f, ensure_ascii=False, indent=4)

        print(f"\n预测完成！结果已保存到: results/sf_bilstm_newdata/")
        print(f"主要指标:")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")

        return full_results

    except Exception as e:
        print(f"预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_prediction()