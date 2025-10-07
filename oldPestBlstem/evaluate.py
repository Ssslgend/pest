import torch
from torch.utils.data import DataLoader
from data.dataset import PestDataset
from model.bilstm import BiLSTMModel
from config.params import MODEL_CONFIG, TRAIN_CONFIG
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    f1_score, precision_recall_curve, average_precision_score,
    precision_score, recall_score, accuracy_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import label_binarize

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def save_plot(fig, filename):
    """保存图片的辅助函数"""
    try:
        os.makedirs('results', exist_ok=True)
        filepath = os.path.join('results', filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {os.path.abspath(filepath)}")
    except Exception as e:
        print(f"保存图片时出错: {e}")

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    """绘制混淆矩阵"""
    try:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes,
                    yticklabels=classes)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        save_plot(plt.gcf(), 'confusion_matrix.png')
        plt.close()
    except Exception as e:
        print(f"绘制混淆矩阵时出错: {e}")

def plot_roc_curve(y_true, y_probs, classes, title='ROC Curves'):
    """绘制ROC曲线"""
    try:
        plt.figure(figsize=(12, 10))
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
            auc = roc_auc_score(y_true[:, i], y_probs[:, i])
            plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        save_plot(plt.gcf(), 'roc_curves.png')
        plt.close()
    except Exception as e:
        print(f"绘制ROC曲线时出错: {e}")

def plot_precision_recall_curve(y_true, y_probs, classes, title='Precision-Recall Curves'):
    """绘制精确率-召回率曲线"""
    try:
        plt.figure(figsize=(10, 8))
        for i in range(len(classes)):
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
            ap = average_precision_score(y_true[:, i], y_probs[:, i])
            plt.plot(recall, precision, label=f'{classes[i]} (AP = {ap:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_plot(plt.gcf(), 'precision_recall_curves.png')
        plt.close()
    except Exception as e:
        print(f"绘制精确率-召回率曲线时出错: {e}")

def evaluate_model():
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model = BiLSTMModel(MODEL_CONFIG).to(device)
    try:
        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("模型加载成功")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return
    
    # 加载数据
    from data.dataset import DataHandler
    handler = DataHandler(TRAIN_CONFIG)
    data_path = "datas/pest_rice_with_features_2_classified.csv"
    
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = handler.load_data(data_path)
        print("数据加载成功")
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return
    
    # 创建测试数据集和数据加载器
    test_dataset = PestDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 评估模型
    model.eval()
    all_test_preds = []
    all_test_labels = []
    all_test_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_test_preds.extend(predicted.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())
            all_test_probs.extend(probs.cpu().numpy())
    
    # 转换为numpy数组
    all_test_preds = np.array(all_test_preds)
    all_test_labels = np.array(all_test_labels)
    all_test_probs = np.array(all_test_probs)
    
    # 获取实际出现的类别
    unique_labels = np.unique(all_test_labels)
    class_names = ['No Risk', 'Low Risk', 'Medium Risk', 'High Risk']
    actual_class_names = [class_names[i] for i in unique_labels]
    
    # 计算各项指标
    accuracy = accuracy_score(all_test_labels, all_test_preds)
    weighted_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')
    precision = precision_score(all_test_labels, all_test_preds, average='weighted')
    recall = recall_score(all_test_labels, all_test_preds, average='weighted')
    
    # 计算ROC AUC和AP
    y_test_bin = label_binarize(all_test_labels, classes=range(4))
    roc_auc = roc_auc_score(y_test_bin, all_test_probs, average='weighted')
    ap = average_precision_score(y_test_bin, all_test_probs, average='weighted')
    
    # 打印评估结果
    print("\n测试集性能:")
    print(f"准确率: {accuracy:.4f}")
    print(f"加权F1分数: {weighted_f1:.4f}")
    print(f"加权精确率: {precision:.4f}")
    print(f"加权召回率: {recall:.4f}")
    print(f"加权ROC AUC: {roc_auc:.4f}")
    print(f"加权平均精确率: {ap:.4f}")
    
    # 打印每个类别的AUC
    print("\n每个类别的AUC:")
    for i, class_name in enumerate(actual_class_names):
        class_auc = roc_auc_score(y_test_bin[:, i], all_test_probs[:, i])
        print(f"{class_name}: {class_auc:.4f}")
    
    print("\n分类报告:")
    print(classification_report(all_test_labels, all_test_preds, 
                              labels=unique_labels,
                              target_names=actual_class_names))
    
    # 绘制评估曲线
    plot_confusion_matrix(all_test_labels, all_test_preds,
                         classes=actual_class_names,
                         title='测试集混淆矩阵')
    
    plot_roc_curve(y_test_bin, all_test_probs, actual_class_names)
    plot_precision_recall_curve(y_test_bin, all_test_probs, actual_class_names)

if __name__ == "__main__":
    evaluate_model() 