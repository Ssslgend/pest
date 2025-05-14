import torch
from torch.utils.data import DataLoader
from data.dataset import PestDataset, DataHandler
from model.bilstm import BiLSTMModel
from config.params import MODEL_CONFIG, TRAIN_CONFIG
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    f1_score, precision_recall_curve, average_precision_score,
    precision_score, recall_score, accuracy_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.preprocessing import label_binarize
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

# 创建保存图片的目录
os.makedirs('results', exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def save_plot(fig, filename):
    """保存图片的辅助函数"""
    try:
        # 确保results目录存在
        os.makedirs('results', exist_ok=True)
        
        # 构建完整的文件路径
        filepath = os.path.join('results', filename)
        
        # 保存图片
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {os.path.abspath(filepath)}")
        
        # 检查文件是否成功保存
        if os.path.exists(filepath):
            print(f"文件大小: {os.path.getsize(filepath)} 字节")
        else:
            print("警告: 文件似乎没有成功保存")
            
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
        
        # 使用辅助函数保存图片
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

def train_model():
    # 初始化组件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化数据处理器和模型
    handler = DataHandler(TRAIN_CONFIG)
    model = BiLSTMModel(MODEL_CONFIG).to(device)
    
    # 加载数据 - 使用新的带有复合特征的数据文件
    data_path = "datas/pest_rice_with_features_2_classified.csv"
    print(f"尝试加载数据: {data_path}")
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 文件 {data_path} 不存在")
        return
    
    # 加载数据
    X_train, X_val, X_test, y_train, y_val, y_test = handler.load_data(data_path)
    
    # 检查数据是否成功加载
    if X_train is None or y_train is None:
        print("数据加载失败，无法继续训练")
        return
    
    # 检查数据维度
    print(f"训练数据维度: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"验证数据维度: X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"测试数据维度: X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # 计算类别权重 - 使用更复杂的权重计算方法
    try:
        class_counts = np.bincount(y_train)
        # 使用逆频率的平方根作为权重，减少极端权重的影响
        class_weights = 1. / np.sqrt(class_counts)
        # 归一化权重
        class_weights = class_weights / np.sum(class_weights) * len(class_weights)
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"类别权重: {class_weights.cpu().numpy()}")
    except Exception as e:
        print(f"计算类别权重时出错: {e}")
        print("使用默认权重")
        criterion = nn.CrossEntropyLoss()
    
    # 使用AdamW优化器，添加权重衰减
    optimizer = AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=0.01,  # L2正则化
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 使用带重启的余弦退火学习率调度器
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=15,  # 第一次重启的周期
        T_mult=2,  # 每次重启后周期翻倍
        eta_min=1e-6
    )
    
    # 创建数据集和数据加载器
    train_dataset = PestDataset(X_train, y_train)
    val_dataset = PestDataset(X_val, y_val)
    test_dataset = PestDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 训练循环
    best_val_acc = 0
    best_val_f1 = 0
    best_val_auc = 0
    train_losses = []
    val_accuracies = []
    val_f1_scores = []
    val_auc_scores = []  # 添加验证集AUC记录
    patience = 20  # 早停耐心值
    patience_counter = 0  # 早停计数器
    
    for epoch in range(TRAIN_CONFIG["num_epochs"]):
        # 训练阶段
        model.train()
        epoch_loss = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []
        all_val_probs = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                all_val_probs.extend(probs.cpu().numpy())
        
        # 计算验证集指标
        all_val_preds = np.array(all_val_preds)
        all_val_labels = np.array(all_val_labels)
        all_val_probs = np.array(all_val_probs)
        
        val_accuracy = val_correct / val_total
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
        
        # 计算验证集AUC
        y_val_bin = label_binarize(all_val_labels, classes=range(4))
        val_auc = roc_auc_score(y_val_bin, all_val_probs, average='weighted')
        val_auc_scores.append(val_auc)
        
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)
        
        # 保存最佳模型 - 使用F1分数作为主要指标
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_accuracy
            best_val_auc = val_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_f1': val_f1,
                'val_auc': val_auc
            }, 'best_model.pth')
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= patience:
            print(f"早停触发: {patience} 个epoch没有改善")
            break
        
        # 打印训练信息
        print(f"Epoch [{epoch+1}/{TRAIN_CONFIG['num_epochs']}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Acc: {val_accuracy:.4f}, "
              f"Val F1: {val_f1:.4f}, "
              f"Val AUC: {val_auc:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # 测试阶段
    try:
        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        test_correct = 0
        test_total = 0
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
                
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                all_test_preds.extend(predicted.cpu().numpy())
                all_test_labels.extend(labels.cpu().numpy())
                all_test_probs.extend(probs.cpu().numpy())
        
        # 计算各项评估指标
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
        print("\nTest Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted F1 Score: {weighted_f1:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print(f"Weighted ROC AUC: {roc_auc:.4f}")
        print(f"Weighted Average Precision: {ap:.4f}")
        
        # 打印每个类别的AUC
        print("\nPer-class AUC:")
        for i, class_name in enumerate(actual_class_names):
            class_auc = roc_auc_score(y_test_bin[:, i], all_test_probs[:, i])
            print(f"{class_name}: {class_auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(all_test_labels, all_test_preds, 
                                  labels=unique_labels,
                                  target_names=actual_class_names))
        
        # 绘制评估曲线
        plot_confusion_matrix(all_test_labels, all_test_preds,
                             classes=actual_class_names,
                             title='Test Set Confusion Matrix')
        
        plot_roc_curve(y_test_bin, all_test_probs, actual_class_names)
        plot_precision_recall_curve(y_test_bin, all_test_probs, actual_class_names)
        
        # 绘制训练曲线
        try:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.plot(train_losses, label='Training Loss')
            plt.title('Training Loss Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 3, 2)
            plt.plot(val_accuracies, label='Validation Accuracy')
            plt.title('Validation Accuracy Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 3, 3)
            plt.plot(val_auc_scores, label='Validation AUC')
            plt.title('Validation AUC Curve')
            plt.xlabel('Epoch')
            plt.ylabel('AUC')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            
            # 使用辅助函数保存图片
            save_plot(plt.gcf(), 'training_curves.png')
            plt.close()
            
        except Exception as e:
            print(f"绘制训练曲线时出错: {e}")
            
    except Exception as e:
        print(f"测试阶段出错: {e}")

if __name__ == "__main__":
    train_model()