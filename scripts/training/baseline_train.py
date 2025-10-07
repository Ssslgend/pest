import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score, accuracy_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

from model import BiLSTMModel
from data_processor import DataProcessor

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 创建保存结果的目录
os.makedirs('baseline/results', exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def save_plot(fig, filename):
    """保存图片的辅助函数"""
    try:
        filepath = os.path.join('baseline/results', filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {os.path.abspath(filepath)}")
    except Exception as e:
        print(f"保存图片时出错: {e}")

def plot_confusion_matrix(y_true, y_pred, classes, title='混淆矩阵'):
    """绘制混淆矩阵"""
    try:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes,
                    yticklabels=classes)
        plt.title(title)
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        save_plot(plt.gcf(), 'confusion_matrix.png')
        plt.close()
    except Exception as e:
        print(f"绘制混淆矩阵时出错: {e}")

def plot_roc_curves(y_true, y_pred_proba, classes):
    """绘制ROC曲线"""
    try:
        plt.figure(figsize=(10, 8))
        
        # 计算每个类别的ROC曲线
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(len(classes)):
            fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_proba[:, i])
            roc_auc[i] = roc_auc_score(y_true == i, y_pred_proba[:, i])
        
        # 绘制每个类别的ROC曲线
        for i in range(len(classes)):
            plt.plot(fpr[i], tpr[i], label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('各风险等级的ROC曲线')
        plt.legend(loc="lower right")
        plt.tight_layout()
        save_plot(plt.gcf(), 'roc_curves.png')
        plt.close()
    except Exception as e:
        print(f"绘制ROC曲线时出错: {e}")

def plot_accuracy_curve(train_accs, val_accs, title='准确率曲线'):
    """绘制准确率曲线"""
    try:
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_accs) + 1)
        plt.plot(epochs, train_accs, 'b-', label='训练集准确率')
        plt.plot(epochs, val_accs, 'r-', label='验证集准确率')
        plt.title(title)
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_plot(plt.gcf(), 'accuracy_curve.png')
        plt.close()
    except Exception as e:
        print(f"绘制准确率曲线时出错: {e}")

def train_model():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据路径
    data_path = "datas/pest_rice_with_features_2_classified.csv"
    
    # 初始化数据处理器
    data_processor = DataProcessor(data_path)
    
    # 加载数据
    X_train, X_val, X_test, y_train, y_val, y_test = data_processor.load_data()
    
    # 获取输入维度和类别数
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = data_processor.create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test,
        batch_size=32
    )
    
    # 初始化模型
    model = BiLSTMModel(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.5
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    
    # 训练参数
    num_epochs = 100
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    # 记录训练和验证准确率
    train_accs = []
    val_accs = []
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # 更新学习率
        scheduler.step()
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        # 计算验证集准确率
        val_acc = val_correct / val_total
        
        # 记录准确率
        train_accs.append(train_correct / train_total)
        val_accs.append(val_acc)
        
        # 打印训练信息
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_correct/train_total:.4f}, '
              f'Val Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'train_accs': train_accs,
                'val_accs': val_accs
            }, 'baseline/best_model.pth')
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= patience:
            print(f"早停触发: {patience} 个epoch没有改善")
            break
    
    # 测试阶段
    print("\n测试阶段:")
    checkpoint = torch.load('baseline/best_model.pth')
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
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_test_preds.extend(predicted.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())
            all_test_probs.extend(probs.cpu().numpy())
    
    # 计算测试集指标
    test_acc = test_correct / test_total
    test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')
    test_precision = precision_score(all_test_labels, all_test_preds, average='weighted')
    test_recall = recall_score(all_test_labels, all_test_preds, average='weighted')
    roc_auc = roc_auc_score(all_test_labels, all_test_probs, multi_class='ovr')
    
    # 打印测试结果
    print(f"测试集准确率: {test_acc:.4f}")
    print(f"测试集F1分数: {test_f1:.4f}")
    print(f"测试集精确率: {test_precision:.4f}")
    print(f"测试集召回率: {test_recall:.4f}")
    print(f"测试集ROC AUC: {roc_auc:.4f}")
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(all_test_labels, all_test_preds))
    
    # 绘制混淆矩阵
    class_names = ['无风险', '低风险', '中风险', '高风险']
    plot_confusion_matrix(all_test_labels, all_test_preds, class_names)
    
    # 绘制ROC曲线
    plot_roc_curves(np.array(all_test_labels), np.array(all_test_probs), class_names)
    
    # 绘制准确率曲线
    plot_accuracy_curve(train_accs, val_accs)

if __name__ == "__main__":
    train_model() 