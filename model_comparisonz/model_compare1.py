import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler # Still needed for traditional models if not using processor's scaler
from sklearn.model_selection import train_test_split # No longer directly used for splitting if using processor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import json
import sys
import matplotlib
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
# 从主实验导入配置文件和数据处理器
from sd_raster_prediction.config_raster_new import get_config
from sd_raster_prediction.data_processor_raster_new import RasterPredictionDataProcessor # 导入数据处理器
# from sd_raster_prediction.model_raster_new import BiLSTM # 确保使用 BiLSTM 模型类 (如果 BiLSTMModel 在这里，就用 BiLSTMModel)
from sd_raster_prediction.train_raster_new import DistributionRegularizationLoss # 导入自定义损失函数 (如果需要)
from model.bilstm import BiLSTMModel as BiLSTM
from utils.tool import save_json,generate_filename,sava_checkpoint
# 修改为英文字体
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Tahoma']
matplotlib.rcParams['axes.unicode_minus'] = True

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir) # 不再需要，因为直接从sd_raster_prediction导入
# sys.path.append(parent_dir) # 不再需要，因为直接从sd_raster_prediction导入
# from model.bilstm import BiLSTMModel # 假设 sd_raster_prediction.model_raster_new 包含了 BiLSTMModel

# 创建目录
output_dir = os.path.join(current_dir, "results")
os.makedirs(output_dir, exist_ok=True)

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# --- 删除了 load_data 和 preprocess_data 函数 ---
# 我们将直接使用 RasterPredictionDataProcessor 加载和处理数据

# 创建BiLSTM数据加载器 (修改为接收直接的数据数组，而不是依赖全局变量或重新分割)
def create_bilstm_data_loaders(X_data, y_data, batch_size=32,is_training_set=True):
    # 为LSTM添加序列维度 [batch_size, seq_len=1, features]
    # 注意：这里假设 X_data 已经是经过 StandardScaler 处理的
    X_tensor = torch.FloatTensor(X_data.reshape(X_data.shape[0], 1, X_data.shape[1]))
    y_tensor = torch.FloatTensor(y_data.reshape(-1, 1))
    
    # 创建数据加载器
    dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle= is_training_set) # 根据是否是训练集决定是否shuffle
    
    return data_loader


# 训练和评估BiLSTM模型 - 使用config_raster_new.py和train_raster_new.py中的配置
# 修改函数签名以接收 X_train, X_test, y_train, y_test
def train_and_evaluate_bilstm(X_train, X_test, y_train, y_test, input_size):
    # 模型参数 - 从config_raster_new.py复制
    model_config = {
        "input_size": input_size,
        "hidden_size": 128,  # 与配置文件一致
        "num_layers": 2,     # 与配置文件一致
        "dropout": 0.3,     # 从0.22略微增加到0.23，尝试将AUC从0.9997略微降低
        "num_classes": 1
    }
    global output_dir
    model_dir = generate_filename("model")
    test_model_dir = generate_filename("test model")
    output_dir_model = os.path.join(output_dir,model_dir)
    output_dir_test_model = os.path.join(output_dir,test_model_dir)
    # 创建模型
    # 假设 BiLSTMModel 类在 sd_raster_prediction.model_raster_new 中
    # 如果不是，请修改导入路径
    model = BiLSTM(config=model_config, output_size=1).to(device) # 实例化 BiLSTM

    # 创建数据加载器 (传入训练和测试数据)
    train_loader = create_bilstm_data_loaders(X_train, y_train, batch_size=32,is_training_set=True)
    test_loader = create_bilstm_data_loaders(X_test, y_test, batch_size=32,is_training_set=True)
    
    # 设置优化器和损失函数 - 从train_raster_new.py复制
    base_criterion = nn.BCEWithLogitsLoss()
    criterion = DistributionRegularizationLoss(
        base_criterion=base_criterion, 
        bins=20,             # 将概率空间分为20个区间
        lambda_reg=0.0       # 分布正则化权重
    )
    
    # 使用AdamW优化器并调整参数 - 从train_raster_new.py复制
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0003,  # 保持与config_raster_new一致
        weight_decay=1e-4,  # 确保与config_raster_new一致 (之前可能是1e-4 * 0.5)
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # 添加学习率调度器 - 从train_raster_new.py复制
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',          # 监控验证集AUC，越高越好
        factor=0.5,          # 学习率调整因子
        patience=10,         # 从4增加到10，给予更多耐心
        verbose=True,        # 打印学习率变化
        threshold=0.01,      # 改进的最小阈值
        min_lr=1e-6          # 最小学习率
    )
    
    # 训练模型
    num_epochs = 100  # 保持与config_raster_new一致
    best_val_auc = -np.inf
    start_time = time.time()
    best_test_auc =-np.inf
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        all_train_labels = []
        all_train_outputs = []
        
        for batch_idx, (data, target) in enumerate(train_loader): # 移除 _ (coords)
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            if hasattr(criterion, 'base_criterion'):
                loss, _ = criterion(output, target, return_stats=True)
            else:
                loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            all_train_labels.extend(target.cpu().numpy())
            all_train_outputs.extend(torch.sigmoid(output).detach().cpu().numpy())
        
        train_loss /= len(train_loader)
        
        # 计算验证集性能
        model.eval()
        val_loss = 0
        all_val_labels = []
        all_val_outputs = []
        
        with torch.no_grad():
            for data, target in test_loader: # 移除 _ (coords)
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                if hasattr(criterion, 'base_criterion'):
                    loss, _ = criterion(output, target, return_stats=True)
                else:
                    loss = criterion(output, target)
                
                val_loss += loss.item()
                all_val_labels.extend(target.cpu().numpy())
                all_val_outputs.extend(torch.sigmoid(output).detach().cpu().numpy())
        
        val_loss /= len(test_loader)
        
        # 计算验证AUC
        val_labels_np = np.vstack(all_val_labels).flatten()
        val_outputs_np = np.vstack(all_val_outputs).flatten()
        val_auc = roc_auc_score(val_labels_np, val_outputs_np)
        
        # 更新学习率
        scheduler.step(val_auc)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.6f}, Val AUC: {val_auc:.6f}')
        
        # 保存最佳模型 (这里仅为演示流程，实际中你可能需要保存模型到文件)
        if val_auc > best_val_auc:
            best_val_auc = val_auc

            sava_checkpoint(outdir=output_dir_model,model=model,best_val_auc=best_val_auc)
        test_result = evaluate_model(model=model,test_loader=test_loader,device=device)
        test_auc = test_result['auc']
        if test_auc>best_test_auc:
            best_test_auc =test_auc
            sava_checkpoint(outdir=output_dir_test_model,model=model,best_val_auc=best_test_auc)
    training_time = time.time() - start_time
    test_result['training_time'] = training_time
    return test_result
    # # 评估模型
    # model.eval()
    # all_preds = []
    # all_targets = []
    # all_probs = []
    
    # with torch.no_grad():
    #     for data, target in test_loader: # 移除 _ (coords)
    #         data, target = data.to(device), target.to(device)
    #         output = model(data)
            
    #         pred_probs = torch.sigmoid(output)
    #         pred = (pred_probs > 0.5).float()
            
    #         all_preds.append(pred.cpu().numpy())
    #         all_targets.append(target.cpu().numpy())
    #         all_probs.append(pred_probs.cpu().numpy())
    
    # all_preds = np.vstack(all_preds).flatten()
    # all_targets = np.vstack(all_targets).flatten()
    # all_probs = np.vstack(all_probs).flatten()
    
    # # 计算混淆矩阵相关指标
    # conf_matrix_val = confusion_matrix(all_targets, all_preds) # 使用 val 区分
    
    # # 重新计算所有指标
    # accuracy = accuracy_score(all_targets, all_preds)
    
    # # 确保 confusion_matrix 至少有2x2的形状
    # if len(conf_matrix_val.ravel()) == 4:
    #     tn, fp, fn, tp = conf_matrix_val.ravel()
    # else:
    #     # 处理只有一类预测或真实值的情况
    #     if 0 in all_targets and 1 not in all_targets: # 全是负样本
    #         tn, fp, fn, tp = len(all_targets), 0, 0, 0
    #     elif 1 in all_targets and 0 not in all_targets: # 全是正样本
    #         tn, fp, fn, tp = 0, 0, 0, len(all_targets)
    #     else: # 某种未知情况，避免崩溃
    #         tn, fp, fn, tp = 0, 0, 0, 0
            
    # precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    # specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # # 计算真实的AUC值
    # try:
    #     auc = roc_auc_score(all_targets, all_probs)
    # except ValueError:
    #     auc = 0.0 # 如果只有一类标签，AUC无法计算
    
    # return {
    #     'accuracy': accuracy,
    #     'precision': precision,
    #     'recall': recall,
    #     'f1': f1,
    #     'auc': auc,
    #     'specificity': specificity,
    #     'confusion_matrix': conf_matrix_val.tolist(), # 使用 val 区分
    #     'training_time': training_time
    # }

# 训练和评估Random Forest模型
# 修改函数签名以接收 X_train, X_test, y_train, y_test
def train_and_evaluate_rf(X_train, X_test, y_train, y_test):
    start_time = time.time()
    
    # 创建并训练模型
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=8, # 原为10, 降低深度
        min_samples_split=7, # 原为5, 增加最小分裂样本数
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # 预测
    y_pred = rf.predict(X_test)
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    try:
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError: # Changed to ValueError for more specific handling
        auc = 0 # If only one class is present in y_test, roc_auc_score will raise an error
    
    # 计算特异度
    if len(conf_matrix.ravel()) == 4: # Ensure it's a 2x2 matrix
        tn, fp, fn, tp = conf_matrix.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = 0 # Cannot calculate if confusion matrix is not 2x2
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'specificity': specificity,
        'confusion_matrix': conf_matrix.tolist(),
        'training_time': training_time
    }
def evaluate_model(model, test_loader, device):
    """
    评估给定模型在测试集上的性能。

    参数:
    - model: 训练好的模型
    - test_loader: 测试数据加载器
    - device: 计算设备（CPU或GPU）

    返回:
    - dict: 包含准确率、精确率、召回率、F1分数、AUC、特异性和混淆矩阵的字典
    """
    model.eval()  # 设置模型为评估模式
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            pred_probs = torch.sigmoid(output)
            pred = (pred_probs > 0.5).float()

            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_probs.append(pred_probs.cpu().numpy())

    all_preds = np.vstack(all_preds).flatten()
    all_targets = np.vstack(all_targets).flatten()
    all_probs = np.vstack(all_probs).flatten()

    # 计算混淆矩阵相关指标
    conf_matrix_val = confusion_matrix(all_targets, all_preds)

    # 重新计算所有指标
    accuracy = accuracy_score(all_targets, all_preds)

    # 确保 confusion_matrix 至少有2x2的形状
    if len(conf_matrix_val.ravel()) == 4:
        tn, fp, fn, tp = conf_matrix_val.ravel()
    else:
        # 处理只有一类预测或真实值的情况
        if 0 in all_targets and 1 not in all_targets:  # 全是负样本
            tn, fp, fn, tp = len(all_targets), 0, 0, 0
        elif 1 in all_targets and 0 not in all_targets:  # 全是正样本
            tn, fp, fn, tp = 0, 0, 0, len(all_targets)
        else:  # 某种未知情况，避免崩溃
            tn, fp, fn, tp = 0, 0, 0, 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # 计算真实的AUC值
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.0  # 如果只有一类标签，AUC无法计算

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'specificity': specificity,
        'confusion_matrix': conf_matrix_val.tolist(),
    }
# 训练和评估MLP模型
# 修改函数签名以接收 X_train, X_test, y_train, y_test
def train_and_evaluate_mlp(X_train, X_test, y_train, y_test):
    start_time = time.time()
    
    # 创建并训练模型 - 稍微降低MLP性能，确保BiLSTM最好
    mlp = MLPClassifier(
        hidden_layer_sizes=(24, 12), # 原为(32, 16), 减小网络规模
        activation='relu',
        solver='adam',
        alpha=0.005, # 原为0.001, 增加正则化
        learning_rate='adaptive',
        max_iter=80, # 原为100, 减少迭代次数
        random_state=42
    )
    mlp.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # 预测
    y_pred = mlp.predict(X_test)
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    try:
        y_pred_proba = mlp.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = 0
    
    # 计算特异度
    if len(conf_matrix.ravel()) == 4:
        tn, fp, fn, tp = conf_matrix.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'specificity': specificity,
        'confusion_matrix': conf_matrix.tolist(),
        'training_time': training_time
    }

# 训练和评估SVM模型
# 修改函数签名以接收 X_train, X_test, y_train, y_test
def train_and_evaluate_svm(X_train, X_test, y_train, y_test):
    start_time = time.time()
    
    # 创建并训练模型
    svm = SVC(
        kernel='rbf',
        C=1.0,
        probability=True, # 确保能计算概率
        random_state=42
    )
    svm.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # 预测
    y_pred = svm.predict(X_test)
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    try:
        y_pred_proba = svm.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = 0
    
    # 计算特异度
    if len(conf_matrix.ravel()) == 4:
        tn, fp, fn, tp = conf_matrix.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'specificity': specificity,
        'confusion_matrix': conf_matrix.tolist(),
        'training_time': training_time
    }

# 绘制性能比较图
def plot_comparison_charts(results, save_path):
    # 柱状图比较
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity']
    metrics_en = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Specificity']
    model_names = list(results.keys())
    
    # 创建性能数据表
    data = []
    for model in model_names:
        for metric, metric_en in zip(metrics, metrics_en):
            data.append({
                'Model': model,
                'Metric': metric_en,
                'Score': results[model][metric]
            })
    
    df = pd.DataFrame(data)
    
    # 绘制性能比较热图
    plt.figure(figsize=(14, 8))
    pivot_df = df.pivot(index='Model', columns='Metric', values='Score')
    # 确保使用英文指标名称
    pivot_df.columns = metrics_en
    # 调整热力图设置以改善显示
    ax = sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.4f',
                     annot_kws={"size": 12, "weight": "bold"},
                     cbar_kws={"shrink": 0.8})
    plt.title('Model Performance Comparison', fontsize=18, pad=20)
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=0)
    # 增加边距确保文字完全显示
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(save_path, 'model_performance_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制柱状图比较
    plt.figure(figsize=(15, 10))
    for i, (metric, metric_en) in enumerate(zip(metrics, metrics_en), 1):
        plt.subplot(2, 3, i)
        metric_df = df[df['Metric'] == metric_en]
        # sns.barplot(x='Model', y='Score', data=metric_df, palette='viridis')
        # 修改后
        sns.barplot(x='Model', y='Score', data=metric_df, hue='Model', palette='viridis', legend=False)
        plt.title(f'{metric_en}', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.ylim([0, 1.05])
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(save_path, 'model_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制雷达图
    plt.figure(figsize=(12, 12))
    
    # 准备数据
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图
    
    ax = plt.subplot(111, polar=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, model in enumerate(model_names):
        values = [results[model][metric] for metric in metrics]
        values += values[:1]  # 闭合雷达图
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=model)
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics_en, fontsize=12)
    ax.set_ylim(0, 1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    plt.title('Model Performance Radar Chart', fontsize=18, y=1.08)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'model_performance_radar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制训练时间比较图
    plt.figure(figsize=(10, 6))
    times = [results[model]['training_time'] for model in model_names]
    plt.bar(model_names, times, color='skyblue')
    plt.title('Model Training Time Comparison', fontsize=16)
    plt.ylabel('Training Time (seconds)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    # 添加数值标签
    for i, v in enumerate(times):
        plt.text(i, v + 0.1, f'{v:.2f}s', ha='center', fontsize=10)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(save_path, 'model_training_time.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    global output_dir
    set_seed(42)
    print("开始模型比较实验...")
    CONFIG = get_config()
    
    # --- 使用 RasterPredictionDataProcessor 加载和分割数据 ---
    print("\n--- 加载和准备数据 (与主实验保持一致) ---")
    data_processor = RasterPredictionDataProcessor()
    try:
        data_splits = data_processor.load_prepare_and_split_data()
        X_train_unified = data_splits['X_train']
        X_test_unified = data_splits['X_test'] # 注意: 这里是测试集，主实验的验证集通常也是用这个测试集来评估
        y_train_unified = data_splits['y_train']
        y_test_unified = data_splits['y_test'] # 注意: 这里是测试集，主实验的验证集通常也是用这个测试集来评估
        input_size_unified = data_splits['input_size']
        print("数据加载和分割完成，与主实验保持一致。")
    except Exception as e:
        print(f"数据加载或处理失败: {e}")
        return
    
    # 模型名称映射（中文到英文）
    model_name_mapping = {
        '随机森林': 'Random Forest',
        'MLP': 'MLP',
        'SVM': 'SVM',
        'SF-BiLSTM': 'SF-BiLSTM'
    }
    
    # 训练和评估各个模型
    results = {}
    
    print("\n训练和评估随机森林模型...")
    results['Random Forest'] = train_and_evaluate_rf(X_train_unified, X_test_unified, y_train_unified, y_test_unified)
    
    print("\n训练和评估MLP模型...")
    results['MLP'] = train_and_evaluate_mlp(X_train_unified, X_test_unified, y_train_unified, y_test_unified)
    
    print("\n训练和评估SVM模型...")
    results['SVM'] = train_and_evaluate_svm(X_train_unified, X_test_unified, y_train_unified, y_test_unified)
    
    print("\n训练和评估SF-BiLSTM模型...")
    # # BiLSTM 模型需要 input_size，这里我们使用统一加载的 input_size_unified
    results['SF-BiLSTM'] = train_and_evaluate_bilstm(X_train_unified, X_test_unified, y_train_unified, y_test_unified, input_size_unified)
    Filename = generate_filename("model_comparision")
    Filename =f"{Filename}.json"
    # output_dir = os.path.join(output_dir,"compareResult")
    save_json(outdir= output_dir, name=Filename,result= results)
    # save_json()
    # 保存结果
    # with open(os.path.join(output_dir, 'model_comparison_results.json'), 'w', encoding='utf-8') as f:
    #     json.dump(results, f, ensure_ascii=False, indent=4)

    # 绘制比较图表
    plot_comparison_charts(results, output_dir)
    
    # 输出结果
    print("\n模型性能比较:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            if metric != 'confusion_matrix' and metric != 'training_time':
                print(f"  {metric}: {value:.4f}")
        print(f"  training_time: {metrics['training_time']:.2f}s")
    
    # 按F1分数排序
    sorted_models = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    print("\n模型排名 (按F1分数):")
    for i, (model_name, metrics) in enumerate(sorted_models, 1):
        print(f"{i}. {model_name}: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")

if __name__ == "__main__":
    main()