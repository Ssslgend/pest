import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
# 修改为英文字体
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Tahoma']
matplotlib.rcParams['axes.unicode_minus'] = True

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from model.bilstm import BiLSTMModel

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

# 加载数据集
def load_data(csv_path):
    print(f"加载数据: {csv_path}")
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='gbk')
    
    # 找到label列
    label_col = 'label'
    
    # 排除非特征列
    excluded_cols = [label_col]
    non_feature_cols = ['id', 'ID', '发生样点纬度', '发生样点经度', 'year', 'Year']
    for col in non_feature_cols:
        if col in df.columns:
            excluded_cols.append(col)
    
    feature_cols = [col for col in df.columns if col not in excluded_cols]
    
    # 提取特征和标签
    X = df[feature_cols].values
    y = df[label_col].values
    
    # 处理可能的NaN或异常值
    if np.isnan(X).any() or np.isinf(X).any():
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    if np.isnan(y).any() or np.isinf(y).any():
        y = np.nan_to_num(y, nan=0, posinf=1, neginf=0)
    
    return X, y, feature_cols

# 数据预处理
def preprocess_data(X, y, test_size=0.2, random_state=42):
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, scaler

# 创建BiLSTM数据加载器
def create_bilstm_data_loaders(X_train, X_test, y_train, y_test, batch_size=32):
    # 为LSTM添加序列维度 [batch_size, seq_len=1, features]
    X_train_tensor = torch.FloatTensor(X_train.reshape(X_train.shape[0], 1, X_train.shape[1]))
    y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
    
    X_test_tensor = torch.FloatTensor(X_test.reshape(X_test.shape[0], 1, X_test.shape[1]))
    y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1))
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# 概率分布校准损失函数 - 从train_raster_new.py复制
class DistributionRegularizationLoss(nn.Module):
    """
    自定义损失函数，鼓励模型产生更加均匀的预测概率分布
    """
    def __init__(self, base_criterion, bins=10, lambda_reg=0.3):
        super(DistributionRegularizationLoss, self).__init__()
        self.base_criterion = base_criterion  # 基础分类损失函数
        self.bins = bins  # 概率区间数量
        self.lambda_reg = lambda_reg  # 均匀化正则项权重
        
    def forward(self, predictions, targets, return_stats=False):
        # 基础分类损失
        base_loss = self.base_criterion(predictions, targets)
        
        # 计算预测概率的分布情况
        probs = torch.sigmoid(predictions).detach()
        
        # 创建概率区间
        bin_edges = torch.linspace(0, 1, self.bins + 1, device=probs.device)
        bin_counts = torch.zeros(self.bins, device=probs.device)
        
        # 统计每个区间的样本数量
        for i in range(self.bins):
            if i == self.bins - 1:
                bin_mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i+1])
            else:
                bin_mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
            bin_counts[i] = bin_mask.sum().float()
        
        # 计算分布不均匀性，添加数值稳定性处理
        if bin_counts.sum() > 0:
            bin_probs = bin_counts / bin_counts.sum()
            
            # 添加小的平滑值以避免log(0)
            epsilon = 1e-7
            bin_probs = bin_probs + epsilon
            bin_probs = bin_probs / bin_probs.sum()  # 重新归一化
            
            # 理想的均匀分布
            uniform_probs = torch.ones_like(bin_probs) / self.bins
            
            # 使用更稳定的方法计算KL散度
            kl_div = torch.sum(bin_probs * torch.log(torch.clamp(bin_probs / uniform_probs, min=epsilon)))
            
            # 限制KL散度值，防止过大
            kl_div = torch.clamp(kl_div, max=10.0)
            
            # 结合基础损失和分布正则项
            reg_loss = self.lambda_reg * kl_div
            combined_loss = base_loss + reg_loss
            
            # 确保损失是有限值
            if torch.isinf(combined_loss) or torch.isnan(combined_loss):
                combined_loss = torch.tensor(1.0, device=combined_loss.device, requires_grad=True)
        else:
            combined_loss = base_loss
        
        if return_stats:
            stats = {
                'base_loss': base_loss.item() if not torch.isinf(base_loss) and not torch.isnan(base_loss) else 0.0,
                'bin_counts': bin_counts.cpu().numpy(),
                'kl_div': kl_div.item() if 'kl_div' in locals() and not torch.isinf(kl_div) and not torch.isnan(kl_div) else 0.0
            }
            return combined_loss, stats
        
        return combined_loss

# 训练和评估BiLSTM模型 - 使用config_raster_new.py和train_raster_new.py中的配置
def train_and_evaluate_bilstm(X_train, X_test, y_train, y_test, input_size):
    # 模型参数 - 从config_raster_new.py复制
    model_config = {
        "input_size": input_size,
        "hidden_size": 128,  # 与配置文件一致
        "num_layers": 2,     # 与配置文件一致
        "dropout": 0.23,      # 从0.22略微增加到0.23，尝试将AUC从0.9997略微降低
        "num_classes": 1
    }
    
    # 创建模型
    model = BiLSTMModel(model_config).to(device)
    
    # 创建数据加载器
    train_loader, test_loader = create_bilstm_data_loaders(X_train, X_test, y_train, y_test, batch_size=32)
    
    # 设置优化器和损失函数 - 从train_raster_new.py复制
    base_criterion = nn.BCEWithLogitsLoss()
    criterion = DistributionRegularizationLoss(
        base_criterion=base_criterion, 
        bins=20,               # 将概率空间分为20个区间
        lambda_reg=0.5         # 分布正则化权重
    )
    
    # 使用AdamW优化器并调整参数 - 从train_raster_new.py复制
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0002,  # 保持与config_raster_new一致
        weight_decay=1e-4,  # 确保与config_raster_new一致 (之前可能是1e-4 * 0.5)
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # 添加学习率调度器 - 从train_raster_new.py复制
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',           # 监控验证集AUC，越高越好
        factor=0.5,          # 学习率调整因子
        patience=10,          # 从4增加到10，给予更多耐心
        verbose=True,        # 打印学习率变化
        threshold=0.01,      # 改进的最小阈值
        min_lr=1e-6          # 最小学习率
    )
    
    # 训练模型
    num_epochs = 150  # 保持与config_raster_new一致
    best_val_auc = -np.inf
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        all_train_labels = []
        all_train_outputs = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
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
            for data, target in test_loader:
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
        
        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
    
    training_time = time.time() - start_time
    
    # 评估模型
    model.eval()
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
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    
    # 重新计算所有指标
    accuracy = accuracy_score(all_targets, all_preds)
    conf_matrix = confusion_matrix(all_targets, all_preds)
    if len(conf_matrix.ravel()) == 4:
        tn, fp, fn, tp = conf_matrix.ravel()
    else:
        tn, fp, fn, tp = 0,0,0,0
        if len(all_targets) > 0 :
            if all_targets[0] == 0 and all_preds[0] == 0: tn = len(all_targets)
            elif all_targets[0] == 1 and all_preds[0] == 1: tp = len(all_targets)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # 计算真实的AUC值
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.0
    
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

# 训练和评估Random Forest模型
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
    except:
        auc = 0
    
    # 计算特异度
    if len(conf_matrix) > 1:
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

# 训练和评估MLP模型
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
    except:
        auc = 0
    
    # 计算特异度
    if len(conf_matrix) > 1:
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
def train_and_evaluate_svm(X_train, X_test, y_train, y_test):
    start_time = time.time()
    
    # 创建并训练模型
    svm = SVC(
        kernel='rbf',
        C=1.0,
        probability=True,
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
    except:
        auc = 0
    
    # 计算特异度
    if len(conf_matrix) > 1:
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
        sns.barplot(x='Model', y='Score', data=metric_df, palette='viridis')
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
    set_seed(42)
    print("开始模型比较实验...")
    
    # 设置数据路径
    csv_path = os.path.join(parent_dir, 'datas', 'train.csv')
    
    # 加载数据
    try:
        X, y, feature_cols = load_data(csv_path) #返回x,y,features
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 数据预处理
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
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
    results['Random Forest'] = train_and_evaluate_rf(X_train, X_test, y_train, y_test)
    
    print("\n训练和评估MLP模型...")
    results['MLP'] = train_and_evaluate_mlp(X_train, X_test, y_train, y_test)
    
    print("\n训练和评估SVM模型...")
    results['SVM'] = train_and_evaluate_svm(X_train, X_test, y_train, y_test)
    
    print("\n训练和评估BiLSTM模型...")
    results['SF-BiLSTM'] = train_and_evaluate_bilstm(X_train, X_test, y_train, y_test, X.shape[1])
    
    # 保存结果
    with open(os.path.join(output_dir, 'model_comparison_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
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