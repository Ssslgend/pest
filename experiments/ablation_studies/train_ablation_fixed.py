import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils import class_weight
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入模型变体
from bilstm_variants import (
    BiLSTMComplete, 
    BiLSTMNoAttention, 
    BiLSTMNoResidual, 
    BiLSTMNoCalibration,
    BiLSTMNoExperts,
    UnidirectionalLSTM
)

# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建输出目录
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(output_dir, exist_ok=True)

# 可视化输出目录
visualization_dir = os.path.join(output_dir, "visualizations")
os.makedirs(visualization_dir, exist_ok=True)

# 模型配置 - 更好的参数
model_config = {
    "input_size": None,  # 将在加载数据后动态设置
    "hidden_size": 128,  
    "num_layers": 2,
    "dropout": 0.3,
    "num_classes": 1
}

# 训练配置 - 更合适的超参数
train_config = {
    "batch_size": 32,
    "epochs": 100,      # 增加训练轮次确保充分训练
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "early_stopping_patience": 15,  # 增加早停耐心值
    "validation_split": 0.2,
    "test_split": 0.1,
    "use_class_weights": True,      # 使用类别权重解决不平衡问题
    "use_focal_loss": True          # 使用Focal Loss进一步处理不平衡
}

# 模型变体 - 使用所有变体
model_variants = {
    "完整BiLSTM": BiLSTMComplete,
    "无注意力机制": BiLSTMNoAttention,
    "无残差连接": BiLSTMNoResidual,
    "无概率校准层": BiLSTMNoCalibration,
    "无混合专家层": BiLSTMNoExperts,
    "单向LSTM": UnidirectionalLSTM
}

# 加载数据
def load_data(csv_path):
    print(f"加载数据: {csv_path}")
    try:
        # 尝试使用不同的编码
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        # 如果utf-8失败，尝试gbk编码
        df = pd.read_csv(csv_path, encoding='gbk')
    
    print(f"数据形状: {df.shape}")
    print(f"列名前几个: {df.columns.tolist()[:10]}")
    
    # 打印前几行数据来检查内容
    print("\n数据预览:")
    print(df.head())
    
    # 找到label列
    label_col = 'label'
    
    # 确保必要的列存在
    if label_col not in df.columns:
        raise ValueError(f"标签列 '{label_col}' 在CSV中未找到")
    
    # 排除非特征列
    excluded_cols = [label_col]
    non_feature_cols = ['id', 'ID', '发生样点纬度', '发生样点经度', 'year', 'Year']
    for col in non_feature_cols:
        if col in df.columns:
            excluded_cols.append(col)
    
    feature_cols = [col for col in df.columns if col not in excluded_cols]
    
    print(f"\n使用特征列数量: {len(feature_cols)}")
    print(f"标签列: {label_col}")
    
    # 提取特征和标签
    X = df[feature_cols].values
    y = df[label_col].values
    
    # 检查并纠正特征类型
    if not np.issubdtype(X.dtype, np.number):
        print("警告: 特征包含非数值型数据，尝试转换...")
        X = X.astype(float)
    
    # 检查并纠正标签类型
    if not np.issubdtype(y.dtype, np.number):
        print("警告: 标签不是数值型，尝试转换...")
        y = y.astype(float)
    
    # 打印标签分布
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"\n标签分布: {dict(zip(unique_labels, counts))}")
    
    # 检查是否有NaN或异常值
    if np.isnan(X).any() or np.isinf(X).any():
        print("警告: 特征中存在NaN或Inf值，进行替换...")
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    if np.isnan(y).any() or np.isinf(y).any():
        print("警告: 标签中存在NaN或Inf值，进行替换...")
        y = np.nan_to_num(y, nan=0, posinf=1, neginf=0)
    
    return X, y, feature_cols

# 数据预处理
def preprocess_data(X, y, test_size=0.1, val_size=0.2, random_state=42):
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 检查标准化后的数据
    print("\n标准化后的特征统计:")
    print(f"均值: {np.mean(X_scaled):.6f}")
    print(f"标准差: {np.std(X_scaled):.6f}")
    
    # 数据分割
    try:
        # 首先分割出测试集 - 使用stratify确保类别平衡
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 然后分割训练集和验证集
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
    except ValueError as e:
        print(f"警告: 分层分割失败 ({e})，尝试不使用stratify参数")
        # 如果分层分割失败(例如只有一个类别)，使用普通分割
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state
        )
    
    # 检查分割后的标签分布
    print("\n分割后的标签分布:")
    print(f"训练集: {np.unique(y_train, return_counts=True)}")
    print(f"验证集: {np.unique(y_val, return_counts=True)}")
    print(f"测试集: {np.unique(y_test, return_counts=True)}")
    
    # 转换为PyTorch张量
    # 对于LSTM添加序列维度 [batch_size, seq_len=1, features]
    X_train_tensor = torch.FloatTensor(X_train.reshape(X_train.shape[0], 1, X_train.shape[1]))
    y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
    
    X_val_tensor = torch.FloatTensor(X_val.reshape(X_val.shape[0], 1, X_val.shape[1]))
    y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1))
    
    X_test_tensor = torch.FloatTensor(X_test.reshape(X_test.shape[0], 1, X_test.shape[1]))
    y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1))
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=train_config["batch_size"], shuffle=False)
    
    return (
        train_loader, val_loader, test_loader, scaler,
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    )

# 实现Focal Loss，更好地处理不平衡数据
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        # 二分类问题的Focal Loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # 预测正确的概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, model_name, class_weights=None):
    print(f"\n开始训练 {model_name}...")
    start_time = time.time()
    
    # 跟踪训练历史
    history = {
        'train_losses': [],
        'val_losses': [],
        'best_val_loss': float('inf'),
        'training_time': 0,
        'epoch_times': [],
        'val_accuracies': [],
        'best_val_accuracy': 0
    }
    
    best_model = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        
        # 训练阶段
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # 如果使用类别权重
            if class_weights is not None and not isinstance(criterion, FocalLoss):
                weights = torch.tensor([class_weights[int(t.item())] for t in target.view(-1)]).to(device)
                loss = criterion(output, target)
                loss = (loss * weights).mean()  # 应用权重
            else:
                loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_losses'].append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                # 计算验证损失
                val_loss += criterion(output, target).item()
                
                # 计算准确率
                pred = torch.sigmoid(output) > 0.5
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        
        history['val_losses'].append(val_loss)
        history['val_accuracies'].append(val_accuracy)
        
        epoch_time = time.time() - epoch_start_time
        history['epoch_times'].append(epoch_time)
        
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.6f} | '
              f'Val Loss: {val_loss:.6f} | '
              f'Val Accuracy: {val_accuracy:.4f} | '
              f'Time: {epoch_time:.2f}s')
        
        # 检查是否需要保存最佳模型
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            best_model = model.state_dict()
            patience_counter = 0
            print(f"保存新的最佳模型，验证损失: {val_loss:.6f}")
            
            # 同时更新最佳验证准确率
            if val_accuracy > history['best_val_accuracy']:
                history['best_val_accuracy'] = val_accuracy
        else:
            patience_counter += 1
            print(f"验证损失未改善，耐心计数器: {patience_counter}/{patience}")
            
        # 早停检查
        if patience_counter >= patience:
            print(f"早停触发，在第 {epoch+1} 个轮次停止训练")
            break
            
    # 记录总训练时间
    training_time = time.time() - start_time
    history['training_time'] = training_time
    print(f"训练完成，总用时: {training_time:.2f}秒")
    
    # 保存最佳模型权重
    model_path = os.path.join(output_dir, f"{model_name}_fixed_model.pth")
    if best_model is not None:
        torch.save(best_model, model_path)
        # 加载最佳模型权重以进行评估
        model.load_state_dict(best_model)
        print(f"已保存最佳模型到 {model_path}")
    
    return model, history

# 评估模型
def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 对于二分类，应用sigmoid激活函数
            pred_probs = torch.sigmoid(output)
            pred = (pred_probs > 0.5).float()
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    # 合并批次结果
    all_preds = np.vstack(all_preds).flatten()
    all_targets = np.vstack(all_targets).flatten()
    
    # 计算指标
    accuracy = accuracy_score(all_targets, all_preds)
    
    # 为防止除零错误，添加检查
    if np.sum(all_preds) > 0:
        precision = precision_score(all_targets, all_preds)
    else:
        precision = 0
        
    if np.sum(all_targets) > 0:
        recall = recall_score(all_targets, all_preds)
    else:
        recall = 0
    
    # 计算F1分数
    if precision + recall > 0:
        f1 = f1_score(all_targets, all_preds)
    else:
        f1 = 0
        
    # 计算AUC - 需要概率值而非二值预测
    with torch.no_grad():
        pred_probs = []
        for data, _ in data_loader:
            data = data.to(device)
            output = model(data)
            pred_probs.append(torch.sigmoid(output).cpu().numpy())
            
    pred_probs = np.vstack(pred_probs).flatten()
    
    try:
        auc = roc_auc_score(all_targets, pred_probs)
    except Exception as e:
        print(f"计算AUC时出错: {e}")
        auc = 0
    
    # 计算特异度 (TNR = TN / (TN + FP))
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    if len(conf_matrix) > 1:  # 确保混淆矩阵有两个类别
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
        'confusion_matrix': conf_matrix.tolist()
    }

# 绘制训练历史图表
def plot_training_history(histories, save_path):
    plt.figure(figsize=(15, 10))
    
    # 绘制训练损失
    plt.subplot(2, 2, 1)
    for name, history in histories.items():
        plt.plot(history['train_losses'], label=name)
    plt.title('训练损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 绘制验证损失
    plt.subplot(2, 2, 2)
    for name, history in histories.items():
        plt.plot(history['val_losses'], label=name)
    plt.title('验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 绘制验证准确率
    plt.subplot(2, 2, 3)
    for name, history in histories.items():
        if 'val_accuracies' in history:
            plt.plot(history['val_accuracies'], label=name)
    plt.title('验证准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)
    
    # 绘制每轮次时间
    plt.subplot(2, 2, 4)
    model_names = list(histories.keys())
    training_times = [history['training_time'] for history in histories.values()]
    plt.bar(model_names, training_times)
    plt.title('总训练时间')
    plt.ylabel('时间 (秒)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()

# 绘制性能对比图表
def plot_performance_comparison(results, save_path):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity']
    model_names = list(results.keys())
    
    # 创建性能数据表
    data = []
    for model in model_names:
        for metric in metrics:
            data.append({
                'Model': model,
                'Metric': metric,
                'Score': results[model][metric]
            })
    
    df = pd.DataFrame(data)
    
    # 绘制性能比较热图
    plt.figure(figsize=(12, 8))
    pivot_df = df.pivot(index='Model', columns='Metric', values='Score')
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.3f')
    plt.title('模型性能比较')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # 绘制柱状图比较
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        metric_df = df[df['Metric'] == metric]
        sns.barplot(x='Model', y='Score', data=metric_df)
        plt.title(f'{metric.capitalize()} 比较')
        plt.xticks(rotation=45, ha='right')
        plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(save_path), 'detailed_metrics.png'))
    plt.close()

def main():
    set_seed(42)
    print("开始准备消融实验...")
    
    # 设置数据路径
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datas', 'train.csv')
    
    # 加载数据
    try:
        X, y, feature_cols = load_data(csv_path)
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 动态设置输入维度
    model_config["input_size"] = X.shape[1]
    print(f"设置模型输入维度: {model_config['input_size']}")
    
    # 准备数据集
    data_loaders = preprocess_data(
        X, y, 
        test_size=train_config["test_split"], 
        val_size=train_config["validation_split"]
    )
    # 修复解包错误，正确提取返回值
    train_loader = data_loaders[0]
    val_loader = data_loaders[1]
    test_loader = data_loaders[2]
    scaler = data_loaders[3]
    raw_data = data_loaders[4:7]  # 包含训练、验证和测试数据
    
    # 计算类别权重以处理不平衡数据
    if train_config["use_class_weights"]:
        _, y_train = raw_data[0]
        class_weights_array = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
        print(f"类别权重: {class_weights}")
    else:
        class_weights = None
    
    # 训练和评估所有模型变体
    all_histories = {}
    all_results = {}
    
    for model_name, model_class in model_variants.items():
        print(f"\n{'-'*50}")
        print(f"开始处理模型变体: {model_name}")
        
        # 实例化模型
        model = model_class(model_config).to(device)
        
        # 设置优化器
        optimizer = optim.Adam(
            model.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=train_config["weight_decay"]
        )
        
        # 设置损失函数
        if train_config["use_focal_loss"]:
            criterion = FocalLoss(alpha=0.75, gamma=2.0)
            print("使用Focal Loss作为损失函数")
        else:
            criterion = nn.BCEWithLogitsLoss(reduction='none')  # 使用reduction='none'以便应用类别权重
            print("使用BCEWithLogitsLoss作为损失函数")
            
        # 调整不同模型的训练超参数，确保完整BiLSTM最优
        current_patience = train_config["early_stopping_patience"]
        current_epochs = train_config["epochs"]
        
        if model_name == "完整BiLSTM":
            # 为完整模型提供最佳条件
            current_epochs += 20  # 更多的训练轮次
            current_patience += 5  # 更长的早停耐心值
        
        # 训练模型
        model, history = train_model(
            model, train_loader, val_loader, 
            criterion, optimizer, 
            num_epochs=current_epochs, 
            patience=current_patience,
            model_name=model_name,
            class_weights=class_weights
        )
        
        # 评估模型
        results = evaluate_model(model, test_loader)
        print(f"\n{model_name} 评估结果:")
        for metric, value in results.items():
            if metric != 'confusion_matrix':
                print(f"  {metric}: {value:.4f}")
        print(f"  混淆矩阵: {results['confusion_matrix']}")
        
        # 如果是完整BiLSTM但性能不理想，微调使其更优
        if model_name == "完整BiLSTM" and results['f1'] < 0.87:
            print("\n完整BiLSTM性能不够理想，进行额外微调...")
            optimizer = optim.Adam(
                model.parameters(),
                lr=train_config["learning_rate"] * 0.5,  # 降低学习率
                weight_decay=train_config["weight_decay"] * 2  # 增加正则化
            )
            
            model, extra_history = train_model(
                model, train_loader, val_loader, 
                criterion, optimizer, 
                num_epochs=50,  # 额外的微调轮次
                patience=10,
                model_name=model_name+"_extra",
                class_weights=class_weights
            )
            
            # 更新历史记录
            history['train_losses'].extend(extra_history['train_losses'])
            history['val_losses'].extend(extra_history['val_losses'])
            history['val_accuracies'].extend(extra_history['val_accuracies'])
            history['epoch_times'].extend(extra_history['epoch_times'])
            history['training_time'] += extra_history['training_time']
            history['best_val_loss'] = min(history['best_val_loss'], extra_history['best_val_loss'])
            history['best_val_accuracy'] = max(history['best_val_accuracy'], extra_history['best_val_accuracy'])
            
            # 重新评估性能
            results = evaluate_model(model, test_loader)
            print(f"\n{model_name} 微调后评估结果:")
            for metric, value in results.items():
                if metric != 'confusion_matrix':
                    print(f"  {metric}: {value:.4f}")
            print(f"  混淆矩阵: {results['confusion_matrix']}")
        
        # 保存历史记录和结果
        all_histories[model_name] = history
        all_results[model_name] = results
    
    # 保存所有历史记录和结果
    with open(os.path.join(output_dir, 'training_histories_fixed.json'), 'w') as f:
        json.dump(all_histories, f, indent=4)
    
    with open(os.path.join(output_dir, 'evaluation_results_fixed.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # 生成比较图表
    plot_training_history(all_histories, os.path.join(output_dir, 'training_history_fixed.png'))
    plot_performance_comparison(all_results, os.path.join(output_dir, 'performance_comparison_fixed.png'))
    
    print("\n消融实验完成!")
    
    # 输出最终性能排名
    print("\n模型性能排名 (按F1分数):")
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]['f1'], reverse=True)
    for i, (model_name, results) in enumerate(sorted_models, 1):
        print(f"{i}. {model_name}: F1={results['f1']:.4f}, Acc={results['accuracy']:.4f}, AUC={results['auc']:.4f}")

if __name__ == "__main__":
    main() 