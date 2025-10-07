# quick_train_county.py
"""
快速县级BiLSTM训练脚本
简化版本，用于快速测试和验证
"""

import torch
import torch.nn as nn
from torch.optim import Adam
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import os
from datetime import datetime

# 检查是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 导入BiLSTM模型
from model.bilstm import BiLSTMModel

def load_and_prepare_data(data_path, sequence_length=8):
    """
    加载和准备数据 - 二分类版本（是否有病虫害）
    """
    print(f"加载数据: {data_path}")
    data = pd.read_csv(data_path)
    
    # 特征列
    feature_columns = [
        'Temperature', 'Humidity', 'Rainfall', 'WS', 'WD', 'Pressure', 
        'Sunshine', 'Visibility', 'Temperature_MA', 'Humidity_MA', 
        'Rainfall_MA', 'Pressure_MA', 'Temp_7day_MA', 'Humidity_7day_MA', 
        'Rainfall_7day_MA', 'Temp_Change', 'Cumulative_Rainfall_7day', 
        'Temp_Humidity_Index'
    ]
    
    # 按县和时间排序
    data = data.sort_values(['county_name', 'year', 'month', 'day'])
    
    # 提取特征和标签
    features = data[feature_columns].values
    
    # 二分类转换：Value_Class > 0 表示有病虫害，Value_Class == 0 表示无病虫害
    original_labels = data['Value_Class'].values
    labels = (original_labels > 0).astype(int)  # 0=无病虫害, 1=有病虫害
    
    print(f"原始标签分布:")
    unique_orig, counts_orig = np.unique(original_labels, return_counts=True)
    for u, c in zip(unique_orig, counts_orig):
        print(f"  原始类别 {u}: {c} 样本")
    
    print(f"二分类标签分布:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        status = "有病虫害" if u == 1 else "无病虫害"
        print(f"  {status} (类别 {u}): {c} 样本")
    
    # 标准化
    scaler = joblib.load('datas/shandong_pest_data/spatial_meteorological_scaler.joblib')
    features = scaler.transform(features)
    
    print(f"特征形状: {features.shape}")
    print(f"标签形状: {labels.shape}")
    
    # 创建序列
    sequences = []
    sequence_labels = []
    
    counties = data['county_name'].unique()
    print(f"处理 {len(counties)} 个县的数据...")
    
    # 使用所有县的数据
    for county in counties:
        county_data = data[data['county_name'] == county]
        county_indices = county_data.index
        
        county_mask = data.index.isin(county_indices)
        county_features = features[county_mask]
        county_labels = labels[county_mask]
        
        # 创建序列
        for i in range(len(county_features) - sequence_length + 1):
            sequences.append(county_features[i:i + sequence_length])
            sequence_labels.append(county_labels[i + sequence_length - 1])
    
    sequences = np.array(sequences)
    sequence_labels = np.array(sequence_labels)
    
    print(f"序列形状: {sequences.shape}")
    print(f"序列标签形状: {sequence_labels.shape}")
    
    # 统计最终标签分布
    unique, counts = np.unique(sequence_labels, return_counts=True)
    print("最终序列标签分布:")
    for u, c in zip(unique, counts):
        status = "有病虫害" if u == 1 else "无病虫害"
        print(f"  {status} (类别 {u}): {c} 样本 ({c/len(sequence_labels)*100:.1f}%)")
    
    return sequences, sequence_labels, len(feature_columns)

def train_model():
    """训练模型 - 基于原始框架"""
    # 加载数据
    sequences, labels, input_size = load_and_prepare_data('datas/shandong_pest_data/spatial_train_data.csv', 8)
    
    # 转换为PyTorch张量
    X = torch.FloatTensor(sequences).to(device)
    y = torch.LongTensor(labels).to(device)
    
    # 创建模型配置 - 二分类配置
    model_config = {
        'input_size': input_size,
        'hidden_size': 256,  # 使用原始框架的大小
        'num_layers': 4,      # 使用原始框架的层数
        'num_classes': 2,     # 二分类：有/无病虫害
        'dropout': 0.3        # 使用原始框架的dropout
    }
    
    # 创建模型
    model = BiLSTMModel(model_config).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试模型输出维度
    test_input = torch.randn(1, 8, input_size).to(device)
    test_output = model(test_input)
    print(f"测试输入形状: {test_input.shape}")
    print(f"测试输出形状: {test_output.shape}")
    print(f"输出范围: {test_output.min():.4f} 到 {test_output.max():.4f}")
    
    # 计算类别权重 - 处理数据不均衡
    class_counts = np.bincount(y.cpu().numpy())
    class_weights = len(y) / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"类别权重: 无病虫害={class_weights[0]:.2f}, 有病虫害={class_weights[1]:.2f}")
    
    # 定义加权损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # 训练循环 - 基于原始框架的训练方式
    print("开始训练...")
    model.train()
    
    epochs = 10  # 增加训练轮数
    batch_size = 64  # 增加批次大小
    
    # 记录训练历史
    train_losses = []
    train_accuracies = []
    best_accuracy = 0
    best_model_state = None
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        # 小批量训练
        num_batches = len(X) // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_x = X[start_idx:end_idx]
            batch_y = y[start_idx:end_idx]
            
            # 前向传播
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / num_batches
        accuracy = 100. * correct / total
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict().copy()
            print(f"  🎯 新最佳准确率: {best_accuracy:.2f}%")
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  损失: {avg_loss:.4f}")
        print(f"  准确率: {accuracy:.2f}%")
        
        # 每10个epoch显示一次详细分析
        if (epoch + 1) % 10 == 0:
            print(f"  标签分布: {np.bincount(batch_y.cpu().numpy())}")
    
    # 评估模型
    print("\n评估模型...")
    model.eval()
    
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)
        
        # 计算指标
        accuracy = accuracy_score(labels, predicted.cpu().numpy())
        f1 = f1_score(labels, predicted.cpu().numpy(), average='weighted')
        precision = precision_score(labels, predicted.cpu().numpy(), average='weighted')
        recall = recall_score(labels, predicted.cpu().numpy(), average='weighted')
        
        print(f"训练集准确率: {accuracy:.4f}")
        print(f"训练集F1分数: {f1:.4f}")
        print(f"训练集精确率: {precision:.4f}")
        print(f"训练集召回率: {recall:.4f}")
        
        # 详细报告
        print("\n分类报告:")
        print(classification_report(
            labels, 
            predicted.cpu().numpy(),
            target_names=['低风险', '中风险', '高风险']
        ))
    
    # 加载最佳模型状态
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ 加载最佳模型状态 (最佳准确率: {best_accuracy:.2f}%)")
    
    # 最终评估
    print("\n最终评估...")
    model.eval()
    
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)
        
        # 计算指标 - 针对不均衡数据的评估
        accuracy = accuracy_score(labels, predicted.cpu().numpy())
        f1 = f1_score(labels, predicted.cpu().numpy(), average='weighted')
        f1_binary = f1_score(labels, predicted.cpu().numpy())
        precision = precision_score(labels, predicted.cpu().numpy(), average='weighted')
        recall = recall_score(labels, predicted.cpu().numpy(), average='weighted')
        
        # 计算每个类别的指标
        precision_per_class = precision_score(labels, predicted.cpu().numpy(), average=None)
        recall_per_class = recall_score(labels, predicted.cpu().numpy(), average=None)
        
        print(f"最终训练集准确率: {accuracy:.4f}")
        print(f"最终训练集F1分数 (weighted): {f1:.4f}")
        print(f"最终训练集F1分数 (binary): {f1_binary:.4f}")
        print(f"最终训练集精确率 (weighted): {precision:.4f}")
        print(f"最终训练集召回率 (weighted): {recall:.4f}")
        
        # 每个类别的详细指标
        print(f"\n各类别详细指标:")
        print(f"无病虫害 - 精确率: {precision_per_class[0]:.4f}, 召回率: {recall_per_class[0]:.4f}")
        print(f"有病虫害 - 精确率: {precision_per_class[1]:.4f}, 召回率: {recall_per_class[1]:.4f}")
        
        # 分析预测分布
        print("\n预测分析:")
        true_dist = np.bincount(labels)
        pred_dist = np.bincount(predicted.cpu().numpy())
        print(f"真实标签分布: 无病虫害={true_dist[0]}, 有病虫害={true_dist[1]}")
        print(f"预测标签分布: 无病虫害={pred_dist[0]}, 有病虫害={pred_dist[1]}")
        
        # 混淆矩阵
        cm = confusion_matrix(labels, predicted.cpu().numpy())
        print(f"\n混淆矩阵:")
        print(f"                预测无病虫害  预测有病虫害")
        print(f"真实无病虫害      {cm[0,0]:6d}      {cm[0,1]:6d}")
        print(f"真实有病虫害      {cm[1,0]:6d}      {cm[1,1]:6d}")
        
        # 详细报告
        print("\n分类报告:")
        print(classification_report(
            labels, 
            predicted.cpu().numpy(),
            target_names=['无病虫害', '有病虫害'],
            zero_division=0
        ))
    
    # 保存模型 - 添加时间戳和性能信息
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'county_level_results/county_bilstm_model_acc_{accuracy:.2f}_{timestamp}.pth'
    os.makedirs('county_level_results', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'input_size': input_size,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'best_accuracy': best_accuracy,
        'final_accuracy': accuracy,
        'timestamp': timestamp
    }, model_path)
    
    print(f"\n模型已保存到: {model_path}")
    
    return model, model_path

def test_prediction():
    """测试预测功能 - 二分类版本"""
    print("\n测试预测功能...")
    
    # 如果没有提供模型路径，查找最新的模型文件
    model_dir = 'county_level_results'
    if not os.path.exists(model_dir):
        print("模型目录不存在，请先训练模型")
        return
    
    # 查找最新的模型文件
    model_files = [f for f in os.listdir(model_dir) if f.startswith('county_bilstm_model_acc_') and f.endswith('.pth')]
    if not model_files:
        print("没有找到训练好的模型文件，请先训练模型")
        return
    
    # 按文件名排序，获取最新的模型
    model_files.sort()
    model_path = os.path.join(model_dir, model_files[-1])
    
    print(f"加载模型: {model_path}")
    
    checkpoint = torch.load(model_path)
    model_config = checkpoint['model_config']
    input_size = checkpoint['input_size']
    
    # 重建模型
    model = BiLSTMModel(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载验证数据
    sequences, labels, _ = load_and_prepare_data(
        'datas/shandong_pest_data/spatial_val_data.csv',
        8
    )
    
    # 转换为PyTorch张量
    X = torch.FloatTensor(sequences).to(device)
    
    # 进行预测
    with torch.no_grad():
        outputs = model(X)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
    
    # 计算准确率和详细指标
    accuracy = accuracy_score(labels, predicted.cpu().numpy())
    f1_binary = f1_score(labels, predicted.cpu().numpy())
    precision_per_class = precision_score(labels, predicted.cpu().numpy(), average=None)
    recall_per_class = recall_score(labels, predicted.cpu().numpy(), average=None)
    
    print(f"验证集准确率: {accuracy:.4f}")
    print(f"验证集F1分数: {f1_binary:.4f}")
    
    # 分析预测分布
    print("\n验证集预测分析:")
    true_dist = np.bincount(labels)
    pred_dist = np.bincount(predicted.cpu().numpy())
    print(f"真实标签分布: 无病虫害={true_dist[0]}, 有病虫害={true_dist[1]}")
    print(f"预测标签分布: 无病虫害={pred_dist[0]}, 有病虫害={pred_dist[1]}")
    
    # 各类别详细指标
    print(f"\n验证集各类别指标:")
    print(f"无病虫害 - 精确率: {precision_per_class[0]:.4f}, 召回率: {recall_per_class[0]:.4f}")
    print(f"有病虫害 - 精确率: {precision_per_class[1]:.4f}, 召回率: {recall_per_class[1]:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(labels, predicted.cpu().numpy())
    print(f"\n验证集混淆矩阵:")
    print(f"                预测无病虫害  预测有病虫害")
    print(f"真实无病虫害      {cm[0,0]:6d}      {cm[0,1]:6d}")
    print(f"真实有病虫害      {cm[1,0]:6d}      {cm[1,1]:6d}")
    
    # 显示一些预测结果
    print("\n预测结果示例:")
    pest_status = ['无病虫害', '有病虫害']
    
    for i in range(min(10, len(predicted))):
        true_label = pest_status[labels[i]]
        pred_label = pest_status[predicted[i].item()]
        confidence = probabilities[i][predicted[i]].item()
        
        print(f"样本 {i+1}: 真实={true_label}, 预测={pred_label}, 置信度={confidence:.2%}")

if __name__ == "__main__":
    print("开始县级BiLSTM快速训练（二分类版本）...")
    
    # 训练模型
    model, model_path = train_model()
    
    # 测试预测
    test_prediction()
    
    print("\n训练完成！")
    print("生成的文件:")
    print(f"  - {model_path}: 训练好的模型")
    print("任务：基于县域气象数据预测美国白蛾病虫害发生情况")
    print("类别：0=无病虫害, 1=有病虫害")