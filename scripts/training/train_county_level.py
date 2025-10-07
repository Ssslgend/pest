# train_county_level.py
"""
县级气象数据BiLSTM训练脚本
使用生成的县级空间气象数据训练害虫风险预测模型
"""

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
import pandas as pd
import os
from tqdm import tqdm
import joblib
from torch.utils.data import Dataset, DataLoader

from model.bilstm import BiLSTMModel

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 创建保存结果的目录
os.makedirs('county_level_results', exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class CountyLevelDataset(Dataset):
    """县级数据集类"""
    def __init__(self, data_path, scaler_path=None, sequence_length=8):
        """
        Args:
            data_path: 数据文件路径
            scaler_path: 标准化器路径
            sequence_length: 序列长度
        """
        self.sequence_length = sequence_length
        
        # 加载数据
        print(f"加载数据: {data_path}")
        self.data = pd.read_csv(data_path)
        print(f"数据形状: {self.data.shape}")
        
        # 加载标准化器
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("加载了预训练的标准化器")
        else:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            print("创建了新的标准化器")
        
        # 特征列
        self.feature_columns = [
            'Temperature', 'Humidity', 'Rainfall', 'WS', 'WD', 'Pressure', 
            'Sunshine', 'Visibility', 'Temperature_MA', 'Humidity_MA', 
            'Rainfall_MA', 'Pressure_MA', 'Temp_7day_MA', 'Humidity_7day_MA', 
            'Rainfall_7day_MA', 'Temp_Change', 'Cumulative_Rainfall_7day', 
            'Temp_Humidity_Index'
        ]
        
        # 准备数据
        self._prepare_data()
    
    def _prepare_data(self):
        """准备训练数据"""
        print("准备训练数据...")
        
        # 按县和时间排序
        self.data = self.data.sort_values(['county_name', 'year', 'month', 'day'])
        
        # 提取特征和标签
        features = self.data[self.feature_columns].values
        labels = self.data['Value_Class'].values - 1  # 转换为0-based索引
        
        # 如果需要，进行标准化
        if not hasattr(self.scaler, 'mean_'):
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)
        
        # 创建序列数据
        self.sequences, self.labels = self._create_sequences(features, labels)
        
        print(f"序列数据形状: {self.sequences.shape}")
        print(f"标签数据形状: {self.labels.shape}")
        
        # 标签分布
        unique, counts = np.unique(self.labels, return_counts=True)
        print("标签分布:")
        for u, c in zip(unique, counts):
            print(f"  类别 {u+1}: {c} 样本")
    
    def _create_sequences(self, features, labels):
        """创建时间序列"""
        sequences = []
        sequence_labels = []
        
        # 按县分组
        counties = self.data['county_name'].unique()
        
        for county in counties:
            county_data = self.data[self.data['county_name'] == county]
            county_indices = county_data.index
            
            # 找到该县在特征数组中的位置
            county_mask = self.data.index.isin(county_indices)
            county_features = features[county_mask]
            county_labels = labels[county_mask]
            
            # 创建序列
            for i in range(len(county_features) - self.sequence_length + 1):
                sequences.append(county_features[i:i + self.sequence_length])
                sequence_labels.append(county_labels[i + self.sequence_length - 1])
        
        return np.array(sequences), np.array(sequence_labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.LongTensor([self.labels[idx]])
        return sequence, label

class CountyLevelTrainer:
    """县级BiLSTM模型训练器"""
    
    def __init__(self, config):
        """
        Args:
            config: 训练配置
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建数据集
        self.train_dataset = CountyLevelDataset(
            config['train_data_path'],
            config['scaler_path'],
            config['sequence_length']
        )
        
        self.val_dataset = CountyLevelDataset(
            config['val_data_path'],
            config['scaler_path'],
            config['sequence_length']
        )
        
        self.test_dataset = CountyLevelDataset(
            config['test_data_path'],
            config['scaler_path'],
            config['sequence_length']
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=2
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
        # 创建模型
        input_size = len(self.train_dataset.feature_columns)
        hidden_size = config['hidden_size']
        num_layers = config['num_layers']
        num_classes = config['num_classes']
        
        self.model = BiLSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=config['dropout']
        ).to(self.device)
        
        # 定义损失函数和优化器
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config['epochs'])
        
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'训练 Epoch {epoch+1}')
        
        for sequences, labels in progress_bar:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device).squeeze()
            
            # 前向传播
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for sequences, labels in self.val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device).squeeze()
                
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # 计算F1分数
        f1 = f1_score(all_labels, np.argmax(all_predictions, axis=1), average='weighted')
        
        return avg_loss, accuracy, f1
    
    def test(self):
        """测试模型"""
        self.model.eval()
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for sequences, labels in self.test_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device).squeeze()
                
                outputs = self.model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy())
        
        # 计算指标
        predictions = np.argmax(all_predictions, axis=1)
        
        accuracy = accuracy_score(all_labels, predictions)
        f1 = f1_score(all_labels, predictions, average='weighted')
        precision = precision_score(all_labels, predictions, average='weighted')
        recall = recall_score(all_labels, predictions, average='weighted')
        
        # 详细报告
        report = classification_report(
            all_labels, predictions, 
            target_names=['低风险', '中风险', '高风险'],
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(all_labels, predictions)
        }
    
    def train(self):
        """训练模型"""
        print("开始训练县级BiLSTM模型...")
        
        best_f1 = 0
        train_history = {'loss': [], 'accuracy': []}
        val_history = {'loss': [], 'accuracy': [], 'f1': []}
        
        for epoch in range(self.config['epochs']):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.config['epochs']}")
            print(f"{'='*50}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            train_history['loss'].append(train_loss)
            train_history['accuracy'].append(train_acc)
            
            # 验证
            val_loss, val_acc, val_f1 = self.validate()
            val_history['loss'].append(val_loss)
            val_history['accuracy'].append(val_acc)
            val_history['f1'].append(val_f1)
            
            # 学习率调度
            self.scheduler.step()
            
            # 保存最佳模型
            if val_f1 > best_f1:
                best_f1 = val_f1
                self.save_model('best_county_bilstm.pth')
                print(f"✅ 保存最佳模型 (F1: {val_f1:.4f})")
            
            print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}")
        
        # 最终测试
        print(f"\n{'='*50}")
        print("最终测试")
        print(f"{'='*50}")
        
        test_results = self.test()
        
        print(f"测试准确率: {test_results['accuracy']:.4f}")
        print(f"测试F1分数: {test_results['f1_score']:.4f}")
        print(f"测试精确率: {test_results['precision']:.4f}")
        print(f"测试召回率: {test_results['recall']:.4f}")
        
        # 保存结果
        self.save_results(train_history, val_history, test_results)
        
        # 绘制图表
        self.plot_training_history(train_history, val_history)
        self.plot_confusion_matrix(test_results['confusion_matrix'])
        
        return train_history, val_history, test_results
    
    def save_model(self, filename):
        """保存模型"""
        model_path = os.path.join('county_level_results', filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'feature_columns': self.train_dataset.feature_columns,
            'scaler': self.train_dataset.scaler
        }, model_path)
    
    def save_results(self, train_history, val_history, test_results):
        """保存训练结果"""
        results = {
            'config': self.config,
            'train_history': train_history,
            'val_history': val_history,
            'test_results': test_results,
            'best_f1': max(val_history['f1'])
        }
        
        results_path = os.path.join('county_level_results', 'training_results.json')
        
        import json
        # 转换numpy类型为Python原生类型
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            return obj
        
        results_converted = {}
        for key, value in results.items():
            results_converted[key] = convert_numpy(value)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, ensure_ascii=False, indent=2)
        
        print(f"训练结果已保存到: {results_path}")
    
    def plot_training_history(self, train_history, val_history):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        axes[0].plot(train_history['loss'], label='训练损失')
        axes[0].plot(val_history['loss'], label='验证损失')
        axes[0].set_title('训练和验证损失')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('损失')
        axes[0].legend()
        axes[0].grid(True)
        
        # 准确率曲线
        axes[1].plot(train_history['accuracy'], label='训练准确率')
        axes[1].plot(val_history['accuracy'], label='验证准确率')
        axes[1].plot(val_history['f1'], label='验证F1分数')
        axes[1].set_title('训练和验证指标')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('分数')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join('county_level_results', 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"训练历史图表已保存")
    
    def plot_confusion_matrix(self, cm):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['低风险', '中风险', '高风险'],
                    yticklabels=['低风险', '中风险', '高风险'])
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig(os.path.join('county_level_results', 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"混淆矩阵已保存")


def main():
    """主函数"""
    # 训练配置
    config = {
        'train_data_path': 'datas/shandong_pest_data/spatial_train_data.csv',
        'val_data_path': 'datas/shandong_pest_data/spatial_val_data.csv',
        'test_data_path': 'datas/shandong_pest_data/spatial_test_data.csv',
        'scaler_path': 'datas/shandong_pest_data/spatial_meteorological_scaler.joblib',
        'sequence_length': 8,  # 使用8天的时间序列
        'batch_size': 64,
        'hidden_size': 256,
        'num_layers': 4,
        'num_classes': 3,  # 3个风险等级
        'dropout': 0.3,
        'learning_rate': 0.001,
        'epochs': 50
    }
    
    # 创建训练器
    trainer = CountyLevelTrainer(config)
    
    # 开始训练
    train_history, val_history, test_results = trainer.train()
    
    print("\n" + "="*50)
    print("训练完成！")
    print("="*50)
    print(f"最佳验证F1分数: {max(val_history['f1']):.4f}")
    print(f"最终测试准确率: {test_results['accuracy']:.4f}")
    print(f"最终测试F1分数: {test_results['f1_score']:.4f}")
    
    # 显示详细分类报告
    print("\n详细分类报告:")
    print(classification_report(
        np.argmax(test_results['classification_report']['macro avg']['f1-score']),
        test_results['classification_report']
    ))


if __name__ == "__main__":
    main()