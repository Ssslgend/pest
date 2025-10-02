import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

class PestDataset(Dataset):
    def __init__(self, X, y, sequence_length=30):
        self.X = torch.FloatTensor(X)
        # 将pandas Series转换为numpy数组，然后再转换为PyTorch张量
        self.y = torch.LongTensor(y.values if hasattr(y, 'values') else y)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.X) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        # 获取时间序列数据
        x = self.X[idx:idx + self.sequence_length]
        y = self.y[idx + self.sequence_length - 1]  # 使用最后一个时间步的标签
        return x, y

class DataProcessor:
    def __init__(self, data_path, test_size=0.2, val_size=0.2, random_state=42, sequence_length=30):
        self.data_path = data_path
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
        # 定义特征列名（与CSV文件中的列名完全匹配）
        self.feature_columns = [
            'MaxT', 'MinT', 'RH1', 'RH2', 'RF', 'WS', 'SSH', 'EVP', 'LTP', 'TF', 'PTR', 'THC'
        ]
        
    def load_data(self):
        """加载数据并进行预处理"""
        try:
            # 读取CSV文件，使用GBK编码
            df = pd.read_csv(self.data_path, encoding='gbk')
            
            # 打印实际的列名
            print("CSV文件中的实际列名:")
            print(df.columns.tolist())
            
            # 检查所需的列是否存在
            missing_columns = [col for col in self.feature_columns + ['Value_Class'] if col not in df.columns]
            if missing_columns:
                raise ValueError(f"CSV文件中缺少以下列: {missing_columns}")
            
            # 分离特征和标签
            X = df[self.feature_columns].copy()
            y = df['Value_Class']
            
            # 处理特殊值
            for col in X.columns:
                # 将非数值类型的值替换为NaN
                X[col] = pd.to_numeric(X[col], errors='coerce')
            
            # 使用均值填充NaN值
            X = X.fillna(X.mean())
            
            # 数据标准化
            X = self.scaler.fit_transform(X)
            
            # 划分训练集、验证集和测试集
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=self.test_size + self.val_size,
                random_state=self.random_state, stratify=y
            )
            
            # 从临时测试集中划分出验证集和测试集
            val_ratio = self.val_size / (self.test_size + self.val_size)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=val_ratio,
                random_state=self.random_state, stratify=y_temp
            )
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            print(f"加载数据时出错: {e}")
            raise
    
    def create_dataloaders(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
        """创建数据加载器"""
        # 创建数据集
        train_dataset = PestDataset(X_train, y_train, self.sequence_length)
        val_dataset = PestDataset(X_val, y_val, self.sequence_length)
        test_dataset = PestDataset(X_test, y_test, self.sequence_length)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader 