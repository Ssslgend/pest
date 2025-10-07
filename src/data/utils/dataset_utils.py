# data/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import os

DATA_PATHS = {
    "climate": r"pest_rice_with_features_2_classified.csv",
    "circulation": "path/to/circulation_data.csv"
}

class PestDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class DataHandler:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        
    def create_sequences(self, data):
        """创建时间序列数据"""
        sequences = []
        labels = []
        
        # 按年份和周排序
        data = data.sort_values(['year', 'week'])
        
        # 选择特征列
        feature_cols = [
            'MaxT', 'MinT', 'RH1', 'RH2', 'RF', 'WS', 'SSH', 'EVP',
            'LTP', 'TF', 'PTR', 'THC'  # 添加复合特征
        ]
        
        # 确保所有特征列存在
        missing_cols = [col for col in feature_cols if col not in data.columns]
        if missing_cols:
            print(f"警告: 以下特征列不存在于数据集中: {missing_cols}")
            print(f"可用的列: {data.columns.tolist()}")
            return None, None
        
        # 提取特征和标签
        try:
            # 先尝试直接提取数值列
            X = data[feature_cols].values
            y = data['Value_Class'].values
        except Exception as e:
            print(f"提取特征和标签时出错: {e}")
            print(f"数据列: {data.columns.tolist()}")
            return None, None
        
        # 检查数据类型
        print(f"特征数据类型: {X.dtype}")
        print(f"标签数据类型: {y.dtype}")
        
        # 检查是否有非数值数据
        if X.dtype == object:
            print("警告: 特征包含非数值数据，尝试转换...")
            try:
                # 创建一个新的数组来存储转换后的数据
                X_converted = np.zeros((X.shape[0], X.shape[1]), dtype=float)
                
                # 逐列转换
                for i in range(X.shape[1]):
                    col_name = feature_cols[i]
                    print(f"转换列 {col_name} 从 {X[:, i].dtype} 到 float")
                    
                    # 使用pandas的to_numeric函数进行转换
                    converted_col = pd.to_numeric(X[:, i], errors='coerce')
                    
                    # 检查是否有NaN值
                    nan_count = np.isnan(converted_col).sum()
                    if nan_count > 0:
                        print(f"警告: 列 {col_name} 中有 {nan_count} 个NaN值，将使用0填充")
                        converted_col = np.nan_to_num(converted_col, nan=0.0)
                    
                    # 存储转换后的列
                    X_converted[:, i] = converted_col
                
                # 使用转换后的数组
                X = X_converted
                
            except Exception as e:
                print(f"转换数据类型时出错: {e}")
                print(f"尝试使用更简单的方法转换...")
                
                try:
                    # 使用更简单的方法，直接替换非数值数据
                    for i in range(X.shape[1]):
                        col_name = feature_cols[i]
                        # 将非数值数据替换为0
                        X[:, i] = pd.to_numeric(X[:, i], errors='coerce').fillna(0)
                    
                    # 确保X是浮点型
                    X = X.astype(float)
                except Exception as e2:
                    print(f"简单转换也失败: {e2}")
                    return None, None
        
        # 标准化特征
        try:
            X = self.scaler.fit_transform(X)
        except Exception as e:
            print(f"标准化特征时出错: {e}")
            print(f"特征形状: {X.shape}")
            print(f"特征示例: {X[:5]}")
            return None, None
        
        # 创建序列
        seq_length = self.config['seq_length']
        for i in range(len(X) - seq_length):
            sequences.append(X[i:i+seq_length])
            labels.append(y[i+seq_length])
        
        return np.array(sequences), np.array(labels)
    
    def balance_data(self, X, y):
        """平衡数据集，使用上采样方法"""
        # 获取每个类别的样本数
        unique_labels, counts = np.unique(y, return_counts=True)
        max_samples = np.max(counts)
        
        # 对每个类别进行上采样
        X_balanced = []
        y_balanced = []
        
        for label in unique_labels:
            # 获取当前类别的样本
            mask = y == label
            X_class = X[mask]
            y_class = y[mask]
            
            # 如果样本数少于最大样本数，进行上采样
            if len(X_class) < max_samples:
                # 使用替换采样
                indices = np.random.choice(len(X_class), max_samples, replace=True)
                X_class = X_class[indices]
                y_class = y_class[indices]
            
            X_balanced.append(X_class)
            y_balanced.append(y_class)
        
        # 合并所有类别的样本
        X_balanced = np.vstack(X_balanced)
        y_balanced = np.concatenate(y_balanced)
        
        # 打乱数据
        indices = np.random.permutation(len(X_balanced))
        return X_balanced[indices], y_balanced[indices]
    
    def load_data(self, data_path):
        """加载和预处理数据"""
        print(f"加载数据: {data_path}")
        
        # 检查文件是否存在
        if not os.path.exists(data_path):
            print(f"错误: 文件 {data_path} 不存在")
            return None, None, None, None, None, None
        
        # 读取CSV文件 - 尝试不同的编码
        encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8']
        df = None
        
        for encoding in encodings:
            try:
                print(f"尝试使用 {encoding} 编码读取文件...")
                df = pd.read_csv(data_path, encoding=encoding)
                print(f"成功使用 {encoding} 编码读取文件")
                break
            except UnicodeDecodeError:
                print(f"{encoding} 编码失败，尝试下一个编码...")
                continue
            except Exception as e:
                print(f"使用 {encoding} 编码时发生错误: {e}")
                continue
        
        if df is None:
            print("无法使用任何编码读取文件")
            return None, None, None, None, None, None
        
        print(f"数据形状: {df.shape}")
        print(f"数据列: {df.columns.tolist()}")
        
        # 重命名列，移除单位符号
        column_mapping = {}
        for col in df.columns:
            if '(' in col and ')' in col:
                # 提取列名中的基本部分（不包含单位）
                base_name = col.split('(')[0]
                column_mapping[col] = base_name
                print(f"将列 '{col}' 重命名为 '{base_name}'")
        
        # 应用列名映射
        if column_mapping:
            print(f"重命名列: {column_mapping}")
            df = df.rename(columns=column_mapping)
            print(f"重命名后的列: {df.columns.tolist()}")
        
        # 检查数据类型
        print("列数据类型:")
        for col in df.columns:
            print(f"{col}: {df[col].dtype}")
        
        # 预处理数值列 - 在创建序列之前处理
        numeric_cols = ['MaxT', 'MinT', 'RH1', 'RH2', 'RF', 'WS', 'SSH', 'EVP', 'LTP', 'TF', 'PTR', 'THC']
        for col in numeric_cols:
            if col in df.columns:
                try:
                    # 尝试转换为数值类型
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # 填充NaN值
                    nan_count = df[col].isna().sum()
                    if nan_count > 0:
                        print(f"列 {col} 中有 {nan_count} 个NaN值，将使用0填充")
                        df[col] = df[col].fillna(0)
                except Exception as e:
                    print(f"处理列 {col} 时出错: {e}")
        
        # 检查必要的列是否存在
        required_cols = ['year', 'week', 'Value_Class']
        for col in required_cols:
            if col not in df.columns:
                print(f"错误: 缺少必要的列 {col}")
                return None, None, None, None, None, None
        
        # 创建序列
        X, y = self.create_sequences(df)
        if X is None or y is None:
            return None, None, None, None, None, None
        
        # 打印类别分布
        unique_labels, counts = np.unique(y, return_counts=True)
        print("原始数据类别分布:")
        for label, count in zip(unique_labels, counts):
            print(f"类别 {label}: {count} 样本 ({count/len(y)*100:.2f}%)")
        
        # 平衡数据
        X_balanced, y_balanced = self.balance_data(X, y)
        
        # 打印平衡后的类别分布
        unique_labels, counts = np.unique(y_balanced, return_counts=True)
        print("平衡后数据类别分布:")
        for label, count in zip(unique_labels, counts):
            print(f"类别 {label}: {count} 样本 ({count/len(y_balanced)*100:.2f}%)")
        
        # 划分数据集
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test