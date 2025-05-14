import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch

class SpatialDataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.feature_columns = [
            'Temperature', 'Humidity', 'Rainfall', 'WS',
            'WD', 'Pressure', 'Sunshine', 'Visibility',
            'Temperature_MA', 'Humidity_MA', 'Rainfall_MA', 'Pressure_MA'
        ]
        self.label_column = 'Value_Class'
        self.coordinate_columns = ['latitude', 'longitude']
        self.scaler = StandardScaler()
        
    def load_data(self):
        """加载数据并返回特征、标签和坐标"""
        try:
            # 读取数据
            df = pd.read_csv(self.data_path)
            
            # 检查必要的列是否存在
            required_columns = self.feature_columns + [self.label_column] + self.coordinate_columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"数据中缺少以下列: {missing_columns}")
            
            # 提取特征、标签和坐标
            X = df[self.feature_columns].values
            y = df[self.label_column].values
            coordinates = df[self.coordinate_columns].values
            
            # 标准化特征
            X = self.scaler.fit_transform(X)
            
            # 划分数据集
            indices = np.random.permutation(len(X))
            train_size = int(0.7 * len(X))
            val_size = int(0.15 * len(X))
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size+val_size]
            test_indices = indices[train_size+val_size:]
            
            # 返回划分后的数据
            return {
                'train': {
                    'X': X[train_indices],
                    'y': y[train_indices],
                    'coordinates': coordinates[train_indices]
                },
                'val': {
                    'X': X[val_indices],
                    'y': y[val_indices],
                    'coordinates': coordinates[val_indices]
                },
                'test': {
                    'X': X[test_indices],
                    'y': y[test_indices],
                    'coordinates': coordinates[test_indices]
                }
            }
            
        except Exception as e:
            print(f"加载数据时出错: {e}")
            raise
            
    def create_dataloaders(self, data_dict, batch_size=32):
        """创建数据加载器"""
        class SpatialDataset(Dataset):
            def __init__(self, X, y, coordinates):
                self.X = torch.FloatTensor(X)
                self.y = torch.LongTensor(y)
                self.coordinates = torch.FloatTensor(coordinates)
                
            def __len__(self):
                return len(self.X)
                
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx], self.coordinates[idx]
        
        # 创建数据集
        train_dataset = SpatialDataset(
            data_dict['train']['X'],
            data_dict['train']['y'],
            data_dict['train']['coordinates']
        )
        val_dataset = SpatialDataset(
            data_dict['val']['X'],
            data_dict['val']['y'],
            data_dict['val']['coordinates']
        )
        test_dataset = SpatialDataset(
            data_dict['test']['X'],
            data_dict['test']['y'],
            data_dict['test']['coordinates']
        )
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader
