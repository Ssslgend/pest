# sd_raster_prediction/data_processor_raster_new.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import os
import joblib
from sd_raster_prediction.config_raster_new import get_config # 导入新的配置文件

# 加载配置
CONFIG = get_config()
DP_CONFIG = CONFIG['data_processing']

class SdPestPresenceAbsenceDataset(Dataset):
    """Dataset for presence/pseudo-absence points for BiLSTM training."""
    def __init__(self, X, y, coordinates, augment=False):
        # Add sequence dim for BiLSTM: (samples, features) -> (samples, 1, features)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.coordinates = torch.tensor(coordinates, dtype=torch.float32)
        self.augment = augment  # 是否应用数据增强
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx].clone()  # 使用克隆避免修改原始数据
        y = self.y[idx].clone()
        coords = self.coordinates[idx].clone()
        
        # 数据增强 - 仅对训练集
        if self.augment and torch.rand(1).item() < 0.5:  # 50%的几率应用增强
            # 对特征添加少量噪声，增加模型鲁棒性
            noise_level = 0.05
            noise = torch.randn_like(X) * noise_level
            X = X + noise
            
        return X, y, coords

class RasterPredictionDataProcessor:
    def __init__(self):
        self.config = get_config()
        self.dp_config = self.config['data_processing']
        self.csv_data_path = self.config['csv_data_path']
        self.scaler_save_path = self.config['scaler_save_path']
        self.scaler = StandardScaler()
        self.feature_columns = []

    def _generate_pseudo_absence(self, presence_df):
        """改进的伪阴性点生成策略，使用更智能的采样方法"""
        num_presence = len(presence_df)
        # 略微增加伪阴性点比例，使数据更平衡
        pseudo_absence_ratio = self.dp_config['pseudo_absence_ratio'] * 1.2
        num_absence = int(num_presence * pseudo_absence_ratio)
        print(f"生成 {num_absence} 个伪阴性点 (使用改进的特征采样方法)...")

        coord_cols = self.dp_config['coordinate_columns']
        min_x, max_x = presence_df[coord_cols[0]].min(), presence_df[coord_cols[0]].max()
        min_y, max_y = presence_df[coord_cols[1]].min(), presence_df[coord_cols[1]].max()

        # 增加缓冲区以覆盖更广泛的区域
        buffer_x = (max_x - min_x) * 0.2  # 从0.15增加到0.2
        buffer_y = (max_y - min_y) * 0.2

        # 生成坐标，确保与阳性点保持一定距离
        presence_coords = presence_df[coord_cols].values
        pseudo_coords = []
        attempts = 0
        max_attempts = num_absence * 15  # 增加最大尝试次数，从10增加到15
        min_distance = 0.05  # 保持最小距离阈值

        while len(pseudo_coords) < num_absence and attempts < max_attempts:
            # 生成候选坐标
            candidate_x = np.random.uniform(min_x - buffer_x, max_x + buffer_x)
            candidate_y = np.random.uniform(min_y - buffer_y, max_y + buffer_y)
            candidate = np.array([candidate_x, candidate_y])

            # 计算与所有阳性点的距离
            distances = np.sqrt(np.sum((presence_coords - candidate)**2, axis=1))

            # 如果与所有阳性点的距离都大于阈值，则接受该点
            if np.min(distances) > min_distance:
                pseudo_coords.append(candidate)

            attempts += 1

        if len(pseudo_coords) < num_absence:
            print(f"警告: 只能生成 {len(pseudo_coords)} 个满足距离条件的伪阴性点 (目标: {num_absence})")
            # 如果生成的点不够，补充随机点
            remaining = num_absence - len(pseudo_coords)
            for _ in range(remaining):
                pseudo_coords.append([np.random.uniform(min_x - buffer_x, max_x + buffer_x),
                                     np.random.uniform(min_y - buffer_y, max_y + buffer_y)])

        pseudo_coords = np.array(pseudo_coords)
        pseudo_coords_x = pseudo_coords[:, 0]
        pseudo_coords_y = pseudo_coords[:, 1]

        # 为伪阴性点生成特征
        pseudo_features = {}
        for col in self.feature_columns:
            # 获取阳性点的特征分布
            presence_values = presence_df[col].dropna()
            if len(presence_values) > 0:
                # 计算特征的均值和标准差
                mean_val = presence_values.mean()
                std_val = presence_values.std()

                if std_val > 0:
                    # 调整伪阴性点的特征分布
                    # 对于阴性点，我们假设分布与阳性点有一定差异
                    adjusted_mean = mean_val * np.random.uniform(0.85, 0.95) # 更多样化的均值调整
                    adjusted_std = std_val * np.random.uniform(1.05, 1.15)   # 更多样化的标准差调整

                    # 从正态分布中采样
                    pseudo_features[col] = np.random.normal(
                        loc=adjusted_mean,
                        scale=adjusted_std,
                        size=num_absence
                    )
                else:
                    # 如果标准差为0，使用均匀分布在均值附近采样
                    pseudo_features[col] = np.random.uniform(
                        low=mean_val * 0.8,
                        high=mean_val * 1.2,
                        size=num_absence
                    )
            else:
                pseudo_features[col] = np.zeros(num_absence)  # 如果列全为NaN，则填充0
                print(f"警告: 特征列 '{col}' 在阳性数据中没有非NaN值。伪阴性点填充为0。")

        # 创建伪阴性点数据框
        pseudo_absence_df = pd.DataFrame(pseudo_features)
        pseudo_absence_df[coord_cols[0]] = pseudo_coords_x
        pseudo_absence_df[coord_cols[1]] = pseudo_coords_y
        pseudo_absence_df[self.dp_config['label_column']] = 0

        return pseudo_absence_df

    def load_prepare_and_split_data(self):
        """加载CSV，标准化，拆分，并保存标准化器。"""
        try:
            print(f"正在加载CSV文件: {self.csv_data_path}")
            # 尝试不同编码方式加载CSV文件
            try:
                df = pd.read_csv(self.csv_data_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(self.csv_data_path, encoding='gbk')
                except:
                    df = pd.read_csv(self.csv_data_path, encoding='latin1')
                    
            print(f"加载的CSV数据形状: {df.shape}")
            
            # 检查数据分布
            print(f"标签分布: {df[self.dp_config['label_column']].value_counts().to_dict()}")

            # 确定特征列
            excluded = (self.dp_config['excluded_cols_from_features'] +
                        self.dp_config['coordinate_columns'] +
                        [self.dp_config['label_column']])
            self.feature_columns = [col for col in df.columns if col not in excluded]
            print(f"使用来自CSV的 {len(self.feature_columns)} 个特征: {self.feature_columns}")
            if not self.feature_columns:
                raise ValueError("未从CSV确定特征列。")

            # 验证所有预期的特征列是否存在于数据框中
            missing_csv_cols = [col for col in self.feature_columns if col not in df.columns]
            if missing_csv_cols:
                raise ValueError(f"CSV缺少预期的特征列: {missing_csv_cols}")
            
            # 检查特征中是否有重复列
            duplicate_features = set([x for x in self.feature_columns if self.feature_columns.count(x) > 1])
            if duplicate_features:
                print(f"警告: 发现重复特征名称: {duplicate_features}")
                # 修复重复列名问题
                unique_features = []
                for idx, feature in enumerate(self.feature_columns):
                    if feature in unique_features:
                        # 如果已存在，从特征列表中删除
                        print(f"删除重复特征: {feature}")
                        self.feature_columns[idx] = None
                    else:
                        unique_features.append(feature)
                # 移除None值
                self.feature_columns = [f for f in self.feature_columns if f is not None]
                print(f"修复后的特征列数: {len(self.feature_columns)}")

            # 直接使用原始数据，不生成伪阴性点
            combined_df = df.sample(frac=1, random_state=self.dp_config['random_state']).reset_index(drop=True)

            X = combined_df[self.feature_columns].values
            y = combined_df[self.dp_config['label_column']].values
            coordinates = combined_df[self.dp_config['coordinate_columns']].values

            if np.isnan(X).any():
                print("警告: 特征中发现NaN。用列均值填充。")
                col_means = np.nanmean(X, axis=0)
                if np.isnan(col_means).any():
                   nan_cols = [self.feature_columns[i] for i, is_nan in enumerate(np.isnan(col_means)) if is_nan]
                   raise ValueError(f"列全是NaN: {nan_cols}")
                inds = np.where(np.isnan(X))
                X[inds] = np.take(col_means, inds[1])

            # 在训练数据上拟合转换缩放器（在拆分之前）
            X_scaled = self.scaler.fit_transform(X)
            print("特征标准化完成。")

            # --- 保存拟合好的缩放器 --- #
            os.makedirs(os.path.dirname(self.scaler_save_path), exist_ok=True)
            joblib.dump(self.scaler, self.scaler_save_path)
            print(f"缩放器保存到: {self.scaler_save_path}")

            # 拆分数据
            test_size = self.dp_config['test_size']
            val_size = self.dp_config['val_size']
            random_state = self.dp_config['random_state']

            X_train_val, X_test, y_train_val, y_test, coords_train_val, coords_test = train_test_split(
                X_scaled, y, coordinates, test_size=test_size, random_state=random_state, stratify=y
            )

            # Further split train_val into train and val
            relative_val_size = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val, coords_train, coords_val = train_test_split(
                X_train_val, y_train_val, coords_train_val, 
                test_size=relative_val_size, random_state=random_state, stratify=y_train_val
            )

            print(f"训练集: {X_train.shape[0]} 个样本, 特征: {X_train.shape[1]}")
            print(f"验证集: {X_val.shape[0]} 个样本")
            print(f"测试集: {X_test.shape[0]} 个样本")
            print(f"阳性样本: {np.sum(y)} / {len(y)}")

            # 设置类属性，以便稍后创建数据加载器
            self.X_train, self.y_train, self.coords_train = X_train, y_train, coords_train
            self.X_val, self.y_val, self.coords_val = X_val, y_val, coords_val
            self.X_test, self.y_test, self.coords_test = X_test, y_test, coords_test

            return {
                'input_size': X_train.shape[1],
                'feature_names': self.feature_columns,
                'X_train': X_train,
                'y_train': y_train,
                'coords_train': coords_train,
                'X_val': X_val,
                'y_val': y_val,
                'coords_val': coords_val,
                'X_test': X_test,
                'y_test': y_test,
                'coords_test': coords_test
            }

        except Exception as e:
            print(f"数据处理过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def get_dataloaders(self, batch_size):
        """为模型训练创建数据加载器。"""
        train_dataset = SdPestPresenceAbsenceDataset(self.X_train, self.y_train, self.coords_train, augment=True)
        val_dataset = SdPestPresenceAbsenceDataset(self.X_val, self.y_val, self.coords_val)
        test_dataset = SdPestPresenceAbsenceDataset(self.X_test, self.y_test, self.coords_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader 