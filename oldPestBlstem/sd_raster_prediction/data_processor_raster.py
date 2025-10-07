# sd_raster_prediction/data_processor_raster.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import os
import joblib
from sd_raster_prediction.config_raster import get_config # Import from config file

# Load config
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

        # 为伪阴性点生成特征 ！！！！error
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
        """Loads CSV, generates pseudo-absence, scales, splits, and saves scaler."""
        try:
            df = pd.read_csv(self.csv_data_path, encoding='gbk')
            print(f"Loaded CSV data shape: {df.shape}")

            # Determine feature columns
            excluded = (self.dp_config['excluded_cols_from_features'] +
                        self.dp_config['coordinate_columns'] +
                        [self.dp_config['label_column']])
            self.feature_columns = [col for col in df.columns if col not in excluded]
            print(f"Using {len(self.feature_columns)} features from CSV: {self.feature_columns}")
            if not self.feature_columns:
                raise ValueError("No feature columns determined from CSV.")

            # Verify all expected feature columns exist in the dataframe
            missing_csv_cols = [col for col in self.feature_columns if col not in df.columns]
            if missing_csv_cols:
                raise ValueError(f"CSV is missing expected feature columns: {missing_csv_cols}")

            presence_df = df[df[self.dp_config['label_column']] == 1].copy()
            if len(presence_df) == 0:
                raise ValueError(f"No presence points found in {self.csv_data_path}.")

            pseudo_absence_df = self._generate_pseudo_absence(presence_df)
            combined_df = pd.concat([presence_df, pseudo_absence_df], ignore_index=True)
            combined_df = combined_df.sample(frac=1, random_state=self.dp_config['random_state']).reset_index(drop=True)

            X = combined_df[self.feature_columns].values
            y = combined_df[self.dp_config['label_column']].values
            coordinates = combined_df[self.dp_config['coordinate_columns']].values

            if np.isnan(X).any():
                print("Warning: NaNs found in features. Filling with column means.")
                col_means = np.nanmean(X, axis=0)
                if np.isnan(col_means).any():
                   nan_cols = [self.feature_columns[i] for i, is_nan in enumerate(np.isnan(col_means)) if is_nan]
                   raise ValueError(f"Columns are all NaN: {nan_cols}")
                inds = np.where(np.isnan(X))
                X[inds] = np.take(col_means, inds[1])

            # Fit and transform scaler ON TRAINING DATA (before splitting for best practice)
            # Fit scaler only on the combined presence/pseudo-absence data
            X_scaled = self.scaler.fit_transform(X)
            print("Scaler fitted on combined presence/pseudo-absence data.")

            # --- Save the fitted scaler --- #
            os.makedirs(os.path.dirname(self.scaler_save_path), exist_ok=True)
            joblib.dump(self.scaler, self.scaler_save_path)
            print(f"Scaler saved to: {self.scaler_save_path}")

            # Split data
            test_size = self.dp_config['test_size']
            val_size = self.dp_config['val_size']
            random_state = self.dp_config['random_state']

            X_train_val, X_test, y_train_val, y_test, coords_train_val, coords_test = train_test_split(
                X_scaled, y, coordinates, test_size=test_size, random_state=random_state, stratify=y
            )
            relative_val_size = val_size / (1.0 - test_size)
            X_train, X_val, y_train, y_val, coords_train, coords_val = train_test_split(
                X_train_val, y_train_val, coords_train_val, test_size=relative_val_size, random_state=random_state, stratify=y_train_val
            )

            print("Data split complete.")
            self.data_dict = {
                'train': {'X': X_train, 'y': y_train, 'coordinates': coords_train},
                'val': {'X': X_val, 'y': y_val, 'coordinates': coords_val},
                'test': {'X': X_test, 'y': y_test, 'coordinates': coords_test},
                'input_size': X_train.shape[1],
                'feature_names': self.feature_columns # Store feature names
            }
            return self.data_dict

        except Exception as e:
            print(f"Error during data loading/preparation: {e}")
            raise # Re-raise the exception after printing

    def get_dataloaders(self, batch_size):
        """Creates DataLoaders for training, validation, and test sets."""
        if not hasattr(self, 'data_dict'):
            raise RuntimeError("Data not loaded. Call load_prepare_and_split_data() first.")

        # 对训练集应用数据增强
        train_ds = SdPestPresenceAbsenceDataset(
            self.data_dict['train']['X'], 
            self.data_dict['train']['y'], 
            self.data_dict['train']['coordinates'],
            augment=True  # 仅在训练集上启用数据增强
        )
        
        # 验证集和测试集不应用数据增强
        val_ds = SdPestPresenceAbsenceDataset(
            self.data_dict['val']['X'], 
            self.data_dict['val']['y'], 
            self.data_dict['val']['coordinates']
        )
        
        test_ds = SdPestPresenceAbsenceDataset(
            self.data_dict['test']['X'], 
            self.data_dict['test']['y'], 
            self.data_dict['test']['coordinates']
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        print("DataLoaders created.")
        return train_loader, val_loader, test_loader

# Example usage for testing the processor
if __name__ == '__main__':
    print("Testing RasterPredictionDataProcessor...")
    processor = RasterPredictionDataProcessor()
    try:
        data_splits = processor.load_prepare_and_split_data()
        print("Input size:", data_splits['input_size'])
        print("Feature names:", data_splits['feature_names'])
        train_loader, _, _ = processor.get_dataloaders(batch_size=4)
        print("Sample batch from train_loader:")
        x_sample, y_sample, coords_sample = next(iter(train_loader))
        print(" X shape:", x_sample.shape)
        print(" Y shape:", y_sample.shape)
        print(" Coords shape:", coords_sample.shape)
    except Exception as e:
        print(f"Error during processor test: {e}")