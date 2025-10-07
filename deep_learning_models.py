#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用BiLSTM和图神经网络（GNN）进行美国白蛾发病情况预测
包含时间序列建模和空间关系建模
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader as GeoDataLoader
import json
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class PestDataset(Dataset):
    """病虫害数据集类，用于时间序列建模"""

    def __init__(self, data, sequence_length=3, feature_cols=None, target_col='Severity_Level'):
        self.data = data
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols
        self.target_col = target_col

        # 按县和年份排序
        self.data = self.data.sort_values(['County', 'Year'])

        # 创建时间序列样本
        self.sequences = []
        self.targets = []
        self.county_ids = []

        for county in self.data['County'].unique():
            county_data = self.data[self.data['County'] == county].reset_index(drop=True)

            if len(county_data) >= sequence_length:
                features = county_data[self.feature_cols].values
                targets = county_data[self.target_col].values

                # 使用滑动窗口创建样本
                for i in range(len(county_data) - sequence_length + 1):
                    seq = features[i:i+sequence_length]
                    # 使用窗口内的最后一个值作为目标，或者下一个值（如果存在）
                    if i + sequence_length < len(county_data):
                        target = targets[i+sequence_length] - 1  # 转换为0,1,2
                    else:
                        target = targets[i+sequence_length-1] - 1  # 使用窗口内最后一个值

                    county_id = county  # 保存县ID

                    self.sequences.append(torch.FloatTensor(seq))
                    self.targets.append(torch.LongTensor([target]))
                    self.county_ids.append(county_id)

        print(f"创建了 {len(self.sequences)} 个时间序列样本")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'target': self.targets[idx],
            'county_id': self.county_ids[idx]
        }

class SpatialTemporalDataset:
    """时空数据集类，用于图神经网络建模"""

    def __init__(self, data, feature_cols=None, target_col='Severity_Level'):
        self.data = data
        self.feature_cols = feature_cols
        self.target_col = target_col

        # 创建县ID映射
        self.counties = sorted(self.data['County'].unique())
        self.county_to_idx = {county: idx for idx, county in enumerate(self.counties)}
        self.num_counties = len(self.counties)

        # 创建年份映射
        self.years = sorted(self.data['Year'].unique())
        self.year_to_idx = {year: idx for idx, year in enumerate(self.years)}
        self.num_years = len(self.years)

        print(f"县域数量: {self.num_counties}, 年份数量: {self.num_years}")

    def create_graph_data(self, year):
        """为特定年份创建图数据"""
        year_data = self.data[self.data['Year'] == year]

        if len(year_data) == 0:
            return None

        # 创建节点特征
        node_features = []
        node_targets = []
        node_counties = []

        for county in self.counties:
            county_data = year_data[year_data['County'] == county]

            if len(county_data) > 0:
                features = county_data[self.feature_cols].iloc[0].values
                target = county_data[self.target_col].iloc[0] - 1  # 转换为0,1,2
            else:
                # 如果该年份没有数据，使用均值填充
                features = self.data[self.data['County'] == county][self.feature_cols].mean().values
                target = 1  # 默认轻度发病

            node_features.append(features)
            node_targets.append(target)
            node_counties.append(county)

        x = torch.FloatTensor(node_features)
        y = torch.LongTensor(node_targets)

        # 创建边（基于地理距离）
        edge_index, edge_attr = self._create_edges(year_data)

        return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr,
                   county_names=node_counties, year=year)

    def _create_edges(self, year_data):
        """基于地理距离创建图的边"""
        counties_data = {}

        for county in self.counties:
            county_info = self.data[self.data['County'] == county]
            if len(county_info) > 0:
                lat = county_info['Latitude'].iloc[0]
                lon = county_info['Longitude'].iloc[0]
                counties_data[county] = (lat, lon)

        # 创建边索引
        edges = []
        edge_attrs = []

        for i, county1 in enumerate(self.counties):
            for j, county2 in enumerate(self.counties):
                if i != j and county1 in counties_data and county2 in counties_data:
                    lat1, lon1 = counties_data[county1]
                    lat2, lon2 = counties_data[county2]

                    # 计算地理距离
                    distance = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

                    # 如果距离小于阈值，则创建边
                    if distance < 2.0:  # 2度约200km
                        edges.append([i, j])
                        edge_attrs.append([distance])

        if len(edges) == 0:
            # 如果没有边，创建自环
            edges = [[i, i] for i in range(len(self.counties))]
            edge_attrs = [[0.0] for _ in range(len(self.counties))]

        edge_index = torch.LongTensor(edges).t().contiguous()
        edge_attr = torch.FloatTensor(edge_attrs)

        return edge_index, edge_attr

class BiLSTMModel(nn.Module):
    """BiLSTM模型用于时间序列预测"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=3, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)

        # BiLSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size * 2)

        # 全连接层
        output = self.fc(last_output)

        return output

class GCNModel(nn.Module):
    """图卷积网络模型"""

    def __init__(self, input_size, hidden_size=64, num_classes=3, num_layers=2, dropout=0.2):
        super(GCNModel, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_size, hidden_size))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_size, hidden_size))

        if num_layers > 1:
            self.convs.append(GCNConv(hidden_size, hidden_size))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_attr.squeeze() if edge_attr is not None else None)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        out = self.classifier(x)
        return out

class GATModel(nn.Module):
    """图注意力网络模型"""

    def __init__(self, input_size, hidden_size=64, num_classes=3, num_heads=4, dropout=0.2):
        super(GATModel, self).__init__()

        self.gat1 = GATConv(input_size, hidden_size, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_size * num_heads, hidden_size, heads=1, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)

        out = self.classifier(x)
        return out

class SpatialTemporalFusionModel(nn.Module):
    """时空融合模型：结合BiLSTM和GNN"""

    def __init__(self, input_size, hidden_size=64, num_classes=3, num_layers=2, dropout=0.2):
        super(SpatialTemporalFusionModel, self).__init__()

        # 时间维度：BiLSTM
        self.bilstm = BiLSTMModel(input_size, hidden_size, num_layers, num_classes, dropout)

        # 空间维度：GCN
        self.gcn = GCNModel(input_size, hidden_size, num_classes, num_layers, dropout)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, temporal_data, spatial_data):
        # 时间维度预测
        temporal_out = self.bilstm(temporal_data)  # (batch_size, num_classes)

        # 空间维度预测
        spatial_out = self.gcn(spatial_data)      # (num_nodes, num_classes)

        # 选择对应县的空间预测
        # 这里需要根据batch中的县ID来选择对应的空间特征
        # 简化处理：使用平均池化
        spatial_pooled = global_mean_pool(spatial_out, torch.zeros(spatial_data.num_nodes, dtype=torch.long))
        spatial_expanded = spatial_pooled.expand(temporal_out.size(0), -1)

        # 融合预测
        combined = torch.cat([temporal_out, spatial_expanded], dim=1)
        output = self.fusion(combined)

        return output

class DeepLearningTrainer:
    """深度学习模型训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载数据
        self.load_data()

        # 特征缩放器
        self.scaler = StandardScaler()

    def load_data(self):
        """加载数据"""
        print("Loading data for deep learning models...")

        self.train_data = pd.read_csv(self.config.TRAIN_DATA_PATH)
        self.val_data = pd.read_csv(self.config.VAL_DATA_PATH)

        # 合并数据用于创建图结构
        self.full_data = pd.concat([self.train_data, self.val_data], ignore_index=True)

        print(f"Training data: {len(self.train_data)} samples")
        print(f"Validation data: {len(self.val_data)} samples")

    def prepare_data(self):
        """准备数据"""
        print("Preparing data for deep learning models...")

        # 特征列
        feature_cols = self.config.ALL_FEATURES

        # 创建时间序列数据集
        self.train_dataset = PestDataset(
            self.train_data,
            sequence_length=3,
            feature_cols=feature_cols
        )
        self.val_dataset = PestDataset(
            self.val_data,
            sequence_length=3,
            feature_cols=feature_cols
        )

        # 创建时空数据集
        self.spatial_temporal_dataset = SpatialTemporalDataset(
            self.full_data,
            feature_cols=feature_cols
        )

        print(f"Time series samples - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")

    def train_bilstm_model(self):
        """训练BiLSTM模型"""
        print("\n=== Training BiLSTM Model ===")

        # 创建数据加载器
        train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=16, shuffle=False)

        # 初始化模型
        input_size = len(self.config.ALL_FEATURES)
        model = BiLSTMModel(input_size=input_size).to(self.device)

        # 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # 训练循环
        num_epochs = 100
        best_val_f1 = 0.0
        patience = 10
        patience_counter = 0

        train_losses = []
        val_losses = []
        val_f1s = []

        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                sequences = batch['sequence'].to(self.device)
                targets = batch['target'].squeeze(-1).to(self.device)  # 移除最后一个维度

                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 验证阶段
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    sequences = batch['sequence'].to(self.device)
                    targets = batch['target'].squeeze(-1).to(self.device)

                    outputs = model(sequences)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

            val_loss /= len(val_loader)
            val_f1 = f1_score(all_targets, all_preds, average='weighted')

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_f1s.append(val_f1)

            # 早停
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), 'models/county_level/bilstm_best.pth')
                print(f"  New best model saved with F1: {val_f1:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')

        # 加载最佳模型（如果存在）
        if os.path.exists('models/county_level/bilstm_best.pth'):
            model.load_state_dict(torch.load('models/county_level/bilstm_best.pth'))
        else:
            # 如果没有保存模型，保存当前模型
            torch.save(model.state_dict(), 'models/county_level/bilstm_best.pth')
            print("No best model found, saving current model")

        # 最终评估
        model.eval()
        final_preds = []
        final_targets = []

        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequence'].to(self.device)
                targets = batch['target'].squeeze(-1).to(self.device)

                outputs = model(sequences)
                preds = torch.argmax(outputs, dim=1)

                final_preds.extend(preds.cpu().numpy())
                final_targets.extend(targets.cpu().numpy())

        final_accuracy = accuracy_score(final_targets, final_preds)
        final_f1 = f1_score(final_targets, final_preds, average='weighted')

        print(f'BiLSTM Final Results - Accuracy: {final_accuracy:.4f}, F1: {final_f1:.4f}')

        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_f1s': val_f1s,
            'final_accuracy': final_accuracy,
            'final_f1': final_f1
        }

    def train_gcn_model(self):
        """训练图卷积网络模型"""
        print("\n=== Training GCN Model ===")

        # 创建图数据
        train_years = [2019, 2020, 2021, 2022]
        val_years = [2023]

        train_graphs = []
        val_graphs = []

        for year in train_years:
            graph_data = self.spatial_temporal_dataset.create_graph_data(year)
            if graph_data is not None:
                train_graphs.append(graph_data)

        for year in val_years:
            graph_data = self.spatial_temporal_dataset.create_graph_data(year)
            if graph_data is not None:
                val_graphs.append(graph_data)

        print(f"Training graphs: {len(train_graphs)}, Validation graphs: {len(val_graphs)}")

        # 初始化模型
        input_size = len(self.config.ALL_FEATURES)
        model = GCNModel(input_size=input_size).to(self.device)

        # 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        # 训练循环
        num_epochs = 200
        best_val_f1 = 0.0
        patience = 20
        patience_counter = 0

        train_losses = []
        val_losses = []
        val_f1s = []

        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0

            for graph_data in train_graphs:
                graph_data = graph_data.to(self.device)
                optimizer.zero_grad()
                outputs = model(graph_data)
                loss = criterion(outputs, graph_data.y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_graphs)

            # 验证阶段
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for graph_data in val_graphs:
                    graph_data = graph_data.to(self.device)
                    outputs = model(graph_data)
                    loss = criterion(outputs, graph_data.y)
                    val_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(graph_data.y.cpu().numpy())

            val_loss /= len(val_graphs)
            if len(all_targets) > 0:
                val_f1 = f1_score(all_targets, all_preds, average='weighted')
            else:
                val_f1 = 0.0

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_f1s.append(val_f1)

            # 早停
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), 'models/county_level/gcn_best.pth')
                print(f"  New best model saved with F1: {val_f1:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')

        # 加载最佳模型（如果存在）
        if os.path.exists('models/county_level/gcn_best.pth'):
            model.load_state_dict(torch.load('models/county_level/gcn_best.pth'))
        else:
            # 如果没有保存模型，保存当前模型
            torch.save(model.state_dict(), 'models/county_level/gcn_best.pth')
            print("No best model found, saving current model")

        # 最终评估
        model.eval()
        final_preds = []
        final_targets = []

        with torch.no_grad():
            for graph_data in val_graphs:
                graph_data = graph_data.to(self.device)
                outputs = model(graph_data)
                preds = torch.argmax(outputs, dim=1)

                final_preds.extend(preds.cpu().numpy())
                final_targets.extend(graph_data.y.cpu().numpy())

        if len(final_targets) > 0:
            final_accuracy = accuracy_score(final_targets, final_preds)
            final_f1 = f1_score(final_targets, final_preds, average='weighted')
        else:
            final_accuracy = 0.0
            final_f1 = 0.0

        print(f'GCN Final Results - Accuracy: {final_accuracy:.4f}, F1: {final_f1:.4f}')

        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_f1s': val_f1s,
            'final_accuracy': final_accuracy,
            'final_f1': final_f1
        }

    def save_results(self, results):
        """保存训练结果"""
        print("\n=== Saving Deep Learning Results ===")

        os.makedirs('results/deep_learning', exist_ok=True)

        # 转换结果为可序列化格式
        serializable_results = {}
        for model_name, result in results.items():
            serializable_results[model_name] = {
                'final_accuracy': float(result['final_accuracy']),
                'final_f1': float(result['final_f1']),
                'train_losses': [float(loss) for loss in result['train_losses']],
                'val_losses': [float(loss) for loss in result['val_losses']],
                'val_f1s': [float(f1) for f1 in result['val_f1s']]
            }

        # 保存结果
        with open('results/deep_learning/deep_learning_results.json', 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        print("Deep learning results saved to results/deep_learning/deep_learning_results.json")

    def run_training(self):
        """运行深度学习模型训练"""
        print("=== Starting Deep Learning Model Training ===")

        # 准备数据
        self.prepare_data()

        # 确保模型目录存在
        os.makedirs('models/county_level', exist_ok=True)

        # 训练BiLSTM模型
        bilstm_results = self.train_bilstm_model()

        # 训练GCN模型
        gcn_results = self.train_gcn_model()

        # 保存结果
        results = {
            'BiLSTM': bilstm_results,
            'GCN': gcn_results
        }
        self.save_results(results)

        # 打印总结
        print("\n" + "="*60)
        print("DEEP LEARNING TRAINING SUMMARY")
        print("="*60)
        print(f"BiLSTM - Accuracy: {bilstm_results['final_accuracy']:.4f}, F1: {bilstm_results['final_f1']:.4f}")
        print(f"GCN - Accuracy: {gcn_results['final_accuracy']:.4f}, F1: {gcn_results['final_f1']:.4f}")
        print("="*60)

        return results

def main():
    """主函数"""
    from county_level_config import CountyLevelConfig

    # 创建配置
    config = CountyLevelConfig()

    # 创建训练器
    trainer = DeepLearningTrainer(config)

    # 运行训练
    results = trainer.run_training()

    print("Deep learning training completed!")

if __name__ == "__main__":
    main()