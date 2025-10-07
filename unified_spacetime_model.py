#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的时空深度学习模型 - 整合GCN和BiLSTM
包含健康县(0级)和发病县(1-3级)的完整山东省数据
训练后生成县级发病热力图
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import json
import os
from county_level_config import CountyLevelConfig
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class UnifiedSpatioTemporalDataset(Dataset):
    """统一的时空数据集，包含健康县和发病县"""

    def __init__(self, data, sequence_length=2, feature_cols=None, target_col='Severity_Level'):
        self.data = data
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols
        self.target_col = target_col

        # 创建时间序列样本
        self.samples = self._create_time_series_samples()

        # 标准化特征
        self.scaler = StandardScaler()
        if len(self.samples) > 0:
            all_features = np.concatenate([sample['features'] for sample in self.samples])
            self.scaler.fit(all_features)

        # 创建图数据
        self.graph_data = self._create_spatial_graph()

    def _create_time_series_samples(self):
        """创建时间序列样本"""
        samples = []
        counties = self.data['County'].unique()

        for county in counties:
            county_data = self.data[self.data['County'] == county].sort_values('Year')

            if len(county_data) >= self.sequence_length:
                for i in range(len(county_data) - self.sequence_length + 1):
                    sequence_data = county_data.iloc[i:i+self.sequence_length]

                    features = sequence_data[self.feature_cols].values
                    target = sequence_data.iloc[-1][self.target_col]
                    year = sequence_data.iloc[-1]['Year']

                    samples.append({
                        'county': county,
                        'year': year,
                        'features': features,
                        'target': target,
                        'sequence_data': sequence_data
                    })

        return samples

    def _create_spatial_graph(self):
        """创建空间图"""
        counties = self.data['County'].unique()
        county_to_idx = {county: idx for idx, county in enumerate(counties)}

        # 获取各县地理坐标
        county_coords = {}
        for county in counties:
            county_info = self.data[self.data['County'] == county].iloc[0]
            county_coords[county] = (county_info['Latitude'], county_info['Longitude'])

        # 创建边连接
        edges = []
        edge_weights = []

        for i, county1 in enumerate(counties):
            for j, county2 in enumerate(counties):
                if i != j:
                    lat1, lon1 = county_coords[county1]
                    lat2, lon2 = county_coords[county2]

                    # 计算地理距离
                    distance = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

                    # 创建连接（距离小于2度约200km）
                    if distance < 2.0:
                        edges.append([i, j])
                        edge_weights.append(1.0 / (1 + distance))

        # 确保图连通
        if len(edges) == 0:
            for i in range(len(counties)):
                edges.append([i, i])
                edge_weights.append(1.0)

        edge_index = torch.LongTensor(edges).t().contiguous()
        edge_attr = torch.FloatTensor(edge_weights)

        return {
            'counties': counties,
            'county_to_idx': county_to_idx,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'county_coords': county_coords
        }

    def create_graph_data_for_year(self, year):
        """为特定年份创建图数据"""
        year_data = self.data[self.data['Year'] == year]

        if len(year_data) == 0:
            return None

        # 创建节点特征
        node_features = []
        node_targets = []

        for county in self.graph_data['counties']:
            county_data = year_data[year_data['County'] == county]

            if len(county_data) > 0:
                features = county_data[self.feature_cols].iloc[0].values
                target = county_data[self.target_col].iloc[0]
            else:
                # 如果该年份没有数据，使用历史均值
                county_hist = self.data[self.data['County'] == county]
                if len(county_hist) > 0:
                    features = county_hist[self.feature_cols].mean().values
                    target = int(county_hist[self.target_col].mode().iloc[0])
                else:
                    features = np.zeros(len(self.feature_cols))
                    target = 0  # 默认健康

            features_scaled = self.scaler.transform([features])[0]
            node_features.append(features_scaled)
            node_targets.append(target)

        x = torch.FloatTensor(node_features)
        y = torch.LongTensor(node_targets)

        return Data(
            x=x,
            y=y,
            edge_index=self.graph_data['edge_index'],
            edge_attr=self.graph_data['edge_attr'],
            county_names=self.graph_data['counties'],
            year=year
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        features_scaled = self.scaler.transform(sample['features'])

        return {
            'county': sample['county'],
            'year': int(sample['year']),
            'sequence': torch.FloatTensor(features_scaled),
            'target': torch.LongTensor([int(sample['target'])])
        }

class UnifiedSpatioTemporalModel(nn.Module):
    """统一的时空深度学习模型"""

    def __init__(self, input_size, hidden_size=32, num_classes=4, dropout=0.3):
        """
        Args:
            input_size: 特征维度
            hidden_size: 隐藏层大小
            num_classes: 类别数 (0=健康, 1=轻度, 2=中度, 3=重度)
            dropout: dropout率
        """
        super(UnifiedSpatioTemporalModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # BiLSTM分支 - 时间建模
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # LSTM注意力机制
        self.lstm_attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # GCN分支 - 空间建模
        self.gcn_conv1 = GCNConv(input_size, hidden_size)
        self.gcn_conv2 = GCNConv(hidden_size, hidden_size)

        # 特征融合
        self.temporal_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.spatial_proj = nn.Linear(hidden_size, hidden_size)

        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, temporal_data, spatial_data):
        """
        Args:
            temporal_data: 时间序列数据 [batch_size, seq_len, features]
            spatial_data: 空间图数据
        """
        # BiLSTM分支
        lstm_out, _ = self.bilstm(temporal_data)  # [batch_size, seq_len, hidden_size*2]

        # 时间注意力
        attention_weights = self.lstm_attention(lstm_out)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_weights, dim=1)

        # 加权平均
        temporal_features = torch.sum(lstm_out * attention_weights, dim=1)  # [batch_size, hidden_size*2]
        temporal_features = self.temporal_proj(temporal_features)  # [batch_size, hidden_size]

        # GCN分支
        x, edge_index, edge_attr = spatial_data.x, spatial_data.edge_index, spatial_data.edge_attr

        x1 = F.relu(self.gcn_conv1(x, edge_index, edge_weight=edge_attr))
        x1 = F.dropout(x1, p=0.2, training=self.training)

        x2 = self.gcn_conv2(x1, edge_index, edge_weight=edge_attr)
        spatial_features = x2  # [num_nodes, hidden_size]

        # 特征融合
        # 对于每个节点，融合其时间特征和空间特征
        batch_size = temporal_features.size(0)
        num_nodes = spatial_features.size(0)

        # 扩展时间特征以匹配所有节点
        temporal_expanded = temporal_features.unsqueeze(1).expand(-1, num_nodes, -1)  # [batch_size, num_nodes, hidden_size]
        spatial_expanded = spatial_features.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_nodes, hidden_size]

        # 拼接特征
        fused_features = torch.cat([temporal_expanded, spatial_expanded], dim=2)  # [batch_size, num_nodes, hidden_size*2]

        # 通过融合层
        fused_features = fused_features.view(-1, self.hidden_size * 2)  # [batch_size*num_nodes, hidden_size*2]
        output = self.fusion_layer(fused_features)  # [batch_size*num_nodes, num_classes]
        output = output.view(batch_size, num_nodes, self.num_classes)  # [batch_size, num_nodes, num_classes]

        return output

class UnifiedModelTrainer:
    """统一模型训练器"""

    def __init__(self):
        self.config = CountyLevelConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载数据
        self.load_unified_data()

    def load_unified_data(self):
        """加载统一的完整数据"""
        print("Loading unified spatiotemporal data...")

        # 加载完整数据
        complete_data = pd.read_csv(self.config.COMPLETE_DATA_PATH)

        # 确保包含所有县的数据（包括健康的县）
        print(f"原始数据: {len(complete_data)} 样本")
        print(f"包含县数: {complete_data['County'].nunique()}")
        print(f"年份范围: {complete_data['Year'].min()}-{complete_data['Year'].max()}")

        # 检查发病程度分布
        severity_dist = complete_data['Severity_Level'].value_counts().sort_index()
        print("发病程度分布:")
        for level, count in severity_dist.items():
            level_name = {0: '健康', 1: '轻度', 2: '中度', 3: '重度'}.get(level, f'级别{level}')
            print(f"  {level_name}({level}级): {count} 样本")

        self.data = complete_data

    def prepare_datasets(self):
        """准备训练和测试数据集"""
        print("Preparing unified datasets...")

        # 分割训练和测试数据
        train_data = self.data[self.data['Year'] <= 2020]
        test_data = self.data[self.data['Year'] > 2020]

        print(f"训练数据: {len(train_data)} 样本 ({train_data['Year'].min()}-{train_data['Year'].max()})")
        print(f"测试数据: {len(test_data)} 样本 ({test_data['Year'].min()}-{test_data['Year'].max()})")

        # 创建数据集
        self.train_dataset = UnifiedSpatioTemporalDataset(
            train_data,
            sequence_length=2,
            feature_cols=self.config.ALL_FEATURES,
            target_col='Severity_Level'
        )

        self.test_dataset = UnifiedSpatioTemporalDataset(
            test_data,
            sequence_length=2,
            feature_cols=self.config.ALL_FEATURES,
            target_col='Severity_Level'
        )

        print(f"训练时间序列样本: {len(self.train_dataset)}")
        print(f"测试时间序列样本: {len(self.test_dataset)}")

        # 创建图数据
        train_years = sorted(train_data['Year'].unique())
        test_years = sorted(test_data['Year'].unique())

        self.train_graphs = []
        for year in train_years:
            graph = self.train_dataset.create_graph_data_for_year(year)
            if graph is not None:
                self.train_graphs.append(graph)

        self.test_graphs = []
        for year in test_years:
            graph = self.test_dataset.create_graph_data_for_year(year)
            if graph is not None:
                self.test_graphs.append(graph)

        print(f"训练图数据: {len(self.train_graphs)} 年份")
        print(f"测试图数据: {len(self.test_graphs)} 年份")

    def train_model(self):
        """训练统一模型"""
        print("\n=== Training Unified Spatiotemporal Model ===")

        # 初始化模型
        model = UnifiedSpatioTemporalModel(
            input_size=len(self.config.ALL_FEATURES),
            hidden_size=32,
            num_classes=4,  # 0=健康, 1=轻度, 2=中度, 3=重度
            dropout=0.3
        ).to(self.device)

        # 优化器和损失函数
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        # 数据加载器
        train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)

        # 训练循环
        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(50):
            model.train()
            total_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                sequences = batch['sequence'].to(self.device)

                # 获取对应年份的图数据
                batch_years = batch['year'].cpu().numpy()
                graph_data = None

                for year in batch_years:
                    for graph in self.train_graphs:
                        if graph.year == year:
                            graph_data = graph.to(self.device)
                            break
                    if graph_data is not None:
                        break

                if graph_data is None:
                    continue

                optimizer.zero_grad()

                # 前向传播
                outputs = model(sequences, graph_data)

                # 计算损失
                batch_size = sequences.size(0)
                num_nodes = graph_data.x.size(0)

                # 扩展目标
                targets = batch['target'].squeeze(-1).to(self.device)
                targets_expanded = targets.unsqueeze(1).expand(-1, num_nodes)  # [batch_size, num_nodes]
                targets_expanded = targets_expanded.reshape(-1)  # [batch_size*num_nodes]

                outputs_flat = outputs.reshape(batch_size * num_nodes, -1)  # [batch_size*num_nodes, num_classes]

                loss = criterion(outputs_flat, targets_expanded)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            scheduler.step(avg_loss)

            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/50], Loss: {avg_loss:.4f}')

            # 早停
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), 'models/unified_spacetime_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

        print(f'Best training loss: {best_loss:.4f}')
        print('Model saved to models/unified_spacetime_best.pth')

        return model

    def test_model(self, model):
        """测试模型并生成预测结果"""
        print("\n=== Testing Model and Generating Predictions ===")

        model.eval()
        all_predictions = []
        all_targets = []
        all_counties = []
        all_years = []
        all_probabilities = []

        with torch.no_grad():
            for graph_data in self.test_graphs:
                graph_data = graph_data.to(self.device)
                year = graph_data.year

                # 获取该年份的时间序列数据
                year_samples = [sample for sample in self.test_dataset.samples if sample['year'] == year]

                if not year_samples:
                    continue

                # 批量处理
                sequences = []
                targets = []
                counties = []

                for sample in year_samples:
                    features_scaled = self.test_dataset.scaler.transform(sample['features'])
                    sequences.append(torch.FloatTensor(features_scaled))
                    targets.append(sample['target'])
                    counties.append(sample['county'])

                if not sequences:
                    continue

                sequences = torch.stack(sequences).to(self.device)

                # 预测
                outputs = model(sequences, graph_data)  # [batch_size, num_nodes, num_classes]
                probabilities = F.softmax(outputs, dim=2)
                predictions = torch.argmax(outputs, dim=2)

                # 收集结果（取第一个样本的预测作为该县的预测）
                for i, (county, target) in enumerate(zip(counties, targets)):
                    # 在图数据中查找县索引
                    county_names = graph_data.county_names.tolist() if hasattr(graph_data.county_names, 'tolist') else graph_data.county_names
                    county_idx = county_names.index(county) if county in county_names else 0

                    pred = predictions[i, county_idx].item()
                    prob = probabilities[i, county_idx].cpu().numpy()

                    all_predictions.append(pred)
                    all_targets.append(target)
                    all_counties.append(county)
                    all_years.append(year)
                    all_probabilities.append(prob)

        # 计算指标
        accuracy = accuracy_score(all_targets, all_predictions)
        f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
        f1_macro = f1_score(all_targets, all_predictions, average='macro')

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test F1-Weighted: {f1_weighted:.4f}")
        print(f"Test F1-Macro: {f1_macro:.4f}")

        # 详细分类报告
        class_names = ['健康(0级)', '轻度(1级)', '中度(2级)', '重度(3级)']
        unique_labels = sorted(list(set(all_targets + all_predictions)))
        valid_class_names = [class_names[i] for i in unique_labels]

        print("\nClassification Report:")
        print(classification_report(all_targets, all_predictions,
                                  labels=unique_labels,
                                  target_names=valid_class_names))

        # 保存预测结果
        results_df = pd.DataFrame({
            'County': all_counties,
            'Year': all_years,
            'Actual_Severity': all_targets,
            'Predicted_Severity': all_predictions,
            'Probability_0': [prob[0] for prob in all_probabilities],
            'Probability_1': [prob[1] for prob in all_probabilities],
            'Probability_2': [prob[2] for prob in all_probabilities],
            'Probability_3': [prob[3] for prob in all_probabilities]
        })

        results_df.to_csv('results/unified_model_predictions.csv', index=False, encoding='utf-8')
        print("Predictions saved to results/unified_model_predictions.csv")

        return results_df

    def generate_heat_map(self, predictions_df):
        """生成山东省县级美国白蛾发病热力图"""
        print("\n=== Generating Shandong Province Heat Map ===")

        # 创建结果目录
        os.makedirs('results/heatmaps', exist_ok=True)

        # 获取最新年份的预测结果
        latest_year = predictions_df['Year'].max()
        year_predictions = predictions_df[predictions_df['Year'] == latest_year]

        print(f"Generating heat map for year {latest_year}")
        print(f"Total counties: {len(year_predictions)}")

        # 统计发病程度分布
        severity_counts = year_predictions['Predicted_Severity'].value_counts().sort_index()
        print("Predicted severity distribution:")
        for level, count in severity_counts.items():
            level_name = {0: '健康', 1: '轻度', 2: '中度', 3: '重度'}.get(level, f'级别{level}')
            print(f"  {level_name}({level}级): {count} 县")

        # 创建热力图
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # 1. 预测发病程度热力图
        pivot_severity = year_predictions.pivot_table(
            values='Predicted_Severity',
            index='County',
            aggfunc='mean'
        ).sort_index()

        sns.heatmap(pivot_severity,
                   annot=True,
                   fmt='.1f',
                   cmap='YlOrRd',
                   ax=axes[0, 0],
                   cbar_kws={'label': '预测发病程度'})
        axes[0, 0].set_title(f'{latest_year}年美国白蛾发病程度预测热力图')
        axes[0, 0].set_xlabel('县')
        axes[0, 0].set_ylabel('发病程度')

        # 2. 发病概率热力图（重度）
        pivot_severe = year_predictions.pivot_table(
            values='Probability_3',
            index='County',
            aggfunc='mean'
        ).sort_index()

        sns.heatmap(pivot_severe,
                   annot=True,
                   fmt='.3f',
                   cmap='Reds',
                   ax=axes[0, 1],
                   cbar_kws={'label': '重度发病概率'})
        axes[0, 1].set_title(f'{latest_year}年重度发病概率热力图')
        axes[0, 1].set_xlabel('县')
        axes[0, 1].set_ylabel('重度发病概率')

        # 3. 各县发病程度分布条形图
        county_severity = year_predictions.groupby('County')['Predicted_Severity'].mean().sort_values(ascending=False)

        # 只显示前20个县
        top_counties = county_severity.head(20)

        bars = axes[1, 0].barh(range(len(top_counties)), top_counties.values,
                               color=['green' if x < 1 else 'yellow' if x < 2 else 'orange' if x < 3 else 'red'
                                     for x in top_counties.values])
        axes[1, 0].set_yticks(range(len(top_counties)))
        axes[1, 0].set_yticklabels(top_counties.index, fontsize=8)
        axes[1, 0].set_xlabel('预测发病程度')
        axes[1, 0].set_title(f'{latest_year}年发病程度前20县')
        axes[1, 0].invert_yaxis()

        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, top_counties.values)):
            axes[1, 0].text(value + 0.02, bar.get_y() + bar.get_height()/2,
                           f'{value:.1f}', ha='left', va='center', fontsize=8)

        # 4. 发病程度分布饼图
        severity_dist = year_predictions['Predicted_Severity'].value_counts().sort_index()
        colors = ['green', 'yellow', 'orange', 'red']
        labels = ['健康', '轻度', '中度', '重度']
        sizes = [severity_dist.get(i, 0) for i in range(4)]

        # 过滤掉数量为0的类别
        non_zero = [(size, label, color) for size, label, color in zip(sizes, labels, colors) if size > 0]
        if non_zero:
            sizes, labels, colors = zip(*non_zero)

        axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title(f'{latest_year}年发病程度分布')

        plt.tight_layout()
        plt.savefig(f'results/heatmaps/shandong_pest_heatmap_{latest_year}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        # 创建时间序列变化图
        if len(predictions_df['Year'].unique()) > 1:
            fig, axes = plt.subplots(2, 1, figsize=(15, 12))

            # 每年平均发病程度变化
            yearly_avg = predictions_df.groupby('Year')['Predicted_Severity'].mean()

            axes[0].plot(yearly_avg.index, yearly_avg.values, 'o-', linewidth=2, markersize=8)
            axes[0].set_title('山东省平均发病程度年际变化')
            axes[0].set_xlabel('年份')
            axes[0].set_ylabel('平均发病程度')
            axes[0].grid(True, alpha=0.3)

            # 各类别数量变化
            yearly_counts = predictions_df.groupby(['Year', 'Predicted_Severity']).size().unstack(fill_value=0)

            # 动态设置列名
            severity_names = {0: '健康', 1: '轻度', 2: '中度', 3: '重度'}
            actual_columns = [severity_names.get(col, f'级别{col}') for col in yearly_counts.columns]
            yearly_counts.columns = actual_columns

            yearly_counts.plot(kind='bar', ax=axes[1], stacked=True)
            axes[1].set_title('山东省发病程度类别数量年际变化')
            axes[1].set_xlabel('年份')
            axes[1].set_ylabel('县数量')
            axes[1].legend(title='发病程度')
            axes[1].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig('results/heatmaps/temporal_trends.png', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Heat maps saved to results/heatmaps/")
        print(f"Main heat map: results/heatmaps/shandong_pest_heatmap_{latest_year}.png")

        # 生成报告
        self.generate_prediction_report(predictions_df, latest_year)

    def generate_prediction_report(self, predictions_df, latest_year):
        """生成预测报告"""
        report = {
            'prediction_date': pd.Timestamp.now().isoformat(),
            'latest_year': int(latest_year),
            'total_counties': len(predictions_df[predictions_df['Year'] == latest_year]),
            'overall_accuracy': float(accuracy_score(
                predictions_df['Actual_Severity'],
                predictions_df['Predicted_Severity']
            )),
            'severity_distribution': {},
            'high_risk_counties': [],
            'model_performance': {}
        }

        # 发病程度分布
        year_data = predictions_df[predictions_df['Year'] == latest_year]
        severity_counts = year_data['Predicted_Severity'].value_counts().sort_index()

        for level, count in severity_counts.items():
            level_name = {0: '健康', 1: '轻度', 2: '中度', 3: '重度'}.get(level, f'级别{level}')
            report['severity_distribution'][level_name] = int(count)

        # 高风险县（中度及以上）
        high_risk = year_data[year_data['Predicted_Severity'] >= 2]
        high_risk_counties = high_risk.sort_values('Probability_3', ascending=False)

        for _, row in high_risk_counties.head(10).iterrows():
            report['high_risk_counties'].append({
                'county': row['County'],
                'predicted_severity': int(row['Predicted_Severity']),
                'severe_probability': float(row['Probability_3'])
            })

        # 模型性能
        report['model_performance'] = {
            'accuracy': float(accuracy_score(predictions_df['Actual_Severity'], predictions_df['Predicted_Severity'])),
            'f1_weighted': float(f1_score(predictions_df['Actual_Severity'], predictions_df['Predicted_Severity'], average='weighted')),
            'f1_macro': float(f1_score(predictions_df['Actual_Severity'], predictions_df['Predicted_Severity'], average='macro'))
        }

        # 保存报告
        with open('results/heatmaps/prediction_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 打印摘要
        print(f"\n=== Prediction Summary for {latest_year} ===")
        print(f"Total counties predicted: {report['total_counties']}")
        print(f"Overall accuracy: {report['overall_accuracy']:.4f}")
        print(f"Severity distribution:")
        for level, count in report['severity_distribution'].items():
            print(f"  {level}: {count} counties")

        if report['high_risk_counties']:
            print(f"\nTop 5 high-risk counties:")
            for i, county_info in enumerate(report['high_risk_counties'][:5], 1):
                print(f"  {i}. {county_info['county']} - Severity: {county_info['predicted_severity']}, "
                      f"Severe probability: {county_info['severe_probability']:.3f}")

    def run_complete_pipeline(self):
        """运行完整的训练-预测-可视化流程"""
        print("=== Starting Complete Unified Model Pipeline ===")

        # 确保模型目录存在
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)

        # 准备数据
        self.prepare_datasets()

        # 训练模型
        model = self.train_model()

        # 测试模型
        predictions_df = self.test_model(model)

        # 生成热力图
        self.generate_heat_map(predictions_df)

        print("\n=== Pipeline Complete ===")
        print("Results:")
        print("- Trained model: models/unified_spacetime_best.pth")
        print("- Predictions: results/unified_model_predictions.csv")
        print("- Heat maps: results/heatmaps/")
        print("- Report: results/heatmaps/prediction_report.json")

        return model, predictions_df

def main():
    """主函数"""
    trainer = UnifiedModelTrainer()
    model, predictions = trainer.run_complete_pipeline()
    return model, predictions

if __name__ == "__main__":
    model, predictions = main()