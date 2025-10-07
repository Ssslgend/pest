#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练好的改进模型预测山东省各县发病概率并生成热力图
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入模型和配置
from improved_model_trainer import ImprovedBiLSTMGCNModel, ImprovedDataset
from improved_county_config import ImprovedCountyLevelConfig

class ShandongHeatmapPredictor:
    """山东省发病概率热力图预测器"""

    def __init__(self):
        self.config = ImprovedCountyLevelConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载训练好的模型
        self.load_trained_model()

        # 加载山东省地理数据
        self.load_shandong_geometry()

    def load_trained_model(self):
        """加载训练好的模型"""
        print("=== 加载训练好的模型 ===")

        # 模型参数
        input_size = self.config.NUM_FEATURES
        hidden_size = 64
        num_classes = self.config.NUM_CLASSES

        # 创建模型
        self.model = ImprovedBiLSTMGCNModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=0.3
        ).to(self.device)

        # 加载模型权重
        model_path = os.path.join(self.config.MODEL_SAVE_DIR, 'improved_bilstm_gcn_model.pth')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"成功加载模型权重: {model_path}")
        else:
            raise FileNotFoundError(f"模型权重文件不存在: {model_path}")

        # 加载模型信息
        info_path = os.path.join(self.config.MODEL_SAVE_DIR, 'improved_model_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                self.model_info = json.load(f)
            print(f"模型训练时间: {self.model_info.get('training_date', 'Unknown')}")

        self.model.eval()

    def load_shandong_geometry(self):
        """加载山东省地理边界数据"""
        print("=== 加载山东省地理数据 ===")

        # 尝试加载山东省县级边界数据
        shandong_geo_path = 'datas/shandong_county_boundaries.geojson'

        if os.path.exists(shandong_geo_path):
            self.shandong_geo = gpd.read_file(shandong_geo_path)
            print(f"成功加载山东省地理边界数据: {len(self.shandong_geo)} 个县")
        else:
            print("山东省地理边界数据不存在，创建模拟地理数据")
            self.create_mock_shandong_geometry()

    def create_mock_shandong_geometry(self):
        """创建模拟的山东省地理数据"""
        # 读取数据中的所有县
        data = pd.read_csv(self.config.ENHANCED_COMPLETE_DATA_PATH)
        counties = data['County'].unique()

        # 创建模拟的地理位置（经纬度）
        np.random.seed(42)
        mock_geometries = []

        for i, county in enumerate(counties):
            # 模拟山东省范围内的经纬度
            lon = np.random.uniform(114.5, 122.5)  # 山东省经度范围
            lat = np.random.uniform(34.5, 38.5)    # 山东省纬度范围

            # 创建一个小的矩形区域代表县
            from shapely.geometry import Polygon
            polygon = Polygon([
                (lon-0.1, lat-0.1),
                (lon+0.1, lat-0.1),
                (lon+0.1, lat+0.1),
                (lon-0.1, lat+0.1)
            ])

            mock_geometries.append({
                'County': county,
                'Longitude': lon,
                'Latitude': lat,
                'geometry': polygon
            })

        self.shandong_geo = gpd.GeoDataFrame(mock_geometries)
        print(f"创建了 {len(self.shandong_geo)} 个县的模拟地理数据")

    def prepare_county_data(self, target_year=2023):
        """准备要预测的县数据"""
        print(f"=== 准备 {target_year} 年的县数据 ===")

        # 读取增强数据
        data = pd.read_csv(self.config.ENHANCED_COMPLETE_DATA_PATH)

        # 获取所有县
        all_counties = data['County'].unique()
        print(f"总共有 {len(all_counties)} 个县")

        # 为每个县准备最新的时间序列数据
        county_predictions = []

        for county in all_counties:
            county_data = data[data['County'] == county].sort_values('Year')

            # 获取最新的几年数据用于预测
            if len(county_data) >= 2:
                # 使用最新的2年数据
                recent_data = county_data.tail(2)

                # 创建时间序列样本
                sequence_features = recent_data[self.config.ALL_FEATURES].values

                county_predictions.append({
                    'county': county,
                    'year': target_year,
                    'features': sequence_features,
                    'sequence_years': list(recent_data['Year'])
                })

        print(f"准备了 {len(county_predictions)} 个县的预测数据")
        return county_predictions

    def predict_county_probabilities(self, county_data):
        """预测各县发病概率"""
        print("=== 预测各县发病概率 ===")

        predictions = []

        # 创建数据集用于标准化
        all_features = np.concatenate([sample['features'] for sample in county_data])
        scaler = StandardScaler()
        scaler.fit(all_features)

        with torch.no_grad():
            for sample in county_data:
                # 标准化特征
                features_scaled = scaler.transform(sample['features'])
                sequence = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)

                # 模型预测
                output = self.model(sequence)
                probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]

                # 获取预测类别
                predicted_class = np.argmax(probabilities)

                predictions.append({
                    'County': sample['county'],
                    'Year': sample['year'],
                    'Predicted_Class': predicted_class,
                    'Predicted_Severity': self.config.CLASS_NAMES[predicted_class],
                    'Probability_0': probabilities[0],
                    'Probability_1': probabilities[1],
                    'Probability_2': probabilities[2],
                    'Probability_3': probabilities[3],
                    'Max_Probability': np.max(probabilities),
                    'Sequence_Years': sample['sequence_years']
                })

        return pd.DataFrame(predictions)

    def merge_with_geometry(self, predictions_df):
        """将预测结果与地理数据合并"""
        print("=== 合并预测结果与地理数据 ===")

        # 确保列名一致
        geo_df = self.shandong_geo.copy()
        if 'County' not in geo_df.columns and 'county' in geo_df.columns:
            geo_df = geo_df.rename(columns={'county': 'County'})

        # 合并数据
        merged_df = geo_df.merge(predictions_df, on='County', how='left')

        # 检查未匹配的县
        unmatched_counties = predictions_df[~predictions_df['County'].isin(geo_df['County'])]
        if len(unmatched_counties) > 0:
            print(f"警告: {len(unmatched_counties)} 个县未能匹配到地理数据")
            print(f"未匹配的县: {list(unmatched_counties['County'])}")

        print(f"成功合并 {len(merged_df)} 个县的数据")
        return merged_df

    def create_heatmap(self, merged_df, value_column='Predicted_Class', title=None):
        """创建热力图"""
        print(f"=== 创建{value_column}热力图 ===")

        os.makedirs('results/heatmaps', exist_ok=True)

        # 创建图形
        fig, ax = plt.subplots(figsize=(15, 10))

        # 处理缺失值
        merged_df[value_column] = merged_df[value_column].fillna(0)

        # 绘制热力图
        if value_column == 'Predicted_Class':
            # 分类变量使用离散颜色
            colors = self.config.CLASS_COLORS
            cmap = plt.cm.colors.ListedColormap(colors)
            merged_df.plot(column=value_column, cmap=cmap, linewidth=0.8,
                          edgecolor='0.8', legend=True, ax=ax,
                          legend_kwds={'label': "预测发病程度",
                                     'orientation': "horizontal",
                                     'shrink': 0.8})
        else:
            # 连续变量使用渐变色
            merged_df.plot(column=value_column, cmap='YlOrRd', linewidth=0.8,
                          edgecolor='0.8', legend=True, ax=ax,
                          legend_kwds={'label': value_column,
                                     'orientation': "horizontal",
                                     'shrink': 0.8})

        # 设置标题和标签
        if title is None:
            title = f'山东省美国白蛾发病程度预测热力图 ({value_column})'
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('经度', fontsize=12)
        ax.set_ylabel('纬度', fontsize=12)

        # 添加网格
        ax.grid(True, alpha=0.3)

        # 保存图片
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'shandong_{value_column}_heatmap_{timestamp}.png'
        filepath = os.path.join('results/heatmaps', filename)

        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"热力图已保存到: {filepath}")
        return filepath

    def create_multiple_visualizations(self, merged_df):
        """创建多种可视化"""
        print("=== 创建多种可视化 ===")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 1. 发病程度预测热力图
        self.create_heatmap(merged_df, 'Predicted_Class',
                          f'山东省美国白蛾发病程度预测 ({datetime.now().year}年)')

        # 2. 最大概率热力图
        self.create_heatmap(merged_df, 'Max_Probability',
                          '预测置信度热力图')

        # 3. 重度发病概率热力图
        self.create_heatmap(merged_df, 'Probability_3',
                          '重度发病概率预测热力图')

        # 4. 创建统计图表
        self.create_statistics_charts(merged_df, timestamp)

        # 5. 保存预测结果
        self.save_predictions(merged_df, timestamp)

    def create_statistics_charts(self, merged_df, timestamp):
        """创建统计图表"""
        os.makedirs('results/heatmaps', exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 发病程度分布饼图
        severity_counts = merged_df['Predicted_Severity'].value_counts()
        colors = self.config.CLASS_COLORS[:len(severity_counts)]
        axes[0, 0].pie(severity_counts.values, labels=severity_counts.index,
                      autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0, 0].set_title('预测发病程度分布')

        # 2. 各县预测概率分布直方图
        axes[0, 1].hist(merged_df['Max_Probability'], bins=20, alpha=0.7,
                       color='skyblue', edgecolor='black')
        axes[0, 1].set_title('预测置信度分布')
        axes[0, 1].set_xlabel('最大概率')
        axes[0, 1].set_ylabel('县数量')

        # 3. 重度发病概率分布
        axes[1, 0].hist(merged_df['Probability_3'], bins=20, alpha=0.7,
                       color='salmon', edgecolor='black')
        axes[1, 0].set_title('重度发病概率分布')
        axes[1, 0].set_xlabel('重度发病概率')
        axes[1, 0].set_ylabel('县数量')

        # 4. 各程度概率箱线图
        prob_columns = ['Probability_0', 'Probability_1', 'Probability_2', 'Probability_3']
        class_names = self.config.CLASS_NAMES
        box_data = [merged_df[col].dropna() for col in prob_columns]

        bp = axes[1, 1].boxplot(box_data, labels=class_names, patch_artist=True)
        for patch, color in zip(bp['boxes'], self.config.CLASS_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[1, 1].set_title('各发病程度概率分布')
        axes[1, 1].set_ylabel('概率')

        plt.tight_layout()
        plt.savefig(f'results/heatmaps/statistics_charts_{timestamp}.png',
                   dpi=300, bbox_inches='tight')
        plt.show()

        print(f"统计图表已保存到: results/heatmaps/statistics_charts_{timestamp}.png")

    def save_predictions(self, merged_df, timestamp):
        """保存预测结果"""
        os.makedirs('results/heatmaps', exist_ok=True)

        # 选择要保存的列
        save_columns = ['County', 'Year', 'Predicted_Class', 'Predicted_Severity',
                       'Probability_0', 'Probability_1', 'Probability_2', 'Probability_3',
                       'Max_Probability', 'Sequence_Years']

        save_df = merged_df[save_columns].copy()

        # 保存为CSV
        csv_path = f'results/heatmaps/shandong_predictions_{timestamp}.csv'
        save_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # 保存为JSON
        json_path = f'results/heatmaps/shandong_predictions_{timestamp}.json'
        save_df.to_json(json_path, orient='records', indent=2, force_ascii=False)

        print(f"预测结果已保存:")
        print(f"  CSV格式: {csv_path}")
        print(f"  JSON格式: {json_path}")

        # 打印统计摘要
        print(f"\n=== 预测结果摘要 ===")
        print(f"总县数: {len(save_df)}")
        print(f"预测年份: {save_df['Year'].iloc[0]}")
        print(f"\n发病程度分布:")
        severity_dist = save_df['Predicted_Severity'].value_counts()
        for severity, count in severity_dist.items():
            percentage = count / len(save_df) * 100
            print(f"  {severity}: {count} 个县 ({percentage:.1f}%)")

        print(f"\n高风险县 (重度发病概率 > 50%):")
        high_risk = save_df[save_df['Probability_3'] > 0.5]
        print(f"  {len(high_risk)} 个县")
        if len(high_risk) > 0:
            print(f"  县名: {', '.join(high_risk['County'].head(10).tolist())}")
            if len(high_risk) > 10:
                print(f"  ... 还有 {len(high_risk) - 10} 个县")

    def run_prediction(self, target_year=2023):
        """运行完整的预测流程"""
        print(f"=== 开始山东省发病概率预测流程 ===")
        print(f"预测年份: {target_year}")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. 准备数据
        county_data = self.prepare_county_data(target_year)

        # 2. 预测概率
        predictions_df = self.predict_county_probabilities(county_data)

        # 3. 合并地理数据
        merged_df = self.merge_with_geometry(predictions_df)

        # 4. 创建可视化
        self.create_multiple_visualizations(merged_df)

        print(f"\n=== 预测完成 ===")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return merged_df

def main():
    """主函数"""
    print("山东省美国白蛾发病概率热力图预测系统")
    print("=" * 50)

    # 创建预测器
    predictor = ShandongHeatmapPredictor()

    # 运行预测
    results = predictor.run_prediction(target_year=2023)

    print("\n热力图预测完成！请查看 results/heatmaps/ 目录下的输出文件。")

    return predictor, results

if __name__ == "__main__":
    predictor, results = main()