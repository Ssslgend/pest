#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度学习模型预测接口
支持BiLSTM和GCN模型预测
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
import joblib
import json
from county_level_config import CountyLevelConfig
import warnings
warnings.filterwarnings('ignore')

# 导入模型定义
from deep_learning_models import BiLSTMModel, GCNModel, SpatialTemporalDataset

class DeepLearningPredictor:
    """深度学习模型预测器"""

    def __init__(self):
        self.config = CountyLevelConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载数据用于图结构构建
        self.load_reference_data()

        # 特征缩放器
        self.scaler = joblib.load('models/county_level/scaler_standard.joblib')

    def load_reference_data(self):
        """加载参考数据"""
        complete_data = pd.read_csv(self.config.COMPLETE_DATA_PATH)
        self.spatial_temporal_dataset = SpatialTemporalDataset(
            complete_data,
            feature_cols=self.config.ALL_FEATURES
        )

    def load_bilstm_model(self, model_path='models/county_level/bilstm_best.pth'):
        """加载BiLSTM模型"""
        try:
            model = BiLSTMModel(input_size=len(self.config.ALL_FEATURES)).to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            self.bilstm_model = model
            print(f"BiLSTM模型加载成功: {model_path}")
            return True
        except Exception as e:
            print(f"BiLSTM模型加载失败: {e}")
            return False

    def load_gcn_model(self, model_path='models/county_level/gcn_best.pth'):
        """加载GCN模型"""
        try:
            model = GCNModel(input_size=len(self.config.ALL_FEATURES)).to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            self.gcn_model = model
            print(f"GCN模型加载成功: {model_path}")
            return True
        except Exception as e:
            print(f"GCN模型加载失败: {e}")
            return False

    def prepare_time_series_data(self, data, sequence_length=3):
        """准备时间序列数据"""
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        # 确保所有必需特征都存在
        missing_features = set(self.config.ALL_FEATURES) - set(data.columns)
        if missing_features:
            print(f"警告: 缺少特征 {missing_features}")
            for feature in missing_features:
                data[feature] = 0

        # 选择特征并应用缩放
        features = data[self.config.ALL_FEATURES].values
        features_scaled = self.scaler.transform(features)

        # 创建时间序列
        if len(features_scaled) >= sequence_length:
            sequence = features_scaled[-sequence_length:]  # 使用最后几个时间步
            return torch.FloatTensor(sequence).unsqueeze(0)  # 添加batch维度
        else:
            # 如果数据不足，重复填充
            padded_sequence = np.tile(features_scaled, (sequence_length // len(features_scaled) + 1, 1))
            sequence = padded_sequence[:sequence_length]
            return torch.FloatTensor(sequence).unsqueeze(0)

    def prepare_graph_data(self, data, year=2023):
        """准备图数据"""
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        # 创建完整年份的图数据
        graph_data = self.spatial_temporal_dataset.create_graph_data(year)

        if graph_data is None:
            print("无法创建图数据")
            return None

        # 更新特征（如果提供了新数据）
        if len(data) > 0:
            for i, county in enumerate(self.spatial_temporal_dataset.counties):
                county_data = data[data['County'] == county]
                if len(county_data) > 0:
                    features = county_data[self.config.ALL_FEATURES].iloc[0].values
                    features_scaled = self.scaler.transform([features])[0]
                    graph_data.x[i] = torch.FloatTensor(features_scaled)

        return graph_data.to(self.device)

    def predict_bilstm(self, data):
        """使用BiLSTM模型进行预测"""
        if not hasattr(self, 'bilstm_model'):
            if not self.load_bilstm_model():
                return None

        # 准备数据
        sequence = self.prepare_time_series_data(data)

        # 预测
        with torch.no_grad():
            output = self.bilstm_model(sequence)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1).item()

        # 转换结果
        result = {
            'prediction': prediction + 1,  # 转换回1,2,3
            'probabilities': probabilities.cpu().numpy()[0].tolist(),
            'interpretation': self._interpret_prediction(prediction + 1, probabilities.cpu().numpy()[0])
        }

        return result

    def predict_gcn(self, data, year=2023, target_county=None):
        """使用GCN模型进行预测"""
        if not hasattr(self, 'gcn_model'):
            if not self.load_gcn_model():
                return None

        # 准备图数据
        graph_data = self.prepare_graph_data(data, year)
        if graph_data is None:
            return None

        # 预测
        with torch.no_grad():
            output = self.gcn_model(graph_data)
            probabilities = torch.softmax(output, dim=1)
            predictions = torch.argmax(output, dim=1)

        # 如果指定了目标县，返回该县的预测
        if target_county:
            if target_county in self.spatial_temporal_dataset.county_to_idx:
                idx = self.spatial_temporal_dataset.county_to_idx[target_county]
                prediction = predictions[idx].item()
                prob = probabilities[idx].cpu().numpy()

                result = {
                    'prediction': prediction + 1,
                    'probabilities': prob.tolist(),
                    'interpretation': self._interpret_prediction(prediction + 1, prob)
                }
            else:
                result = {'error': f'County {target_county} not found'}
        else:
            # 返回所有县的预测
            results = {}
            for i, county in enumerate(self.spatial_temporal_dataset.counties):
                prediction = predictions[i].item()
                prob = probabilities[i].cpu().numpy()

                results[county] = {
                    'prediction': prediction + 1,
                    'probabilities': prob.tolist(),
                    'interpretation': self._interpret_prediction(prediction + 1, prob)
                }

            result = results

        return result

    def _interpret_prediction(self, prediction, probabilities):
        """解释预测结果"""
        severity_descriptions = {
            1: '轻度发病 - 建议常规监测',
            2: '中度发病 - 建议加强防控措施',
            3: '重度发病 - 需要立即采取综合防控措施'
        }

        interpretation = {
            'predicted_severity': int(prediction),
            'severity_description': severity_descriptions.get(int(prediction), '未知级别'),
            'confidence': float(np.max(probabilities)),
            'class_probabilities': {
                '1级（轻度）': float(probabilities[0]) if len(probabilities) > 0 else 0.0,
                '2级（中度）': float(probabilities[1]) if len(probabilities) > 1 else 0.0,
                '3级（重度）': float(probabilities[2]) if len(probabilities) > 2 else 0.0
            }
        }

        return interpretation

    def predict_ensemble(self, data, year=2023, target_county=None):
        """集成预测（结合BiLSTM和GCN）"""
        # BiLSTM预测
        bilstm_result = self.predict_bilstm(data)

        # GCN预测
        gcn_result = self.predict_gcn(data, year, target_county)

        if bilstm_result is None and gcn_result is None:
            return {'error': 'Both models failed to load'}

        # 如果只有一个模型成功
        if bilstm_result is None:
            return {'model_used': 'GCN', 'result': gcn_result}
        if gcn_result is None:
            return {'model_used': 'BiLSTM', 'result': bilstm_result}

        # 集成两个模型的预测
        if 'error' not in gcn_result:
            # 平均概率
            avg_probabilities = np.array(bilstm_result['probabilities']) * 0.5 + \
                              np.array(gcn_result['probabilities']) * 0.5
            ensemble_prediction = np.argmax(avg_probabilities) + 1

            ensemble_result = {
                'prediction': ensemble_prediction,
                'probabilities': avg_probabilities.tolist(),
                'interpretation': self._interpret_prediction(ensemble_prediction, avg_probabilities),
                'bilstm_prediction': bilstm_result['prediction'],
                'gcn_prediction': gcn_result['prediction'],
                'agreement': bilstm_result['prediction'] == gcn_result['prediction']
            }

            return {
                'model_used': 'Ensemble (BiLSTM + GCN)',
                'result': ensemble_result
            }
        else:
            return {'model_used': 'BiLSTM', 'result': bilstm_result}

    def demonstrate_predictions(self):
        """演示预测功能"""
        print("=== 深度学习模型预测演示 ===\n")

        # 示例数据
        sample_data = [
            {
                'County': '济南市历下区',
                'Temperature_mean': 1.2, 'Temperature_std': 0.3, 'Temperature_min': 0.5, 'Temperature_max': 1.9, 'Temperature_median': 1.2,
                'Humidity_mean': 0.8, 'Humidity_std': 0.7, 'Humidity_min': -0.9, 'Humidity_max': 2.5, 'Humidity_median': 0.8,
                'Rainfall_mean': 0.7, 'Rainfall_std': 1.0, 'Rainfall_min': -0.5, 'Rainfall_max': 2.8, 'Rainfall_median': 0.9,
                'Pressure_mean': 0.0, 'Pressure_std': 1.0, 'Pressure_min': -2.3, 'Pressure_max': 2.3, 'Pressure_median': 0.0,
                'Temp_Humidity_Index_mean': 1.2, 'Temp_Humidity_Index_std': 0.4, 'Temp_Humidity_Index_min': 0.3,
                'Temp_Humidity_Index_max': 2.3, 'Temp_Humidity_Index_median': 1.2,
                'Latitude': 36.7, 'Longitude': 117.0
            },
            {
                'County': '青岛市市南区',
                'Temperature_mean': 1.3, 'Temperature_std': 0.35, 'Temperature_min': 0.6, 'Temperature_max': 2.0, 'Temperature_median': 1.3,
                'Humidity_mean': 0.9, 'Humidity_std': 0.8, 'Humidity_min': -1.0, 'Humidity_max': 2.7, 'Humidity_median': 0.9,
                'Rainfall_mean': 0.8, 'Rainfall_std': 1.1, 'Rainfall_min': -0.6, 'Rainfall_max': 3.0, 'Rainfall_median': 1.0,
                'Pressure_mean': -0.1, 'Pressure_std': 0.9, 'Pressure_min': -2.1, 'Pressure_max': 2.1, 'Pressure_median': -0.1,
                'Temp_Humidity_Index_mean': 1.3, 'Temp_Humidity_Index_std': 0.45, 'Temp_Humidity_Index_min': 0.4,
                'Temp_Humidity_Index_max': 2.4, 'Temp_Humidity_Index_median': 1.3,
                'Latitude': 36.1, 'Longitude': 120.4
            }
        ]

        print("示例数据:")
        for i, data in enumerate(sample_data, 1):
            print(f"{i}. {data['County']}")
            print(f"   温度均值: {data['Temperature_mean']:.1f}, 湿度均值: {data['Humidity_mean']:.1f}")

        print("\n" + "="*60)

        # BiLSTM预测
        print("\n【BiLSTM模型预测结果】")
        for i, data in enumerate(sample_data, 1):
            result = self.predict_bilstm(data)
            if result:
                interpretation = result['interpretation']
                print(f"\n{data['County']}:")
                print(f"  预测发病程度: {interpretation['predicted_severity']}级")
                print(f"  预测描述: {interpretation['severity_description']}")
                print(f"  置信度: {interpretation['confidence']:.1%}")
                print(f"  概率分布: {interpretation['class_probabilities']}")

        # GCN预测
        print("\n【GCN模型预测结果】")
        for i, data in enumerate(sample_data, 1):
            result = self.predict_gcn(data, target_county=data['County'])
            if result and 'error' not in result:
                interpretation = result['interpretation']
                print(f"\n{data['County']}:")
                print(f"  预测发病程度: {interpretation['predicted_severity']}级")
                print(f"  预测描述: {interpretation['severity_description']}")
                print(f"  置信度: {interpretation['confidence']:.1%}")
                print(f"  概率分布: {interpretation['class_probabilities']}")

        # 集成预测
        print("\n【集成模型预测结果】")
        for i, data in enumerate(sample_data, 1):
            ensemble_result = self.predict_ensemble(data, target_county=data['County'])
            if ensemble_result and 'result' in ensemble_result:
                result = ensemble_result['result']
                interpretation = result['interpretation']
                print(f"\n{data['County']}:")
                print(f"  使用的模型: {ensemble_result['model_used']}")
                print(f"  预测发病程度: {interpretation['predicted_severity']}级")
                print(f"  预测描述: {interpretation['severity_description']}")
                print(f"  置信度: {interpretation['confidence']:.1%}")
                if 'agreement' in result:
                    agreement = "一致" if result['agreement'] else "不一致"
                    print(f"  模型一致性: {agreement}")

        print("\n" + "="*60)
        print("预测演示完成!")

def main():
    """主函数"""
    predictor = DeepLearningPredictor()
    predictor.demonstrate_predictions()

if __name__ == "__main__":
    main()