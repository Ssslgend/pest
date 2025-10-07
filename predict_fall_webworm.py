#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
美国白蛾第一代发病情况预测脚本
使用训练好的模型进行预测
"""

import pandas as pd
import numpy as np
import joblib
import json
from county_level_config import CountyLevelConfig

class FallWebwormPredictor:
    """美国白蛾发病情况预测器"""

    def __init__(self):
        self.config = CountyLevelConfig()
        self.model = None
        self.scaler = None
        self.model_type = None

    def load_model(self, model_path=None, model_type='classification'):
        """
        加载训练好的模型
        Args:
            model_path: 模型文件路径，如果为None则使用最佳模型
            model_type: 'classification' 或 'regression'
        """
        if model_path is None:
            # 使用最佳模型
            if model_type == 'classification':
                model_path = f'{self.config.MODEL_DIR}/classification_SVM.joblib'
            else:
                model_path = f'{self.config.MODEL_DIR}/regression_Lasso.joblib'

        try:
            self.model = joblib.load(model_path)
            self.model_type = model_type

            # 加载对应的scaler
            if 'SVM' in model_path or 'LogisticRegression' in model_path or 'MLP' in model_path:
                self.scaler = joblib.load(f'{self.config.MODEL_DIR}/scaler_standard.joblib')
            elif 'LinearRegression' in model_path or 'Ridge' in model_path or 'Lasso' in model_path or 'SVR' in model_path:
                self.scaler = joblib.load(f'{self.config.MODEL_DIR}/scaler_standard.joblib')
            else:
                self.scaler = None

            print(f"成功加载模型: {model_path}")
            print(f"模型类型: {model_type}")

        except Exception as e:
            print(f"加载模型失败: {e}")
            return False

        return True

    def prepare_features(self, data):
        """
        准备特征数据
        Args:
            data: 包含特征的DataFrame或字典
        Returns:
            处理后的特征数组
        """
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        # 确保所有必需特征都存在
        missing_features = set(self.config.ALL_FEATURES) - set(data.columns)
        if missing_features:
            print(f"警告: 缺少特征 {missing_features}")
            # 用0填充缺失特征
            for feature in missing_features:
                data[feature] = 0

        # 选择特征并按正确顺序排列
        features = data[self.config.ALL_FEATURES]

        # 应用特征缩放（如果需要）
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
            return features_scaled
        else:
            return features.values

    def predict(self, features):
        """
        进行预测
        Args:
            features: 特征数据
        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("请先加载模型")

        # 准备特征
        X = self.prepare_features(features)

        # 进行预测
        if self.model_type == 'classification':
            # 多类分类（0,1,2 对应 1,2,3级）
            predictions = self.model.predict(X)
            # 转换回原始级别（1,2,3）
            predictions = predictions + 1

            # 获取预测概率（如果模型支持）
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)
                return {
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'interpretation': self._interpret_classification(predictions, probabilities)
                }
            else:
                return {
                    'predictions': predictions,
                    'interpretation': self._interpret_classification(predictions)
                }

        else:  # regression
            predictions = self.model.predict(X)
            return {
                'predictions': predictions,
                'interpretation': self._interpret_regression(predictions)
            }

    def _interpret_classification(self, predictions, probabilities=None):
        """解释分类预测结果"""
        result = []
        for i, pred in enumerate(predictions):
            interpretation = {
                'predicted_severity': int(pred),
                'severity_level': self._get_severity_description(int(pred))
            }

            if probabilities is not None:
                interpretation['class_probabilities'] = {
                    '1级（轻度）': float(probabilities[i, 0]) if probabilities.shape[1] > 0 else 0.0,
                    '2级（中度）': float(probabilities[i, 1]) if probabilities.shape[1] > 1 else 0.0,
                    '3级（重度）': float(probabilities[i, 2]) if probabilities.shape[1] > 2 else 0.0
                }

            result.append(interpretation)

        return result

    def _interpret_regression(self, predictions):
        """解释回归预测结果"""
        result = []
        for pred in predictions:
            # 将连续预测值转换为离散级别
            if pred < 1.5:
                severity = 1
                level = '轻度'
            elif pred < 2.5:
                severity = 2
                level = '中度'
            else:
                severity = 3
                level = '重度'

            result.append({
                'predicted_value': float(pred),
                'predicted_severity': severity,
                'severity_level': level
            })

        return result

    def _get_severity_description(self, severity_level):
        """获取发病程度描述"""
        descriptions = {
            1: '轻度发病 - 需要常规监测',
            2: '中度发病 - 建议加强防控措施',
            3: '重度发病 - 需要立即采取综合防控措施'
        }
        return descriptions.get(severity_level, '未知级别')

    def predict_from_sample_data(self):
        """使用示例数据进行预测演示"""
        print("=== 美国白蛾发病情况预测演示 ===\n")

        # 示例数据：不同地区的气象特征
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
            },
            {
                'County': '烟台市芝罘区',
                'Temperature_mean': 1.1, 'Temperature_std': 0.25, 'Temperature_min': 0.4, 'Temperature_max': 1.8, 'Temperature_median': 1.1,
                'Humidity_mean': 0.7, 'Humidity_std': 0.6, 'Humidity_min': -0.8, 'Humidity_max': 2.3, 'Humidity_median': 0.7,
                'Rainfall_mean': 0.6, 'Rainfall_std': 0.9, 'Rainfall_min': -0.4, 'Rainfall_max': 2.5, 'Rainfall_median': 0.8,
                'Pressure_mean': 0.1, 'Pressure_std': 1.1, 'Pressure_min': -2.5, 'Pressure_max': 2.5, 'Pressure_median': 0.1,
                'Temp_Humidity_Index_mean': 1.1, 'Temp_Humidity_Index_std': 0.35, 'Temp_Humidity_Index_min': 0.2,
                'Temp_Humidity_Index_max': 2.2, 'Temp_Humidity_Index_median': 1.1,
                'Latitude': 37.5, 'Longitude': 121.4
            }
        ]

        # 转换为DataFrame
        df_samples = pd.DataFrame(sample_data)

        print("示例数据:")
        for i, sample in enumerate(sample_data, 1):
            print(f"{i}. {sample['County']}")
            print(f"   温度均值: {sample['Temperature_mean']:.1f}, 湿度均值: {sample['Humidity_mean']:.1f}, 降雨均值: {sample['Rainfall_mean']:.1f}")
            print(f"   位置: 纬度 {sample['Latitude']:.1f}°, 经度 {sample['Longitude']:.1f}°")

        print("\n" + "="*60)

        # 分类预测
        print("\n【多类分类预测结果】")
        self.load_model(model_type='classification')
        class_result = self.predict(df_samples)

        for i, interpretation in enumerate(class_result['interpretation']):
            county = sample_data[i]['County']
            severity = interpretation['predicted_severity']
            level = interpretation['severity_level']
            print(f"\n{county}:")
            print(f"  预测发病程度: {severity}级 ({level})")
            if 'class_probabilities' in interpretation:
                probs = interpretation['class_probabilities']
                print(f"  概率分布: 轻度 {probs['1级（轻度）']:.1%}, 中度 {probs['2级（中度）']:.1%}, 重度 {probs['3级（重度）']:.1%}")

        print("\n" + "="*60)

        # 回归预测
        print("\n【回归预测结果】")
        self.load_model(model_type='regression')
        reg_result = self.predict(df_samples)

        for i, interpretation in enumerate(reg_result['interpretation']):
            county = sample_data[i]['County']
            value = interpretation['predicted_value']
            severity = interpretation['predicted_severity']
            level = interpretation['severity_level']
            print(f"\n{county}:")
            print(f"  预测连续值: {value:.2f}")
            print(f"  预测发病程度: {severity}级 ({level})")

        print("\n" + "="*60)
        print("预测完成!")

def main():
    """主函数"""
    predictor = FallWebwormPredictor()
    predictor.predict_from_sample_data()

if __name__ == "__main__":
    main()