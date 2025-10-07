#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
山东省县域美国白蛾第一代发病情况预测模型训练脚本
包含分类和回归任务，使用多种机器学习算法
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
from datetime import datetime

# 机器学习库
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score
)

# XGBoost (如果安装了)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed, will skip XGBoost models")

# 本地导入
from county_level_config import CountyLevelConfig
import joblib

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False

class CountyLevelModelTrainer:
    """县域级别模型训练器"""

    def __init__(self):
        self.config = CountyLevelConfig()
        self.config.ensure_directories()
        self.scalers = {}
        self.models = {}
        self.evaluation_results = {}
        self.feature_importance = {}

        # 加载数据
        self.load_data()

    def load_data(self):
        """加载训练数据"""
        print("Loading training data...")

        self.train_data = pd.read_csv(self.config.TRAIN_DATA_PATH)
        self.val_data = pd.read_csv(self.config.VAL_DATA_PATH)

        # 合并训练和验证数据用于交叉验证
        self.full_data = pd.concat([self.train_data, self.val_data], ignore_index=True)

        print(f"Training data: {len(self.train_data)} samples")
        print(f"Validation data: {len(self.val_data)} samples")
        print(f"Full data: {len(self.full_data)} samples")

    def preprocess_data(self, data, fit_scaler=False):
        """数据预处理"""
        processed_data = data.copy()

        # 特征和目标变量
        X = processed_data[self.config.ALL_FEATURES].copy()

        # 分类目标
        y_class = processed_data[self.config.TARGET_CLASSIFICATION].copy()

        # 回归目标
        y_reg = processed_data[self.config.TARGET_REGRESSION].copy()

        # 特征缩放
        if fit_scaler:
            # 训练集时拟合scaler
            self.scalers['standard'] = StandardScaler().fit(X)
            self.scalers['robust'] = RobustScaler().fit(X)
            self.scalers['minmax'] = MinMaxScaler().fit(X)

        # 使用不同的scaler
        X_standard = self.scalers['standard'].transform(X)
        X_robust = self.scalers['robust'].transform(X)
        X_minmax = self.scalers['minmax'].transform(X)

        return {
            'X_original': X,
            'X_standard': X_standard,
            'X_robust': X_robust,
            'X_minmax': X_minmax,
            'y_class': y_class,
            'y_reg': y_reg
        }

    def train_classification_models(self):
        """训练分类模型（多类分类：预测发病程度1-3级）"""
        print("\n=== Training Multi-class Classification Models ===")

        # 预处理数据
        train_processed = self.preprocess_data(self.train_data, fit_scaler=True)
        val_processed = self.preprocess_data(self.val_data, fit_scaler=False)

        # 定义模型（适合多类分类）
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, max_iter=1000, multi_class='multinomial'
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, alpha=0.01
            ),
            'SVM': SVC(
                random_state=42, probability=True, decision_function_shape='ovo'
            )
        }

        # 添加XGBoost模型
        if HAS_XGBOOST:
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            )

        # 训练和评估模型
        for name, model in models.items():
            print(f"\nTraining {name}...")

            # 选择合适的特征缩放方式
            if name in ['LogisticRegression', 'SVM', 'MLP']:
                X_train = train_processed['X_standard']
                X_val = val_processed['X_standard']
            else:
                X_train = train_processed['X_original']
                X_val = val_processed['X_original']

            # 使用发病程度作为多类目标
            y_train_multi = train_processed['y_reg'] - 1  # 转换为0,1,2
            y_val_multi = val_processed['y_reg'] - 1

            # 训练模型
            model.fit(X_train, y_train_multi)

            # 预测
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            # 处理概率预测
            if hasattr(model, 'predict_proba'):
                y_val_proba = model.predict_proba(X_val)
            else:
                y_val_proba = None

            # 评估
            train_metrics = self.evaluate_multiclass_classification(
                y_train_multi, y_train_pred, y_val_proba=None
            )
            val_metrics = self.evaluate_multiclass_classification(
                y_val_multi, y_val_pred, y_val_proba
            )

            # 保存结果
            self.models[f'classification_{name}'] = model
            self.evaluation_results[f'classification_{name}'] = {
                'train': train_metrics,
                'validation': val_metrics
            }

            # 交叉验证
            cv_scores = self.cross_validate_multiclass_classification(
                self.full_data, name, model
            )
            self.evaluation_results[f'classification_{name}']['cv'] = cv_scores

            print(f"  Train Accuracy: {train_metrics['accuracy']:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1-Score: {val_metrics['f1_weighted']:.4f}")
            print(f"  CV F1-Score: {cv_scores['f1_mean']:.4f} (+/- {cv_scores['f1_std']:.4f})")

    def train_regression_models(self):
        """训练回归模型"""
        print("\n=== Training Regression Models ===")

        # 预处理数据
        train_processed = self.preprocess_data(self.train_data, fit_scaler=False)
        val_processed = self.preprocess_data(self.val_data, fit_scaler=False)

        # 定义模型
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            ),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'Lasso': Lasso(alpha=1.0, random_state=42),
            'MLP': MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, alpha=0.01
            ),
            'SVR': SVR()
        }

        # 添加XGBoost模型
        if HAS_XGBOOST:
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            )

        # 训练和评估模型
        for name, model in models.items():
            print(f"\nTraining {name}...")

            # 选择合适的特征缩放方式
            if name in ['LinearRegression', 'Ridge', 'Lasso', 'SVR', 'MLP']:
                X_train = train_processed['X_standard']
                X_val = val_processed['X_standard']
            else:
                X_train = train_processed['X_original']
                X_val = val_processed['X_original']

            # 训练模型
            model.fit(X_train, train_processed['y_reg'])

            # 预测
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            # 评估
            train_metrics = self.evaluate_regression(
                train_processed['y_reg'], y_train_pred
            )
            val_metrics = self.evaluate_regression(
                val_processed['y_reg'], y_val_pred
            )

            # 保存结果
            self.models[f'regression_{name}'] = model
            self.evaluation_results[f'regression_{name}'] = {
                'train': train_metrics,
                'validation': val_metrics
            }

            # 交叉验证
            cv_scores = self.cross_validate_regression(
                self.full_data, name, model
            )
            self.evaluation_results[f'regression_{name}']['cv'] = cv_scores

            print(f"  Train R2: {train_metrics['r2']:.4f}")
            print(f"  Val R2: {val_metrics['r2']:.4f}")
            print(f"  Val RMSE: {val_metrics['rmse']:.4f}")
            print(f"  CV R2: {cv_scores['r2_mean']:.4f} (+/- {cv_scores['r2_std']:.4f})")

    def evaluate_multiclass_classification(self, y_true, y_pred, y_val_proba=None):
        """评估多类分类模型"""
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
                'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
                'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
                'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }

            # 添加每个类别的F1分数
            try:
                class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                for i in range(3):  # 0,1,2 对应1,2,3级
                    class_key = str(i)
                    if class_key in class_report:
                        metrics[f'f1_class_{i+1}'] = class_report[class_key]['f1-score']
                    else:
                        metrics[f'f1_class_{i+1}'] = 0.0
            except:
                for i in range(3):
                    metrics[f'f1_class_{i+1}'] = 0.0

        except Exception as e:
            print(f"Warning: Error in multiclass classification evaluation: {e}")
            metrics = {
                'accuracy': 0.0, 'precision_macro': 0.0, 'recall_macro': 0.0, 'f1_macro': 0.0,
                'precision_weighted': 0.0, 'recall_weighted': 0.0, 'f1_weighted': 0.0
            }
            for i in range(3):
                metrics[f'f1_class_{i+1}'] = 0.0

        return metrics

    def cross_validate_multiclass_classification(self, data, model_name, model):
        """多类分类模型交叉验证"""
        # 预处理完整数据
        full_processed = self.preprocess_data(data, fit_scaler=False)

        # 选择特征
        if model_name in ['LogisticRegression', 'SVM', 'MLP']:
            X = full_processed['X_standard']
        else:
            X = full_processed['X_original']

        y = full_processed['y_reg'] - 1  # 转换为0,1,2

        # 交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')

        return {
            'f1_mean': scores.mean(),
            'f1_std': scores.std(),
            'scores': scores.tolist()
        }

    def evaluate_regression(self, y_true, y_pred):
        """评估回归模型"""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        return metrics

    def cross_validate_classification(self, data, model_name, model):
        """分类模型交叉验证"""
        # 预处理完整数据
        full_processed = self.preprocess_data(data, fit_scaler=False)

        # 选择特征
        if model_name in ['LogisticRegression', 'SVM', 'MLP']:
            X = full_processed['X_standard']
        else:
            X = full_processed['X_original']

        y = full_processed['y_class']

        # 交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')

        return {
            'f1_mean': scores.mean(),
            'f1_std': scores.std(),
            'scores': scores.tolist()
        }

    def cross_validate_regression(self, data, model_name, model):
        """回归模型交叉验证"""
        # 预处理完整数据
        full_processed = self.preprocess_data(data, fit_scaler=False)

        # 选择特征
        if model_name in ['LinearRegression', 'Ridge', 'Lasso', 'SVR', 'MLP']:
            X = full_processed['X_standard']
        else:
            X = full_processed['X_original']

        y = full_processed['y_reg']

        # 交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

        return {
            'r2_mean': scores.mean(),
            'r2_std': scores.std(),
            'scores': scores.tolist()
        }

    def analyze_feature_importance(self):
        """分析特征重要性"""
        print("\n=== Analyzing Feature Importance ===")

        # 获取树模型的特征重要性
        tree_models = ['RandomForest', 'GradientBoosting']
        if HAS_XGBOOST:
            tree_models.append('XGBoost')

        for model_type in ['classification', 'regression']:
            for model_name in tree_models:
                full_model_name = f'{model_type}_{model_name}'
                if full_model_name in self.models:
                    model = self.models[full_model_name]

                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                        feature_imp = pd.DataFrame({
                            'feature': self.config.ALL_FEATURES,
                            'importance': importance
                        }).sort_values('importance', ascending=False)

                        self.feature_importance[full_model_name] = feature_imp

                        print(f"\n{full_model_name} Top 10 Features:")
                        print(feature_imp.head(10))

    def create_visualizations(self):
        """创建可视化图表"""
        print("\n=== Creating Visualizations ===")

        # 1. 模型性能比较
        self._plot_model_performance_comparison()

        # 2. 特征重要性可视化
        self._plot_feature_importance()

        # 3. 预测结果分布
        self._plot_prediction_distributions()

        # 4. 混淆矩阵
        self._plot_confusion_matrices()

    def _plot_model_performance_comparison(self):
        """绘制模型性能比较图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 分类模型性能
        class_models = {k: v for k, v in self.evaluation_results.items() if 'classification' in k}
        class_names = [k.replace('classification_', '') for k in class_models.keys()]
        class_acc = [v['validation']['accuracy'] for v in class_models.values()]
        class_f1 = [v['validation']['f1_weighted'] for v in class_models.values()]

        axes[0, 0].bar(class_names, class_acc)
        axes[0, 0].set_title('Multi-class Classification - Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)

        axes[0, 1].bar(class_names, class_f1)
        axes[0, 1].set_title('Multi-class Classification - Weighted F1-Score')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 回归模型性能
        reg_models = {k: v for k, v in self.evaluation_results.items() if 'regression' in k}
        reg_names = [k.replace('regression_', '') for k in reg_models.keys()]
        reg_r2 = [v['validation']['r2'] for v in reg_models.values()]
        reg_rmse = [v['validation']['rmse'] for v in reg_models.values()]

        axes[1, 0].bar(reg_names, reg_r2)
        axes[1, 0].set_title('Regression Models - R2')
        axes[1, 0].set_ylabel('R2')
        axes[1, 0].tick_params(axis='x', rotation=45)

        axes[1, 1].bar(reg_names, reg_rmse)
        axes[1, 1].set_title('Regression Models - RMSE')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f'{self.config.RESULTS_DIR}/model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_feature_importance(self):
        """绘制特征重要性图"""
        # 选择最佳模型的特征重要性
        best_class_model = max(
            {k: v for k, v in self.evaluation_results.items() if 'classification' in k}.items(),
            key=lambda x: x[1]['validation']['f1_weighted']
        )[0]

        if best_class_model in self.feature_importance:
            plt.figure(figsize=(12, 8))
            feature_imp = self.feature_importance[best_class_model].head(15)

            plt.barh(range(len(feature_imp)), feature_imp['importance'])
            plt.yticks(range(len(feature_imp)), feature_imp['feature'])
            plt.xlabel('Importance')
            plt.title(f'Feature Importance - {best_class_model}')
            plt.gca().invert_yaxis()

            plt.tight_layout()
            plt.savefig(f'{self.config.RESULTS_DIR}/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_prediction_distributions(self):
        """绘制预测结果分布"""
        val_processed = self.preprocess_data(self.val_data, fit_scaler=False)

        # 选择最佳分类模型
        best_class_model = max(
            {k: v for k, v in self.evaluation_results.items() if 'classification' in k}.items(),
            key=lambda x: x[1]['validation']['f1_weighted']
        )[0]

        # 选择最佳回归模型
        best_reg_model = max(
            {k: v for k, v in self.evaluation_results.items() if 'regression' in k}.items(),
            key=lambda x: x[1]['validation']['r2']
        )[0]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 多类分类预测分布
        model = self.models[best_class_model]
        if 'RandomForest' in best_class_model or 'GradientBoosting' in best_class_model or 'XGBoost' in best_class_model:
            X_val = val_processed['X_original']
        else:
            X_val = val_processed['X_standard']

        y_pred = model.predict(X_val)
        y_true = val_processed['y_reg'] - 1  # 转换为0,1,2

        # 分类结果
        classes = np.unique(y_true)
        axes[0].hist([y_true[y_true==c] for c in classes],
                    bins=len(classes), alpha=0.7, label=[f'Class {c}' for c in classes])
        axes[0].set_title(f'Multi-class Distribution - {best_class_model}')
        axes[0].set_xlabel('Severity Class (0-2)')
        axes[0].set_ylabel('Count')
        axes[0].legend()

        # 回归预测分布
        reg_model = self.models[best_reg_model]
        if 'RandomForest' in best_reg_model or 'GradientBoosting' in best_reg_model or 'XGBoost' in best_reg_model:
            X_val = val_processed['X_original']
        else:
            X_val = val_processed['X_standard']

        y_reg_pred = reg_model.predict(X_val)
        y_reg_true = val_processed['y_reg']

        axes[1].scatter(y_reg_true, y_reg_pred, alpha=0.6)
        axes[1].plot([y_reg_true.min(), y_reg_true.max()],
                    [y_reg_true.min(), y_reg_true.max()], 'r--', lw=2)
        axes[1].set_xlabel('Actual Severity')
        axes[1].set_ylabel('Predicted Severity')
        axes[1].set_title(f'Regression Prediction - {best_reg_model}')

        plt.tight_layout()
        plt.savefig(f'{self.config.RESULTS_DIR}/prediction_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrices(self):
        """绘制混淆矩阵"""
        val_processed = self.preprocess_data(self.val_data, fit_scaler=False)

        # 选择前3个分类模型
        class_models = {k: v for k, v in self.models.items() if 'classification' in k}
        top_models = sorted(class_models.items(),
                           key=lambda x: self.evaluation_results[x[0]]['validation']['f1_weighted'],
                           reverse=True)[:3]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, (model_name, model) in enumerate(top_models):
            if 'RandomForest' in model_name or 'GradientBoosting' in model_name or 'XGBoost' in model_name:
                X_val = val_processed['X_original']
            else:
                X_val = val_processed['X_standard']

            y_pred = model.predict(X_val)
            y_true = val_processed['y_reg'] - 1  # 转换为0,1,2

            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues')
            axes[idx].set_title(model_name.replace('classification_', ''))
            axes[idx].set_xlabel('Predicted Class')
            axes[idx].set_ylabel('Actual Class')

        plt.tight_layout()
        plt.savefig(f'{self.config.RESULTS_DIR}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_models_and_results(self):
        """保存模型和结果"""
        print("\n=== Saving Models and Results ===")

        # 保存模型
        for name, model in self.models.items():
            model_path = f'{self.config.MODEL_DIR}/{name}.joblib'
            joblib.dump(model, model_path)
            print(f"Saved model: {model_path}")

        # 保存scalers
        for name, scaler in self.scalers.items():
            scaler_path = f'{self.config.MODEL_DIR}/scaler_{name}.joblib'
            joblib.dump(scaler, scaler_path)
            print(f"Saved scaler: {scaler_path}")

        # 保存评估结果
        results_path = f'{self.config.RESULTS_DIR}/evaluation_results.json'

        # 转换numpy类型为Python原生类型以便JSON序列化
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj

        converted_results = convert_types(self.evaluation_results)

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)
        print(f"Saved evaluation results: {results_path}")

        # 保存特征重要性
        if self.feature_importance:
            importance_path = f'{self.config.RESULTS_DIR}/feature_importance.csv'

            # 合并所有模型的特征重要性
            all_importance = pd.DataFrame()
            for model_name, importance_df in self.feature_importance.items():
                importance_df = importance_df.copy()
                importance_df.columns = ['feature', f'importance_{model_name}']
                if all_importance.empty:
                    all_importance = importance_df
                else:
                    all_importance = all_importance.merge(importance_df, on='feature', how='outer')

            all_importance.to_csv(importance_path, index=False)
            print(f"Saved feature importance: {importance_path}")

        # 创建训练报告
        self._create_training_report()

    def _create_training_report(self):
        """创建训练报告"""
        report = {
            'training_date': datetime.now().isoformat(),
            'dataset_info': {
                'train_samples': len(self.train_data),
                'val_samples': len(self.val_data),
                'features': len(self.config.ALL_FEATURES),
                'target_distribution': self.train_data['Severity_Level'].value_counts().to_dict()
            },
            'best_models': self._get_best_models(),
            'model_count': {
                'classification': len([k for k in self.models.keys() if 'classification' in k]),
                'regression': len([k for k in self.models.keys() if 'regression' in k])
            }
        }

        report_path = f'{self.config.RESULTS_DIR}/training_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved training report: {report_path}")

    def _get_best_models(self):
        """获取最佳模型"""
        best_models = {}

        # 最佳分类模型
        class_models = {k: v for k, v in self.evaluation_results.items() if 'classification' in k}
        if class_models:
            best_class_model = max(class_models.items(), key=lambda x: x[1]['validation']['f1_weighted'])
            best_models['classification'] = {
                'name': best_class_model[0],
                'validation_f1_weighted': best_class_model[1]['validation']['f1_weighted'],
                'validation_accuracy': best_class_model[1]['validation']['accuracy']
            }

        # 最佳回归模型
        reg_models = {k: v for k, v in self.evaluation_results.items() if 'regression' in k}
        if reg_models:
            best_reg_model = max(reg_models.items(), key=lambda x: x[1]['validation']['r2'])
            best_models['regression'] = {
                'name': best_reg_model[0],
                'validation_r2': best_reg_model[1]['validation']['r2'],
                'validation_rmse': best_reg_model[1]['validation']['rmse']
            }

        return best_models

    def run_training_pipeline(self):
        """运行完整的训练流程"""
        print("=== Starting Model Training Pipeline ===")

        # 训练分类模型
        self.train_classification_models()

        # 训练回归模型
        self.train_regression_models()

        # 分析特征重要性
        self.analyze_feature_importance()

        # 创建可视化
        self.create_visualizations()

        # 保存模型和结果
        self.save_models_and_results()

        # 打印总结
        self._print_training_summary()

    def _print_training_summary(self):
        """打印训练总结"""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)

        # 最佳模型
        best_models = self._get_best_models()

        if 'classification' in best_models:
            print(f"\nBest Multi-class Classification Model: {best_models['classification']['name']}")
            print(f"  Validation Weighted F1-Score: {best_models['classification']['validation_f1_weighted']:.4f}")
            print(f"  Validation Accuracy: {best_models['classification']['validation_accuracy']:.4f}")

        if 'regression' in best_models:
            print(f"\nBest Regression Model: {best_models['regression']['name']}")
            print(f"  Validation R2: {best_models['regression']['validation_r2']:.4f}")
            print(f"  Validation RMSE: {best_models['regression']['validation_rmse']:.4f}")

        print(f"\nModels trained: {len(self.models)}")
        print(f"Results saved to: {self.config.RESULTS_DIR}")
        print("="*60)

def main():
    """主函数"""
    trainer = CountyLevelModelTrainer()
    trainer.run_training_pipeline()

if __name__ == "__main__":
    main()