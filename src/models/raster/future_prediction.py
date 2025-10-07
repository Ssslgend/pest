import torch
import rasterio
import numpy as np
import joblib
import os
import sys
import time
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import shutil
import glob
import logging
from dateutil.relativedelta import relativedelta
from visualization import visualize_prediction

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from config_future import get_future_config
from model.bilstm import BiLSTMModel
from feature_extrapolation import FeatureExtrapolator
from predict_raster import probability_to_risk_class, calculate_risk_distribution, save_risk_distribution_to_csv
from utils import load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model(model_path, config):
    try:
        # 使用 config_future.py 中的特征定义来确定 input_size
        input_size = len(config['features']['prediction_features'])
        logger.info(f"根据config_future计算得到的模型输入大小: {input_size}")

        # 根据错误日志推断检查点使用的hidden_size，并使用config_future中的dropout
        # 注意：这仍然可能因为模型结构不匹配（如BatchNorm vs LayerNorm）而失败
        hidden_size_from_checkpoint = 128 # 从错误日志推断
        dropout_from_config = config['model']['dropout_rate']
        num_layers_assumed = 2 # 假设层数为2，与之前代码一致

        logger.warning(f"尝试使用推断的 hidden_size={hidden_size_from_checkpoint} 和 config_future 中的 dropout={dropout_from_config} 加载模型。")
        logger.warning("如果仍然失败，可能是模型结构定义 (model/bilstm.py) 与检查点不匹配。")

        model_config = {
            "input_size": input_size,
            "hidden_size": hidden_size_from_checkpoint,
            "num_layers": num_layers_assumed, # 保持层数假设
            "dropout": dropout_from_config,
            "num_classes": 1 # 假设输出类别为1
        }

        model = BiLSTMModel(model_config)

        # 加载检查点，注意weights_only参数以提高安全性
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False) # 保持 False 以兼容旧格式，但生产环境建议 True

        # 尝试加载 state_dict，设置 strict=False 允许部分不匹配
        # 这可能会加载部分层，但如果关键层不匹配仍然会报错或导致预测不准确
        logger.warning("尝试使用 strict=False 加载模型状态字典，忽略部分不匹配的键。")
        load_result = model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)

        logger.info(f"模型状态字典加载结果: {load_result}")
        if load_result.missing_keys:
            logger.warning(f"加载时丢失的键: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            logger.warning(f"加载时多余的键: {load_result.unexpected_keys}")

        model.eval()
        logger.info(f"已尝试加载PyTorch模型：{model_path}")
        return model
    except Exception as e:
        logger.error(f"模型加载失败：{e}")
        raise

def load_scaler(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
        logger.info(f"成功加载归一化器：{scaler_path}")
        return scaler
    except Exception as e:
        logger.error(f"归一化器加载失败：{e}")
        raise

def load_historical_data(config):
    historical_data = {}
    features_dir = config['future']['historical_features_dir']

    try:
        for feature in config['features']['prediction_features']:
            feature_files = sorted(glob.glob(os.path.join(features_dir, f"{feature}_*.tif")))

            if not feature_files:
                logger.warning(f"未找到特征 {feature} 的历史数据文件")
                continue

            input_length = config['common']['input_length']
            recent_files = feature_files[-input_length:]

            feature_data = []
            for file in recent_files:
                with rasterio.open(file) as src:
                    feature_data.append(src.read(1))

                    if feature == config['features']['prediction_features'][0] and feature == recent_files[0]:
                        historical_data['meta'] = src.meta.copy()

            historical_data[feature] = np.array(feature_data)

        logger.info(f"成功加载历史数据，包含 {len(historical_data)-1} 个特征")
        return historical_data

    except Exception as e:
        logger.error(f"历史数据加载失败：{e}")
        raise

def load_future_features(config):
    future_features = {}
    features_dir = config['future']['future_features_dir']
    future_periods = config['future']['future_periods']

    try:
        end_date = datetime.strptime(config['future']['historical_end_date'], '%Y-%m')

        for period in range(future_periods):
            future_date = end_date + relativedelta(months=period+1)
            future_date_str = future_date.strftime('%Y-%m')

            period_features = {}
            for feature in config['features']['prediction_features']:
                feature_file = os.path.join(features_dir, f"{feature}_{future_date_str}.tif")

                if os.path.exists(feature_file):
                    with rasterio.open(feature_file) as src:
                        period_features[feature] = src.read(1)

                        if feature == config['features']['prediction_features'][0]:
                            if 'meta' not in future_features:
                                future_features['meta'] = src.meta.copy()
                else:
                    logger.warning(f"未找到特征 {feature} 在 {future_date_str} 的数据文件")

            future_features[f"period_{period+1}"] = period_features

        logger.info(f"成功加载未来特征数据，共 {len(future_features)-1} 个时期")
        return future_features

    except Exception as e:
        logger.error(f"未来特征数据加载失败：{e}")
        raise

def prepare_prediction_input(historical_data, future_data, config, period_idx=0):
    try:
        input_length = config['common']['input_length']
        features = config['features']['prediction_features']

        shape = historical_data[features[0]].shape[1:]
        pixels = shape[0] * shape[1]

        X = np.zeros((pixels, input_length, len(features)))

        if period_idx == 0:
            for i, feature in enumerate(features):
                feature_data = historical_data[feature]
                X[:, :, i] = feature_data.reshape(-1, input_length).T
        else:
            pass

        if config['common']['normalize_data']:
            scaler_path = os.path.join(config['common']['model_dir'], 'feature_scaler.pkl')
            scaler = load_scaler(scaler_path)

            X_reshaped = X.reshape(-1, len(features))
            X_normalized = scaler.transform(X_reshaped)
            X = X_normalized.reshape(-1, input_length, len(features))

        logger.info(f"成功准备第 {period_idx+1} 个未来时期的预测输入数据")
        return X, shape

    except Exception as e:
        logger.error(f"预测输入数据准备失败：{e}")
        raise

def predict_future_periods(config, output_dir=None):
    try:
        if output_dir is None:
            output_dir = config['future']['future_output_dir']

        os.makedirs(output_dir, exist_ok=True)

        log_file = os.path.join(output_dir, 'prediction.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        logger.info("开始未来时期预测流程")

        model_path = os.path.join(config['common']['model_dir'], config['model']['model_checkpoint'])
        model = load_model(model_path, config)

        historical_data = load_historical_data(config)

        try:
            future_features = load_future_features(config)
        except Exception as e:
            logger.warning(f"加载未来特征失败（可能不需要）：{e}")
            future_features = None

        meta = historical_data['meta']

        predictions = {}

        future_periods = config['future']['future_periods']
        for period in range(future_periods):
            logger.info(f"预测第 {period+1}/{future_periods} 个未来时期")

            X, shape = prepare_prediction_input(historical_data, future_features, config, period)

            X_tensor = torch.FloatTensor(X)

            with torch.no_grad():
                outputs = model(X_tensor)
                y_pred = outputs.cpu().numpy()

            prediction = y_pred.reshape(shape)

            period_name = config['future']['period_names'][period]
            predictions[period_name] = prediction

            output_file = os.path.join(output_dir, f"prediction_{period_name}.tif")

            with rasterio.open(output_file, 'w', **meta) as dst:
                dst.write(prediction, 1)

            logger.info(f"已保存第 {period+1} 个未来时期的预测结果到 {output_file}")

            visualize_prediction(
                prediction,
                os.path.join(output_dir, f"prediction_{period_name}.png"),
                title=f"害虫风险预测 - {period_name}",
                config=config
            )

            for i, feature in enumerate(config['features']['prediction_features']):
                historical_data[feature] = np.roll(historical_data[feature], -1, axis=0)
                if i == 0:
                    historical_data[feature][-1] = prediction

        create_prediction_gif(predictions, config, os.path.join(output_dir, "prediction_series.gif"))

        logger.info(f"未来时期预测完成，共预测 {future_periods} 个时期")
        logger.info(f"所有结果已保存到 {output_dir}")

        return predictions, output_dir

    except Exception as e:
        logger.error(f"未来预测失败：{e}")
        raise

def create_prediction_gif(predictions, config, output_path):
    try:
        import imageio

        images = []
        for period_name in config['future']['period_names']:
            if period_name in predictions:
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(predictions[period_name], cmap=config['future']['visualization']['cmap'])
                plt.colorbar(im, ax=ax)
                plt.title(f"害虫风险预测 - {period_name}", fontsize=config['future']['visualization']['title_fontsize'])
                plt.tight_layout()

                temp_file = f"temp_{period_name}.png"
                plt.savefig(temp_file, dpi=config['future']['visualization']['dpi'])
                plt.close()

                images.append(imageio.imread(temp_file))

                os.remove(temp_file)

        if images:
            imageio.mimsave(output_path, images, duration=config['future']['visualization']['gif_duration']/1000)
            logger.info(f"已创建预测时间序列GIF：{output_path}")
        else:
            logger.warning("没有足够的预测结果来创建GIF")

    except Exception as e:
        logger.error(f"创建预测GIF失败：{e}")

def run_future_prediction(periods=None, output_dir=None):
    config = get_future_config()

    if periods is not None:
        config['future']['future_periods'] = periods
        config['future']['period_names'] = [f"Month_{i+1}" for i in range(periods)]

    if output_dir is not None:
        config['future']['future_output_dir'] = output_dir

    return predict_future_periods(config)

if __name__ == "__main__":
    predictions, output_dir = run_future_prediction(periods=3)
    print(f"预测结果已保存到 {output_dir}")