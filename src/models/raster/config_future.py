import os
import yaml
from datetime import datetime
from config_raster import get_config as get_base_config

def get_future_config():
    config = {
        'common': {
            'random_seed': 42,
            'model_dir': os.path.join('E:/code/0424/pestBIstm/pestBIstm/sd_raster_prediction', 'results', 'trained_model'),
            'input_length': 12,
            'output_length': 1,
            'normalize_data': True,
        },

        'features': {
            'meteorological': [
                'avg_tmp', 'max_tmp', 'min_tmp', 'precipitation',
                'wind_speed', 'rel_humidity', 'radiation'
            ],
            'geographical': [
                'dem', 'slope', 'aspect'
            ],
            'vegetation': [
                'ndvi', 'evi'
            ],
            'soil': [
                'soil_moisture', 'soil_temperature'
            ],
            'prediction_features': [
                'avg_tmp', 'precipitation', 'rel_humidity',
                'dem', 'ndvi', 'soil_moisture'
            ]
        },

        'future': {
            'future_periods': 6,
            'period_names': ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6'],

            'future_features_dir': os.path.join('data', 'future_features'),

            'future_output_dir': os.path.join('results', f'future_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),

            'historical_features_dir': os.path.join('data', 'historical_features'),
            'historical_end_date': '2023-12',

            'visualization': {
                'cmap': 'RdYlGn_r',
                'risk_levels': 5,
                'risk_labels': ['极低', '低', '中', '高', '极高'],
                'dpi': 300,
                'title_fontsize': 14,
                'gif_duration': 1000,
                'map_background': True,
            }
        },

        'model': {
            'lstm_units': 64,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 10,
            'model_checkpoint': 'sd_bilstm_presence_pseudo.pth',
        }
    }

    config_file = os.path.join('config', 'future_config.yaml')
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                custom_config = yaml.safe_load(f)

                def update_dict(d, u):
                    for k, v in u.items():
                        if isinstance(v, dict):
                            d[k] = update_dict(d.get(k, {}), v)
                        else:
                            d[k] = v
                    return d

                config = update_dict(config, custom_config)
                print(f"已加载自定义配置文件: {config_file}")
        except Exception as e:
            print(f"加载自定义配置文件失败: {e}，使用默认配置")

    return config

if __name__ == '__main__':
    config = get_future_config()
    print("未来预测配置：")
    print(f"预测时期数: {config['future']['future_periods']}")
    print(f"预测特征: {config['features']['prediction_features']}")
    print(f"输出目录: {config['future']['future_output_dir']}")