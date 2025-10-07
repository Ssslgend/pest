# check_features.py
from sd_raster_prediction.config_raster_new import get_config
import pandas as pd

# 获取配置信息
config = get_config()
feature_map = config['feature_raster_map']

# 打印特征映射信息
print("特征映射中的特征数量:", len(feature_map))
print("\n特征列表:")
for i, (name, path) in enumerate(feature_map.items()):
    print(f"{i+1}. {name}")

# 检查是否有重复的特征名称
unique_features = set(feature_map.keys())
if len(unique_features) != len(feature_map):
    print(f"\n警告: 特征映射中有重复的特征名称! 唯一名称数量: {len(unique_features)}")
    # 找出重复的特征
    from collections import Counter
    feature_counts = Counter(feature_map.keys())
    duplicates = [name for name, count in feature_counts.items() if count > 1]
    print(f"重复的特征名称: {duplicates}")

# 打印数据处理配置信息
dp_config = config['data_processing']
print("\n数据处理配置:")
print(f"坐标列: {dp_config['coordinate_columns']}")
print(f"标签列: {dp_config['label_column']}")
print(f"排除的特征列: {dp_config['excluded_cols_from_features']}")

# 尝试读取CSV文件
try:
    csv_path = config['csv_data_path']
    print(f"\n尝试读取CSV文件: {csv_path}")
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(csv_path, encoding='gbk')
        except:
            df = pd.read_csv(csv_path, encoding='latin1')
    
    print(f"CSV文件形状: {df.shape}")
    print(f"CSV列名 ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")
    
    # 检查标签分布
    label_col = dp_config['label_column']
    if label_col in df.columns:
        print(f"\n标签分布: {df[label_col].value_counts().to_dict()}")
    
except Exception as e:
    print(f"读取CSV文件时出错: {e}") 