# sd_raster_prediction/test_config_new.py
import os
import sys
import pandas as pd

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入新的配置文件
from config_raster_new import get_config

def test_config():
    """测试新配置是否正确加载"""
    print("正在加载新配置...")
    config = get_config()
    
    # 打印关键配置信息
    print(f"CSV数据路径: {config['csv_data_path']}")
    print(f"模型保存路径: {config['model_save_path']}")
    print(f"设备: {config['training']['device']}")
    
    # 打印数据处理配置
    dp_config = config['data_processing']
    print("\n数据处理配置:")
    print(f"坐标列: {dp_config['coordinate_columns']}")
    print(f"标签列: {dp_config['label_column']}")
    print(f"排除的特征列: {dp_config['excluded_cols_from_features']}")
    
    # 测试CSV文件是否可以加载
    print("\n测试CSV文件加载...")
    try:
        # 尝试不同编码方式加载CSV文件
        try:
            df = pd.read_csv(config['csv_data_path'], encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(config['csv_data_path'], encoding='gbk')
            except:
                df = pd.read_csv(config['csv_data_path'], encoding='latin1')
        
        print(f"CSV文件加载成功，形状: {df.shape}")
        
        # 检查坐标列和标签列是否存在
        coord_cols = dp_config['coordinate_columns']
        label_col = dp_config['label_column']
        
        for col in coord_cols:
            if col in df.columns:
                print(f"坐标列 '{col}' 存在")
            else:
                print(f"警告: 坐标列 '{col}' 不存在!")
        
        if label_col in df.columns:
            print(f"标签列 '{label_col}' 存在")
            print(f"标签值分布: {df[label_col].value_counts().to_dict()}")
        else:
            print(f"警告: 标签列 '{label_col}' 不存在!")
        
        # 检查预期特征列
        excluded = (dp_config['excluded_cols_from_features'] + 
                   dp_config['coordinate_columns'] + 
                   [dp_config['label_column']])
        feature_cols = [col for col in df.columns if col not in excluded]
        print(f"\n使用 {len(feature_cols)} 个特征列:")
        print(f"特征列前10个: {feature_cols[:10]}")
        print(f"总特征列数: {len(feature_cols)}")
        
    except Exception as e:
        print(f"加载CSV文件时出错: {str(e)}")
    
    print("\n配置测试完成")

if __name__ == "__main__":
    test_config() 