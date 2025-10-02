# display_feature_info.py - 显示特征信息
import os
import sys
import torch
import pandas as pd
import numpy as np
import traceback

# 确保将当前目录添加到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def print_separator():
    print("=" * 80)

# 1. 检查模型
print_separator()
print("检查训练好的模型中的特征数量")
print_separator()

model_path = 'results/trained_model/sd_bilstm_presence_pseudo.pth'
try:
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # 查找LSTM权重以确定输入大小
        for key, value in checkpoint['state_dict'].items():
            if 'lstm.weight_ih_l0' in key:
                input_size = value.shape[1]
                print(f"从模型权重推断输入大小: {input_size}")
                break
    else:
        # 尝试直接获取
        for key, value in checkpoint.items():
            if 'lstm.weight_ih_l0' in key:
                input_size = value.shape[1]
                print(f"从模型权重推断输入大小: {input_size}")
                break
except Exception as e:
    print(f"读取模型时出错: {e}")
    traceback.print_exc()

# 2. 检查配置文件
print_separator()
print("检查配置文件中的特征映射")
print_separator()

try:
    # 添加sd_raster_prediction到路径
    sd_path = os.path.join(current_dir, 'sd_raster_prediction')
    if sd_path not in sys.path:
        sys.path.append(sd_path)
    
    # 尝试两种导入方式
    try:
        from sd_raster_prediction.config_raster_new import get_config
    except ImportError:
        from config_raster_new import get_config
    
    config = get_config()
    feature_map = config['feature_raster_map']
    print(f"配置文件中的特征数量: {len(feature_map)}")
    
    # 检查是否有重复的特征名称
    from collections import Counter
    feature_counts = Counter(feature_map.keys())
    duplicates = [name for name, count in feature_counts.items() if count > 1]
    if duplicates:
        print(f"警告: 发现重复的特征名称: {duplicates}")
    
    # 打印特征列表
    print("特征列表:")
    for i, name in enumerate(feature_map.keys()):
        print(f"{i+1}. {name}")
    
    # 检查特征数量与模型输入是否匹配
    if 'input_size' in locals() and input_size != len(feature_map):
        print(f"警告: 配置文件中的特征数量({len(feature_map)})与模型输入大小({input_size})不匹配!")
except Exception as e:
    print(f"读取配置文件时出错: {str(e)}")
    traceback.print_exc()

# 3. 检查CSV数据
print_separator()
print("检查训练数据")
print_separator()

try:
    csv_path = 'datas/train.csv'
    
    # 尝试不同的编码
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"成功使用 {encoding} 编码读取CSV文件")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        print("无法使用已知编码读取CSV文件")
    else:
        print(f"CSV数据形状: {df.shape}")
        print(f"CSV列名 ({len(df.columns)}):")
        for i, col in enumerate(df.columns):
            print(f"{i+1}. {col}")
except Exception as e:
    print(f"读取CSV数据时出错: {str(e)}")
    traceback.print_exc()

print_separator()
print("特征信息分析完成")
print_separator() 