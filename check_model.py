
# check_model.py - 检查训练好的模型

import os
import sys
import torch
import numpy as np
import joblib
from sd_raster_prediction.config_raster_new import get_config

def check_model():
    """检查模型文件和配置"""
    print("===== 检查模型和配置 =====")
    
    # 获取配置信息
    config = get_config()
    
    # 检查路径
    paths_to_check = [
        ("输出基础目录", config['prediction_output_dir']),
        ("模型文件", config['model_save_path']),
        ("标准化器文件", config['scaler_save_path']),
        ("特征栅格1", list(config['feature_raster_map'].values())[0] if config['feature_raster_map'] else "无特征文件")
    ]
    
    for name, path in paths_to_check:
        exists = os.path.exists(path)
        print(f"{name}: {path} - {'存在' if exists else '不存在'}")
    
    # 创建输出目录
    try:
        os.makedirs(config['prediction_output_dir'], exist_ok=True)
        print(f"已创建输出目录：{config['prediction_output_dir']}")
    except Exception as e:
        print(f"创建输出目录失败: {e}")
    
    # 尝试加载模型
    if os.path.exists(config['model_save_path']):
        try:
            checkpoint = torch.load(config['model_save_path'], map_location='cpu')
            print("成功加载模型")
            
            # 检查模型结构
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            print("模型参数:")
            for key in state_dict.keys():
                if 'weight' in key:
                    print(f"  {key}: 形状 {state_dict[key].shape}")
        except Exception as e:
            print(f"加载模型失败: {e}")
    
    # 尝试加载标准化器
    if os.path.exists(config['scaler_save_path']):
        try:
            scaler = joblib.load(config['scaler_save_path'])
            print("成功加载标准化器")
        except Exception as e:
            print(f"加载标准化器失败: {e}")
    
    print("\n===== 检查完成 =====")

if __name__ == "__main__":
    try:
        check_model()
    except Exception as e:
        print(f"检查过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 