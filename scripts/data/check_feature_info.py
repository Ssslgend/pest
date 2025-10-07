#!/usr/bin/env python
# check_feature_info.py - 简化版特征信息检查

import os
import sys
import torch
import pandas as pd
import numpy as np
from collections import Counter

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def print_header(text):
    print("=" * 80)
    print(text)
    print("=" * 80)

def check_model_features():
    """检查模型特征"""
    print_header("1. 模型特征检查")
    
    model_path = 'results/trained_model/sd_bilstm_presence_pseudo.pth'
    input_size = None
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return input_size
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # 从权重推断输入大小
            for key, value in checkpoint['state_dict'].items():
                if 'lstm.weight_ih_l0' in key:
                    input_size = value.shape[1]
                    print(f"从模型权重推断输入大小: {input_size}")
                    break
        else:
            # 直接从模型权重推断
            for key, value in checkpoint.items():
                if 'lstm.weight_ih_l0' in key:
                    input_size = value.shape[1]
                    print(f"从模型权重推断输入大小: {input_size}")
                    break
    except Exception as e:
        print(f"读取模型时出错: {e}")
    
    return input_size

def check_config_features():
    """检查配置文件中的特征"""
    print_header("2. 配置文件特征检查")
    
    feature_map = None
    
    try:
        # 尝试直接导入
        try:
            from sd_raster_prediction.config_raster_new import get_config
        except ImportError:
            # 备选导入
            sys.path.append(os.path.join(current_dir, 'sd_raster_prediction'))
            from config_raster_new import get_config
        
        config = get_config()
        feature_map = config['feature_raster_map']
        print(f"配置文件中的特征数量: {len(feature_map)}")
        print(f"配置文件中的特征名称:")
        for i, name in enumerate(feature_map.keys()):
            print(f"{i+1}. {name}")
        
        # 检查重复特征
        feature_counts = Counter(feature_map.keys())
        duplicates = [name for name, count in feature_counts.items() if count > 1]
        if duplicates:
            print(f"\n警告: 发现重复的特征名称: {duplicates}")
    except Exception as e:
        print(f"读取配置文件时出错: {e}")
    
    return feature_map

def check_csv_features():
    """检查CSV文件中的特征"""
    print_header("3. CSV文件特征检查")
    
    feature_cols = None
    csv_path = 'datas/train.csv'
    
    if not os.path.exists(csv_path):
        print(f"错误: CSV文件不存在: {csv_path}")
        return feature_cols
    
    try:
        # 尝试不同编码
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
            return feature_cols
        
        # 识别特征列
        exclude_cols = ['发生样点纬度', '发生样点经度', 'label', 'year', 'Unnamed']
        feature_cols = [col for col in df.columns if not any(excl in col for excl in exclude_cols)]
        
        print(f"CSV文件中可能的特征列数量: {len(feature_cols)}")
        print(f"可能的特征列:")
        for i, col in enumerate(feature_cols):
            print(f"{i+1}. {col}")
    except Exception as e:
        print(f"处理CSV文件时出错: {e}")
    
    return feature_cols

def main():
    """主函数"""
    # 检查模型
    input_size = check_model_features()
    
    # 检查配置
    feature_map = check_config_features()
    
    # 检查CSV
    feature_cols = check_csv_features()
    
    # 汇总比较
    print_header("特征信息汇总")
    if input_size:
        print(f"模型输入特征数量: {input_size}")
    
    if feature_map:
        print(f"配置文件中的特征数量: {len(feature_map)}")
        
    if feature_cols:
        print(f"CSV文件中的可能特征数量: {len(feature_cols)}")
    
    # 比较是否匹配
    if input_size and feature_map:
        if len(feature_map) != input_size:
            print(f"\n警告: 配置文件中的特征数量({len(feature_map)})与模型输入大小({input_size})不匹配!")
        else:
            print(f"\n配置文件中的特征数量与模型输入大小匹配: {input_size}")
    
    if input_size and feature_cols:
        if len(feature_cols) != input_size:
            print(f"\n警告: CSV中的特征数量({len(feature_cols)})与模型输入大小({input_size})不匹配!")
        else:
            print(f"\nCSV中的特征数量与模型输入大小匹配: {input_size}")
    
    print("\n特征信息检查完成!")

if __name__ == "__main__":
    main() 