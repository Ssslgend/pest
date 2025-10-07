#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SF-BiLSTM快速运行脚本 - 专门测试真实县域数据
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加路径
sys.path.append('oldPestBlstem/ablation_study')

def quick_test():
    """快速测试SF-BiLSTM模型"""
    print("开始SF-BiLSTM真实数据快速测试...")

    # 检查数据文件
    data_files = [
        'datas/shandong_pest_data/county_level_firstgen_complete.csv',
        'datas/shandong_pest_data/real_occurrence_complete_data.csv'
    ]

    available_files = []
    for file_path in data_files:
        if os.path.exists(file_path):
            available_files.append(file_path)
            print(f"找到数据文件: {file_path}")
        else:
            print(f"数据文件不存在: {file_path}")

    if not available_files:
        print("错误: 没有找到可用的数据文件!")
        return

    # 导入训练模块
    try:
        from train_sf_bilstm_real_data import main as train_main
        print("成功导入训练模块")
    except ImportError as e:
        print(f"导入训练模块失败: {e}")
        return

    # 运行训练
    try:
        print("\n开始运行SF-BiLSTM训练...")
        train_main()
        print("\nSF-BiLSTM训练完成!")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()