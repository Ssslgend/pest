#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SF-BiLSTM快速测试脚本 - 使用真实数据
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
    data_file = 'datas/shandong_pest_data/real_occurrence_complete_data.csv'

    if not os.path.exists(data_file):
        print(f"错误: 数据文件不存在: {data_file}")
        return

    print(f"找到数据文件: {data_file}")

    # 导入训练模块
    try:
        from train_sf_bilstm_real_data import SF_BiLSTM_Trainer, RealDataProcessor
        print("成功导入训练模块")
    except ImportError as e:
        print(f"导入训练模块失败: {e}")
        return

    # 配置
    config = {
        'output_dir': 'results/sf_bilstm_test',
        'hidden_size': 64,  # 减小模型规模
        'num_layers': 1,    # 减少层数
        'dropout': 0.3,
        'epochs': 20,       # 减少训练轮次
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'batch_size': 64,   # 增大批次大小
        'test_size': 0.2,
        'val_size': 0.2,
        'use_focal_loss': True,
        'random_seed': 42
    }

    # 创建训练器
    trainer = SF_BiLSTM_Trainer(config)

    # 创建数据处理器
    processor = RealDataProcessor('real_occurrence')

    # 加载和预处理数据
    try:
        print("加载数据...")
        df = processor.load_real_occurrence_data(data_file)

        print("预处理数据...")
        X, y, feature_cols = processor.preprocess_real_occurrence_data(df)

        print("创建数据加载器...")
        train_loader, val_loader, test_loader, train_data, val_data, test_data = processor.create_data_loaders(
            X, y,
            test_size=config['test_size'],
            val_size=config['val_size'],
            batch_size=config['batch_size']
        )

        print(f"数据准备完成 - 特征数: {len(feature_cols)}")

        # 更新输出目录
        os.makedirs(config['output_dir'], exist_ok=True)
        trainer.output_dir = config['output_dir']

        # 只测试完整模型和基线模型
        from bilstm_variants import BiLSTMComplete, BiLSTMNoAttention

        models_to_test = {
            "完整SF-BiLSTM": BiLSTMComplete,
            "无注意力机制": BiLSTMNoAttention
        }

        results = {}

        for model_name, model_class in models_to_test.items():
            print(f"\n{'-'*50}")
            print(f"训练模型: {model_name}")

            # 创建模型
            model, model_config = trainer.create_model(model_class, len(feature_cols))

            # 训练模型
            model, history = trainer.train_model(
                model, train_loader, val_loader, model_name
            )

            # 评估模型
            model_results = trainer.evaluate_model(model, test_loader)

            print(f"{model_name} 测试结果:")
            for metric, value in model_results.items():
                if metric != 'confusion_matrix':
                    print(f"  {metric}: {value:.4f}")

            results[model_name] = model_results

        # 保存结果
        import json
        with open(os.path.join(config['output_dir'], 'quick_test_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"\n快速测试完成! 结果保存到: {config['output_dir']}")

        # 打印对比结果
        print(f"\n{'='*60}")
        print("性能对比结果:")
        print(f"{'='*60}")

        for model_name, metrics in results.items():
            print(f"{model_name}:")
            print(f"  F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}, "
                  f"Acc: {metrics['accuracy']:.4f}")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()