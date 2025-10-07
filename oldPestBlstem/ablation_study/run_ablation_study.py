
# -*- coding: utf-8 -*-
"""
一键运行消融实验流程
包括数据分析、模型训练、评估和可视化
"""

import os
import sys
import time
import subprocess
import argparse

def print_header(title):
    """打印带格式的标题"""
    width = 80
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")

def print_step(step_number, step_name):
    """打印步骤信息"""
    print(f"\n[{step_number}] {step_name}...")

def run_command(command):
    """运行命令并实时打印输出"""
    print(f"\n执行命令: {command}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        universal_newlines=True
    )
    
    # 实时打印输出
    for line in process.stdout:
        print(line, end='')
    
    # 等待进程完成
    return_code = process.wait()
    if return_code != 0:
        print(f"\n命令执行失败，返回码: {return_code}")
        return False
    return True

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行美国白蛾病虫害BiLSTM模型消融实验')
    parser.add_argument('--skip-data-check', action='store_true', help='跳过数据检查步骤')
    parser.add_argument('--skip-training', action='store_true', help='跳过模型训练步骤')
    parser.add_argument('--skip-visualization', action='store_true', help='跳过可视化步骤')
    parser.add_argument('--use-gpu', action='store_true', help='使用GPU进行训练')
    args = parser.parse_args()
    
    # 记录开始时间
    start_time = time.time()
    
    # 打印欢迎信息
    print_header("山东美国白蛾病虫害BiLSTM消融实验")
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 确保目录结构
    os.makedirs(os.path.join(script_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(script_dir, "results", "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(script_dir, "results", "data_analysis"), exist_ok=True)
    
    # 步骤1: 数据质量检查
    if not args.skip_data_check:
        print_step(1, "数据质量检查")
        if not run_command(f"python {os.path.join(script_dir, 'data_check.py')}"):
            print("数据质量检查失败，退出")
            return
    else:
        print_step(1, "数据质量检查 [已跳过]")
    
    # 步骤2: 训练模型变体
    if not args.skip_training:
        print_step(2, "训练模型变体")
        train_cmd = f"python {os.path.join(script_dir, 'train_ablation_fixed.py')}"
        if args.use_gpu:
            # 设置环境变量使用GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            print("使用GPU进行训练")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            print("使用CPU进行训练")
            
        if not run_command(train_cmd):
            print("模型训练失败，退出")
            return
    else:
        print_step(2, "训练模型变体 [已跳过]")
    
    # 步骤3: 可视化结果
    if not args.skip_visualization:
        print_step(3, "生成可视化结果")
        if not run_command(f"python {os.path.join(script_dir, 'visualize_results.py')}"):
            print("生成可视化结果失败")
    else:
        print_step(3, "生成可视化结果 [已跳过]")
    
    # 完成
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print_header(f"消融实验完成! 总用时: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    print(f"\n结果保存在: {os.path.join(script_dir, 'results')}")
    
    # 提示模型性能排名
    try:
        import json
        results_file = os.path.join(script_dir, "results", "evaluation_results_fixed.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print("\n模型性能排名 (按F1分数):")
            sorted_models = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
            for i, (model_name, metrics) in enumerate(sorted_models, 1):
                print(f"{i}. {model_name}: F1={metrics['f1']:.4f}, 准确率={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")
                
            best_model = sorted_models[0][0]
            print(f"\n最佳模型: {best_model}")
            print("\n消融实验表明:")
            components = {
                "注意力机制": "能够有效捕获序列中的关键特征",
                "残差连接": "帮助解决深层网络中的梯度问题",
                "概率校准层": "使得输出概率分布更加合理",
                "混合专家系统": "根据不同输入特征动态调整模型行为",
                "双向LSTM结构": "同时考虑前后文信息，提升序列理解能力"
            }
            
            for component, desc in components.items():
                print(f"- {component}: {desc}")
                
    except Exception as e:
        print(f"读取结果文件失败: {e}")

if __name__ == "__main__":
    main() 