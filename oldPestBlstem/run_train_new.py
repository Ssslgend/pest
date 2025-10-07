#!/usr/bin/env python
# run_train_new.py - 启动新的训练过程
import os
import sys
import subprocess

def print_header(text):
    """打印带有格式的标题"""
    line = "=" * 80
    print(f"\n{line}")
    print(f"{text.center(80)}")
    print(f"{line}\n")

def main():
    """主函数，提供选项让用户选择要运行的脚本"""
    print_header("栅格预测模型训练工具 - 新版本")
    
    print("这个脚本用于运行基于新的train.csv数据集的BiLSTM栅格预测模型训练。")
    print("你可以先测试配置，然后再开始训练。\n")
    
    while True:
        print("\n可用选项:")
        print("1. 测试配置 (测试数据集加载和配置是否正确)")
        print("2. 开始训练 (使用train.csv数据集训练新的BiLSTM模型)")
        print("3. 退出")
        
        choice = input("\n请选择 [1-3]: ")
        
        if choice == '1':
            print_header("测试新配置")
            try:
                # 执行测试配置脚本
                script_path = os.path.join('sd_raster_prediction', 'test_config_new.py')
                subprocess.run([sys.executable, script_path], check=True)
                print("\n配置测试完成。如果没有错误，可以继续进行训练。")
            except subprocess.CalledProcessError:
                print("\n配置测试失败。请检查错误信息并修复问题。")
        
        elif choice == '2':
            print_header("开始训练新模型")
            try:
                # 执行训练脚本
                script_path = os.path.join('sd_raster_prediction', 'train_raster_new.py')
                print("启动训练过程，这可能需要一段时间...\n")
                subprocess.run([sys.executable, script_path], check=True)
                print("\n训练完成。结果已保存到指定的输出目录。")
            except subprocess.CalledProcessError:
                print("\n训练过程中出现错误。请检查错误信息。")
        
        elif choice == '3':
            print("\n感谢使用！再见。")
            break
        
        else:
            print("\n无效选择，请输入1-3之间的数字。")

if __name__ == "__main__":
    main() 