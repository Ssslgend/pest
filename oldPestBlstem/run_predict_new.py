# run_predict_new.py - 启动新的预测过程
import os
import sys
import subprocess
import traceback
import time

def print_header(text):
    """打印带有格式的标题"""
    line = "=" * 80
    print(f"\n{line}")
    print(f"{text.center(80)}")
    print(f"{line}\n")

def main():
    """主函数，运行预测脚本"""
    print_header("栅格预测工具 - 使用新训练的BiLSTM模型")
    
    print("这个脚本使用新训练的BiLSTM模型生成山东地区的害虫风险栅格预测。")
    print("它会加载训练好的模型和标准化器，读取输入栅格数据，然后生成预测结果。\n")
    
    script_path = os.path.join('sd_raster_prediction', 'predict_raster_new.py')
    
    # 验证脚本是否存在
    if not os.path.exists(script_path):
        print(f"错误: 预测脚本 {script_path} 不存在。")
        return
    
    # 检查输入和输出目录
    try:
        import sys
        sys.path.append(os.path.abspath('.'))
        from sd_raster_prediction.config_raster_new import get_config
        
        config = get_config()
        input_dir = os.path.dirname(list(config['feature_raster_map'].values())[0])
        output_dir = config['prediction_output_dir']
        
        print(f"配置信息:")
        print(f"- 输入栅格目录: {input_dir}")
        print(f"- 输出目录: {output_dir}")
        print(f"- 模型文件: {config['model_save_path']}")
        print(f"- 标准化器文件: {config['scaler_save_path']}")
        
        # 检查模型和标准化器是否存在
        if not os.path.exists(config['model_save_path']):
            print(f"警告: 模型文件不存在: {config['model_save_path']}")
        else:
            print(f"模型文件存在")
            
        if not os.path.exists(config['scaler_save_path']):
            print(f"警告: 标准化器文件不存在: {config['scaler_save_path']}")
        else:
            print(f"标准化器文件存在")
            
        # 检查输入目录是否存在
        if not os.path.exists(input_dir):
            print(f"警告: 输入栅格目录不存在: {input_dir}")
            print("请确保输入数据已正确准备")
        else:
            files = os.listdir(input_dir)
            tif_files = [f for f in files if f.endswith('.tif')]
            print(f"输入目录包含 {len(tif_files)} 个TIF文件，共 {len(files)} 个文件")
            
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录已准备: {output_dir}")
        
    except Exception as e:
        print(f"配置验证过程中出错: {e}")
        traceback.print_exc()
    
    print("\n开始运行预测过程，这可能需要一些时间...")
    start_time = time.time()
    try:
        # 直接导入并调用预测函数，而不是使用subprocess
        print("直接导入预测模块...")
        from sd_raster_prediction.predict_raster_new import predict_raster
        print("调用预测函数...")
        predict_raster()
        end_time = time.time()
        print(f"\n预测完成！总用时: {end_time-start_time:.2f} 秒")
    except Exception as e:
        print(f"\n运行预测过程时出现错误: {e}")
        traceback.print_exc()
    
    print("\n预测结果包括:")
    print("1. 概率栅格 (.tif) - 显示每个位置的害虫发生概率")
    print("2. 风险等级栅格 (.tif) - 将概率划分为不同风险等级")
    print("3. 风险分布统计表 (.csv) - 不同风险等级的区域比例")
    print("4. 增强显示风险栅格 (.tif) - 使用更鲜明的色彩显示风险等级")
    print("5. 概率分布直方图 (.csv) - 概率值的分布统计")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        traceback.print_exc()
        print("程序异常终止。") 