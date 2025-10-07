
# run_all_with_smoothing.py - 整合预测、平滑和可视化的完整流程

import os
import sys
import subprocess
import time
import traceback

def print_header(text):
    """打印带有格式的标题"""
    line = "=" * 80
    print(f"\n{line}")
    print(f"{text.center(80)}")
    print(f"{line}\n")

def run_script(script_path, script_name):
    """
    运行指定的Python脚本
    
    参数:
        script_path: 脚本路径
        script_name: 脚本名称（用于显示）
    
    返回:
        成功执行返回True，否则返回False
    """
    print_header(f"运行{script_name}")
    
    try:
        # 使用Python解释器执行脚本
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # 实时输出脚本执行结果
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                
        # 获取返回码
        return_code = process.poll()
        
        if return_code == 0:
            print(f"\n{script_name}执行成功！")
            return True
        else:
            error = process.stderr.read()
            print(f"\n{script_name}执行失败，错误码: {return_code}")
            if error:
                print(f"错误信息:\n{error}")
            return False
            
    except Exception as e:
        print(f"\n运行{script_name}时出错: {e}")
        traceback.print_exc()
        return False

def main():
    """完整工作流程"""
    print_header("害虫风险预测与空间平滑工作流")
    
    print("这个脚本将自动执行完整的害虫风险预测工作流，包括:")
    print("1. 执行风险预测")
    print("2. 应用空间平滑处理")
    print("3. 生成对比可视化")
    print("4. 可视化平滑后的结果\n")
    
    start_time = time.time()
    
    # 步骤1: 运行预测
    predict_script = "run_predict_new.py"
    if not os.path.exists(predict_script):
        print(f"错误: 找不到预测脚本 {predict_script}")
        return
    
    print("开始执行预测过程...")
    if not run_script(predict_script, "预测脚本"):
        print("由于预测失败，流程终止。")
        return
    
    # 步骤2: 应用空间平滑处理
    smooth_script = "run_smooth.py"
    if not os.path.exists(smooth_script):
        print(f"错误: 找不到平滑处理脚本 {smooth_script}")
        return
    
    print("\n开始执行空间平滑处理...")
    if not run_script(smooth_script, "平滑处理脚本"):
        print("警告: 平滑处理失败，但将继续执行流程。")
    
    # 步骤3: 创建平滑前后的对比可视化
    compare_script = "run_smoothed_visualize.py"
    if not os.path.exists(compare_script):
        print(f"错误: 找不到对比可视化脚本 {compare_script}")
        return
    
    print("\n开始创建平滑前后的对比可视化...")
    if not run_script(compare_script, "对比可视化脚本"):
        print("警告: 对比可视化失败，但将继续执行流程。")
    
    # 步骤4: 可视化平滑后的结果
    visualize_script = "run_visualize.py"
    if not os.path.exists(visualize_script):
        print(f"错误: 找不到可视化脚本 {visualize_script}")
        return
    
    print("\n开始可视化平滑后的结果...")
    if not run_script(visualize_script, "可视化脚本"):
        print("警告: 结果可视化失败。")
    
    # 计算总运行时间
    end_time = time.time()
    run_time = end_time - start_time
    hours, remainder = divmod(run_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print_header("工作流程完成")
    print(f"总运行时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
    print("\n所有任务已完成！请检查结果文件以确保预期效果。")
    print("结果文件位置:")
    print("- 预测结果: ./sd_raster_prediction/results/")
    print("- 平滑后结果: ./sd_raster_prediction/results/smoothed/")
    print("- 对比可视化: ./sd_raster_prediction/results/smoothed/visualizations/")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        traceback.print_exc()
        print("程序异常终止。") 