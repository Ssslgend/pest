
# run_smooth.py - 应用空间平滑处理改进风险预测地图

import os
import sys
import traceback
import time
from sd_raster_prediction.config_raster_new import get_config
from sd_raster_prediction.spatial_smoothing import smooth_raster_file, process_batch_smoothing

def print_header(text):
    """打印带有格式的标题"""
    line = "=" * 80
    print(f"\n{line}")
    print(f"{text.center(80)}")
    print(f"{line}\n")

def main():
    """应用空间平滑处理到风险预测地图"""
    print_header("害虫风险预测空间平滑处理工具")
    
    print("这个脚本对害虫风险预测结果应用空间平滑处理，以改善风险空间分布的连续性。")
    print("它支持多种平滑方法，包括高斯平滑、中值平滑和局部自适应平滑。\n")
    
    # 获取配置
    try:
        config = get_config()
        input_dir = config['prediction_output_dir']
    except Exception as e:
        print(f"读取配置时出错: {e}")
        traceback.print_exc()
        return
    
    # 创建平滑结果输出目录
    output_dir = os.path.join(input_dir, 'smoothed')
    os.makedirs(output_dir, exist_ok=True)
    
    # 显示可用的平滑方法
    print("可用的平滑方法:")
    print("1. 高斯平滑 (gaussian) - 适用于连续数据，保持边缘过渡平滑")
    print("2. 中值平滑 (median) - 适用于去除离群点，保持边界锐利")
    print("3. 均值平滑 (mean) - 简单区域平均，效果介于高斯和中值之间")
    print("4. 自适应平滑 (adaptive) - 根据局部特征自动调整平滑强度\n")
    
    # 默认参数
    default_method = 'gaussian'
    default_sigma = 2.0
    default_size = 5
    
    # 检查预测结果文件
    probability_file = os.path.join(input_dir, 'sd_probability.tif')
    risk_file = os.path.join(input_dir, 'sd_risk_class.tif')
    raw_probability_file = os.path.join(input_dir, 'sd_raw_probability.tif')
    
    files_to_process = []
    if os.path.exists(probability_file):
        files_to_process.append(('概率栅格', probability_file))
    else:
        print(f"警告: 找不到概率文件 {probability_file}")
    
    if os.path.exists(risk_file):
        files_to_process.append(('风险等级栅格', risk_file))
    else:
        print(f"警告: 找不到风险等级文件 {risk_file}")
    
    if os.path.exists(raw_probability_file):
        files_to_process.append(('原始概率栅格', raw_probability_file))
    
    if not files_to_process:
        print("错误: 找不到任何预测结果文件，请先运行预测过程")
        return
    
    # 处理每个文件
    for file_desc, file_path in files_to_process:
        print(f"\n正在处理{file_desc}: {os.path.basename(file_path)}")
        
        # 为不同类型文件选择适当的平滑参数
        if 'risk_class' in file_path:
            # 风险等级栅格（分类数据）使用中值平滑以保持类别边界
            method = 'median'
            output_path = os.path.join(output_dir, f"smoothed_{os.path.basename(file_path)}")
            print(f"对分类数据使用中值平滑 (size={default_size})")
            try:
                smooth_raster_file(file_path, output_path, method, size=default_size, visualize=True)
            except Exception as e:
                print(f"平滑处理出错: {e}")
                traceback.print_exc()
        else:
            # 概率栅格（连续数据）使用高斯平滑
            method = default_method
            output_path = os.path.join(output_dir, f"smoothed_{os.path.basename(file_path)}")
            print(f"对连续数据使用高斯平滑 (sigma={default_sigma})")
            try:
                smooth_raster_file(file_path, output_path, method, sigma=default_sigma, visualize=True)
            except Exception as e:
                print(f"平滑处理出错: {e}")
                traceback.print_exc()
                
            # 对概率栅格额外应用自适应平滑
            if 'probability' in file_path:
                adaptive_output_path = os.path.join(output_dir, f"adaptive_{os.path.basename(file_path)}")
                print(f"对概率数据应用自适应平滑")
                try:
                    smooth_raster_file(file_path, adaptive_output_path, 'adaptive', 
                                       adapt_method='gaussian', sigma=default_sigma, visualize=True)
                except Exception as e:
                    print(f"自适应平滑处理出错: {e}")
                    traceback.print_exc()
    
    # 尝试重新生成风险等级栅格
    try:
        # 查找平滑后的概率栅格
        smoothed_probability = os.path.join(output_dir, f"smoothed_{os.path.basename(probability_file)}")
        if os.path.exists(smoothed_probability):
            print("\n基于平滑后的概率重新生成风险等级栅格...")
            # 导入必要的模块
            from sd_raster_prediction.predict_raster_new import predict_raster
            
            # 临时修改设置以使用平滑后的概率文件重新计算风险等级
            # 这部分在实际运行中可能需要调整
            print("请注意: 这一功能需要修改predict_raster_new.py以支持从已有概率文件重新计算风险等级")
    except Exception as e:
        print(f"重新生成风险等级栅格时出错: {e}")
        traceback.print_exc()
    
    print("\n所有平滑处理已完成！")
    print(f"平滑结果已保存到: {output_dir}")
    print("现在您可以使用run_visualize.py对平滑后的结果生成可视化图像")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        traceback.print_exc()
        print("程序异常终止。") 