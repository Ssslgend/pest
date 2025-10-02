# run_visualize.py - 调用可视化辅助工具
import os
import sys
import traceback
import numpy as np

def print_header(text):
    """打印带有格式的标题"""
    line = "=" * 80
    print(f"\n{line}")
    print(f"{text.center(80)}")
    print(f"{line}\n")

def apply_probability_equalization(probabilities, bins=100, min_prob=0.01, max_prob=0.99):
    """完全均匀化概率分布"""
    print("\n应用强制均匀概率分布...")
    
    # 直接使用排序映射方法实现完美均匀分布
    sorted_indices = np.argsort(probabilities)
    ranks = np.zeros_like(probabilities)
    ranks[sorted_indices] = np.linspace(0, 1, len(probabilities))
    
    # 映射到目标范围
    equalized_probs = min_prob + (max_prob - min_prob) * ranks
    
    # 显示均匀化效果
    display_bins = 10
    display_edges = np.linspace(0, 1, display_bins + 1)
    
    print("均匀化前概率分布:")
    for i in range(display_bins):
        bin_start = display_edges[i]
        bin_end = display_edges[i + 1]
        bin_count = np.sum((probabilities >= bin_start) & (probabilities < bin_end))
        bin_percent = (bin_count / len(probabilities)) * 100
        print(f"  概率区间 [{bin_start:.1f}-{bin_end:.1f}): {bin_count} ({bin_percent:.2f}%)")
    
    print("\n均匀化后概率分布:")
    for i in range(display_bins):
        bin_start = display_edges[i]
        bin_end = display_edges[i + 1]
        bin_count = np.sum((equalized_probs >= bin_start) & (equalized_probs < bin_end))
        bin_percent = (bin_count / len(equalized_probs)) * 100
        print(f"  概率区间 [{bin_start:.1f}-{bin_end:.1f}): {bin_count} ({bin_percent:.2f}%)")
    
    return equalized_probs

def main():
    """主函数，运行可视化工具"""
    print_header("Shandong Province Pest Risk Visualization Tool")
    
    print("This script generates visualization images for pest risk prediction in Shandong Province.")
    print("It will create probability distribution maps, risk level maps, histograms and pie charts based on prediction results.\n")
    
    # 检查可视化脚本是否存在
    script_path = os.path.join('sd_raster_prediction', 'visualization_helper.py')
    if not os.path.exists(script_path):
        print(f"Error: Visualization script {script_path} does not exist.")
        return
    
    # 导入并运行可视化函数
    try:
        sys.path.append(os.path.abspath('.'))
        from sd_raster_prediction.visualization_helper import create_all_visualizations
        
        print("Starting visualization generation...")
        create_all_visualizations()
        print("\nVisualization completed!")
        
        # 获取配置信息
        from sd_raster_prediction.config_raster_new import get_config
        config = get_config()
        output_dir = config['prediction_output_dir']
        
        print(f"\nGenerated images are saved to: {output_dir}")
        print("Generated images include:")
        print("1. probability_map.png - Pest occurrence probability map")
        print("2. risk_map.png - Pest risk level distribution map")
        print("3. probability_histogram.png - Probability distribution histogram")
        print("4. risk_distribution_pie.png - Risk level pie chart")
        
    except ImportError:
        print("Error: Unable to import required modules. Please make sure all dependencies are installed.")
        print("Required dependencies include: matplotlib, numpy, pandas, rasterio, geopandas")
        traceback.print_exc()
    except Exception as e:
        print(f"Error during visualization generation: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()
        print("Program terminated abnormally.") 