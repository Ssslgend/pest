# run_multi_year_prediction.py - 多年份批量预测脚本
import os
import sys
import argparse
import traceback
from tqdm import tqdm

def print_header(text):
    """打印带有格式的标题"""
    line = "=" * 80
    print(f"\n{line}")
    print(f"{text.center(80)}")
    print(f"{line}\n")

def run_prediction_for_year(year):
    """为指定年份运行预测"""
    print_header(f"开始预测 {year} 年数据")
    
    try:
        # 添加项目根目录到路径
        sys.path.append(os.path.abspath('.'))
        
        # 导入预测函数和配置
        from sd_raster_prediction.predict_raster_new import predict_raster
        from sd_raster_prediction.config_raster_new import get_config
        
        # 获取指定年份的配置
        config = get_config(prediction_year=year)
        
        # 显示当前配置路径
        input_dir = config['input_raster_base']
        output_dir = config['prediction_output_dir']
        
        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        
        # 检查输入目录是否存在
        if not os.path.exists(input_dir):
            print(f"错误: 输入目录 {input_dir} 不存在!")
            print(f"请确保 {year} 年的输入数据已准备好。")
            return False
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 运行预测
        print(f"使用年份特定配置开始预测 {year} 年数据...")
        predict_raster(config)
        
        print(f"{year} 年数据预测完成!\n")
        return True
        
    except Exception as e:
        print(f"预测 {year} 年数据时出错: {e}")
        traceback.print_exc()
        return False

def run_visualizations(years):
    """为多个年份生成可视化"""
    successful_years = []
    
    for year in years:
        try:
            # 导入配置和可视化工具
            from sd_raster_prediction.config_raster_new import get_config
            from sd_raster_prediction.visualization_helper import create_all_visualizations
            
            # 获取指定年份的配置
            config = get_config(prediction_year=year)
            
            # 设置当前工作配置为指定年份的配置
            print(f"为 {year} 年生成可视化图表...")
            create_all_visualizations(config)
            
            successful_years.append(year)
            
        except ImportError:
            print(f"警告: 无法导入可视化模块，跳过 {year} 年数据的可视化。")
            print("请确保已安装所需的依赖库: matplotlib, numpy, pandas, rasterio, geopandas")
            
        except Exception as e:
            print(f"为 {year} 年生成可视化时出错: {e}")
            traceback.print_exc()
    
    return successful_years

def main():
    """主函数 - 解析命令行参数并执行批量预测"""
    print_header("山东美国白蛾病虫害多年份批量预测工具")
    print("此脚本可以批量预测多个年份的美国白蛾病虫害风险分布。")
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='多年份美国白蛾病虫害风险预测工具')
    parser.add_argument('-y', '--years', type=int, nargs='+', help='要预测的年份列表，例如: 2019 2020 2021')
    parser.add_argument('--all', action='store_true', help='预测2019-2024所有年份')
    parser.add_argument('--skip-vis', action='store_true', help='跳过可视化步骤')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 确定要处理的年份
    years_to_process = []
    
    if args.all:
        years_to_process = list(range(2019, 2025))  # 2019-2024
        print("将预测 2019-2024 所有年份的数据")
    elif args.years:
        years_to_process = args.years
        print(f"将预测以下年份的数据: {years_to_process}")
    else:
        # 如果没有指定年份，提供交互式选择
        print("\n请输入要预测的年份，用空格分隔（如: 2019 2020 2023）")
        print("或者输入 'all' 预测所有年份 (2019-2024)")
        user_input = input("> ").strip()
        
        if user_input.lower() == 'all':
            years_to_process = list(range(2019, 2025))
            print("将预测 2019-2024 所有年份的数据")
        else:
            try:
                years_to_process = [int(year) for year in user_input.split()]
                print(f"将预测以下年份的数据: {years_to_process}")
            except ValueError:
                print("输入格式错误。请输入有效的年份数字。")
                return
    
    # 验证年份是否有效
    valid_years = [year for year in years_to_process if 2019 <= year <= 2024]
    if len(valid_years) != len(years_to_process):
        invalid_years = [year for year in years_to_process if year not in valid_years]
        print(f"警告: 忽略无效年份 {invalid_years}。仅支持 2019-2024 范围内的年份。")
        years_to_process = valid_years
    
    if not years_to_process:
        print("没有有效的年份可处理。")
        return
    
    # 开始批量预测
    successful_years = []
    for year in years_to_process:
        success = run_prediction_for_year(year)
        if success:
            successful_years.append(year)
    
    # 汇总结果
    print_header("预测完成汇总")
    print(f"成功预测的年份: {successful_years if successful_years else '无'}")
    
    failed_years = [year for year in years_to_process if year not in successful_years]
    if failed_years:
        print(f"失败的年份: {failed_years}")
    
    # 生成可视化
    if not args.skip_vis and successful_years:
        print_header("开始生成可视化图表")
        vis_years = run_visualizations(successful_years)
        print(f"成功生成可视化的年份: {vis_years if vis_years else '无'}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序执行时出错: {e}")
        traceback.print_exc() 