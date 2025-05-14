import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import glob

# 添加项目根目录到Python路径，以便导入ModelLogger（如果需要）
# project_root = os.path.dirname(os.path.abspath(__file__))
# if project_root not in sys.path:
#     sys.path.append(project_root)

# 设置英文字体
def set_plot_style():
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        print("Plot style set to English")
    except Exception as e:
         print(f"Error setting plot style: {e}")

set_plot_style()

# 定义输入和输出目录
DATA_DIR = "./data" # 日志CSV文件所在目录
OUTPUT_DIR = "./output" # 图表输出目录
MODEL_DIRS = { # 各模型训练脚本的根目录（用于查找日志）
    'MLP': ' pestMLP/output/logs/mlp_training_log.txt',
    'LSTM': 'pestLSTM/output/logs/lstm_training_log.txt',
    'BiLSTM': 'sd_raster_prediction/results/analysis/training_history.png'
}

def find_and_load_logs():
    """查找并加载各个模型的训练日志CSV文件"""
    dfs = []
    print(f"Searching for training logs in {DATA_DIR} directory...")
    found_files = glob.glob(os.path.join(DATA_DIR, '*_training_log.csv'))

    if not found_files:
        print("Warning: No 'training_log.csv' files found in 'data' directory.")
        print("Please make sure you've run the modified training scripts and generated logs.")
        return None

    for file_path in found_files:
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            if not df.empty:
                # 从文件名或列中提取模型名称
                model_name = "Unknown"
                if 'Model' in df.columns:
                    model_name = df['Model'].iloc[0]
                else:
                    filename = os.path.basename(file_path)
                    for name_key in MODEL_DIRS.keys():
                        if name_key.lower() in filename.lower():
                            model_name = name_key
                            df['Model'] = model_name # 添加列
                            break
               
                print(f"Successfully loaded {model_name} log: {file_path} ({len(df)} rows)")
                dfs.append(df)
            else:
                print(f"Warning: Empty file {file_path}")
        except Exception as e:
            print(f"Failed to load or process file {file_path}: {e}")

    if not dfs:
        print("Error: Could not load any valid training log data.")
        return None

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined data has {len(combined)} records, covering {combined['Model'].nunique()} models: {combined['Model'].unique().tolist()}")
    return combined

def get_model_params(model_name):
    """估算模型参数量 (需要根据实际情况调整)"""
    # 这是一个简化的示例，实际参数量需要从模型定义中获取
    if model_name == 'MLP':
        return 0.1 # 约10万
    elif model_name == 'LSTM':
        return 0.5 # 约50万
    elif model_name == 'BiLSTM':
        return 1.5 # 约150万
    else:
        return np.nan

def generate_comparison_plots(combined_df):
    """生成模型对比的可视化图表"""
    if combined_df is None or combined_df.empty:
        print("Error: No data available for generating charts.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating charts in {OUTPUT_DIR} directory...")

    # 定义模型颜色
    model_colors = {
        'MLP': '#FF6666', # 红色系
        'LSTM': '#6699FF', # 蓝色系
        'BiLSTM': '#66CC99', # 绿色系
        'Unknown': '#AAAAAA'  # 灰色
    }

    models_present = combined_df['Model'].unique()

    # --- 绘制AUC对比图 ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_df, x='epoch', y='val_auc', hue='Model', palette=model_colors,
                 style='Model', markers=True, dashes=False, linewidth=2, markersize=6)

    # 添加参数量标注
    for model in models_present:
        model_data = combined_df[combined_df['Model'] == model]
        if not model_data.empty:
            last_epoch_data = model_data.loc[model_data['epoch'].idxmax()]
            params = get_model_params(model)
            if pd.notna(params):
                plt.annotate(f"{params:.1f}M params",
                             xy=(last_epoch_data['epoch'], last_epoch_data['val_auc']),
                             xytext=(5, -10), textcoords='offset points',
                             fontsize=9, color=model_colors.get(model, 'gray'))

    plt.title('Model Validation AUC Comparison', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.ylim(bottom=max(0.4, combined_df['val_auc'].min() - 0.05), top=1.05) # 动态调整Y轴下限
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Model', loc='lower right')
    plt.tight_layout()
    auc_plot_path = os.path.join(OUTPUT_DIR, 'model_auc_comparison.png')
    plt.savefig(auc_plot_path, dpi=300)
    plt.close()
    print(f"AUC comparison chart saved: {auc_plot_path}")

    # --- 绘制损失对比图 ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_df, x='epoch', y='val_loss', hue='Model', palette=model_colors,
                 style='Model', markers=True, dashes=False, linewidth=2, markersize=6)

    plt.title('Model Validation Loss Comparison', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.ylim(bottom=0, top=combined_df['val_loss'].max() * 1.1) # 动态调整Y轴上限
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Model', loc='upper right')
    plt.tight_layout()
    loss_plot_path = os.path.join(OUTPUT_DIR, 'model_loss_comparison.png')
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"Loss comparison chart saved: {loss_plot_path}")

    # --- 生成性能摘要表 ---
    summary_list = []
    for model in models_present:
        model_data = combined_df[combined_df['Model'] == model].copy() # 使用副本避免SettingWithCopyWarning
        if not model_data.empty:
            # 最佳AUC
            best_auc_idx = model_data['val_auc'].idxmax()
            best_auc_row = model_data.loc[best_auc_idx]
            # 最低Loss
            best_loss_idx = model_data['val_loss'].idxmin()
            best_loss_row = model_data.loc[best_loss_idx]
            # 最终指标
            final_row = model_data.loc[model_data['epoch'].idxmax()]

            summary_list.append({
                'Model': model,
                'Parameters(M)': get_model_params(model),
                'Total Epochs': final_row['epoch'],
                'Best Val AUC': best_auc_row['val_auc'],
                'Best AUC Epoch': best_auc_row['epoch'],
                'Best Val Loss': best_loss_row['val_loss'],
                'Best Loss Epoch': best_loss_row['epoch'],
                'Final Val AUC': final_row['val_auc'],
                'Final Val Loss': final_row['val_loss']
            })

    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        summary_df = summary_df.round(4) # 保留4位小数
        summary_path = os.path.join(OUTPUT_DIR, 'model_performance_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"Performance summary saved: {summary_path}")
        print("\nModel Performance Summary:\n", summary_df.to_string(index=False))
    else:
        print("Warning: Could not generate performance summary table.")

def main():
    # 确保脚本在正确的目录下运行
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Current working directory: {os.getcwd()}")

    print("\n=== Model Performance Comparison Tool ===")

    # 1. 加载处理好的日志数据
    combined_data = find_and_load_logs()

    # 2. 生成对比图表和摘要
    if combined_data is not None:
        generate_comparison_plots(combined_data)
        print("\nAnalysis complete! Results saved in ./output directory")
    else:
        print("\nAnalysis aborted because no training log data could be loaded.")

if __name__ == "__main__":
    main()
