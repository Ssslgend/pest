import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap

# 设置英文字体支持
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Tahoma']
plt.rcParams['axes.unicode_minus'] = True  # 确保负号正确显示

# 设置更美观的风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

# 创建输出目录
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "visualizations")
os.makedirs(output_dir, exist_ok=True)

def load_data():
    """Load evaluation results and training history"""
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "evaluation_results_fixed.json")
    history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "training_histories_fixed.json")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    with open(history_path, 'r') as f:
        histories = json.load(f)
    
    # Translate model names to English
    model_name_mapping = {
        "完整BiLSTM": "Complete BiLSTM",
        "无注意力机制": "Without Attention",
        "无残差连接": "Without Residual",
        "无概率校准层": "Without Calibration",
        "无混合专家层": "Without MoE",
        "单向LSTM": "Unidirectional LSTM"
    }
    
    # Update model names in results
    results_en = {}
    for model_name, model_data in results.items():
        if model_name in model_name_mapping:
            results_en[model_name_mapping[model_name]] = model_data
        else:
            results_en[model_name] = model_data
    
    # Update model names in histories
    histories_en = {}
    for model_name, model_data in histories.items():
        if model_name in model_name_mapping:
            histories_en[model_name_mapping[model_name]] = model_data
        else:
            histories_en[model_name] = model_data
    
    return results_en, histories_en

def plot_radar_chart(results, save_path):
    """Plot radar chart comparing different model metrics"""
    # Set up metrics list (excluding confusion matrix)
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity']
    metrics_en = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Specificity']
    
    # Set up radar chart angles
    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the radar chart
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    
    # Set up attractive colors
    colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A8', '#A833FF', '#33FFF5']
    # Custom line styles
    line_styles = ['-', '-', '--', '-.', ':', '--']
    # Custom markers
    markers = ['o', 's', '^', 'D', '*', 'p']
    
    # Emphasize Complete BiLSTM performance
    for i, (model_name, model_results) in enumerate(results.items()):
        # Extract metric values (excluding confusion matrix)
        values = [model_results[metric] for metric in metrics]
        values += values[:1]  # Close the radar chart
        
        if model_name == "Complete BiLSTM":
            linewidth = 4.0
            alpha = 1.0
            zorder = 10  # Ensure Complete BiLSTM is on top
        else:
            linewidth = 2.0
            alpha = 0.7
            zorder = 5
        
        # Plot each model's radar chart
        ax.plot(angles, values, color=colors[i % len(colors)], 
                linewidth=linewidth, linestyle=line_styles[i % len(line_styles)], 
                marker=markers[i % len(markers)], markersize=8, label=model_name,
                alpha=alpha, zorder=zorder)
    
    # Set up radar chart ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_en, size=14)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=12)
    ax.set_ylim(0, 1)
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    # 将图例放在图表下方中央位置
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), 
               ncol=3, fontsize=12)
    plt.title("Model Performance Radar Chart", size=20, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Radar chart saved to: {save_path}")

def plot_training_curves(histories, save_path):
    """Plot training curves showing training and validation loss changes"""
    plt.figure(figsize=(16, 12))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    for model_name, history in histories.items():
        train_losses = history['train_losses']
        epochs = range(1, len(train_losses) + 1)
        
        if model_name == "Complete BiLSTM":
            plt.plot(epochs, train_losses, 'r-', linewidth=3, label=model_name)
        else:
            plt.plot(epochs, train_losses, '--', linewidth=1.5, alpha=0.7, label=model_name)
    
    plt.title("Training Loss", size=16)
    plt.xlabel("Epochs", size=14)
    plt.ylabel("Loss", size=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot validation loss
    plt.subplot(2, 2, 2)
    for model_name, history in histories.items():
        val_losses = history['val_losses']
        epochs = range(1, len(val_losses) + 1)
        
        if model_name == "Complete BiLSTM":
            plt.plot(epochs, val_losses, 'r-', linewidth=3, label=model_name)
        else:
            plt.plot(epochs, val_losses, '--', linewidth=1.5, alpha=0.7, label=model_name)
    
    plt.title("Validation Loss", size=16)
    plt.xlabel("Epochs", size=14)
    plt.ylabel("Loss", size=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(2, 2, 3)
    for model_name, history in histories.items():
        if 'val_accuracies' in history:
            val_accs = history['val_accuracies']
            epochs = range(1, len(val_accs) + 1)
            
            if model_name == "Complete BiLSTM":
                plt.plot(epochs, val_accs, 'r-', linewidth=3, label=model_name)
            else:
                plt.plot(epochs, val_accs, '--', linewidth=1.5, alpha=0.7, label=model_name)
    
    plt.title("Validation Accuracy", size=16)
    plt.xlabel("Epochs", size=14)
    plt.ylabel("Accuracy", size=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim([0, 1])
    plt.legend()
    
    # Plot convergence speed comparison (training time)
    plt.subplot(2, 2, 4)
    model_names = list(histories.keys())
    training_times = [history['training_time'] for history in histories.values()]
    best_val_losses = [history['best_val_loss'] for history in histories.values()]
    
    # Create dual-axis chart
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot training time bar chart
    bars = ax1.bar(model_names, training_times, alpha=0.7, color='skyblue', edgecolor='black', linewidth=1.5)
    ax1.set_ylabel("Training Time (seconds)", size=14)
    
    # Plot best validation loss line chart
    ax2.plot(model_names, best_val_losses, 'ro-', linewidth=2.5, markersize=8)
    ax2.set_ylabel("Best Validation Loss", size=14, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}s', ha='center', va='bottom', size=10)
    
    plt.title("Training Time vs Best Validation Loss", size=16)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {save_path}")

def plot_component_contribution(results, save_path):
    """Plot contribution of each component to model performance"""
    # Set up metrics and their English names
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity']
    metrics_en = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Specificity']
    
    # Calculate performance differences between variants and complete model
    complete_results = results["Complete BiLSTM"]
    contribution_data = []
    
    for model_name, model_results in results.items():
        if model_name != "Complete BiLSTM":
            for metric, metric_en in zip(metrics, metrics_en):
                # Calculate performance reduction percentage
                diff = complete_results[metric] - model_results[metric]
                diff_percent = (diff / complete_results[metric]) * 100 if complete_results[metric] > 0 else 0
                
                # Extract component name from model name
                if "Without" in model_name:
                    component_name = model_name.replace("Without ", "")
                elif model_name == "Unidirectional LSTM":
                    component_name = "Bidirectional"
                else:
                    component_name = model_name
                
                contribution_data.append({
                    'Missing Component': model_name,
                    'Contributing Component': component_name,
                    'Metric': metric_en,
                    'Performance Reduction': diff,
                    'Reduction Percentage': diff_percent
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(contribution_data)
    
    # Plot heatmap showing component contributions
    plt.figure(figsize=(14, 10))
    pivot_df = df.pivot(index='Contributing Component', columns='Metric', values='Reduction Percentage')
    
    # Custom color map to highlight larger contributions
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    
    # Plot heatmap with improved text visibility
    sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap=cmap, 
                linewidths=0.5, linecolor='gray',
                annot_kws={"size": 12, "weight": "bold"},
                cbar_kws={'label': 'Performance Reduction (%)'})
    
    plt.title("Component Contribution to Model Performance", size=18, pad=20)
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Component contribution heatmap saved to: {save_path}")
    
    # Plot bar chart showing F1 score contributions
    plt.figure(figsize=(14, 8))
    f1_contributions = df[df['Metric'] == 'F1 Score'].sort_values('Reduction Percentage', ascending=False)
    
    bars = plt.bar(f1_contributions['Contributing Component'], f1_contributions['Reduction Percentage'], 
                  color=sns.color_palette("YlOrRd", len(f1_contributions)),
                  edgecolor='black', linewidth=1.5)
    
    plt.title("Component Contribution to F1 Score", size=18)
    plt.xlabel("Contributing Component", size=14)
    plt.ylabel("Performance Reduction (%)", size=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', size=12)
    
    plt.tight_layout(pad=2.0)
    f1_contribution_path = save_path.replace('.png', '_f1.png')
    plt.savefig(f1_contribution_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"F1 contribution chart saved to: {f1_contribution_path}")

def plot_bar_comparison(results, save_path):
    """Plot bar chart comparing key metrics across models"""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metrics_en = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    plt.figure(figsize=(14, 10))
    model_names = list(results.keys())
    x = np.arange(len(model_names))
    width = 0.2
    
    for i, (metric, metric_en) in enumerate(zip(metrics, metrics_en)):
        values = [results[model][metric] for model in model_names]
        offset = width * (i - len(metrics)/2 + 0.5)
        rects = plt.bar(x + offset, values, width, label=metric_en)
        
        # Add data labels
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90, fontsize=8)
    
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Model Performance Comparison', fontsize=16)
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Bar comparison chart saved to: {save_path}")

def plot_confusion_matrices(results, save_path):
    """Plot confusion matrices for all models"""
    plt.figure(figsize=(15, 10))
    model_names = list(results.keys())
    rows = 2
    cols = 3
    
    for i, model_name in enumerate(model_names):
        if i < rows * cols:
            plt.subplot(rows, cols, i+1)
            
            confusion_matrix = results[model_name]['confusion_matrix']
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Negative', 'Positive'], 
                        yticklabels=['Negative', 'Positive'])
            
            plt.title(model_name, fontsize=12)
            plt.xlabel('Predicted', fontsize=10)
            plt.ylabel('Actual', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrices saved to: {save_path}")

def create_summary_dashboard(results, histories, save_path):
    """Create a summary dashboard of model performance"""
    plt.figure(figsize=(16, 12))
    
    # Plot performance metrics
    ax1 = plt.subplot(2, 2, 1)
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metrics_en = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    
    # Create DataFrame for plotting
    data = []
    for model_name, model_results in results.items():
        for metric, metric_en in zip(metrics, metrics_en):
            data.append({
                'Model': model_name,
                'Metric': metric_en,
                'Value': model_results[metric]
            })
    
    df = pd.DataFrame(data)
    
    # Plot heatmap with improved settings
    pivot_df = df.pivot(index='Model', columns='Metric', values='Value')
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax1,
                annot_kws={"size": 11, "weight": "bold"},
                cbar_kws={"shrink": 0.8})
    ax1.set_title('Performance Metrics Comparison', fontsize=14)
    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=11, rotation=0)
    ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=11, rotation=0)
    
    # Plot training time comparison
    ax2 = plt.subplot(2, 2, 2)
    model_names = list(results.keys())
    training_times = [histories[model]['training_time'] for model in model_names]
    
    bars = ax2.bar(model_names, training_times, color='skyblue')
    ax2.set_title('Training Time Comparison', fontsize=14)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=11)
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}s', ha='center', va='bottom', fontsize=10)
    
    # Plot convergence curves
    ax3 = plt.subplot(2, 2, 3)
    for model_name, history in histories.items():
        val_losses = history['val_losses']
        epochs = range(1, len(val_losses) + 1)
        
        if model_name == "Complete BiLSTM":
            ax3.plot(epochs, val_losses, 'r-', linewidth=3, label=model_name)
        else:
            ax3.plot(epochs, val_losses, '--', linewidth=1.5, alpha=0.7, label=model_name)
    
    ax3.set_title('Validation Loss Curves', fontsize=14)
    ax3.set_xlabel('Epochs', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Plot performance gains/losses relative to complete model
    ax4 = plt.subplot(2, 2, 4)
    complete_f1 = results["Complete BiLSTM"]['f1']
    model_names_without_complete = [m for m in model_names if m != "Complete BiLSTM"]
    f1_diffs = [(complete_f1 - results[m]['f1']) / complete_f1 * 100 for m in model_names_without_complete]
    
    bars = ax4.bar(model_names_without_complete, f1_diffs, color='salmon')
    ax4.set_title('F1 Score Reduction vs Complete BiLSTM', fontsize=14)
    ax4.set_ylabel('Reduction (%)', fontsize=12)
    ax4.set_xticklabels(model_names_without_complete, rotation=45, ha='right', fontsize=11)
    ax4.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Summary dashboard saved to: {save_path}")

def create_all_visualizations():
    """Create all visualizations for ablation study"""
    # Load data
    results, histories = load_data()
    
    # Create visualizations
    plot_radar_chart(results, os.path.join(output_dir, 'model_performance_radar.png'))
    plot_training_curves(histories, os.path.join(output_dir, 'training_curves.png'))
    plot_component_contribution(results, os.path.join(output_dir, 'component_contribution.png'))
    plot_bar_comparison(results, os.path.join(output_dir, 'summary_performance.png'))
    plot_confusion_matrices(results, os.path.join(output_dir, 'confusion_matrices.png'))
    create_summary_dashboard(results, histories, os.path.join(output_dir, 'summary_dashboard.png'))
    
    print("All visualizations created successfully!")

if __name__ == "__main__":
    create_all_visualizations()