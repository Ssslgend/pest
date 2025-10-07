# sd_raster_prediction/utils.py
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import os
import pandas as pd
from torch.utils.data import DataLoader # Needed for importance calculation loader
from sd_raster_prediction.data_processor_raster_new import SdPestPresenceAbsenceDataset # Needed for temporary dataset
from sd_raster_prediction.config_raster_new import get_config
import seaborn as sns

CONFIG = get_config()

def evaluate_model(model, loader, device, criterion=None):
    """Evaluates the model on a given data loader, returns metrics and predictions."""
    model.eval()
    all_labels = []
    all_outputs = []
    all_coords = []
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Evaluating", leave=False)
        for inputs, labels, coords in progress_bar:
            # inputs expected shape: (batch, 1, features)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if criterion:
                try:
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                    progress_bar.set_postfix(loss=loss.item())
                except Exception as e:
                    print(f"Error calculating loss during evaluation: {e}")

            all_labels.extend(labels.cpu().numpy().flatten())
            all_outputs.extend(outputs.cpu().numpy().flatten()) # Probabilities
            all_coords.extend(coords.cpu().numpy())
            dataset_size += inputs.size(0)

    avg_loss = running_loss / dataset_size if criterion and dataset_size > 0 else None
    labels_np = np.array(all_labels)
    outputs_np = np.array(all_outputs)
    coords_np = np.array(all_coords)

    auc = 0.5 # Default if calculation fails or not possible
    if len(np.unique(labels_np)) >= 2:
        try:
            auc = roc_auc_score(labels_np, outputs_np)
        except ValueError as e:
            print(f"Warning: Could not calculate AUC: {e}. Returning 0.5")
    else:
         print("Warning: Only one class present during evaluation, AUC is 0.5.")

    # Calculate other metrics based on 0.5 threshold
    threshold = 0.5
    preds_binary = (outputs_np >= threshold).astype(int)
    accuracy = accuracy_score(labels_np, preds_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, preds_binary, average='binary', zero_division=0
    )
    cm = confusion_matrix(labels_np, preds_binary)

    metrics = {
        'loss': avg_loss,
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

    return metrics, labels_np, outputs_np, coords_np

def calculate_and_plot_permutation_importance(model, data_processor, baseline_auc, device, plot_path=None, csv_path=None):
    """Calculates and plots feature importance using permutation."""
    print("\n--- Calculating Permutation Feature Importance --- ")
    model.eval()
    importances = {}
    if not hasattr(data_processor, 'feature_columns') or not data_processor.feature_columns:
        print("Error: feature_columns not found in data_processor.")
        return None
    feature_names = data_processor.feature_columns

    if not hasattr(data_processor, 'data_dict') or 'test' not in data_processor.data_dict:
        print("Error: Test data not found in data_processor.")
        return None
    X_test_orig = data_processor.data_dict['test']['X']
    y_test_orig = data_processor.data_dict['test']['y']
    coords_test_orig = data_processor.data_dict['test']['coordinates']

    print(f"Calculating importance for {len(feature_names)} features...")
    pbar = tqdm(range(X_test_orig.shape[1]), desc="Permuting Features")
    for i in pbar:
        feature_name = feature_names[i]
        pbar.set_postfix_str(f"Feature: {feature_name}")

        X_test_permuted = copy.deepcopy(X_test_orig)
        np.random.shuffle(X_test_permuted[:, i])

        permuted_dataset = SdPestPresenceAbsenceDataset(X_test_permuted, y_test_orig, coords_test_orig)
        permuted_loader = DataLoader(permuted_dataset, batch_size=CONFIG['training']['batch_size'] * 2, shuffle=False)

        # Use evaluate_model, we only need the AUC from the metrics dict
        permuted_metrics, _, _, _ = evaluate_model(model, permuted_loader, device)
        permuted_auc = permuted_metrics['auc']

        importance_score = baseline_auc - permuted_auc
        importances[feature_name] = importance_score

    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    print("\n--- Permutation Importance Results (Drop in AUC) --- ")
    for feature, score in sorted_importances.items():
        print(f"  {feature}: {score:.4f}")

    # Save importance scores
    try:
        importance_df = pd.DataFrame(list(sorted_importances.items()), columns=['Feature', 'Importance_AUC_Drop'])
        # 使用传入的csv路径或配置中的路径
        importance_path = csv_path if csv_path else CONFIG['feature_importance_csv_path']
        os.makedirs(os.path.dirname(importance_path), exist_ok=True)
        importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
        print(f"Feature importance scores saved to: {importance_path}")
    except Exception as e:
        print(f"Error saving feature importance scores: {e}")

    # Plotting feature importance
    try:
        # 使用英文字体
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.unicode_minus'] = True
        plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
        plt.barh(list(sorted_importances.keys()), list(sorted_importances.values()), color='skyblue')
        # 使用英文标签
        plt.xlabel("Feature Importance (AUC Drop)")
        plt.ylabel("Features")
        plt.title("Permutation Feature Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        # 使用传入的图像路径或配置中的路径
        plot_path_to_use = plot_path if plot_path else CONFIG['feature_importance_plot_path']
        os.makedirs(os.path.dirname(plot_path_to_use), exist_ok=True)
        plt.savefig(plot_path_to_use)
        plt.close()
        print(f"Feature importance plot saved to: {plot_path_to_use}")
    except Exception as e:
        print(f"Error plotting feature importance: {e}")

    return sorted_importances

def plot_training_history(history, save_path):
    """Plots training and validation loss and AUC from history dictionary."""
    epochs = range(1, len(history['train_loss']) + 1)

    # 使用英文字体
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = True
    plt.figure(figsize=(12, 5))

    # 处理可能的无限或NaN值
    train_loss = np.array(history['train_loss'])
    val_loss = np.array(history['val_loss'])
    
    # 替换无限值和NaN值，但保留波动
    train_loss = np.array([x if (not np.isinf(x) and not np.isnan(x)) else np.nan for x in train_loss])
    val_loss = np.array([x if (not np.isinf(x) and not np.isnan(x)) else np.nan for x in val_loss])
    
    # 仅用于确定y轴范围的临时数组（忽略NaN值）
    train_loss_valid = train_loss[~np.isnan(train_loss)]
    val_loss_valid = val_loss[~np.isnan(val_loss)]
    
    if len(train_loss_valid) > 0 and len(val_loss_valid) > 0:
        # 使用更宽松的百分位数，保留更多波动
        ymin = min(np.min(train_loss_valid), np.min(val_loss_valid)) * 0.95
        ymax = max(np.percentile(train_loss_valid, 99), np.percentile(val_loss_valid, 99)) * 1.05
        
        # 确保有足够的间距显示波动
        range_y = ymax - ymin
        if range_y < 0.1:  # 如果范围太小，扩大它
            mean_val = (ymax + ymin) / 2
            ymin = mean_val - 0.1
            ymax = mean_val + 0.1
    else:
        # 如果没有有效值，使用默认范围
        ymin, ymax = 0, 1
    
    # NaN值的实际替换（仅用于绘图）
    # 使用前一个有效值填充NaN，保留趋势
    for i in range(1, len(train_loss)):
        if np.isnan(train_loss[i]):
            train_loss[i] = train_loss[i-1] if not np.isnan(train_loss[i-1]) else 0
    for i in range(1, len(val_loss)):
        if np.isnan(val_loss[i]):
            val_loss[i] = val_loss[i-1] if not np.isnan(val_loss[i-1]) else 0
    
    # Plot Loss - 使用点线结合，更清晰地显示波动
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'bo-', label='Training Loss', markersize=4)
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss', markersize=4)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (BCELoss)')
    plt.ylim(ymin, ymax)  # 设置合理的y轴范围
    plt.legend()
    plt.grid(True, alpha=0.3)  # 降低网格线透明度，更好地显示波动

    # Plot AUC - 也使用点线结合
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_auc'], 'bo-', label='Training AUC', markersize=4)
    plt.plot(epochs, history['val_auc'], 'ro-', label='Validation AUC', markersize=4)
    plt.title('Training and Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)  # 增加DPI获得更清晰的图像
    plt.close()
    print(f"Training history plot saved to: {save_path}")

def save_checkpoint(state, filepath):
    """Saves model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)

def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """Loads model checkpoint."""
    if not os.path.exists(filepath):
        print(f"Warning: Checkpoint file not found at {filepath}")
        return None, None # Return None for checkpoint and epoch
    checkpoint = torch.load(filepath, map_location=device)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print(f"Error loading state dict: {e}. Model architecture might have changed.")
        return None, None

    if optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded.")
        except ValueError as e:
            print(f"Warning: Could not load optimizer state. {e}")

    epoch = checkpoint.get('epoch', None)
    best_val_metric = checkpoint.get('best_val_auc', None)
    metric_info = f", Best Val AUC: {best_val_metric:.4f}" if best_val_metric else ""
    epoch_info = f" (Epoch {epoch}{metric_info})" if epoch else ""
    print(f"Checkpoint loaded from {filepath}{epoch_info}")
    return checkpoint, epoch 

def plot_confusion_matrix(y_true, y_pred, save_path):
    """绘制混淆矩阵可视化"""
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 绘制混淆矩阵热图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

def plot_roc_curve(y_true, y_scores, save_path):
    """绘制ROC曲线"""
    from sklearn.metrics import roc_curve, auc
    
    # 计算ROC曲线点
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"ROC curve saved to: {save_path}")

def plot_metrics_comparison(metrics_dict, save_path):
    """绘制模型评估指标比较图"""
    # 准备数据
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    values = [metrics_dict['accuracy'], metrics_dict['precision'], 
              metrics_dict['recall'], metrics_dict['f1'], metrics_dict['auc']]
    
    # 绘制条形图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color='skyblue')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.ylim(0, 1.1)
    plt.title('Model Evaluation Metrics')
    plt.ylabel('Score')
    plt.grid(axis='y', alpha=0.3)
    
    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Metrics comparison saved to: {save_path}")

def plot_top_features_impact(model, X_test, y_test, feature_names, importances, save_path):
    """分析并可视化前五个最重要特征的影响"""
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import roc_auc_score
    
    # 获取前五个最重要特征
    top_n = 5
    top_features = list(dict(sorted(importances.items(), 
                               key=lambda item: item[1], 
                               reverse=True)).keys())[:top_n]
    
    # 创建DataFrame以便分析
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    y_true = y_test
    
    # 创建子图
    fig, axes = plt.subplots(top_n, 1, figsize=(10, 4*top_n))
    
    for i, feature in enumerate(top_features):
        # 获取特征值
        feature_values = X_test_df[feature].values
        
        # 将特征值分成10个bin
        bins = np.linspace(np.min(feature_values), np.max(feature_values), 11)
        bin_indices = np.digitize(feature_values, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins)-2)
        
        # 计算每个bin中的平均目标值
        bin_means = np.zeros(len(bins)-1)
        bin_counts = np.zeros(len(bins)-1)
        
        for bin_idx in range(len(bins)-1):
            mask = bin_indices == bin_idx
            if np.sum(mask) > 0:
                bin_means[bin_idx] = np.mean(y_true[mask])
                bin_counts[bin_idx] = np.sum(mask)
        
        # 计算每个bin的边界值
        bin_edges = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]
        bin_centers = [(edge[0] + edge[1])/2 for edge in bin_edges]
        
        # 绘制特征值与目标关系
        ax = axes[i]
        ax.bar(bin_centers, bin_means, width=(bins[1]-bins[0])*0.8, alpha=0.6, color='skyblue')
        
        # 添加样本数量信息
        for j, (center, count) in enumerate(zip(bin_centers, bin_counts)):
            if count > 0:
                ax.text(center, bin_means[j] + 0.03, f'n={int(count)}', 
                        ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Average Target Value')
        ax.set_title(f'Impact of {feature} (Importance: {importances[feature]:.4f})')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Top features impact analysis saved to: {save_path}")

def plot_feature_interactions(X_test, feature_names, importances, save_path):
    """分析并可视化特征间的相互作用"""
    import pandas as pd
    import seaborn as sns
    
    # 获取前5个最重要特征
    top_n = min(5, len(importances))
    top_features = list(dict(sorted(importances.items(), 
                               key=lambda item: item[1], 
                               reverse=True)).keys())[:top_n]
    
    # 创建DataFrame以便分析
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # 计算相关性矩阵（仅包含重要特征）
    top_corr = X_test_df[top_features].corr()
    
    # 创建热图
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(top_corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # 绘制相关性热图
    sns.heatmap(top_corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt='.2f', square=True, linewidths=.5)
    
    plt.title('Feature Interactions (Pearson Correlation)')
    plt.tight_layout()
    
    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Feature interactions heatmap saved to: {save_path}")

def plot_threshold_impact(y_true, y_scores, save_path):
    """分析并可视化分类阈值对各个指标的影响"""
    from sklearn.metrics import precision_recall_curve, accuracy_score, f1_score
    
    # 计算不同阈值下的精确率、召回率
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    thresholds = np.append(thresholds, 1.0)  # 添加阈值1.0
    
    # 计算不同阈值下的准确率和F1分数
    accuracies = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        accuracies.append(accuracy_score(y_true, y_pred))
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
    
    # 绘制阈值影响图
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision, 'b-', label='Precision')
    plt.plot(thresholds, recall, 'g-', label='Recall')
    plt.plot(thresholds, accuracies, 'r-', label='Accuracy')
    plt.plot(thresholds, f1_scores, 'y-', label='F1 Score')
    
    # 找到F1分数最大的阈值
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    
    # 标记最佳阈值
    plt.axvline(x=best_threshold, color='k', linestyle='--', 
                label=f'Best Threshold = {best_threshold:.2f}')
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Impact of Classification Threshold on Metrics')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    
    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Threshold impact analysis saved to: {save_path}") 