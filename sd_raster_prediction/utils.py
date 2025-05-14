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
from data_processor_raster import SdPestPresenceAbsenceDataset # Needed for temporary dataset
from config_raster import get_config

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

def calculate_and_plot_permutation_importance(model, data_processor, baseline_auc, device):
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
        importance_path = CONFIG['feature_importance_csv_path']
        os.makedirs(os.path.dirname(importance_path), exist_ok=True)
        importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
        print(f"Feature importance scores saved to: {importance_path}")
    except Exception as e:
        print(f"Error saving feature importance scores: {e}")

    # Plotting feature importance
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
        plt.barh(list(sorted_importances.keys()), list(sorted_importances.values()), color='skyblue')
        plt.xlabel("特征重要性 (AUC 下降值)")
        plt.ylabel("特征变量")
        plt.title("置换特征重要性排序")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plot_path = CONFIG['feature_importance_plot_path']
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        print(f"Feature importance plot saved to: {plot_path}")
    except Exception as e:
        print(f"Error plotting feature importance: {e}")

    return sorted_importances

def plot_training_history(history, save_path):
    """Plots training and validation loss and AUC from history dictionary."""
    epochs = range(1, len(history['train_loss']) + 1)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='训练损失')
    plt.plot(epochs, history['val_loss'], 'r-', label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失 (BCELoss)')
    plt.legend()
    plt.grid(True)

    # Plot AUC
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_auc'], 'b-', label='训练 AUC')
    plt.plot(epochs, history['val_auc'], 'r-', label='验证 AUC')
    plt.title('训练和验证 AUC')
    plt.xlabel('轮次')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
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