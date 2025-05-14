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
# !!! NOTE: These imports might need adjustment depending on data loading in pestLSTM !!!
# from data_processor_raster import SdPestPresenceAbsenceDataset # Needed for temporary dataset
# from config_raster import get_config # Should use config_lstm

#CONFIG = get_config() # Should load LSTM config

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
        # Assuming loader yields (inputs, labels, coords)
        # Adapt if your LSTM data loader structure is different
        for batch_data in progress_bar:
            # Adjust unpacking based on your DataLoader output for LSTM
            if len(batch_data) == 3:
                inputs, labels, coords = batch_data
            elif len(batch_data) == 2:
                inputs, labels = batch_data
                coords = None # No coordinates provided
            else:
                raise ValueError("Unexpected data format from DataLoader")

            # inputs expected shape: (batch, 1, features) or (batch, features)
            # Model should handle the expected shape
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) # Model forward pass

            if criterion:
                try:
                    # Ensure labels have the same shape as outputs if needed by loss
                    loss = criterion(outputs.squeeze(), labels.float().squeeze()) # Adjust shapes as needed
                    running_loss += loss.item() * inputs.size(0)
                    progress_bar.set_postfix(loss=loss.item())
                except Exception as e:
                    print(f"Error calculating loss during evaluation: {e}")
                    print(f"Output shape: {outputs.shape}, Label shape: {labels.shape}")

            all_labels.extend(labels.cpu().numpy().flatten())
            all_outputs.extend(outputs.cpu().numpy().flatten()) # Probabilities
            if coords is not None:
                all_coords.extend(coords.cpu().numpy())
            dataset_size += inputs.size(0)

    avg_loss = running_loss / dataset_size if criterion and dataset_size > 0 else None
    labels_np = np.array(all_labels)
    outputs_np = np.array(all_outputs)
    coords_np = np.array(all_coords) if all_coords else None

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
    # Handle case where there might be only one class predicted/actual
    unique_labels = np.unique(labels_np)
    unique_preds = np.unique(preds_binary)
    if len(unique_labels) < 2 or len(unique_preds) < 2:
         # Ensure cm calculation handles single class case if necessary, though sklearn might handle it
        cm = confusion_matrix(labels_np, preds_binary, labels=[0, 1]) # Specify labels for consistent shape
    else:
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

# --- Permutation Importance (Requires Adaptation) ---
# This function needs significant adaptation for the LSTM project:
# 1. Needs the LSTM config (paths for saving results).
# 2. Needs the specific data loading/processing used in train_lstm.py.
# 3. Needs the correct way to create temporary datasets/loaders for LSTM.
# Commenting out for now, requires specific implementation details.
"""
def calculate_and_plot_permutation_importance(model, data_processor, baseline_auc, device, config_lstm):
    '''Calculates and plots feature importance using permutation.'''
    print("\n--- Calculating Permutation Feature Importance (LSTM Adaptation Needed) --- ")
    model.eval()
    importances = {}

    # --- !!! ADAPTATION NEEDED HERE !!! --- 
    # Get feature names, test data (X, y, coords) correctly for the LSTM setup
    # feature_names = config_lstm['feature_names']
    # X_test_orig, y_test_orig, coords_test_orig = get_lstm_test_data(data_processor) # Placeholder
    # --- End Adaptation Needed --- 

    print(f"Calculating importance for {len(feature_names)} features...")
    pbar = tqdm(range(X_test_orig.shape[-1]), desc="Permuting Features") # Assuming features are last dim
    for i in pbar:
        feature_name = feature_names[i]
        pbar.set_postfix_str(f"Feature: {feature_name}")

        X_test_permuted = copy.deepcopy(X_test_orig)
        # Permute along the feature dimension
        np.random.shuffle(X_test_permuted[..., i]) # Adapt index if necessary

        # --- !!! ADAPTATION NEEDED HERE !!! ---
        # Create a temporary Dataset and DataLoader compatible with your LSTM training
        # permuted_dataset = LSTMPestDataset(X_test_permuted, y_test_orig, coords_test_orig) # Placeholder
        # permuted_loader = DataLoader(permuted_dataset, batch_size=config_lstm['training']['batch_size'] * 2, shuffle=False)
        # --- End Adaptation Needed --- 

        permuted_metrics, _, _, _ = evaluate_model(model, permuted_loader, device)
        permuted_auc = permuted_metrics['auc']

        importance_score = baseline_auc - permuted_auc
        importances[feature_name] = importance_score

    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    print("\n--- Permutation Importance Results (Drop in AUC) --- ")
    for feature, score in sorted_importances.items():
        print(f"  {feature}: {score:.4f}")

    # Save importance scores (Adapt paths from config_lstm)
    try:
        importance_df = pd.DataFrame(list(sorted_importances.items()), columns=['Feature', 'Importance_AUC_Drop'])
        # importance_path = config_lstm['feature_importance_csv_path'] # Get path from LSTM config
        importance_path = os.path.join(config_lstm['output_dir'], 'lstm_feature_importance.csv') # Example path
        os.makedirs(os.path.dirname(importance_path), exist_ok=True)
        importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
        print(f"Feature importance scores saved to: {importance_path}")
    except Exception as e:
        print(f"Error saving feature importance scores: {e}")

    # Plotting feature importance (Adapt paths from config_lstm)
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
        plt.barh(list(sorted_importances.keys()), list(sorted_importances.values()), color='skyblue')
        plt.xlabel("特征重要性 (AUC 下降值)")
        plt.ylabel("特征变量")
        plt.title("置换特征重要性排序 (LSTM)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        # plot_path = config_lstm['feature_importance_plot_path'] # Get path from LSTM config
        plot_path = os.path.join(config_lstm['output_dir'], 'lstm_feature_importance.png') # Example path
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        print(f"Feature importance plot saved to: {plot_path}")
    except Exception as e:
        print(f"Error plotting feature importance: {e}")

    return sorted_importances
"""

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
    plt.title('训练和验证损失 (LSTM)')
    plt.xlabel('轮次')
    plt.ylabel('损失 (BCELoss)') # Adjust if different loss used
    plt.legend()
    plt.grid(True)

    # Plot AUC
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_auc'], 'b-', label='训练 AUC')
    plt.plot(epochs, history['val_auc'], 'r-', label='验证 AUC')
    plt.title('训练和验证 AUC (LSTM)')
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
        # Add more specific error handling if needed (e.g., strict=False)
        try:
            print("Attempting to load with strict=False")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Loaded state dict with strict=False. Check for missing/unexpected keys.")
        except Exception as inner_e:
            print(f"Failed to load state dict even with strict=False: {inner_e}")
            return None, None

    if optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded.")
        except ValueError as e:
            print(f"Warning: Could not load optimizer state. {e}")

    epoch = checkpoint.get('epoch', None)
    # Use 'best_val_auc' or 'best_val_loss' depending on what's saved
    best_val_metric = checkpoint.get('best_val_auc', checkpoint.get('best_val_loss', None))
    metric_name = 'AUC' if 'best_val_auc' in checkpoint else 'Loss' if 'best_val_loss' in checkpoint else 'Metric'
    metric_info = f", Best Val {metric_name}: {best_val_metric:.4f}" if best_val_metric is not None else ""
    epoch_info = f" (Epoch {epoch}{metric_info})" if epoch is not None else ""
    print(f"Checkpoint loaded from {filepath}{epoch_info}")
    return checkpoint, epoch 