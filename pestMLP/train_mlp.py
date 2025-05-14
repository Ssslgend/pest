import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import os
import time
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score # Import roc_auc_score
from tqdm import tqdm
import logging
import random

# Project specific imports
from config_mlp import get_config
from model.mlp import MLPModel # Import MLP model
from utils import evaluate_model, plot_training_history, save_checkpoint, load_checkpoint

# --- Configuration ---
CONFIG = get_config()
DEVICE = CONFIG['training']['device']
SEED = CONFIG['training']['seed']

# --- Setup Logging ---
log_file = CONFIG['log_file']
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# --- Set Random Seeds ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
logger.info(f"Using device: {DEVICE}")
logger.info(f"Random seed set to: {SEED}")

def load_data_from_csv(csv_path):
    """Loads data from CSV, identifies features, target, coords, and handles NaNs."""
    logger.info(f"Loading data from: {csv_path}")
    if not os.path.exists(csv_path):
        logger.error(f"Data file not found: {csv_path}")
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    try:
        # Try common encodings
        encodings_to_try = ['utf-8', 'gbk', 'utf-8-sig']
        df = None
        last_error = None
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                logger.info(f"Successfully loaded CSV with encoding: {enc}")
                break # Stop if successful
            except UnicodeDecodeError:
                logger.warning(f"Failed to decode CSV with encoding: {enc}")
                last_error = f"UnicodeDecodeError with {enc}"
            except Exception as e:
                 logger.warning(f"Error reading CSV with encoding {enc}: {e}") # Catch other potential read errors
                 last_error = e

        if df is None:
             raise ValueError(f"Could not read CSV file {csv_path} with any attempted encodings ({encodings_to_try}). Last error: {last_error}")

        logger.info(f"Data loaded successfully with shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")

        # Define Columns (Adapt if needed)
        target_col = 'BYQ_baie'
        coord_cols = ['X', 'Y']
        # Exclude non-feature columns
        exclude_cols = [target_col] + coord_cols + ['sd_station']
        # Robustly find feature columns (numeric, not excluded)
        feature_cols = [col for col in df.select_dtypes(include=np.number).columns
                        if col not in exclude_cols]

        logger.info(f"Target column: {target_col}")
        logger.info(f"Coordinate columns: {coord_cols}")
        logger.info(f"Identified feature columns ({len(feature_cols)}): {feature_cols}")

        # Validations
        if target_col not in df.columns: raise ValueError(f"Target column '{target_col}' not found.")
        if not all(c in df.columns for c in coord_cols): raise ValueError("Coordinate columns not found.")
        if not feature_cols: raise ValueError("No numeric feature columns identified.")

        # Extract Data
        X_data = df[feature_cols].values
        y_data = df[target_col].values
        coords_data = df[coord_cols].values

        # Handle Missing Values
        nan_counts = np.isnan(X_data).sum()
        if nan_counts > 0:
            logger.warning(f"Found {nan_counts} NaN values in features. Imputing with mean.")
            imputer = SimpleImputer(strategy='mean')
            X_data = imputer.fit_transform(X_data)
        else:
            logger.info("No NaN values found in features.")

        # Convert types
        X_data = X_data.astype(np.float32)
        y_data = y_data.astype(np.float32)
        coords_data = coords_data.astype(np.float64)

        # Check Target Variable
        unique_targets, counts = np.unique(y_data, return_counts=True)
        logger.info(f"Unique values in target column '{target_col}': {dict(zip(unique_targets, counts))}")
        if not np.all(np.isin(unique_targets, [0, 1])):
            logger.warning(f"Target column contains values other than 0 and 1.")
        if len(unique_targets) < 2:
             logger.warning(f"Target column contains only one class value. Training/Evaluation might be problematic.")

        return X_data, y_data, coords_data, feature_cols

    except Exception as e:
        logger.error(f"Error loading or processing data from {csv_path}: {e}", exc_info=True)
        raise

def train_model():
    """Main function to train the MLP model."""
    logger.info("--- Starting MLP Model Training ---")
    start_time = time.time()

    # --- 1. Load and Preprocess Data ---
    csv_path = CONFIG['data_csv_path']
    logger.info(f"Attempting to load data from path in config: {csv_path}")

    try:
        X_data_raw, y_data, coords_data, feature_names_loaded = load_data_from_csv(csv_path)
        logger.info(f"Data shapes - X: {X_data_raw.shape}, y: {y_data.shape}, Coords: {coords_data.shape}")

        # Update config with actual features found
        CONFIG['feature_names'] = feature_names_loaded
        CONFIG['model']['input_size'] = X_data_raw.shape[1]
        num_features = X_data_raw.shape[1]

        # Scale Features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_data_raw)
        logger.info("Features scaled using StandardScaler.")

        # Save Scaler
        scaler_path = CONFIG['scaler_save_path']
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

        # --- NO Reshaping needed for MLP input ---
        # X input shape for MLP should be (num_samples, num_features)
        y_reshaped = y_data.reshape(-1, 1) # Ensure y is (num_samples, 1)

        logger.info(f"Data ready for MLP: X shape {X_scaled.shape}, y shape {y_reshaped.shape}")

        # Create PyTorch Dataset
        full_dataset = TensorDataset(
            torch.tensor(X_scaled, dtype=torch.float32), # Use scaled X directly
            torch.tensor(y_reshaped, dtype=torch.float32),
            torch.tensor(coords_data, dtype=torch.float64)
        )
        logger.info(f"Total dataset size: {len(full_dataset)}")

    except FileNotFoundError:
        logger.error(f"Exiting due to missing data file.")
        return
    except Exception as e:
        logger.error(f"Exiting due to data loading error: {e}", exc_info=True)
        return

    # --- 2. Split Data ---
    val_split = CONFIG['training']['validation_split']
    if not (0 < val_split < 1):
        logger.error("validation_split must be between 0 and 1")
        return
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    if train_size <= 0 or val_size <= 0:
        logger.error(f"Dataset size ({len(full_dataset)}) too small for validation split ({val_split}).")
        return

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['training']['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['training']['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")

    # --- 3. Initialize Model, Loss, Optimizer, Scheduler ---
    model = MLPModel(CONFIG['model']).to(DEVICE)
    logger.info(f"MLP Model initialized with input_size={num_features}:\n{model}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Loss function
    if CONFIG['training']['loss_function'] == 'BCELoss':
        criterion = nn.BCELoss()
    elif CONFIG['training']['loss_function'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    else:
        logger.error(f"Unsupported loss function: {CONFIG['training']['loss_function']}")
        return
    logger.info(f"Using loss function: {CONFIG['training']['loss_function']}")

    # Optimizer
    lr = CONFIG['training']['learning_rate']
    optimizer_name = CONFIG['training']['optimizer']
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        logger.error(f"Unsupported optimizer: {optimizer_name}")
        return
    logger.info(f"Using optimizer: {optimizer_name} with LR: {lr}")

    # Scheduler
    scheduler_step = CONFIG['training']['scheduler_step_size']
    scheduler_gamma = CONFIG['training']['scheduler_gamma']
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    logger.info(f"Using StepLR scheduler: step_size={scheduler_step}, gamma={scheduler_gamma}")

    # --- 4. Training Loop ---
    epochs = CONFIG['training']['epochs']
    patience = CONFIG['training']['early_stopping_patience']
    save_best = CONFIG['training']['save_best_model']
    model_save_path = CONFIG['model_save_path']

    best_val_metric = -np.inf
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}
    start_epoch = 0

    logger.info("--- Starting Training Loop ---")
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        train_labels_epoch = []
        train_outputs_epoch = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for batch_data in progress_bar:
            inputs, labels, _ = batch_data # Assuming (X, y, coords) format
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs) # MLP takes (batch, features)

            try:
                loss = criterion(outputs.squeeze(), labels.float().squeeze())
                loss.backward()
                optimizer.step()
            except Exception as e:
                 logger.error(f"Error during training step: {e}")
                 logger.error(f"Output shape: {outputs.shape}, Label shape: {labels.shape}")
                 continue

            running_loss += loss.item() * inputs.size(0)
            train_labels_epoch.extend(labels.cpu().numpy().flatten())
            train_outputs_epoch.extend(outputs.detach().cpu().numpy().flatten())
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_dataset) if len(train_dataset) > 0 else 0.0
        epoch_train_auc = 0.5
        if train_labels_epoch:
            unique_train_labels = np.unique(train_labels_epoch)
            if len(unique_train_labels) > 1:
                try:
                    epoch_train_auc = roc_auc_score(np.array(train_labels_epoch), np.array(train_outputs_epoch))
                except ValueError:
                    logger.warning(f"Epoch {epoch+1}: Could not calculate train AUC (ValueError).")
            else:
                logger.warning(f"Epoch {epoch+1}: Only one class ({unique_train_labels}) in training labels.")
        else:
             logger.warning(f"Epoch {epoch+1}: No labels collected during training.")

        val_metrics, _, _, _ = evaluate_model(model, val_loader, DEVICE, criterion)
        val_loss = val_metrics.get('loss')
        val_auc = val_metrics.get('auc', 0.5)

        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(epoch_train_auc)
        history['val_auc'].append(val_auc)

        epoch_duration = time.time() - epoch_start_time
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
        logger.info(f"Epoch {epoch+1}/{epochs} - Dur: {epoch_duration:.2f}s - "
                    f"Train Loss: {epoch_loss:.4f}, Train AUC: {epoch_train_auc:.4f} - "
                    f"Val Loss: {val_loss_str}, Val AUC: {val_auc:.4f}")

        scheduler.step()

        current_metric = val_auc
        if current_metric > best_val_metric:
            logger.info(f"Val AUC improved ({best_val_metric:.4f} --> {current_metric:.4f}).")
            best_val_metric = current_metric
            epochs_no_improve = 0
            if save_best:
                logger.info(f"Saving best model to {model_save_path}")
                checkpoint_state = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_auc': best_val_metric,
                    'config': CONFIG['model'],
                    'feature_names': CONFIG['feature_names']
                }
                save_checkpoint(checkpoint_state, model_save_path)
        else:
            epochs_no_improve += 1
            logger.info(f"Val AUC did not improve for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs.")
            break

    total_training_time = time.time() - start_time
    logger.info(f"--- Training Finished --- Total Time: {total_training_time:.2f}s")

    # --- 5. Plot Training History ---
    history_plot_path = os.path.join(CONFIG['output_dir'], 'mlp_training_history.png')
    os.makedirs(os.path.dirname(history_plot_path), exist_ok=True)
    if history['train_loss']:
         plot_training_history(history, history_plot_path)
    else:
         logger.warning("Training history is empty, skipping plot generation.")

    # --- 6. Final Evaluation on Validation Set ---
    if save_best and os.path.exists(model_save_path):
        logger.info("--- Evaluating Best Model on Validation Set ---")
        try:
            checkpoint_load = torch.load(model_save_path, map_location='cpu')
            saved_model_config = checkpoint_load.get('config', CONFIG['model'])
            # Ensure input size from config matches data used
            saved_model_config['input_size'] = num_features

            best_model = MLPModel(saved_model_config).to(DEVICE)
            checkpoint, _ = load_checkpoint(model_save_path, best_model, device=DEVICE)
            if checkpoint:
                epoch_saved = checkpoint.get('epoch', 'N/A')
                auc_saved = checkpoint.get('best_val_auc', 'N/A')
                auc_saved_str = f"{auc_saved:.4f}" if isinstance(auc_saved, (int, float)) else str(auc_saved)
                logger.info(f"Loaded best model from epoch {epoch_saved} with Val AUC {auc_saved_str}")

                final_val_metrics, _, _, _ = evaluate_model(best_model, val_loader, DEVICE, criterion)
                logger.info("Best Model Validation Set Metrics:")
                for key, value in final_val_metrics.items():
                    if key == 'confusion_matrix':
                        logger.info(f"  {key}: \n{value}")
                    elif value is not None:
                        if isinstance(value, (int, float, np.number)):
                           logger.info(f"  {key}: {value:.4f}")
                        else:
                           logger.info(f"  {key}: {value}")
            else:
                logger.error("Could not load the best model checkpoint for final validation.")
        except Exception as e:
            logger.error(f"Error during final model evaluation: {e}", exc_info=True)

    elif not save_best:
         logger.info("Skipping final evaluation as 'save_best_model' was False.")
    else:
         logger.warning("Best model file not found, skipping final evaluation.")

if __name__ == '__main__':
    train_model()
