# pestLSTM/train_lstm.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd # Import pandas
import os
import time
import joblib # For saving scaler
from sklearn.preprocessing import StandardScaler # Example scaler
from sklearn.impute import SimpleImputer # To handle potential NaNs
from sklearn.metrics import roc_auc_score # <-- Import added here
from tqdm import tqdm
import logging
import random

# Project specific imports
from config_lstm import get_config
from model.lstm import LSTMModel
from utils import evaluate_model, plot_training_history, save_checkpoint, load_checkpoint

# --- Configuration ---
CONFIG = get_config()
DEVICE = CONFIG['training']['device']
SEED = CONFIG['training']['seed']

# --- Setup Logging ---
log_file = CONFIG['log_file']
# Ensure log directory exists
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
    # Ensure reproducibility (may impact performance)
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
        # Specify encoding, try 'utf-8' again for this file
        df = pd.read_csv(csv_path, encoding='utf-8') 
        logger.info(f"Data loaded successfully with shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")

        # --- Define Columns ---
        # Adapt these based on your actual CSV column names
        target_col = 'BYQ_baie'
        coord_cols = ['X', 'Y']
        # Assume all other numeric columns are features
        feature_cols = [col for col in df.columns if col not in [target_col] + coord_cols + ['sd_station']] # Exclude station ID too
        logger.info(f"Target column: {target_col}")
        logger.info(f"Coordinate columns: {coord_cols}")
        logger.info(f"Identified feature columns ({len(feature_cols)}): {feature_cols}")

        # Ensure target and coords exist
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in CSV.")
        if not all(col in df.columns for col in coord_cols):
             raise ValueError(f"One or more coordinate columns '{coord_cols}' not found in CSV.")
        if not feature_cols:
             raise ValueError("No feature columns identified.")

        # --- Extract Data ---
        X_data = df[feature_cols].values
        y_data = df[target_col].values
        coords_data = df[coord_cols].values

        # --- Handle Missing Values (Imputation) ---
        # Check for NaNs before imputation
        nan_counts = np.isnan(X_data).sum()
        if nan_counts > 0:
            logger.warning(f"Found {nan_counts} NaN values in features. Imputing with mean.")
            imputer = SimpleImputer(strategy='mean')
            X_data = imputer.fit_transform(X_data)
        else:
            logger.info("No NaN values found in features.")

        # Convert to appropriate types
        X_data = X_data.astype(np.float32)
        y_data = y_data.astype(np.float32) # Target for BCELoss should be float
        coords_data = coords_data.astype(np.float64) # Coords often double

        # --- Check Target Variable ---
        unique_targets = np.unique(y_data)
        logger.info(f"Unique values in target column '{target_col}': {unique_targets}")
        if not np.all(np.isin(unique_targets, [0, 1])):
            logger.warning(f"Target column '{target_col}' contains values other than 0 and 1. Ensure it's suitable for binary classification.")
        if len(unique_targets) < 2:
             logger.warning(f"Target column '{target_col}' contains only one class value. Training/Evaluation might be problematic.")

        return X_data, y_data, coords_data, feature_cols

    except Exception as e:
        logger.error(f"Error loading or processing data from {csv_path}: {e}", exc_info=True)
        raise


def train_model():
    """Main function to train the LSTM model."""
    logger.info("--- Starting LSTM Model Training ---")
    start_time = time.time()

    # --- 1. Load and Preprocess Data ---
    # Construct the path to the CSV file relative to the project root
    # Assumes 'datas' directory is at the same level as 'pestLSTM'
    # Or adjust DATA_DIR in config_lstm.py and use that
    csv_filename = 'sd_all_2.csv' # <-- Changed filename here
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datas', csv_filename)) # More robust path finding
    logger.info(f"Attempting to load data from resolved path: {csv_path}")

    try:
        X_data_raw, y_data, coords_data, feature_names_loaded = load_data_from_csv(csv_path)
        logger.info(f"Data shapes - X: {X_data_raw.shape}, y: {y_data.shape}, Coords: {coords_data.shape}")

        # Verify loaded features match config (optional but recommended)
        if set(feature_names_loaded) != set(CONFIG['feature_names']):
             logger.warning("Feature names loaded from CSV differ from those in config_lstm.py. Using features from CSV.")
             # Update config's feature count if needed, though it should be auto-detected by model
             CONFIG['model']['input_size'] = len(feature_names_loaded)
             CONFIG['feature_names'] = feature_names_loaded # Use names from data


        # --- Scale Features ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_data_raw)
        logger.info("Features scaled using StandardScaler.")

        # Save the fitted scaler
        scaler_path = CONFIG['scaler_save_path']
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

        # --- Reshape for LSTM ---
        # Assuming each sample (row in CSV) is an independent sequence of length 1
        seq_len = 1
        num_samples = X_scaled.shape[0]
        num_features = X_scaled.shape[1]
        X_reshaped = X_scaled.reshape(num_samples, seq_len, num_features)
        y_reshaped = y_data.reshape(-1, 1) # Ensure y is (num_samples, 1)

        logger.info(f"Data reshaped for LSTM: X shape {X_reshaped.shape}, y shape {y_reshaped.shape}")

        # --- Create PyTorch Dataset ---
        full_dataset = TensorDataset(
            torch.tensor(X_reshaped, dtype=torch.float32),
            torch.tensor(y_reshaped, dtype=torch.float32),
            torch.tensor(coords_data, dtype=torch.float64) # Coords often double precision
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

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['training']['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['training']['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
    # Optional: test_loader = DataLoader(test_dataset, ...)

    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")


    # --- 3. Initialize Model, Loss, Optimizer, Scheduler ---
    # Update input size in config just in case it changed due to CSV features
    CONFIG['model']['input_size'] = num_features
    model = LSTMModel(CONFIG['model']).to(DEVICE)
    logger.info(f"LSTM Model initialized with input_size={num_features}:\n{model}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Loss function
    if CONFIG['training']['loss_function'] == 'BCELoss':
        criterion = nn.BCELoss() # Assumes model has Sigmoid output
    elif CONFIG['training']['loss_function'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss() # Assumes model outputs raw logits
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
        optimizer = optim.AdamW(model.parameters(), lr=lr) # Consider adding weight_decay
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9) # Consider adding momentum
    else:
        logger.error(f"Unsupported optimizer: {optimizer_name}")
        return
    logger.info(f"Using optimizer: {optimizer_name} with LR: {lr}")

    # Scheduler
    scheduler_step = CONFIG['training']['scheduler_step_size']
    scheduler_gamma = CONFIG['training']['scheduler_gamma']
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step,
        gamma=scheduler_gamma
    )
    logger.info(f"Using StepLR scheduler: step_size={scheduler_step}, gamma={scheduler_gamma}")

    # --- 4. Training Loop ---
    epochs = CONFIG['training']['epochs']
    patience = CONFIG['training']['early_stopping_patience']
    save_best = CONFIG['training']['save_best_model']
    model_save_path = CONFIG['model_save_path']

    best_val_metric = -np.inf # Initialize for maximizing AUC
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}
    start_epoch = 0

    # Optional: Load checkpoint to resume
    # checkpoint, start_epoch_load = load_checkpoint(model_save_path, model, optimizer, DEVICE)
    # if checkpoint and start_epoch_load is not None:
    #     start_epoch = start_epoch_load
    #     best_val_metric = checkpoint.get('best_val_auc', best_val_metric) # Or best_val_loss
    #     logger.info(f"Resuming training from epoch {start_epoch + 1}")

    logger.info("--- Starting Training Loop ---")
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        train_labels_epoch = []
        train_outputs_epoch = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for batch_data in progress_bar:
            # Adapt unpacking based on your DataLoader output
            inputs, labels, _ = batch_data # Assuming (X, y, coords) format

            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE) # Move data to device

            optimizer.zero_grad() # Zero gradients
            outputs = model(inputs) # Forward pass

            # Calculate loss - ensure shapes match criterion expectations
            try:
                loss = criterion(outputs.squeeze(), labels.float().squeeze())
                loss.backward() # Backward pass
                optimizer.step() # Update weights
            except Exception as e:
                 logger.error(f"Error during training step: {e}")
                 logger.error(f"Output shape: {outputs.shape}, Label shape: {labels.shape}")
                 # Optionally skip this batch or raise the error
                 continue

            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            train_labels_epoch.extend(labels.cpu().numpy().flatten())
            train_outputs_epoch.extend(outputs.detach().cpu().numpy().flatten())
            progress_bar.set_postfix(loss=loss.item())

        # Calculate epoch training metrics
        epoch_loss = running_loss / len(train_dataset)
        epoch_train_auc = 0.5
        # Ensure there are labels before calculating AUC
        if train_labels_epoch:
            unique_train_labels = np.unique(train_labels_epoch)
            if len(unique_train_labels) > 1:
                try:
                    epoch_train_auc = roc_auc_score(np.array(train_labels_epoch), np.array(train_outputs_epoch))
                except ValueError:
                    logger.warning(f"Epoch {epoch+1}: Could not calculate train AUC (ValueError).")
            else:
                logger.warning(f"Epoch {epoch+1}: Only one class ({unique_train_labels}) present in training batch labels, train AUC set to 0.5.")
        else:
             logger.warning(f"Epoch {epoch+1}: No labels collected during training epoch.")


        # Validation step
        val_metrics, _, _, _ = evaluate_model(model, val_loader, DEVICE, criterion)
        val_loss = val_metrics.get('loss') # Use .get for safety
        val_auc = val_metrics.get('auc', 0.5) # Default to 0.5 if AUC key missing

        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(epoch_train_auc)
        history['val_auc'].append(val_auc)

        epoch_duration = time.time() - epoch_start_time
        # Handle potential None val_loss
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
        logger.info(f"Epoch {epoch+1}/{epochs} - Duration: {epoch_duration:.2f}s - "
                    f"Train Loss: {epoch_loss:.4f}, Train AUC: {epoch_train_auc:.4f} - "
                    f"Val Loss: {val_loss_str}, Val AUC: {val_auc:.4f}")

        scheduler.step() # Step the scheduler

        # Early Stopping & Checkpointing (based on validation AUC)
        current_metric = val_auc
        if current_metric > best_val_metric:
            logger.info(f"Validation AUC improved ({best_val_metric:.4f} --> {current_metric:.4f}).")
            best_val_metric = current_metric
            epochs_no_improve = 0
            if save_best:
                logger.info(f"Saving best model to {model_save_path}")
                checkpoint_state = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_auc': best_val_metric, # Save the metric value
                    'config': CONFIG['model'], # Save model config
                    'feature_names': CONFIG['feature_names'] # Save feature names used for training
                }
                save_checkpoint(checkpoint_state, model_save_path)
        else:
            epochs_no_improve += 1
            logger.info(f"Validation AUC did not improve for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs.")
            break

    total_training_time = time.time() - start_time
    logger.info(f"--- Training Finished --- Total Time: {total_training_time:.2f}s")

    # --- 5. Plot Training History ---
    history_plot_path = os.path.join(CONFIG['output_dir'], 'lstm_training_history.png')
    os.makedirs(os.path.dirname(history_plot_path), exist_ok=True)
    # Ensure history is not empty before plotting
    if history['train_loss']:
         plot_training_history(history, history_plot_path)
    else:
         logger.warning("Training history is empty, skipping plot generation.")

    # --- 6. Final Evaluation (Optional - using best saved model on validation set) ---
    if save_best and os.path.exists(model_save_path):
        logger.info("--- Evaluating Best Model on Validation Set ---")
        try:
            # Need to load the config used by the saved model
            checkpoint_load = torch.load(model_save_path, map_location='cpu')
            saved_model_config = checkpoint_load.get('config', CONFIG['model']) # Fallback to current config
            # Ensure input size matches
            saved_model_config['input_size'] = X_reshaped.shape[2] # Use actual feature count from data

            best_model = LSTMModel(saved_model_config).to(DEVICE)
            checkpoint, _ = load_checkpoint(model_save_path, best_model, device=DEVICE)
            if checkpoint:
                logger.info(f"Loaded best model from epoch {checkpoint.get('epoch', 'N/A')} with Val AUC {checkpoint.get('best_val_auc', 'N/A'):.4f}")
                final_val_metrics, _, _, _ = evaluate_model(best_model, val_loader, DEVICE, criterion)
                logger.info("Best Model Validation Set Metrics:")
                for key, value in final_val_metrics.items():
                    if key == 'confusion_matrix':
                        logger.info(f"  {key}: \n{value}")
                    elif value is not None:
                        # Check if value is numeric before formatting
                        if isinstance(value, (int, float, np.number)):
                           logger.info(f"  {key}: {value:.4f}")
                        else:
                           logger.info(f"  {key}: {value}") # Print non-numeric as is
            else:
                logger.error("Could not load the best model checkpoint for final validation.")
        except Exception as e:
            logger.error(f"Error during final model evaluation: {e}", exc_info=True)

    elif not save_best:
         logger.info("Skipping final evaluation as 'save_best_model' was False.")
    else:
         logger.warning("Best model file not found, skipping final evaluation.")

    # Add evaluation on a separate test_loader here if you have one


if __name__ == '__main__':
    train_model()