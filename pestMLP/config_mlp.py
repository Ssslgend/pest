# pestMLP/config_mlp.py
import torch
import os

def get_config():
    """Returns the configuration dictionary for the MLP project."""

    # --- Project Root --- #
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    # --- Data Paths --- #
    # Assumes data CSV is in a sibling directory 'datas'
    DATA_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'datas'))
    CSV_FILENAME = 'sd_all_2.csv' # Or the CSV you intend to use
    DATA_CSV_PATH = os.path.join(DATA_DIR, CSV_FILENAME)

    # Feature names will be loaded dynamically from CSV in train_mlp.py
    # but you can predefine them if needed for consistency checks
    FEATURE_NAMES = [] # Let training script populate this

    # --- Output Paths --- #
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
    PREDICTION_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'prediction_output') # Keep for potential future use
    MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    SCALER_SAVE_DIR = os.path.join(OUTPUT_DIR, 'scalers')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(SCALER_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)

    # --- Model Configuration --- #
    MODEL_CONFIG = {
        "input_size": -1, # Will be set dynamically after loading data
        "hidden_layers": [128, 64, 32], # Example hidden layer sizes for MLP
        "dropout": 0.3,               # Dropout rate for MLP layers
        "num_classes": 1,             # Output size (1 for binary classification/probability)
        "model_type": "mlp"         # Identifier
    }

    # --- Training Configuration --- #
    TRAINING_CONFIG = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 100, # MLPs might train faster or require more epochs
        "batch_size": 512, # Often possible to use larger batch size for MLP
        "learning_rate": 0.001,
        "optimizer": "AdamW", # AdamW or Adam often work well
        "loss_function": "BCELoss", # Binary Cross Entropy for sigmoid output
        "scheduler_step_size": 20,
        "scheduler_gamma": 0.5,
        "early_stopping_patience": 15,
        "validation_split": 0.15,  # Adjust split if needed
        "seed": 42,
        "save_best_model": True,
        "model_save_path": os.path.join(MODEL_SAVE_DIR, f"mlp_model_best.pth"),
        "scaler_save_path": os.path.join(SCALER_SAVE_DIR, "mlp_scaler.joblib"),
        "log_file": os.path.join(LOG_DIR, "mlp_training_log.txt")
    }

    # --- Prediction Configuration (Placeholder) --- #
    # Keep structure but might not be used if only training
    PREDICTION_CONFIG = {
        "prediction_batch_size": 8192,
        # Add paths if prediction script is created later
        # "prediction_output_path": ...
    }

    # --- Combine all configurations --- #
    config = {
        "project_root": PROJECT_ROOT,
        "data_csv_path": DATA_CSV_PATH,
        "feature_names": FEATURE_NAMES, # Will be updated in training script
        "output_dir": OUTPUT_DIR,
        "prediction_output_dir": PREDICTION_OUTPUT_DIR,
        "model": MODEL_CONFIG,
        "training": TRAINING_CONFIG,
        "prediction": PREDICTION_CONFIG,
    }

    # -- Add derived paths directly to the main config for easier access -- #
    config['model_save_path'] = TRAINING_CONFIG['model_save_path']
    config['scaler_save_path'] = TRAINING_CONFIG['scaler_save_path']
    config['log_file'] = TRAINING_CONFIG['log_file']

    return config

# Example usage:
# if __name__ == '__main__':
#     CONFIG = get_config()
#     import json
#     print(json.dumps(CONFIG, indent=4)) 