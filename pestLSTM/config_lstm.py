# pestLSTM/config_lstm.py
import torch
import os

def get_config():
    """Returns the configuration dictionary for the LSTM project."""

    # --- Project Root --- #
    # Assumes this config file is in the project root
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    # --- Data Paths --- #
    # !! MUST BE ADAPTED to your specific data setup !!
    # Example: Using pre-aligned data from the BiLSTM project
    # BASE_DATA_DIR = os.path.join(PROJECT_ROOT, '..', 'pestBIstm', 'data', 'aligned_features') # Example
    BASE_DATA_DIR = os.path.join(PROJECT_ROOT, 'data') # Default placeholder
    FEATURE_NAMES = [ # Must match the features the model was trained on / will be trained on
        'bio01', 'bio02', 'bio03', 'bio04', 'bio05', 'bio06', 'bio07', 'bio08',
        'bio09', 'bio10', 'bio11', 'bio12', 'bio13', 'bio14', 'bio15', 'bio16',
        'bio17', 'bio18', 'bio19',
        'dem', 'slope', 'aspect',
        'landcover',
        'soil_type'
    ]
    # Map feature names to their corresponding ALIGNED raster files
    feature_raster_map = {
        name: os.path.join(BASE_DATA_DIR, f"{name}_aligned.tif")
        for name in FEATURE_NAMES
    }
    # Target variable raster (for training/evaluation if applicable)
    # TARGET_RASTER_PATH = os.path.join(PROJECT_ROOT, '..', 'pestBIstm', 'data', 'target', 'target_aligned.tif') # Example
    TARGET_RASTER_PATH = os.path.join(BASE_DATA_DIR, 'target_aligned.tif') # Default placeholder

    # --- Output Paths --- #
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
    PREDICTION_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'prediction_output')
    MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    SCALER_SAVE_DIR = os.path.join(OUTPUT_DIR, 'scalers')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(SCALER_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)

    # --- Model Configuration --- #
    MODEL_CONFIG = {
        "input_size": len(FEATURE_NAMES), # Automatically set based on features
        "hidden_size": 64,            # LSTM hidden units (adjust as needed)
        "num_layers": 1,              # Number of LSTM layers (adjust as needed)
        "dropout": 0.1,               # Dropout rate
        "num_classes": 1,             # Output size (1 for binary classification/probability)
        "model_type": "lstm"        # Identifier
    }

    # --- Training Configuration --- #
    TRAINING_CONFIG = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 50,
        "batch_size": 1024,
        "learning_rate": 0.001,
        "optimizer": "Adam", # Options: Adam, AdamW, SGD
        "loss_function": "BCELoss", # Binary Cross Entropy for sigmoid output
        "scheduler_step_size": 10, # For StepLR scheduler
        "scheduler_gamma": 0.5,    # For StepLR scheduler
        "early_stopping_patience": 10,
        "validation_split": 0.1,   # Percentage of data for validation
        "seed": 42,                # Random seed for reproducibility
        "save_best_model": True,   # Save only the best model based on validation loss
        "model_save_path": os.path.join(MODEL_SAVE_DIR, f"lstm_model_best.pth"),
        "scaler_save_path": os.path.join(SCALER_SAVE_DIR, "lstm_scaler.joblib"),
        "log_file": os.path.join(LOG_DIR, "lstm_training_log.txt")
    }

    # --- Prediction Configuration --- #
    PREDICTION_CONFIG = {
        "prediction_batch_size": 4096, # Larger batch size for prediction is often possible
        "processing_block_size": 512, # Size of blocks for raster processing
        "prediction_tif_path": os.path.join(PREDICTION_OUTPUT_DIR, "lstm_prediction_probability.tif"),
        "prediction_risk_class_tif_path": os.path.join(PREDICTION_OUTPUT_DIR, "lstm_risk_classification.tif"),
        "risk_stats_csv_path": os.path.join(PREDICTION_OUTPUT_DIR, "lstm_risk_distribution_statistics.csv"),
        # --- Risk Classification Thresholds --- #
        "risk_thresholds": {
            'no_risk': 0.1,       # Probability < 0.1
            'low_risk': 0.3,      # 0.1 <= Probability < 0.3
            'medium_risk': 0.5,   # 0.3 <= Probability < 0.5
            'high_risk': 0.7,     # 0.5 <= Probability < 0.7
            # 'extreme_risk': implicitly >= 0.7
        },
        "risk_class_values": {
            'no_risk': 0,
            'low_risk': 1,
            'medium_risk': 2,
            'high_risk': 3,
            'extreme_risk': 4,
        },
         # Optional Probability Calibration (adjust or remove if not needed)
        "probability_calibration": {
            "scale_factor": 1.0,
            "shift_factor": 0.0
        }
    }

    # --- Raster Output Configuration --- #
    RASTER_OUTPUT_CONFIG = {
        "tif_driver": "GTiff",
        "tif_nodata_value": -9999.0,  # NoData value for float probability output
        "risk_class_nodata_value": 255 # NoData value for uint8 risk class output
    }

    # --- Combine all configurations --- #
    config = {
        "project_root": PROJECT_ROOT,
        "feature_names": FEATURE_NAMES,
        "feature_raster_map": feature_raster_map,
        "target_raster_path": TARGET_RASTER_PATH,
        "output_dir": OUTPUT_DIR,
        "prediction_output_dir": PREDICTION_OUTPUT_DIR,
        "model": MODEL_CONFIG,
        "training": TRAINING_CONFIG,
        "prediction": PREDICTION_CONFIG,
        "raster_output": RASTER_OUTPUT_CONFIG
    }

    # -- Add derived paths directly to the main config for easier access -- #
    config['model_save_path'] = TRAINING_CONFIG['model_save_path']
    config['scaler_save_path'] = TRAINING_CONFIG['scaler_save_path']
    config['log_file'] = TRAINING_CONFIG['log_file']
    config['prediction_tif_path'] = PREDICTION_CONFIG['prediction_tif_path']
    config['prediction_risk_class_tif_path'] = PREDICTION_CONFIG['prediction_risk_class_tif_path']
    config['risk_stats_csv_path'] = PREDICTION_CONFIG['risk_stats_csv_path']

    return config

# Example usage:
# if __name__ == '__main__':
#     CONFIG = get_config()
#     import json
#     print(json.dumps(CONFIG, indent=4)) 