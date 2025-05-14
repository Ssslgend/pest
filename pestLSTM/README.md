# Pest Prediction with Unidirectional LSTM

This project trains and uses a Unidirectional LSTM model to predict pest risk based on raster data.

## Structure

- `model/lstm.py`: Defines the LSTM model.
- `config_lstm.py`: Configuration file.
- `train_lstm.py`: Script for training the model.
- `predict_lstm_raster.py`: Script for generating prediction rasters.
- `utils.py`: Utility functions.
- `requirements.txt`: Dependencies.
- `data/`: Placeholder for input data.
- `output/`: Stores trained models and logs.
- `prediction_output/`: Stores prediction results (GeoTIFFs, stats).

## Usage

1.  **Setup:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configure:** Edit `config_lstm.py` to set paths, hyperparameters, etc.
3.  **Train:**
    ```bash
    python train_lstm.py
    ```
4.  **Predict:**
    ```bash
    python predict_lstm_raster.py
    ``` 