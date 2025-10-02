# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a BiLSTM-based pest risk prediction system that uses meteorological, geographical, vegetation, and soil features to predict pest risk levels. The project contains two main implementations:

1. **baseline/** - Basic BiLSTM model for point-based predictions
2. **sd_raster_prediction/** - Advanced spatial raster prediction system

## Key Commands

### Training and Prediction
```bash
# Train baseline model
python baseline/train.py

# Evaluate baseline model  
python baseline/evaluate.py

# Run spatial data processing
python sd.py

# Run raster prediction training
python sd_raster_prediction/train_raster.py

# Generate future risk maps
python sd_raster_prediction/generate_future_risk_maps.py
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Key dependencies: torch>=1.8.0, pandas>=1.2.0, scikit-learn>=0.24.0, numpy>=1.20.0
```

## Architecture Overview

### Configuration System
- **config/params.py** - Main model configuration (hidden_size=256, num_layers=4, dropout=0.3)
- **baseline/config.py** - Baseline-specific configuration
- **sd_raster_prediction/config_future.py** - Future prediction configuration

### Model Architecture
The main BiLSTM model in `model/bilstm.py` includes:
- AttentionLayer for capturing important sequence information
- ResidualBlock for gradient flow improvement  
- BiLSTM with configurable attention and residual connections
- Supports both binary and multi-class classification

### Data Processing
- **SpatialDataProcessor** in `sd.py` - Handles meteorological features, coordinates, and risk classification
- **Raster processing** in `sd_raster_prediction/` - Handles geospatial TIFF data processing
- Features include: temperature, humidity, rainfall, DEM, NDVI, soil moisture

### Key Features
- **Input Features**: 12-dimensional (8 original + 4 composite features)
- **Risk Levels**: 4-class classification (Value_Class column)
- **Temporal Processing**: Sequence length of 8 time steps
- **Spatial Processing**: Uses historical and future raster data in GeoTIFF format

## Directory Structure

```
├── baseline/           # Basic BiLSTM implementation
├── model/             # Advanced BiLSTM model with attention
├── config/            # Configuration files
├── sd_raster_prediction/  # Spatial raster prediction system
├── data/              # Feature data (historical and future GeoTIFFs)
├── results/           # Model outputs and predictions
└── utils/             # Utility functions
```

## Development Notes

- The project follows the .cursorrules for AI-friendly development practices
- All functions should include type annotations and docstrings
- Model configurations are centralized in the config/ directory
- Use relative imports for consistency across modules
- Results are automatically timestamped and stored in results/ directory