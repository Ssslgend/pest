# visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import rasterio
from rasterio.plot import show
import pandas as pd
from datetime import datetime
import logging

# Setup logging
logger = logging.getLogger(__name__)

def get_risk_colors():
    """Get colors corresponding to risk levels"""
    return {
        0: '#2ca02c',  # Green - Extremely Low Risk
        1: '#1f77b4',  # Blue - Low Risk
        2: '#ff7f0e',  # Orange - Medium Risk
        3: '#d62728',  # Red - High Risk
        4: '#9467bd',  # Purple - Extremely High Risk
        255: '#f0f0f0'  # Gray - NoData
    }

def get_risk_labels(config=None):
    """Get labels corresponding to risk levels"""
    if config and 'future' in config and 'visualization' in config['future'] and 'risk_labels' in config['future']['visualization']:
        labels = config['future']['visualization']['risk_labels']
        if len(labels) >= 5:
            return {
                0: labels[0],
                1: labels[1],
                2: labels[2],
                3: labels[3],
                4: labels[4],
                255: 'NoData'
            }
    
    # Default labels
    return {
        0: 'Extremely Low Risk',
        1: 'Low Risk',
        2: 'Medium Risk',
        3: 'High Risk',
        4: 'Extremely High Risk',
        255: 'NoData'
    }

def visualize_prediction(prediction, output_path, title=None, add_timestamp=True, config=None):
    """
    Visualize prediction results and save as image
    
    Args:
        prediction: Prediction result array
        output_path: Output image path
        title: Image title, if None use default title
        add_timestamp: Whether to add timestamp
        config: Configuration information
    
    Returns:
        bool: Whether the image was successfully saved
    """
    try:
        # Get configuration parameters
        cmap_name = 'RdYlGn_r'  # Default color map
        dpi = 300  # Default DPI
        title_fontsize = 14  # Default title font size
        
        if config and 'future' in config and 'visualization' in config['future']:
            viz_config = config['future']['visualization']
            if 'cmap' in viz_config:
                cmap_name = viz_config['cmap']
            if 'dpi' in viz_config:
                dpi = viz_config['dpi']
            if 'title_fontsize' in viz_config:
                title_fontsize = viz_config['title_fontsize']
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # If prediction values are between 0-1, use continuous color mapping
        if np.nanmin(prediction) >= 0 and np.nanmax(prediction) <= 1:
            # Probability map
            im = plt.imshow(prediction, cmap=cmap_name, vmin=0, vmax=1)
            plt.colorbar(im, label='Risk Probability')
        else:
            # Risk classification map
            risk_colors = get_risk_colors()
            risk_labels = get_risk_labels(config)
            
            # Create custom color map
            cmap = mcolors.ListedColormap(
                [risk_colors[0], risk_colors[1], risk_colors[2], 
                 risk_colors[3], risk_colors[4]]
            )
            bounds = [0, 1, 2, 3, 4, 5]
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            
            # Draw risk classification map
            im = plt.imshow(prediction, cmap=cmap, norm=norm)
            
            # Add color bar and legend
            cbar = plt.colorbar(im, ticks=[0.5, 1.5, 2.5, 3.5, 4.5])
            cbar.set_ticklabels([risk_labels[0], risk_labels[1], risk_labels[2], 
                                risk_labels[3], risk_labels[4]])
        
        # Add title
        if title:
            plt.title(title, fontsize=title_fontsize)
        else:
            plt.title('Pest Risk Prediction', fontsize=title_fontsize)
        
        # Turn off axes
        plt.axis('off')
        
        # Add timestamp
        if add_timestamp:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            plt.figtext(0.99, 0.01, f'Generated: {timestamp}', ha='right', fontsize=8)
        
        # Save image
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization result saved to: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error visualizing prediction result: {e}")
        return False

def visualize_comparison(predictions, output_path, title=None, config=None):
    """
    Visualize comparison of predictions for multiple time periods
    
    Args:
        predictions: Dictionary, keys are period names, values are prediction result arrays
        output_path: Output image path
        title: Image title, if None use default title
        config: Configuration information
    
    Returns:
        bool: Whether the image was successfully saved
    """
    try:
        # Get configuration parameters
        cmap_name = 'RdYlGn_r'  # Default color map
        dpi = 300  # Default DPI
        
        if config and 'future' in config and 'visualization' in config['future']:
            viz_config = config['future']['visualization']
            if 'cmap' in viz_config:
                cmap_name = viz_config['cmap']
            if 'dpi' in viz_config:
                dpi = viz_config['dpi']
        
        # Determine subplot layout
        n_periods = len(predictions)
        if n_periods <= 3:
            n_cols = n_periods
            n_rows = 1
        elif n_periods <= 6:
            n_cols = 3
            n_rows = 2
        else:
            n_cols = 4
            n_rows = (n_periods + 3) // 4
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        
        # Draw prediction results for each period
        for i, (period_name, prediction) in enumerate(predictions.items()):
            if i < len(axes):
                ax = axes[i]
                im = ax.imshow(prediction, cmap=cmap_name)
                ax.set_title(period_name)
                ax.axis('off')
        
        # Hide extra subplots
        for i in range(len(predictions), len(axes)):
            axes[i].axis('off')
        
        # Add shared color bar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Risk Level')
        
        # Add overall title
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle('Multi-period Pest Risk Prediction Comparison', fontsize=16)
        
        # Save image
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison visualization result saved to: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error visualizing comparison result: {e}")
        return False

def create_risk_distribution_plot(risk_distributions, output_path, title=None, config=None):
    """
    Create risk distribution trend chart
    
    Args:
        risk_distributions: Dictionary, keys are period names, values are risk distribution dictionaries
        output_path: Output image path
        title: Image title, if None use default title
        config: Configuration information
    
    Returns:
        bool: Whether the image was successfully saved
    """
    try:
        # Get configuration parameters
        dpi = 300  # Default DPI
        
        if config and 'future' in config and 'visualization' in config['future']:
            viz_config = config['future']['visualization']
            if 'dpi' in viz_config:
                dpi = viz_config['dpi']
        
        # Prepare data
        periods = list(risk_distributions.keys())
        risk_levels = get_risk_labels(config)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Draw a line for each risk level
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
        markers = ['o', 's', '^', 'd', 'p']
        
        for i, risk_level in enumerate(range(5)):  # 0-4 risk levels
            risk_label = risk_levels[risk_level]
            values = []
            
            for period in periods:
                if period in risk_distributions:
                    period_data = risk_distributions[period]
                    risk_value = period_data.get(risk_level, 0)
                    values.append(risk_value)
                else:
                    values.append(0)
            
            plt.plot(
                periods, 
                values, 
                marker=markers[i], 
                label=risk_label, 
                linewidth=2, 
                color=colors[i]
            )
        
        # Add title and labels
        if title:
            plt.title(title, fontsize=16)
        else:
            plt.title('Multi-period Risk Level Distribution Trend', fontsize=16)
        
        plt.xlabel('Prediction Period')
        plt.ylabel('Proportion (%)')
        plt.legend(title='Risk Level')
        plt.grid(True, alpha=0.3)
        
        # Save image
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Risk distribution trend chart saved to: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error creating risk distribution trend chart: {e}")
        return False

def visualize_sensitivity(sensitivity_results, output_path, title=None, config=None):
    """
    Visualize sensitivity analysis results
    
    Args:
        sensitivity_results: Dictionary, keys are feature names, values are sensitivity values (e.g., change rate)
        output_path: Output image path
        title: Image title, if None use default title
        config: Configuration information
    
    Returns:
        bool: Whether the image was successfully saved
    """
    try:
        # Get configuration parameters
        dpi = 300  # Default DPI
        
        if config and 'future' in config and 'visualization' in config['future']:
            viz_config = config['future']['visualization']
            if 'dpi' in viz_config:
                dpi = viz_config['dpi']
        
        # Prepare data
        features = list(sensitivity_results.keys())
        values = list(sensitivity_results.values())
        
        # Sort features by sensitivity value
        sorted_indices = np.argsort(values)
        features = [features[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Draw horizontal bar chart
        bars = plt.barh(features, values, color='skyblue')
        
        # Use different colors for positive and negative values
        for i, v in enumerate(values):
            if v < 0:
                bars[i].set_color('salmon')
        
        # Add title and labels
        if title:
            plt.title(title, fontsize=16)
        else:
            plt.title('Feature Sensitivity Analysis', fontsize=16)
        
        plt.xlabel('Sensitivity Value (Risk Change Rate)')
        plt.ylabel('Feature')
        plt.grid(True, axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(values):
            plt.text(v + (0.01 if v >= 0 else -0.01), 
                    i, 
                    f'{v:.4f}', 
                    va='center', 
                    ha='left' if v >= 0 else 'right')
        
        # Save image
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Sensitivity analysis result saved to: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error visualizing sensitivity analysis result: {e}")
        return False 