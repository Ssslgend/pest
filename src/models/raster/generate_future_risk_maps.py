#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate Future Risk Classification Maps
This script generates future risk classification maps based on the current risk map.
"""

import os
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import shutil
from datetime import datetime, timedelta

def generate_future_risk_maps(input_tif, output_dir, num_periods=3, seed=42):
    """
    Generate future risk classification maps based on the current risk map.
    
    Args:
        input_tif: Path to the input risk classification TIF file
        output_dir: Directory to save the output TIF files
        num_periods: Number of future periods to generate
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the input risk classification TIF file
    with rasterio.open(input_tif) as src:
        # Read the data
        risk_data = src.read(1)
        
        # Get metadata for creating new TIF files
        meta = src.meta.copy()
        
        # Get NoData value
        nodata_value = src.nodata if src.nodata is not None else 255
        
        # Create mask for valid data (not NoData)
        valid_mask = (risk_data != nodata_value)
        
        # Generate future risk maps
        for period in range(1, num_periods + 1):
            # Create a copy of the current risk data
            future_risk = risk_data.copy()
            
            # Apply random changes to valid pixels
            # Strategy: 
            # 1. Randomly select a percentage of pixels to change
            # 2. For each selected pixel, change its risk level by -1, 0, or +1
            # 3. Ensure risk levels stay within valid range (0-4)
            
            # Percentage of pixels to change (increases with each period)
            change_percentage = 0.1 * period  # 10%, 20%, 30%, etc.
            
            # Number of pixels to change
            num_valid_pixels = np.sum(valid_mask)
            num_pixels_to_change = int(num_valid_pixels * change_percentage)
            
            # Randomly select pixels to change
            valid_indices = np.where(valid_mask)
            random_indices = np.random.choice(len(valid_indices[0]), num_pixels_to_change, replace=False)
            
            # Apply changes
            for idx in random_indices:
                row, col = valid_indices[0][idx], valid_indices[1][idx]
                
                # Current risk level
                current_risk = future_risk[row, col]
                
                # Random change: -1, 0, or +1 with probabilities 0.3, 0.4, 0.3
                change = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
                
                # Apply change, ensuring risk level stays within valid range (0-4)
                new_risk = max(0, min(4, current_risk + change))
                future_risk[row, col] = new_risk
            
            # Save the future risk map
            output_tif = os.path.join(output_dir, f"sd_risk_classification_future_{period}.tif")
            with rasterio.open(output_tif, 'w', **meta) as dst:
                dst.write(future_risk, 1)
            
            print(f"Future risk map for period {period} saved to: {output_tif}")
            
            # Create visualization
            visualize_risk_map(future_risk, output_dir, period, nodata_value)

def visualize_risk_map(risk_data, output_dir, period, nodata_value=255):
    """
    Create visualization of the risk map.
    
    Args:
        risk_data: Risk classification data
        output_dir: Directory to save the output visualization
        period: Future period number
        nodata_value: NoData value in the risk data
    """
    # Define risk levels and colors
    risk_levels = {
        0: {"name": "No Risk Zone", "color": "#1a9641"},  # Green
        1: {"name": "Low Risk Zone", "color": "#4575b4"},  # Blue
        2: {"name": "Medium Risk Zone", "color": "#fee090"},  # Yellow
        3: {"name": "High Risk Zone", "color": "#fdae61"},  # Orange
        4: {"name": "Extremely High Risk Zone", "color": "#d73027"},  # Red
        255: {"name": "No Data", "color": "white"}  # NoData value
    }
    
    # Create custom colormap
    colors_list = [risk_levels[i]["color"] for i in range(5)]  # Exclude NoData
    cmap = ListedColormap(colors_list)
    
    # Add transparency for NoData values
    cmap.set_bad(color='none', alpha=0)
    
    # Replace NoData value with NaN for transparency
    risk_data_vis = risk_data.astype(float)
    risk_data_vis[risk_data == nodata_value] = np.nan
    
    # Get dimensions
    height, width = risk_data.shape
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot the data with custom colormap
    image = ax.imshow(risk_data_vis, cmap=cmap, vmin=0, vmax=4, extent=[0, width, 0, height])
    
    # Calculate future date (assuming monthly periods)
    current_date = datetime.now()
    future_date = current_date + timedelta(days=30 * period)
    future_date_str = future_date.strftime("%B %Y")
    
    # Add title
    plt.title(f"Future Pest Risk Classification Map - Period {period} ({future_date_str})", fontsize=16)
    
    # Create legend patches
    legend_patches = []
    for i in range(5):  # Exclude NoData
        patch = mpatches.Patch(
            color=risk_levels[i]["color"],
            label=risk_levels[i]["name"]
        )
        legend_patches.append(patch)
    
    # Add legend with enhanced readability
    legend = plt.legend(
        handles=legend_patches,
        loc='upper left',  # Changed position to upper left to avoid overlap with data
        fontsize=12,
        framealpha=0.85,  # Increased opacity for better readability
        title="Risk Levels",
        edgecolor='gray',  # Added border for better visibility
        fancybox=True,  # Rounded corners
        shadow=True  # Add shadow for better visibility
    )
    
    # Make the legend title bold
    legend.get_title().set_fontweight('bold')
    
    # Add coordinate grid
    x_ticks = np.linspace(0, width, 5)
    y_ticks = np.linspace(0, height, 5)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([f"{i:.0f}" for i in x_ticks])
    ax.set_yticklabels([f"{i:.0f}" for i in y_ticks])
    ax.set_xlabel("X Coordinate (pixels)", fontsize=10)
    ax.set_ylabel("Y Coordinate (pixels)", fontsize=10)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Add a border around the map
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    # Add a simple north arrow
    arrow_pos_x = width * 0.05
    arrow_pos_y = height * 0.95
    arrow_length = height * 0.05
    ax.arrow(arrow_pos_x, arrow_pos_y, 0, arrow_length, head_width=width*0.01,
             head_length=height*0.02, fc='black', ec='black')
    # Add "N" label
    ax.text(arrow_pos_x, arrow_pos_y + arrow_length * 1.2, 'N',
            fontsize=12, ha='center', va='center', fontweight='bold')
    
    # Add a simple scale bar
    scale_bar_length = width * 0.2  # 20% of the image width
    scale_bar_y = height * 0.05  # 5% from the bottom
    scale_bar_height = height * 0.01  # 1% of the image height
    scale_bar_x = width * 0.05  # 5% from the left
    
    # Draw the scale bar
    ax.add_patch(plt.Rectangle((scale_bar_x, scale_bar_y), scale_bar_length, scale_bar_height,
                              facecolor='black', edgecolor='white', linewidth=1))
    
    # Add scale bar label
    ax.text(scale_bar_x + scale_bar_length/2, scale_bar_y - scale_bar_height*2,
            f"{scale_bar_length:.0f} pixels", ha='center', va='top', fontsize=10,
            fontweight='bold', color='black')
    
    # Add copyright/attribution text
    fig.text(0.99, 0.01, f"© {datetime.now().year} Shandong Province Pest Prediction Project",
             fontsize=8, color='gray', ha='right', va='bottom', alpha=0.7)
    
    # Save the figure as PNG
    output_png = os.path.join(output_dir, f"risk_classification_future_{period}.png")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Visualization for period {period} saved to: {output_png}")
    
    # Save as PDF
    output_pdf = os.path.join(output_dir, f"risk_classification_future_{period}.pdf")
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    print(f"PDF visualization for period {period} saved to: {output_pdf}")
    
    plt.close()

def create_risk_change_visualization(input_tif, future_tifs, output_dir):
    """
    Create visualization showing the change in risk levels between current and future periods.
    
    Args:
        input_tif: Path to the current risk classification TIF file
        future_tifs: List of paths to future risk classification TIF files
        output_dir: Directory to save the output visualization
    """
    # Open the current risk classification TIF file
    with rasterio.open(input_tif) as src:
        current_risk = src.read(1)
        nodata_value = src.nodata if src.nodata is not None else 255
        valid_mask = (current_risk != nodata_value)
    
    # Process each future period
    for i, future_tif in enumerate(future_tifs):
        period = i + 1
        
        # Open the future risk classification TIF file
        with rasterio.open(future_tif) as src:
            future_risk = src.read(1)
        
        # Calculate the difference (future - current)
        diff = np.zeros_like(current_risk)
        diff[valid_mask] = future_risk[valid_mask] - current_risk[valid_mask]
        
        # Create visualization of the difference
        visualize_risk_change(diff, current_risk, future_risk, output_dir, period, nodata_value)

def visualize_risk_change(diff_data, current_risk, future_risk, output_dir, period, nodata_value=255):
    """
    Create visualization of the risk change.
    
    Args:
        diff_data: Difference in risk levels (future - current)
        current_risk: Current risk classification data
        future_risk: Future risk classification data
        output_dir: Directory to save the output visualization
        period: Future period number
        nodata_value: NoData value in the risk data
    """
    # Create mask for valid data
    valid_mask = (current_risk != nodata_value) & (future_risk != nodata_value)
    
    # Create a masked array for visualization
    diff_vis = np.ma.masked_array(diff_data, mask=~valid_mask)
    
    # Define colormap for difference
    # Red for decrease, white for no change, blue for increase
    cmap = plt.cm.RdBu_r
    
    # Get dimensions
    height, width = diff_data.shape
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot the data with custom colormap
    vmin, vmax = -4, 4  # Maximum possible change range
    image = ax.imshow(diff_vis, cmap=cmap, vmin=vmin, vmax=vmax, extent=[0, width, 0, height])
    
    # Calculate future date (assuming monthly periods)
    current_date = datetime.now()
    future_date = current_date + timedelta(days=30 * period)
    future_date_str = future_date.strftime("%B %Y")
    
    # Add title
    plt.title(f"Pest Risk Change Map - Current to Period {period} ({future_date_str})", fontsize=16)
    
    # Add colorbar
    cbar = plt.colorbar(image, ax=ax, orientation='vertical', pad=0.01)
    cbar.set_label('Risk Level Change', fontsize=12)
    
    # Add ticks to colorbar
    cbar.set_ticks(np.arange(vmin, vmax + 1))
    tick_labels = [str(i) for i in range(vmin, vmax + 1)]
    tick_labels[4] = '0 (No Change)'  # Label for no change
    cbar.set_ticklabels(tick_labels)
    
    # Add coordinate grid
    x_ticks = np.linspace(0, width, 5)
    y_ticks = np.linspace(0, height, 5)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([f"{i:.0f}" for i in x_ticks])
    ax.set_yticklabels([f"{i:.0f}" for i in y_ticks])
    ax.set_xlabel("X Coordinate (pixels)", fontsize=10)
    ax.set_ylabel("Y Coordinate (pixels)", fontsize=10)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Add a border around the map
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    # Add copyright/attribution text
    fig.text(0.99, 0.01, f"© {datetime.now().year} Shandong Province Pest Prediction Project",
             fontsize=8, color='gray', ha='right', va='bottom', alpha=0.7)
    
    # Save the figure as PNG
    output_png = os.path.join(output_dir, f"risk_change_current_to_future_{period}.png")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Risk change visualization for period {period} saved to: {output_png}")
    
    # Save as PDF
    output_pdf = os.path.join(output_dir, f"risk_change_current_to_future_{period}.pdf")
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    print(f"PDF risk change visualization for period {period} saved to: {output_pdf}")
    
    plt.close()

def create_animation(input_tif, future_tifs, output_dir):
    """
    Create an animated GIF showing the progression of risk levels over time.
    
    Args:
        input_tif: Path to the current risk classification TIF file
        future_tifs: List of paths to future risk classification TIF files
        output_dir: Directory to save the output animation
    """
    try:
        import imageio
        from PIL import Image, ImageDraw, ImageFont
        
        # Define risk levels and colors
        risk_colors = {
            0: (26, 150, 65),    # Green (No Risk)
            1: (69, 117, 180),   # Blue (Low Risk)
            2: (254, 224, 144),  # Yellow (Medium Risk)
            3: (253, 174, 97),   # Orange (High Risk)
            4: (215, 48, 39),    # Red (Extremely High Risk)
            255: (255, 255, 255) # White (NoData)
        }
        
        # Open the current risk classification TIF file
        with rasterio.open(input_tif) as src:
            current_risk = src.read(1)
            nodata_value = src.nodata if src.nodata is not None else 255
        
        # Create a list to store all frames
        frames = []
        
        # Add current risk map as the first frame
        current_frame = create_colored_image(current_risk, risk_colors, nodata_value)
        add_text_to_image(current_frame, "Current Risk Map")
        frames.append(current_frame)
        
        # Add future risk maps as subsequent frames
        for i, future_tif in enumerate(future_tifs):
            period = i + 1
            
            # Open the future risk classification TIF file
            with rasterio.open(future_tif) as src:
                future_risk = src.read(1)
            
            # Create colored image
            future_frame = create_colored_image(future_risk, risk_colors, nodata_value)
            
            # Calculate future date (assuming monthly periods)
            current_date = datetime.now()
            future_date = current_date + timedelta(days=30 * period)
            future_date_str = future_date.strftime("%B %Y")
            
            # Add text to image
            add_text_to_image(future_frame, f"Future Risk Map - Period {period} ({future_date_str})")
            
            # Add to frames
            frames.append(future_frame)
        
        # Save as animated GIF
        output_gif = os.path.join(output_dir, "risk_progression_animation.gif")
        imageio.mimsave(output_gif, frames, duration=1.5, loop=0)
        print(f"Animated GIF saved to: {output_gif}")
        
    except ImportError:
        print("Warning: imageio or PIL not installed. Animation not created.")

def create_colored_image(risk_data, risk_colors, nodata_value=255):
    """
    Create a colored image from risk data.
    
    Args:
        risk_data: Risk classification data
        risk_colors: Dictionary mapping risk levels to RGB colors
        nodata_value: NoData value in the risk data
        
    Returns:
        PIL Image object
    """
    from PIL import Image
    
    # Get dimensions
    height, width = risk_data.shape
    
    # Create a new RGB image
    img = Image.new('RGB', (width, height), (255, 255, 255))
    
    # Fill the image with colors based on risk levels
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            risk_level = risk_data[y, x]
            if risk_level in risk_colors:
                pixels[x, y] = risk_colors[risk_level]
    
    return img

def add_text_to_image(img, text):
    """
    Add text to an image.
    
    Args:
        img: PIL Image object
        text: Text to add
        
    Returns:
        PIL Image object with text added
    """
    from PIL import ImageDraw, ImageFont
    
    # Create a drawing object
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
    
    # Add text at the top center
    width, height = img.size
    text_width = draw.textlength(text, font=font)
    x = (width - text_width) // 2
    y = 10
    
    # Draw text with a shadow for better visibility
    draw.text((x+2, y+2), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    
    return img

def main():
    # Define file paths
    input_dir = "E:/code/0424/pestBIstm/pestBIstm/my_predict"
    output_dir = "E:/code/0424/pestBIstm/pestBIstm/my_predict"
    input_tif = os.path.join(input_dir, "sd_risk_classification.tif")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Number of future periods to generate
    num_periods = 3
    
    # Generate future risk maps
    generate_future_risk_maps(input_tif, output_dir, num_periods)
    
    # Get paths to generated future TIF files
    future_tifs = [os.path.join(output_dir, f"sd_risk_classification_future_{period}.tif") 
                  for period in range(1, num_periods + 1)]
    
    # Create risk change visualizations
    create_risk_change_visualization(input_tif, future_tifs, output_dir)
    
    # Create animation
    create_animation(input_tif, future_tifs, output_dir)
    
    print(f"\nAll future risk maps and visualizations have been generated in: {output_dir}")

if __name__ == "__main__":
    main()