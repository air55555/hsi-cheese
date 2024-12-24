import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from spectral import open_image
from scipy.stats import skew, kurtosis, iqr, mode, variation, entropy
import json
from matplotlib.widgets import RectangleSelector
import cv2
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
import os
#pip install opencv-python
from save_mat import *

# Load the hyperspectral cube from ENVI format

import numpy as np

from utils import *

# Example Usage
# file_path = 'example_image.hdr'  # Replace with your ENVI file path
# hsi_cube = load_hsi(file_path)




# Calculate extensive statistical metrics



def calculate_statistics(hsi_cube):
    """
    Calculate an extensive set of statistical metrics for each pixel across spectral bands with a progress bar.
    """
    from tqdm import tqdm  # Import tqdm inside the function to ensure it's available only when needed.

    # Initialize an empty dictionary to store results
    stats = {}

    # Create a list of calculations with their respective names and functions
    calculations = [
        ("Mean", lambda x: np.mean(x, axis=2)),
        ("Median", lambda x: np.median(x, axis=2)),
        ("Mode", lambda x: mode(x, axis=2, nan_policy='omit')[0].squeeze()),
        ("Variance", lambda x: np.var(x, axis=2)),
        ("Standard Deviation", lambda x: np.std(x, axis=2)),
        ("Range", lambda x: np.ptp(x, axis=2)),
        ("IQR", lambda x: iqr(x, axis=2)),
        ("CV", lambda x: variation(x, axis=2)),
        ("Skewness", lambda x: skew(x, axis=2, nan_policy='omit')),
        ("Kurtosis", lambda x: kurtosis(x, axis=2, nan_policy='omit')),
        ("Min", lambda x: np.min(x, axis=2)),
        ("Max", lambda x: np.max(x, axis=2)),
        ("Sum", lambda x: np.sum(x, axis=2)),
        ("Product", lambda x: np.prod(x + 1e-8, axis=2)),
        #("Cumulative Product", lambda x: np.cumprod(x + 1e-8, axis=2)[..., -1]),
        ("Entropy", lambda x: entropy(x + 1e-8, axis=2)),
        ("Spectral Energy", lambda x: np.sum(np.square(x), axis=2)),
        ("Spectral Flatness", lambda x: np.exp(np.mean(np.log(x + 1e-8), axis=2)) / np.mean(x, axis=2)),
        ("Spectral Brightness", lambda x: np.linalg.norm(x, axis=2)),
        ("25th Percentile", lambda x: np.percentile(x, 25, axis=2)),
        ("50th Percentile", lambda x: np.percentile(x, 50, axis=2)),
        ("75th Percentile", lambda x: np.percentile(x, 75, axis=2)),
        ("10th Quantile", lambda x: np.percentile(x, 10, axis=2)),
        ("90th Quantile", lambda x: np.percentile(x, 90, axis=2)),
        ("Z-Score Std", lambda x: np.std((x - np.mean(x, axis=2, keepdims=True)) / (np.std(x, axis=2, keepdims=True) + 1e-8), axis=2)),
        ("RMS", lambda x: np.sqrt(np.mean(np.square(x), axis=2))),
        ("Geometric Mean", lambda x: np.exp(np.mean(np.log(x + 1e-8), axis=2))),
        ("Harmonic Mean", lambda x: len(x[0, 0, :]) / np.sum(1.0 / (x + 1e-8), axis=2)),
        ("SNR", lambda x: np.mean(x, axis=2) / (np.std(x, axis=2) + 1e-8)),
        ("3rd Moment", lambda x: np.mean((x - np.mean(x, axis=2, keepdims=True))**3, axis=2)),
        ("4th Moment", lambda x: np.mean((x - np.mean(x, axis=2, keepdims=True))**4, axis=2)),
        ("Cumulative Sum", lambda x: np.cumsum(x, axis=2)[..., -1]),
        ("Cumulative Product", lambda x: np.cumprod(x + 1e-8, axis=2)[..., -1]),
        ("Cumulative Product", lambda x: np.cumprod(x + 1e-8, axis=2)[..., -1]),
    ]

    # Use tqdm to iterate over calculations
    for name, func in tqdm(calculations, desc="Calculating statistics", unit="stat", total=len(calculations)):
        try:
            stats[name] = func(hsi_cube)
        except Exception as e:
            print(f"Error calculating {name}: {e}")
            stats[name] = None  # Handle error gracefully and assign None if calculation fails

    return stats



# Generate an RGB image
def generate_rgb(hsi_cube):
    """
    Generate an RGB image from the hyperspectral cube.
    """
    r_band = hsi_cube[:, :, 17]
    g_band = hsi_cube[:, :, 5]
    b_band = hsi_cube[:, :, 10]
    r_band = cv2.normalize(r_band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    g_band = cv2.normalize(g_band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    b_band = cv2.normalize(b_band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.stack((r_band, g_band, b_band), axis=2)


# Save configuration to JSON
def save_config(file_path, roi=None, config_path="config.json"):
    config = {"hsi_file": file_path, "roi": roi}
    with open(config_path, 'w') as config_file:
        json.dump(config, config_file, indent=4)


# Select ROI
def select_roi(rgb_image):
    global roi_coords

    def on_select(eclick, erelease):
        global roi_coords
        x_start, y_start = int(eclick.xdata), int(eclick.ydata)
        x_end, y_end = int(erelease.xdata), int(erelease.ydata)
        roi_coords = (x_start, x_end, y_start, y_end)

    roi_coords = None
    fig, ax = plt.subplots()
    ax.imshow(rgb_image)
    ax.set_title("Select ROI and close window.")
    rect_selector = RectangleSelector(ax, on_select, interactive=True, useblit=True)
    plt.show()
    return roi_coords
def load_config(config_path="config.json"):
    """
    Load the file path and region of interest (ROI) from the configuration file.
    """
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            return config.get("hsi_file", None), config.get("roi", None)
    except FileNotFoundError:
        return None, None


# Display images
def display_images(rgb_image, **stat_maps):
    num_stats = len(stat_maps)
    plt.figure(figsize=(15, 10))
    cols = 4
    rows = -(-num_stats // cols)
    plt.subplot(rows, cols, 1)
    plt.imshow(rgb_image)
    plt.title("RGB Image")
    plt.axis('off')
    for idx, (stat_name, stat_map) in enumerate(stat_maps.items(), start=2):
        if idx==33:
            break
        plt.subplot(rows, cols, idx)
        plt.imshow(stat_map, cmap='viridis')
        plt.title(stat_name)
        plt.colorbar()
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def select_file():
    """
    Open a file dialog to select an HDR file.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select Hyperspectral HDR File",
        filetypes=[("ENVI HDR files", "*.hdr"), ("All Files", "*.*")]
    )
    return file_path
# Select region of interest (ROI)
def select_roi(rgb_image):
    """
    Use an interactive Matplotlib plot to select a rectangular ROI.
    Returns the selected ROI as (x_start, x_end, y_start, y_end).
    """
    global roi_coords

    def on_select(eclick, erelease):
        """
        Callback function to capture ROI selection.
        """
        global roi_coords
        x_start, y_start = int(eclick.xdata), int(eclick.ydata)
        x_end, y_end = int(erelease.xdata), int(erelease.ydata)
        roi_coords = (x_start, x_end, y_start, y_end)

    roi_coords = None
    fig, ax = plt.subplots()
    ax.imshow(rgb_image)
    ax.set_title("Draw a rectangle to select ROI, then close the window")

    # Rectangle selector without `drawtype`
    rect_selector = RectangleSelector(
        ax, on_select, interactive=True, useblit=True,
        button=[1], minspanx=5, minspany=5, spancoords='pixels'
    )

    plt.show()
    return roi_coords

# Main workflow
def main():


    tk.Tk().withdraw()
    # Load file path from configuration or prompt user to select a file
    file_path, roi = load_config()
    if not file_path:
        print("No file selected. Please choose a hyperspectral HDR file.")
        file_path = select_file()
        if file_path:
            save_config(file_path)
        else:
            print("No file selected. Exiting...")
            return
    hsi_cube = load_hsi(file_path)
    rgb_image = generate_rgb(hsi_cube)
    if not roi:
        roi = select_roi(rgb_image)
        if roi:
            save_config(file_path, roi)

    if roi:
        x_start, x_end, y_start, y_end = roi
        hsi_cube = hsi_cube[y_start:y_end, x_start:x_end, :]
        print(f"Using ROI: x=({x_start}, {x_end}), y=({y_start}, {y_end}) {hsi_cube.shape}")
        # Save the HSI cube under a new name with a generated header
        base, ext = os.path.splitext(file_path)
        output_file_path = f"{base}_{str(hsi_cube.shape).replace(' ', '')}_cropped{ext}"
        #normalize
        cube_min = hsi_cube.min()
        cube_max = hsi_cube.max()
        normalized_data = (hsi_cube - cube_min) / (cube_max - cube_min)
        hsi_cube = normalized_data

        save_hsi_as(hsi_cube, output_file_path)
        envi_to_matlab(output_file_path)
        print(f'Saved mat from {output_file_path},{hsi_cube.shape}')
    save_config(file_path, roi)
    stats = calculate_statistics(hsi_cube)

    #regenerate rgb with roi
    rgb_image = generate_rgb(hsi_cube)

    display_images(rgb_image, **stats)


# Run the app
if __name__ == "__main__":
    main()
