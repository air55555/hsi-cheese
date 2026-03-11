import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import spectral
import cv2
from scipy.stats import skew, kurtosis
from scipy.stats import entropy
import json
import tkinter as tk
from tkinter import filedialog


# Load the hyperspectral cube from ENVI format
def load_hsi(file_path):
    """
    Load the hyperspectral image cube from an ENVI file.
    """
    try:
        hsi_cube = spectral.open_image(file_path).load()
        return hsi_cube
    except Exception as e:
        raise IOError(f"Error loading file: {file_path}. Details: {e}")


# Save selected file path to a configuration file
def save_config(file_path, config_path="config.json"):
    """
    Save the selected file path to a configuration file.
    """
    with open(config_path, 'w') as config_file:
        json.dump({"hsi_file": file_path}, config_file)


# Load the file path from a configuration file
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



# File selection dialog
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


# Calculate pixel-wise statistics
def calculate_statistics(hsi_cube):
    """
    Calculate statistical metrics for each pixel across the spectral bands.
    """
    print("Calculating....")
    mean_map = np.mean(hsi_cube, axis=2)
    median_map = np.median(hsi_cube, axis=2)
    variance_map = np.var(hsi_cube, axis=2)
    skewness_map = skew(hsi_cube, axis=2)
    kurtosis_map = kurtosis(hsi_cube, axis=2)
    min_map = np.min(hsi_cube, axis=2)
    max_map = np.max(hsi_cube, axis=2)
    range_map = max_map - min_map
    sum_map = np.sum(hsi_cube, axis=2)
    entropy_map = entropy(hsi_cube + 1e-8, axis=2)  # Avoid log(0)
    return (mean_map, median_map, variance_map, skewness_map, kurtosis_map,
            min_map, max_map, range_map, sum_map, entropy_map)


# Generate an RGB image from the hyperspectral cube
def generate_rgb(hsi_cube, r_band_idx=30, g_band_idx=20, b_band_idx=10):
    """
    Generate an RGB image from the hyperspectral cube.
    """
    r_band = hsi_cube[:, :, r_band_idx]
    g_band = hsi_cube[:, :, g_band_idx]
    b_band = hsi_cube[:, :, b_band_idx]

    r_band = cv2.normalize(r_band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    g_band = cv2.normalize(g_band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    b_band = cv2.normalize(b_band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    rgb_image = np.stack((r_band, g_band, b_band), axis=2)
    return rgb_image


# Display images
def display_images(rgb_image, **stat_maps):
    """
    Display the RGB image and statistical maps.
    """
    num_stats = len(stat_maps)
    plt.figure(figsize=(15, 10))
    cols = 4
    rows = -(-num_stats // cols)  # Round up

    # Display RGB image
    plt.subplot(rows, cols, 1)
    plt.imshow(rgb_image)
    plt.title("RGB Image")
    plt.axis('off')

    # Display statistical maps
    for idx, (stat_name, stat_map) in enumerate(stat_maps.items(), start=2):
        plt.subplot(rows, cols, idx)
        plt.imshow(stat_map, cmap='viridis')
        plt.title(f"{stat_name} Map")
        plt.colorbar()
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Main workflow
def main():
    """
    Main function to process the hyperspectral image cube.
    """
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

    # Load hyperspectral cube
    hsi_cube = load_hsi(file_path)
    # Select ROI if not already saved
    if not roi:
        roi = select_roi(rgb_image)
        if roi:
            save_config(file_path, roi)

    if roi:
        x_start, x_end, y_start, y_end = roi
        hsi_cube = hsi_cube[y_start:y_end, x_start:x_end, :]
        print(f"Using ROI: x=({x_start}, {x_end}), y=({y_start}, {y_end})")

    # Calculate statistics
    stats = calculate_statistics(hsi_cube)
    stat_names = [
        "Mean", "Median", "Variance", "Skewness", "Kurtosis",
        "Min", "Max", "Range", "Sum", "Entropy"
    ]
    stat_maps = dict(zip(stat_names, stats))

    # Generate RGB image
    rgb_image = generate_rgb(hsi_cube)

    # Display images
    display_images(rgb_image, **stat_maps)


# Run the app
if __name__ == "__main__":
    main()
