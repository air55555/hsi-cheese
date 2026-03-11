import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import spectral
import cv2
from scipy.stats import skew, kurtosis
from scipy.stats import entropy
#pip install opencv-python

def downsample_hsi(hsi_cube, factor=5, method='mean'):
    """
    Downsamples the hyperspectral cube by a specified factor.

    Parameters:
        hsi_cube (numpy.ndarray): The original HSI cube (height, width, bands).
        factor (int): The downsampling factor for spatial dimensions.
        method (str): Downsampling method ('mean' or 'nearest').

    Returns:
        numpy.ndarray: The downsampled HSI cube.
    """
    h, w, bands = hsi_cube.shape
    new_h, new_w = h // factor, w // factor

    if method == 'mean':
        # Use average pooling
        downsampled_cube = hsi_cube[:new_h * factor, :new_w * factor, :].reshape(
            new_h, factor, new_w, factor, bands
        ).mean(axis=(1, 3))
    elif method == 'nearest':
        # Use nearest-neighbor downsampling
        downsampled_cube = hsi_cube[::factor, ::factor, :]
    else:
        raise ValueError("Unsupported downsampling method. Choose 'mean' or 'nearest'.")

    return downsampled_cube

# Load the hyperspectral cube from ENVI format
def load_hsi(file_path):
    """
    Load the hyperspectral image cube from an ENVI file.
    """
    hsi_cube = spectral.open_image(file_path).load()
    return hsi_cube

# Calculate pixel-wise statistics
def calculate_statistics(hsi_cube):
    """
    Calculate statistical metrics for each pixel across the spectral bands.
    """
    mean_map = np.mean(hsi_cube, axis=2)
    median_map = np.median(hsi_cube, axis=2)
    variance_map = np.var(hsi_cube, axis=2)
    skewness_map = skew(hsi_cube, axis=2)
    kurtosis_map = kurtosis(hsi_cube, axis=2)
    min_map = np.min(hsi_cube, axis=2)
    max_map = np.max(hsi_cube, axis=2)
    range_map = max_map - min_map
    sum_map = np.sum(hsi_cube, axis=2)
    small_cube = downsample_hsi(hsi_cube, factor=5, method='nearest')#'mean')
    entropy_map = entropy(small_cube + 1e-8, axis=2)  # Avoid log(0) with small epsilon
    return (mean_map, median_map, variance_map, skewness_map, kurtosis_map,
            min_map, max_map, range_map, sum_map, entropy_map)

# Generate an RGB image from the hyperspectral cube
def generate_rgb(hsi_cube):
    """
    Generate an RGB image from the hyperspectral cube.
    Assumes bands are [Red, Green, Blue] indices.
    """
    # Example: Map specific bands to RGB. Adjust indices for your dataset.
    r_band = hsi_cube[:, :, 130]  # Replace 30 with a red band index
    g_band = hsi_cube[:, :, 50]  # Replace 20 with a green band index
    b_band = hsi_cube[:, :, 10]  # Replace 10 with a blue band index

    # Normalize bands
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
    rows = -(-num_stats // cols)  # Round up for number of rows

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
def main(file_path):
    """
    Main function to process the hyperspectral image cube.
    """
    hsi_cube = load_hsi(file_path)
    stats = calculate_statistics(hsi_cube)
    stat_names = [
        "Mean", "Median", "Variance", "Skewness", "Kurtosis",
        "Min", "Max", "Range", "Sum", "Entropy"
    ]
    stat_maps = dict(zip(stat_names, stats))

    rgb_image = generate_rgb(hsi_cube)
    display_images(rgb_image, **stat_maps)

# Run the app
if __name__ == "__main__":
    file_path = "2024913/2024-09-13_08-04-13_white_circ_t.hdr"  # Replace with the path to your .hdr file
    main(file_path)
