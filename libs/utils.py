from spectral import open_image
from spectral.io.envi import save_image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import zoom
from sklearn.decomposition import PCA

def super_resolve_hsi(hsi_cube, scale_factor):
    """
    Performs super-resolution on a hyperspectral image cube using bicubic interpolation.

    Args:
        hsi_cube (ndarray): Input HSI cube (lines, samples, bands).
        scale_factor (float or tuple): Scale factor (single or (scale_y, scale_x)).

    Returns:
        ndarray: Super-resolved HSI cube.

    Raises:
        TypeError, ValueError: If inputs are invalid.
    """
    if not isinstance(hsi_cube, np.ndarray):
        raise TypeError("hsi_cube must be a NumPy array.")

    if hsi_cube.ndim != 3:
        raise ValueError("hsi_cube must have 3 dimensions (lines, samples, bands).")

    if isinstance(scale_factor, (int, float)):
        scale_factor = (scale_factor, scale_factor)
    elif not isinstance(scale_factor, tuple) or len(scale_factor) != 2 or not all(
            isinstance(x, (int, float)) for x in scale_factor):
        raise TypeError("scale_factor must be a float, int or a tuple of two floats/ints.")

    if any(s <= 0 for s in scale_factor):
        raise ValueError("Scale factors must be positive.")
    spatial_zoom = (*scale_factor, 1)
    hsi_super_res = zoom(hsi_cube, spatial_zoom, order=3)
    return hsi_super_res


def restore_hsi(hsi_cube_hr, scale_factor):
    """
    Restores a super-resolved HSI cube to its original spatial dimensions.

    Args:
        hsi_cube_hr (ndarray): The high-resolution (super-resolved) HSI cube.
        scale_factor (float or tuple): The scale factor used for super-resolution.

    Returns:
        ndarray: The restored HSI cube with the original spatial dimensions.

    Raises:
        TypeError: if inputs are of incorrect type.
        ValueError: If `scale_factor` has incorrect length or values or if the high resolution cube has incorrect dimensions.
    """
    if not isinstance(hsi_cube_hr, np.ndarray):
        raise TypeError("hsi_cube_hr must be a NumPy array.")
    if hsi_cube_hr.ndim != 3:
        raise ValueError("hsi_cube_hr must have 3 dimensions (lines, samples, bands).")
    if isinstance(scale_factor, (int, float)):
        scale_factor = (scale_factor, scale_factor)
    elif not isinstance(scale_factor, tuple) or len(scale_factor) != 2 or not all(
            isinstance(x, (int, float)) for x in scale_factor):
        raise TypeError("scale_factor must be a float, int or a tuple of two floats/ints.")
    if any(s <= 0 for s in scale_factor):
        raise ValueError("Scale factors must be positive.")

    # Calculate the inverse scale factor for downsampling
    inv_scale_factor = (1 / scale_factor[0], 1 / scale_factor[1])
    spatial_zoom = (*inv_scale_factor, 1)
    hsi_restored = zoom(hsi_cube_hr, spatial_zoom, order=3)
    return hsi_restored


def load_hsi(file_path):
    """
    Load the hyperspectral image cube from an ENVI file.
    """
    hsi_cube = open_image(file_path).load()
    return hsi_cube
def generate_rgb(hsi_cube,rb,gb,bb):
    """
    Generate an RGB image from the hyperspectral cube.
    r g b band nums
    """
    r_band = hsi_cube[:, :, rb]
    g_band = hsi_cube[:, :, gb]
    b_band = hsi_cube[:, :, bb]
    r_band = cv2.normalize(r_band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    g_band = cv2.normalize(g_band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    b_band = cv2.normalize(b_band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.stack((r_band, g_band, b_band), axis=2)



def create_header_from_cube(hsi_cube):
    """
    Create a header for the ENVI file based on the content of the HSI cube.

    Parameters:
        hsi_cube: ndarray
            The hyperspectral image data.

    Returns:
        header: dict
            A dictionary containing the ENVI header information.
    """
    header = {
        'samples': hsi_cube.shape[1],  # Number of columns (pixels per line)
        'lines': hsi_cube.shape[0],  # Number of rows (lines)
        'bands': hsi_cube.shape[2],  # Number of spectral bands
        'interleave': 'bil',  # Default interleave format (Band Interleaved by Line)
        'data type': 4,  # Default data type (32-bit float for numpy.float32)
        'byte order': 0  # Default byte order (little-endian)
    }

    # Set wavelength information if available
    # header['wavelength'] = ['wavelength_1', 'wavelength_2', ...]  # Example

    return header


def save_hsi_as(hsi_cube, output_file_path):
    """
    Save the hyperspectral image cube to an ENVI file with a new name.

    Parameters:
        hsi_cube: ndarray
            The hyperspectral image data to save.
        output_file_path: str
            The path to save the new ENVI file.
    """
    # Create a header from the cube
    header = create_header_from_cube(hsi_cube)

    # Save the HSI cube with the new header
    save_image(output_file_path, hsi_cube, metadata=header, force=True)

def read_hsi_files(file_paths):
    """
    Reads HSI files and returns an array of data cubes.

    Parameters:
        file_paths (list): List of file paths to HSI data.
        read_hsi (function): Function to read an HSI file. Should return a 3D array.

    Returns:
        list: A list of 3D NumPy arrays (data cubes).
    """
    data_cubes = []

    for file_path in file_paths:
        # Read the HSI data
        print(file_path)
        hsi_data = load_hsi(file_path)

        # Verify that the data is 3D
        if hsi_data.ndim != 3:
            raise ValueError(f"File {file_path} does not contain 3D HSI data.")

        # Append to the list of cubes
        data_cubes.append(hsi_data)

    return data_cubes

def synergize_hsi(datasets, align=True, method="average"):
    """
    Synergize multiple HSI datasets.

    Parameters:
        datasets (list of numpy.ndarray): List of HSI datasets to synergize.
                                          Each dataset should be a 3D array (rows, cols, bands).
        align (bool): Whether to align datasets based on spectral range and resolution.
        method (str): Method for synergizing datasets. Options are:
                      "average" - Compute the mean spectrum across datasets.
                      "stack" - Stack datasets along the band axis.

    Returns:
        numpy.ndarray: Synergized HSI dataset.
    """
    if not datasets:
        raise ValueError("The list of datasets is empty.")

    # Ensure all datasets have the same shape if alignment is not requested
    if not align and any(dataset.shape != datasets[0].shape for dataset in datasets):
       raise ValueError("All datasets must have the same shape if alignment is disabled.")

    # Align datasets based on spectral range and resolution (if required)
    if align:
        min_bands = min(dataset.shape[2] for dataset in datasets)
        aligned_datasets = [dataset[:, :, :min_bands] for dataset in datasets]
    else:
        aligned_datasets = datasets

    # Synergize datasets based on the selected method
    if method == "average":
        synergized = np.mean(aligned_datasets, axis=0)
    elif method == "stack":
        synergized = np.concatenate(aligned_datasets, axis=2)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return synergized


def read_log_file(file_path):
    """
    Reads the content of a log file and returns it as a string.

    Args:
        file_path (str): The path to the log file.

    Returns:
        str: The content of the log file as a string.
    """
    try:
        with open(file_path, 'r') as file:
            log_content = file.read()
        return log_content
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


import csv


import ast

def parse_log(log_text):
    log_entries = []
    entries = log_text.split('***************')

    for entry in entries:
        if "task:" in entry:
            try:
                # Extract task type
                task = entry.split('task:')[1].split('\n')[0].strip()

                # Extract parameters
                params_start = entry.find('params:')
                params_end = entry.find('solver:')
                params = entry[params_start:params_end].strip().splitlines()

                param_dict = {}
                for param in params:
                    if param.strip():
                        # Split only on the first colon
                        key, value = param.split(':', 1)
                        param_dict[key.strip()] = value.strip()

                # Extract the metrics before and after
                before_metrics = {}
                after_metrics = {}
                if 'Before |' in entry and 'After |' in entry:
                    before_str = entry.split('Before |')[1].split('After |')[0].strip()
                    after_str = entry.split('After |')[1].strip()

                    # Convert string representations of dictionaries into actual dictionaries
                    before_metrics = ast.literal_eval(before_str)
                    after_metrics = ast.literal_eval(after_str)

                # Store the parsed entry
                log_entries.append({
                    'task': task,
                    **param_dict,
                    **before_metrics,
                    **after_metrics
                })
            except Exception as e:
                print(f"Error parsing entry: {e}")
                continue  # Skip to the next entry in case of an error

    return log_entries





def reduce_hsi_bands_ndarray(data, n_bands=50):
    """
    Reduces the number of bands in a hyperspectral image stored in a NumPy ndarray.

    Parameters:
    data (np.ndarray): HSI data with shape (rows, cols, bands).
    n_bands (int): Number of bands to reduce to.

    Returns:
    np.ndarray: HSI data reduced to the specified number of bands.
    """
    try:
        # Validate input dimensions
        if data.ndim != 3:
            raise ValueError("Input data must have 3 dimensions (rows, cols, bands).")

        rows, cols, bands = data.shape
        print(f"Original dimensions: {rows} x {cols} x {bands}")

        # Flatten spatial dimensions for PCA (reshape to [pixels, bands])
        data_reshaped = data.reshape(-1, bands)

        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=n_bands)
        reduced_data = pca.fit_transform(data_reshaped)

        # Reshape reduced data back to spatial dimensions
        reduced_hsi = reduced_data.reshape(rows, cols, n_bands)
        print(f"Reduced dimensions: {reduced_hsi.shape}")

        return reduced_hsi
    except Exception as e:
        print(f"Error: {e}")
        return None


def write_to_csv(log_entries, output_file):
    # Define CSV headers based on the keys of the log entries
    if log_entries:
        headers = log_entries[0].keys()

        # Write data to CSV file
        with open(output_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(log_entries)
        print(f"CSV file saved to {output_file}")


import numpy as np
from sklearn.decomposition import PCA
import spectral

import numpy as np
from sklearn.decomposition import PCA
import spectral

def reduce_bands_and_save_hsi_envi(data, target_bands=50, output_prefix="reduced_hsi"):
    """
    Processes hyperspectral data using six reduction methods and saves the results in ENVI format.

    Parameters:
    data (np.ndarray): Input HSI data of shape (rows, cols, bands).
    target_bands (int): Number of bands to reduce to.
    output_prefix (str): Prefix for output file names.

    Saves:
    ENVI files (.hdr and .dat) for each reduction method.
    """
    try:
        # Validate input dimensions
        if data.ndim != 3:
            raise ValueError("Input data must have 3 dimensions (rows, cols, bands).")

        rows, cols, bands = data.shape
        print(f"Processing HSI data with dimensions: {rows} x {cols} x {bands}")

        methods = ["mean", "max", "min", "median", "std", "pca"]

        for method in methods:
            if method == "pca":
                # PCA Reduction
                data_reshaped = data.reshape(-1, bands)
                pca = PCA(n_components=target_bands)
                reduced_data = pca.fit_transform(data_reshaped)
                reduced_hsi = reduced_data.reshape(rows, cols, target_bands)
            else:
                # Aggregation-based methods
                group_sizes = [bands // target_bands] * target_bands
                for i in range(bands % target_bands):
                    group_sizes[i] += 1

                start = 0
                reduced_hsi = np.zeros((rows, cols, target_bands))

                for i in range(target_bands):
                    end = start + group_sizes[i]
                    if method == "mean":
                        reduced_hsi[:, :, i] = np.mean(data[:, :, start:end], axis=2)
                    elif method == "max":
                        reduced_hsi[:, :, i] = np.max(data[:, :, start:end], axis=2)
                    elif method == "min":
                        reduced_hsi[:, :, i] = np.min(data[:, :, start:end], axis=2)
                    elif method == "median":
                        reduced_hsi[:, :, i] = np.median(data[:, :, start:end], axis=2)
                    elif method == "std":
                        reduced_hsi[:, :, i] = np.std(data[:, :, start:end], axis=2)
                    start = end

            # Save the reduced data in ENVI format
            output_file = f"{output_prefix}_{method}_{target_bands}bands"
            spectral.envi.save_image(f"{output_file}.hdr", reduced_hsi, dtype=np.float32, interleave='bsq')
            print(f"Saved {method} reduced data to ENVI format: {output_file}.hdr and {output_file}.dat")

        print("Processing complete. All methods have been saved.")

    except Exception as e:
        print(f"Error: {e}")


