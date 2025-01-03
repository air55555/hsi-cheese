from spectral import open_image
from spectral.io.envi import save_image
import os
import numpy as np
from spectral import open_image
from spectral.io.envi import save_image
from scipy.ndimage import gaussian_filter

def extract_31_bands(file_path, start_band=1, total_bands=31):
    """
    Extract exactly 31 bands from an ENVI hyperspectral image and save with a new header.

    Parameters:
        file_path (str): Path to the original ENVI header file.
        start_band (int): Starting band index (1-based).
        total_bands (int): Total number of bands to extract.
    """
    # Load the hyperspectral image
    hsi = open_image(file_path)
    hsi_data = hsi.load()

    # Convert 1-based indexing to 0-based indexing for Python
    start_band_idx = start_band - 1

    # Calculate step size to evenly sample bands
    num_original_bands = hsi_data.shape[2]
    step = max(1, (num_original_bands - start_band_idx) // total_bands)

    # Extract 31 bands
    selected_band_indices = list(range(start_band_idx, num_original_bands, step))[:total_bands]
    selected_bands = hsi_data[:, :, selected_band_indices]

    # Adjust header metadata
    new_header = hsi.metadata.copy()
    new_header['bands'] = len(selected_band_indices)
    new_header['wavelength'] = np.array(new_header['wavelength'])[selected_band_indices].tolist()
    new_header['fwhm'] = np.array(new_header['fwhm'])[selected_band_indices].tolist()

    # Generate new file name
    base_name, ext = os.path.splitext(file_path)
    new_file_path = f"{base_name}_31bands{ext}"

    # Save the new hyperspectral image with updated header
    save_image(new_file_path, selected_bands, dtype=hsi_data.dtype, interleave=new_header['interleave'],
               metadata=new_header)
    print(f"New file saved as: {new_file_path}")



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


# Example Usage
#file_path = "example_image.hdr"  # Replace with your actual file path
#extract_31_bands(file_path, start_band=1, total_bands=31)


def load_hsi(file_path):
    """
    Load the hyperspectral image cube from an ENVI file.
    """
    hsi_cube = open_image(file_path).load()
    return hsi_cube


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



def spatial_downsampling(hsi, factor=2):
    """
    Perform five spatial downsampling methods on a hyperspectral image (HSI).

    Parameters:
        hsi (numpy.ndarray): Input hyperspectral image (3D array).
        factor (int): Downsampling factor for spatial dimensions.

    Returns:
        dict: Dictionary containing five downsampled HSIs.
    """
    downsampled_variants = {}

    # 1. Spatial Averaging (Block Reduction)
    def spatial_averaging(img, factor):
        return img.reshape(
            img.shape[0] // factor, factor,
            img.shape[1] // factor, factor,
            img.shape[2]
        ).mean(axis=(1, 3))

    downsampled_variants['spatial_averaging'] = spatial_averaging(hsi, factor)

    # 2. Spatial Subsampling
    downsampled_variants['spatial_subsampling'] = hsi[::factor, ::factor, :]

    # 3. Gaussian Blurring + Subsampling
    def gaussian_downsampling(img, factor):
        blurred = gaussian_filter(img, sigma=(factor, factor, 0))  # Blur spatially
        return blurred[::factor, ::factor, :]

    downsampled_variants['gaussian_blur_subsampling'] = gaussian_downsampling(hsi, factor)

    # 4. Max Pooling
    def max_pooling(img, factor):
        return img.reshape(
            img.shape[0] // factor, factor,
            img.shape[1] // factor, factor,
            img.shape[2]
        ).max(axis=(1, 3))

    downsampled_variants['max_pooling'] = max_pooling(hsi, factor)

    # 5. Median Pooling
    def median_pooling(img, factor):
        return np.median(
            img.reshape(
                img.shape[0] // factor, factor,
                img.shape[1] // factor, factor,
                img.shape[2]
            ),
            axis=(1, 3)
        )

    downsampled_variants['median_pooling'] = median_pooling(hsi, factor)

    return [downsampled_variants['median_pooling'],
            downsampled_variants['max_pooling'],
            downsampled_variants['gaussian_blur_subsampling'],
            downsampled_variants['spatial_subsampling'],
            downsampled_variants['spatial_averaging']]

# Example usage
#hsi = np.random.rand(200, 200, 100)  # Simulate a random HSI of size 200x200 with 100 bands
#downsampled_hsis = downsample_hsi(hsi, spatial_factor=2, spectral_factor=2)

#for method, downsampled_hsi in downsampled_hsis.items():
#    print(f"{method}: {downsampled_hsi.shape}")


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

