from spectral import open_image
from spectral.io.envi import save_image

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
