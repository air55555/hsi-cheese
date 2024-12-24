import numpy as np
from spectral import open_image
from scipy.io import savemat
import os


def envi_to_matlab(envi_file, mat_file=None):
    """
    Convert an ENVI file to a MATLAB .mat file.

    Parameters:
        envi_file (str): Path to the input ENVI file (with associated .hdr file).
        mat_file (str): Path to the output MATLAB .mat file.
    """
    # Open the ENVI file
    envi_data = open_image(envi_file)

    # Read the data into a NumPy array
    array = envi_data.load()  # Loads the full hyperspectral cube into memory

    # Save the array as a MATLAB .mat file with the variable name 'gt'
    base, ext = os.path.splitext(envi_file)
    if not mat_file:
        mat_file = f"{base}.mat"
    savemat(mat_file, {'gt': array})

    print(f"ENVI file '{envi_file}' successfully saved as MATLAB file '{mat_file}'.")


# Example usage
#envi_file = r'C:\Users\1\PycharmProjects\DPHSIRmy\input\64_10_circ_t_cropped.hdr'
# Path to the ENVI file
#mat_file = "output_data.mat"  # Path to the MATLAB .mat file
#envi_to_matlab(envi_file)
