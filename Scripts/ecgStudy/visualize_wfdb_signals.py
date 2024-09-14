"""
visualize_wfdb_signals.py

This script is designed to load, compress, recover, and visualize ECG signals from the MIT-BIH Arrhythmia 
Database using various dictionary learning methods and sparse recovery techniques. It allows the user 
to apply different types of measurement matrices and dictionary learning algorithms to test their 
performance on reconstructing the original ECG signal. The signal is divided into training and test sets 
for dictionary learning and signal recovery.

Outputs will be in (root)/studyOutputs

The following dictionaries are supported:
    - DCT (Discrete Cosine Transform)
    - MOD (Method of Optimal Directions)
    - K-SVD (K-Singular Value Decomposition)

The following measurement matrices are supported:
    - DBBD (Deterministic Diagonally Blocked Block Diagonal)
    - Gaussian
    - Scaled Binary
    - Unscaled Binary

The recovered signals can be compared to the original signals using Signal-to-Noise Ratio (SNR) 
and visually plotted to observe the reconstruction quality.

Parameters
----------
record_name : str
    The record name from the MIT-BIH Arrhythmia Database (e.g., '100', '109').
duration_minutes : int
    Duration of the signal to load in minutes.
chosen_method : str
    Dictionary learning method ('DCT', 'MOD', or 'KSVD').
measurement_matrix : str
    Type of measurement matrix to use ('DBBD', 'gaussian', 'scaled_binary', 'unscaled_binary').
training_percentage : float
    Percentage of the signal to be used for training (the rest will be used for testing).
num_samples : int
    Number of samples to plot in the reconstructed vs original signal comparison.
mod_params : dict
    Parameters for the MOD algorithm.
ksvd_params : dict
    Parameters for the K-SVD algorithm.
sl0_params : dict
    Parameters for the SL0 algorithm.

Usage
-----
To execute this script, simply run it as a Python script. Modify the parameters in the `if __name__ == "__main__":`
block to set the desired configuration for the experiment. Ensure that the necessary ECG record files 
from the MIT-BIH Arrhythmia Database are available for loading.

Example:
    python visualize_wfdb_signals.py

The script will load the signal, apply the chosen dictionary learning method and measurement matrix, 
and plot the reconstructed signal along with the original signal for visual comparison.

Dependencies
------------
- numpy
- matplotlib
- scipy
- CompSensePack (custom module with the `compressedSensing` class and `load_signal_from_wfdb` function)

Output
------
The script plots the original and reconstructed ECG signals for visual comparison. SNR values between 
the original and reconstructed signals are computed and displayed in the plot. The reconstruction 
quality can be adjusted by modifying parameters such as the dictionary learning method, measurement 
matrix, and SL0 algorithm parameters. Outputs are saved to (root)/studyOutputs.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io
from CompSensePack import compressedSensing, load_signal_from_wfdb

def main(record_name, duration_minutes, chosen_method, measurement_matrix, training_percentage, num_samples, mod_params, ksvd_params, sl0_params):
    
    # Load the signal using wfdb
    signal, signal_name = load_signal_from_wfdb(record_name, duration_minutes=duration_minutes)

    # Instantiate the compressedSensing class with the chosen measurement matrix
    cs = compressedSensing(signal=signal, matrix_type=measurement_matrix)

    # Divide the signal into training and testing
    cs.divide_signal(training_percentage=training_percentage)

    # Compress the test set
    cs.compress_test_set()

    # Generate the dictionary based on the chosen method
    if chosen_method == 'DCT':
        cs.generate_dictionary(dictionary_type='dct')
    elif chosen_method == 'MOD':
        cs.generate_dictionary(dictionary_type='mod', mod_params=mod_params)
    elif chosen_method == 'KSVD':
        cs.generate_dictionary(dictionary_type='ksvd', ksvd_params=ksvd_params)
    else:
        raise ValueError("Invalid method chosen. Use 'DCT', 'MOD', or 'KSVD'.")

    # Recover the signal using SL0
    cs.recover_signal(sl0_params=sl0_params)

    # Ensure the output folder exists regardless of where the script is launched
    script_dir = Path(__file__).resolve().parent  # Path to the directory where this script is located
    root_dir = script_dir.parents[1]  # Going two levels up to reach the project root
    output_folder = root_dir / 'studyOutputs'  # Path to the 'studyOutputs' folder
    output_folder.mkdir(parents=True, exist_ok=True)

    # Plot the reconstructed signal vs original using the built-in method
    output_file = output_folder / f"reconstructed_{chosen_method}_{measurement_matrix}.png"
    cs.plot_reconstructed_vs_original(
        start_pct=0.0,
        num_samples=num_samples,
        reconstructed_label=f"Reconstructed Signal ({chosen_method}) with {measurement_matrix} Matrix",
        save_path=output_folder,
        filename=output_file.name  # Save with the custom filename
    )


if __name__ == "__main__":
    # Parameters to set up the experiment
    record_name = '109'  # Record from MIT-BIH Arrhythmia Database
    duration_minutes = 2 * 60  # Load 2 hours of signal
    chosen_method = 'KSVD'  # Choose 'DCT', 'MOD', or 'KSVD'
    measurement_matrix = 'unscaled_binary'  # Choose your matrix type ('DBBD', 'gaussian', 'scaled_binary', 'unscaled_binary')
    training_percentage = 0.45  # Percentage of the signal used for training
    num_samples = 2048  # Number of samples to plot

    # MOD parameters
    mod_params = {
        'redundancy': 1, 
        'num_iterations': 10, 
        'L': 4, 
        'initialization_method': 'DataElements'
    }

    # K-SVD parameters
    ksvd_params = {
        'redundancy': 1, 
        'num_iterations': 10, 
        'L': 4, 
        'initialization_method': 'DataElements',  
        'preserve_dc_atom': 0
    }

    # SL0 parameters
    sl0_params = {
        'sigma_min': 1e-3, 
        'sigma_decrease_factor': 0.5, 
        'mu_0': 2, 
        'L': 3, 
        'showProgress': False
    }

    # Run the main function
    main(record_name, duration_minutes, chosen_method, measurement_matrix, training_percentage, num_samples, mod_params, ksvd_params, sl0_params)
