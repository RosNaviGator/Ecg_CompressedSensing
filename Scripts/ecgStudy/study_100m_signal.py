"""
study_100m_signal.py - Comparative Analysis of Dictionary Learning Techniques for Signal Compression and Recovery

This script performs a comprehensive analysis of various dictionary learning techniques for signal 
compression and recovery, using the 100m.mat ECG signal from the MIT-BIH Arrhythmia Database. 
The analysis compares different methods such as the Discrete Cosine Transform (DCT), MOD (Method 
of Optimal Directions), and K-SVD (K-Singular Value Decomposition), as well as their performance 
with standard and Kronecker compression techniques.

The effectiveness of each technique is evaluated using the Signal-to-Noise Ratio (SNR), which 
quantifies the quality of signal recovery. Multiple repetitions of the experiment can be run to 
compute average, minimum, and maximum SNR values for each method. The script allows the user to 
customize key parameters such as the measurement matrix type, training percentage, Kronecker factor, 
and SL0 algorithm parameters.

Outputs will be in (root)/studyOutputs

Supported Dictionaries:
--------------------------------------
- DCT (Discrete Cosine Transform)
- MOD (Method of Optimal Directions)
- K-SVD (K-Singular Value Decomposition)

Supported Measurement Matrices:
-------------------------------
- DBBD (Deterministic Diagonally Blocked Block Diagonal)
- Gaussian
- Scaled Binary
- Unscaled Binary

Compression Methods:
--------------------
- Standard compression
- Kronecker compression (controlled by the KRON_FACT parameter)

Key Features:
-------------
- Allows for multiple repetitions of the experiment to assess variability in SNR.
- Supports visualization of the original vs. reconstructed signals.
- Outputs average, minimum, and maximum SNR values in both graphical (histogram) and CSV formats.

Parameters
----------
- REPS : int
    Number of repetitions for the experiment. If REPS == 1, detailed results are plotted for each method.
- show_snr_box : bool
    Toggle to display the SNR value on the reconstructed vs original signal plots.
- matrix_type : str
    Type of matrix used for the measurement matrix. Options: 'DBBD', 'gaussian', 'scaled_binary', 'unscaled_binary'.
- training_percentage : float
    The percentage of the signal used for training the dictionary.
- signal_duration : int
    Duration of the signal in seconds. Determines the portion of the signal used for the experiment.
- KRON_FACT : int
    Kronecker factor used for Kronecker compression.
- mod_params : dict
    Parameters for the MOD algorithm, including redundancy, number of iterations, initialization method, 
    and the number of non-zero coefficients.
- ksvd_params : dict
    Parameters for the K-SVD algorithm, including redundancy, number of iterations, initialization method, 
    and the number of non-zero coefficients.
- sl0_params : dict
    Parameters for the SL0 (Smoothed L0) algorithm, including sigma_min, sigma_decrease_factor, mu_0, number of 
    inner loop iterations (L), and showProgress toggle.

Usage
-----
1. Customize the parameters under the `if __name__ == "__main__":` section to control the experiment settings.
2. Run the script to execute the analysis and produce results.

Example:
    $ python study_100m_signal.py

Outputs
-------
- Plots of reconstructed vs original signals for each method (DCT, MOD, K-SVD) with and without Kronecker compression.
- If REPS > 1, a histogram of the average, minimum, and maximum SNR values across all repetitions.
- CSV file containing SNR results for each method.
- PNG files of histograms and signal comparisons are saved in the (root)/studyOutputs directory.

"""


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import scipy.io
from CompSensePack import compressedSensing


def main(REPS, show_snr_box, matrix_type, training_percentage, signal_duration, mod_params, ksvd_params, sl0_params, KRON_FACT):

    # Load the signal using scipy.io (from 100m.mat)
    script_dir = Path(__file__).resolve().parent  # Path to the directory where this script is located
    root_dir = script_dir.parents[1]  # Going two levels up to reach the project root
    mat_file = root_dir / 'data' / '100m.mat'  # Path to the data file

    data = scipy.io.loadmat(mat_file)
    signal = data['val'][0]  # Assuming the signal is stored in 'val' key
    signal = signal[360 * 0: 360 * signal_duration]  # Slice the signal based on the duration argument

    # Initialize accumulators for SNR results
    snr_dct_total = snr_mod_total = snr_ksvd_total = 0
    snr_dct_min = snr_mod_min = snr_ksvd_min = float('inf')
    snr_dct_max = snr_mod_max = snr_ksvd_max = float('-inf')

    snr_dct_kron_total = snr_mod_kron_total = snr_ksvd_kron_total = 0
    snr_dct_kron_min = snr_mod_kron_min = snr_ksvd_kron_min = float('inf')
    snr_dct_kron_max = snr_mod_kron_max = snr_ksvd_kron_max = float('-inf')

    # Loop through the number of repetitions
    for rep in range(REPS):
        print(f"Iteration {rep + 1} just started")

        # Instantiate the class
        cs = compressedSensing(signal=signal, matrix_type=matrix_type)

        # Divide the signal
        cs.divide_signal(training_percentage=training_percentage)

        # Compress the test set
        cs.compress_test_set()

        # ----------------- Without Kronecker Compression -----------------

        # ----------------- DCT-Based Dictionary Recovery -----------------
        cs.generate_dictionary(dictionary_type='dct')
        cs.recover_signal(sl0_params=sl0_params)
        snr_dct = cs.get_snr()
        snr_dct_total += snr_dct
        snr_dct_min = min(snr_dct_min, snr_dct)
        snr_dct_max = max(snr_dct_max, snr_dct)

        # If REPS == 1, plot the detailed reconstruction vs original for DCT
        if REPS == 1:
            cs.plot_reconstructed_vs_original(
                start_pct=0.0,
                num_samples=None,
                reconstructed_label="Reconstructed Signal (DCT)",
                show_snr_box=show_snr_box
            )

        # ----------------- MOD-Based Dictionary Recovery -----------------
        cs.generate_dictionary(dictionary_type='mod', mod_params=mod_params)
        cs.recover_signal(sl0_params=sl0_params)
        snr_mod = cs.get_snr()
        snr_mod_total += snr_mod
        snr_mod_min = min(snr_mod_min, snr_mod)
        snr_mod_max = max(snr_mod_max, snr_mod)

        # If REPS == 1, plot the detailed reconstruction vs original for MOD
        if REPS == 1:
            cs.plot_reconstructed_vs_original(
                start_pct=0.0,
                num_samples=None,
                reconstructed_label="Reconstructed Signal (MOD)",
                show_snr_box=show_snr_box
            )

        # ----------------- K-SVD-Based Dictionary Recovery -----------------
        cs.generate_dictionary(dictionary_type='ksvd', ksvd_params=ksvd_params)
        cs.recover_signal(sl0_params=sl0_params)
        snr_ksvd = cs.get_snr()
        snr_ksvd_total += snr_ksvd
        snr_ksvd_min = min(snr_ksvd_min, snr_ksvd)
        snr_ksvd_max = max(snr_ksvd_max, snr_ksvd)

        # If REPS == 1, plot the detailed reconstruction vs original for KSVD
        if REPS == 1:
            cs.plot_reconstructed_vs_original(
                start_pct=0.0,
                num_samples=None,
                reconstructed_label="Reconstructed Signal (KSVD)",
                show_snr_box=show_snr_box
            )

        # ----------------- Activate Kronecker Compression -----------------
        cs.kronecker_activate(KRON_FACT)

        # ----------------- DCT-Based Dictionary Recovery with Kronecker -----------------
        cs.generate_dictionary(dictionary_type='dct')
        cs.recover_signal(sl0_params=sl0_params)
        snr_dct_kron = cs.get_snr()
        snr_dct_kron_total += snr_dct_kron
        snr_dct_kron_min = min(snr_dct_kron_min, snr_dct_kron)
        snr_dct_kron_max = max(snr_dct_kron_max, snr_dct_kron)

        # If REPS == 1, plot the detailed reconstruction vs original for DCT + Kronecker
        if REPS == 1:
            cs.plot_reconstructed_vs_original(
                start_pct=0.0,
                num_samples=None,
                reconstructed_label="Reconstructed Signal (DCT + Kronecker)",
                show_snr_box=show_snr_box
            )

        # ----------------- MOD-Based Dictionary Recovery with Kronecker -----------------
        cs.generate_dictionary(dictionary_type='mod', mod_params=mod_params)
        cs.recover_signal(sl0_params=sl0_params)
        snr_mod_kron = cs.get_snr()
        snr_mod_kron_total += snr_mod_kron
        snr_mod_kron_min = min(snr_mod_kron_min, snr_mod_kron)
        snr_mod_kron_max = max(snr_mod_kron_max, snr_mod_kron)

        # If REPS == 1, plot the detailed reconstruction vs original for MOD + Kronecker
        if REPS == 1:
            cs.plot_reconstructed_vs_original(
                start_pct=0.0,
                num_samples=None,
                reconstructed_label="Reconstructed Signal (MOD + Kronecker)",
                show_snr_box=show_snr_box
            )

        # ----------------- K-SVD-Based Dictionary Recovery with Kronecker -----------------
        cs.generate_dictionary(dictionary_type='ksvd', ksvd_params=ksvd_params)
        cs.recover_signal(sl0_params=sl0_params)
        snr_ksvd_kron = cs.get_snr()
        snr_ksvd_kron_total += snr_ksvd_kron
        snr_ksvd_kron_min = min(snr_ksvd_kron_min, snr_ksvd_kron)
        snr_ksvd_kron_max = max(snr_ksvd_kron_max, snr_ksvd_kron)

        # If REPS == 1, plot the detailed reconstruction vs original for KSVD + Kronecker
        if REPS == 1:
            cs.plot_reconstructed_vs_original(
                start_pct=0.0,
                num_samples=None,
                reconstructed_label="Reconstructed Signal (KSVD + Kronecker)",
                show_snr_box=show_snr_box
            )

    # If more than one repetition, calculate and plot average SNR values
    if REPS > 1:
        avg_snrs = [
            snr_dct_total / REPS,
            snr_mod_total / REPS,
            snr_ksvd_total / REPS,
            snr_dct_kron_total / REPS,
            snr_mod_kron_total / REPS,
            snr_ksvd_kron_total / REPS,
        ]

        snr_min = {
            'dct': snr_dct_min,
            'mod': snr_mod_min,
            'ksvd': snr_ksvd_min,
            'dct_kron': snr_dct_kron_min,
            'mod_kron': snr_mod_kron_min,
            'ksvd_kron': snr_ksvd_kron_min,
        }

        snr_max = {
            'dct': snr_dct_max,
            'mod': snr_mod_max,
            'ksvd': snr_ksvd_max,
            'dct_kron': snr_dct_kron_max,
            'mod_kron': snr_mod_kron_max,
            'ksvd_kron': snr_ksvd_kron_max,
        }

        labels = [
            f"{matrix_type}-DCT", f"{matrix_type}-MOD", f"{matrix_type}-KSVD",
            f"{matrix_type}-DCT-KRON", f"{matrix_type}-MOD-KRON", f"{matrix_type}-KSVD-KRON"
        ]

        # Create a bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, avg_snrs, color='teal')

        for bar, label in zip(bars, labels):
            method_key = label.split("-")[1].lower()
            if "kron" in label.lower():
                method_key += "_kron"
            min_val = snr_min[method_key]
            max_val = snr_max[method_key]
            bar_center = bar.get_x() + bar.get_width() / 2
            plt.plot([bar_center - bar.get_width()/4, bar_center + bar.get_width()/4], [min_val, min_val], color='red', lw=2)
            plt.plot([bar_center - bar.get_width()/4, bar_center + bar.get_width()/4], [max_val, max_val], color='lime', lw=2)

        plt.xlabel('Methods')
        plt.ylabel('Average SNR (dB)')
        plt.title(f'Average SNR for {matrix_type} with Min/Max Values')
        plt.xticks(rotation=45)

        output_folder = root_dir / 'studyOutputs'
        output_folder.mkdir(parents=True, exist_ok=True)

        output_file = output_folder / f'snr_histogram_with_min_max_{matrix_type}.png'
        plt.tight_layout()
        plt.savefig(output_file)
        plt.show()

        print(f"Histogram saved to '{output_file}'")

        # Save CSV with results
        csv_output_file = output_folder / f'snr_results_{matrix_type}.csv'
        results_data = {
            'Method': labels,
            'Average SNR': avg_snrs,
            'Min SNR': [snr_min[method_key] for method_key in ['dct', 'mod', 'ksvd', 'dct_kron', 'mod_kron', 'ksvd_kron']],
            'Max SNR': [snr_max[method_key] for method_key in ['dct', 'mod', 'ksvd', 'dct_kron', 'mod_kron', 'ksvd_kron']]
        }
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(csv_output_file, index=False)

        print(f"CSV saved to '{csv_output_file}'")



if __name__ == "__main__":
    # Modify these values to control the execution
    REPS = 1  # Number of repetitions
    show_snr_box = True  # Toggle SNR box display
    matrix_type = 'unscaled_binary'  # Matrix type for the measurement matrix
    training_percentage = 0.45  # Percentage of signal to use for training
    signal_duration = 60 * 2  # Duration of signal in seconds (e.g., 2 minutes)
    KRON_FACT = 8  # Kronecker factor

    # MOD parameters
    mod_params = {
        'redundancy': 1,
        'num_iterations': 10,
        'initialization_method': 'DataElements',
        'L': 4
    }

    # K-SVD parameters
    ksvd_params = {
        'redundancy': 1,
        'num_iterations': 10,
        'initialization_method': 'DataElements',
        'L': 4,
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

    main(REPS, show_snr_box, matrix_type, training_percentage, signal_duration, mod_params, ksvd_params, sl0_params, KRON_FACT)
