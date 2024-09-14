import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import scipy.io
from pathlib import Path

from CompSensePack import compressedSensing


# Toggle for showing min and max lines
show_min_max = True  # Set this to False if you don't want to show min/max lines

# Number of repetitions, REPS=1 will plot the results, REPS>1 will plot averages
REPS = 1

# Define matrix type (can be changed)
matrix_type = 'unscaled_binary'  # Use 'DBBD', 'gaussian', 'scaled_binary', 'unscaled_binary'

# MOD and K-SVD dictionary learning parameters
mod_params = {
    'redundancy': 1,  # This will translate to 'K = redundancy * BLOCK_LEN'
    'num_iterations': 10,
    'initialization_method': 'DataElements',  # Use 'DataElements' for initialization
    'L': 4  # Number of non-zero coefficients to use in OMP
}

ksvd_params = {
    'redundancy': 1,  # This will translate to 'K = redundancy * BLOCK_LEN'
    'num_iterations': 10,
    'initialization_method': 'DataElements',  # Use 'DataElements' for initialization
    'L': 4,  # Number of non-zero coefficients to use in OMP
    'preserve_dc_atom': 0  # Whether to preserve DC atom (0 for no)
}

# SL0 algorithm parameters
sl0_params = {
    'sigma_min': 1e-3,
    'sigma_decrease_factor': 0.5,
    'mu_0': 2,
    'L': 3,
    'showProgress': False
}

# Load the signal using scipy.io (from 100m.mat)
script_dir = Path(__file__).resolve().parent  # Path to the directory where this script is located
root_dir = script_dir.parents[1]  # Going two levels up to reach the project root
mat_file = root_dir / 'data' / '100m.mat'  # Path to the data file
data = scipy.io.loadmat(mat_file)
signal = data['val'][0]  # Assuming the signal is stored in 'val' key
signal = signal[360 * 0: 360 * 60 * 2]  # Example duration of 2 minutes

# Initialize SNR accumulators and min/max values
snr_dct_total = snr_mod_total = snr_ksvd_total = 0
snr_dct_min = snr_mod_min = snr_ksvd_min = float('inf')
snr_dct_max = snr_mod_max = snr_ksvd_max = float('-inf')
snr_dct_kron_total = snr_mod_kron_total = snr_ksvd_kron_total = 0
snr_dct_kron_min = snr_mod_kron_min = snr_ksvd_kron_min = float('inf')
snr_dct_kron_max = snr_mod_kron_max = snr_ksvd_kron_max = float('-inf')

# SNR box toggle (set to True by default)
show_snr_box = True  # Set this to False if you don't want to display the SNR on the plots

# Loop through the number of repetitions
for rep in range(REPS):
    print(f"Iteration {rep} just started")

    # Instantiate the class (no need to re-load the signal)
    cs = compressedSensing(signal=signal, matrix_type=matrix_type)

    # Divide the signal
    cs.divide_signal(training_percentage=0.45)

    # Compress the test set
    cs.compress_test_set()

    # ----------------- Without Kronecker Compression -----------------

    # ----------------- DCT-Based Dictionary Recovery -----------------
    cs.generate_dictionary(dictionary_type='dct')
    cs.recover_signal(sl0_params=sl0_params)  # Pass SL0 parameters here
    snr_dct = cs.get_snr()
    snr_dct_total += snr_dct
    snr_dct_min = min(snr_dct_min, snr_dct)
    snr_dct_max = max(snr_dct_max, snr_dct)

    # If REPS == 1, plot the detailed reconstruction vs original for DCT
    if REPS == 1:
        cs.plot_reconstructed_vs_original(
            start_pct=0.0,
            num_samples=None,  # Plot all samples
            reconstructed_label=f"Reconstructed Signal (DCT) on 100m.mat",
            show_snr_box=show_snr_box  # Use the SNR box toggle
        )

    # ----------------- MOD-Based Dictionary Recovery -----------------
    cs.generate_dictionary(dictionary_type='mod', mod_params=mod_params)
    cs.recover_signal(sl0_params=sl0_params)  # Pass SL0 parameters here
    snr_mod = cs.get_snr()
    snr_mod_total += snr_mod
    snr_mod_min = min(snr_mod_min, snr_mod)
    snr_mod_max = max(snr_mod_max, snr_mod)

    # If REPS == 1, plot the detailed reconstruction vs original for MOD
    if REPS == 1:
        cs.plot_reconstructed_vs_original(
            start_pct=0.0,
            num_samples=None,  # Plot all samples
            reconstructed_label=f"Reconstructed Signal (MOD) on 100m.mat",
            show_snr_box=show_snr_box  # Use the SNR box toggle
        )

    # ----------------- K-SVD-Based Dictionary Recovery -----------------
    cs.generate_dictionary(dictionary_type='ksvd', ksvd_params=ksvd_params)
    cs.recover_signal(sl0_params=sl0_params)  # Pass SL0 parameters here
    snr_ksvd = cs.get_snr()
    snr_ksvd_total += snr_ksvd
    snr_ksvd_min = min(snr_ksvd_min, snr_ksvd)
    snr_ksvd_max = max(snr_ksvd_max, snr_ksvd)

    # If REPS == 1, plot the detailed reconstruction vs original for KSVD
    if REPS == 1:
        cs.plot_reconstructed_vs_original(
            start_pct=0.0,
            num_samples=None,  # Plot all samples
            reconstructed_label=f"Reconstructed Signal (KSVD) on 100m.mat",
            show_snr_box=show_snr_box  # Use the SNR box toggle
        )

    # ----------------- Activate Kronecker Compression -----------------
    KRON_FACT = 8  # Example Kronecker factor
    cs.kronecker_activate(KRON_FACT)

    # ----------------- DCT-Based Dictionary Recovery with Kronecker -----------------
    cs.generate_dictionary(dictionary_type='dct')
    cs.recover_signal(sl0_params=sl0_params)  # Pass SL0 parameters here
    snr_dct_kron = cs.get_snr()
    snr_dct_kron_total += snr_dct_kron
    snr_dct_kron_min = min(snr_dct_kron_min, snr_dct_kron)
    snr_dct_kron_max = max(snr_dct_kron_max, snr_dct_kron)

    # If REPS == 1, plot the detailed reconstruction vs original for DCT + Kronecker
    if REPS == 1:
        cs.plot_reconstructed_vs_original(
            start_pct=0.0,
            num_samples=None,  # Plot all samples
            reconstructed_label=f"Reconstructed Signal (DCT + Kronecker) on 100m.mat",
            show_snr_box=show_snr_box  # Use the SNR box toggle
        )

    # ----------------- MOD-Based Dictionary Recovery with Kronecker -----------------
    cs.generate_dictionary(dictionary_type='mod', mod_params=mod_params)
    cs.recover_signal(sl0_params=sl0_params)  # Pass SL0 parameters here
    snr_mod_kron = cs.get_snr()
    snr_mod_kron_total += snr_mod_kron
    snr_mod_kron_min = min(snr_mod_kron_min, snr_mod_kron)
    snr_mod_kron_max = max(snr_mod_kron_max, snr_mod_kron)

    # If REPS == 1, plot the detailed reconstruction vs original for MOD + Kronecker
    if REPS == 1:
        cs.plot_reconstructed_vs_original(
            start_pct=0.0,
            num_samples=None,  # Plot all samples
            reconstructed_label=f"Reconstructed Signal (MOD + Kronecker) on 100m.mat",
            show_snr_box=show_snr_box  # Use the SNR box toggle
        )

    # ----------------- K-SVD-Based Dictionary Recovery with Kronecker -----------------
    cs.generate_dictionary(dictionary_type='ksvd', ksvd_params=ksvd_params)
    cs.recover_signal(sl0_params=sl0_params)  # Pass SL0 parameters here
    snr_ksvd_kron = cs.get_snr()
    snr_ksvd_kron_total += snr_ksvd_kron
    snr_ksvd_kron_min = min(snr_ksvd_kron_min, snr_ksvd_kron)
    snr_ksvd_kron_max = max(snr_ksvd_kron_max, snr_ksvd_kron)

    # If REPS == 1, plot the detailed reconstruction vs original for KSVD + Kronecker
    if REPS == 1:
        cs.plot_reconstructed_vs_original(
            start_pct=0.0,
            num_samples=None,  # Plot all samples
            reconstructed_label=f"Reconstructed Signal (KSVD + Kronecker) on 100m.mat",
            show_snr_box=show_snr_box  # Use the SNR box toggle
        )

# For REPS > 1, plot the histogram
if REPS > 1:
    # Compute average, min, and max values for each method
    avg_snrs = [
        snr_dct_total / REPS,
        snr_mod_total / REPS,
        snr_ksvd_total / REPS,
        snr_dct_kron_total / REPS,
        snr_mod_kron_total / REPS,
        snr_ksvd_kron_total / REPS
    ]

    snr_min = {
        'dct': snr_dct_min,
        'mod': snr_mod_min,
        'ksvd': snr_ksvd_min,
        'dct_kron': snr_dct_kron_min,
        'mod_kron': snr_mod_kron_min,
        'ksvd_kron': snr_ksvd_kron_min
    }

    snr_max = {
        'dct': snr_dct_max,
        'mod': snr_mod_max,
        'ksvd': snr_ksvd_max,
        'dct_kron': snr_dct_kron_max,
        'mod_kron': snr_mod_kron_max,
        'ksvd_kron': snr_ksvd_kron_max
    }

    # Labels for each method
    labels = [
        f"{matrix_type}-DCT", f"{matrix_type}-MOD", f"{matrix_type}-KSVD",
        f"{matrix_type}-DCT-KRON", f"{matrix_type}-MOD-KRON", f"{matrix_type}-KSVD-KRON"
    ]

    # Create a bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, avg_snrs, color='teal')

    # Add min and max lines if toggle is True
    if show_min_max:
        for bar, label in zip(bars, labels):
            method_key = label.split("-")[1].lower()  # Extract the dictionary type (e.g., 'dct', 'mod', 'ksvd')
            if "kron" in label.lower():
                method_key += "_kron"  # Add the '_kron' suffix for Kronecker methods
            
            # Get min and max values
            min_val = snr_min[method_key]
            max_val = snr_max[method_key]
            
            # Get the bar center and width
            bar_center = bar.get_x() + bar.get_width() / 2
            
            # Plot min and max lines with bright green and red
            plt.plot([bar_center - bar.get_width()/4, bar_center + bar.get_width()/4], [min_val, min_val], color='red', lw=2)  # Min line (bright red)
            plt.plot([bar_center - bar.get_width()/4, bar_center + bar.get_width()/4], [max_val, max_val], color='lime', lw=2)  # Max line (bright green)

    # Labels and titles
    plt.xlabel('Methods')
    plt.ylabel('Average SNR (dB)')
    plt.title(f'Average SNR for {matrix_type} on 100m.mat with Min/Max Values' if show_min_max else f'Average SNR for {matrix_type} on 100m.mat')
    plt.xticks(rotation=45)


    # Define the path to the studyOutputs folder relative to the root
    output_folder = root_dir / 'studyOutputs'

    # Ensure the output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save the plot with the matrix type in the filename
    output_file = output_folder / f'snr_histogram_with_min_max_{matrix_type}_100m.png'
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

    print(f"Histogram saved to '{output_file}'")

    # Save CSV with results
    csv_output_file = output_folder / f'snr_results_{matrix_type}_100m.csv'
    results_data = {
        'Method': labels,
        'Average SNR': avg_snrs,
        'Min SNR': [snr_min[method_key] for method_key in ['dct', 'mod', 'ksvd', 'dct_kron', 'mod_kron', 'ksvd_kron']],
        'Max SNR': [snr_max[method_key] for method_key in ['dct', 'mod', 'ksvd', 'dct_kron', 'mod_kron', 'ksvd_kron']]
    }
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(csv_output_file, index=False)

    print(f"CSV saved to '{csv_output_file}'")
