# testEval.py - Testing script for the eval module of CompSensePack
# This script is supposed to stay in (root)/Scripts/functionalityTesting and outputs are always
# saved to (root)/testingOutputs/eval_plots, regardless of where the script is launched from.

import numpy as np
from CompSensePack import calculate_snr, plot_signals  # Updated import
from pathlib import Path

# Function to determine the absolute path for the output directory
def get_output_dir():
    # Determine the directory where this script is located (i.e., (root)/Scripts/functionalityTesting)
    script_dir = Path(__file__).resolve().parent
    
    # Navigate up to the root of the project and define the testingOutputs path
    root_dir = script_dir.parents[1]  # Go up to the project root (2 levels up from this script)
    output_dir = root_dir / 'testingOutputs' / 'eval_plots'
    
    return output_dir

# Test for calculate_snr function
def test_calculate_snr():
    print("Testing calculate_snr...")
    
    # Create an original signal and a reconstructed signal with some noise
    original_signal = np.random.rand(1000)
    noise = np.random.normal(0, 0.1, 1000)
    reconstructed_signal = original_signal + noise
    
    # Calculate SNR
    snr = calculate_snr(original_signal, reconstructed_signal)
    
    print(f"Calculated SNR: {snr:.2f} dB")
    print("Test complete for calculate_snr.\n")

# Test for plot_signals function
def test_plot_signals():
    print("Testing plot_signals...")
    
    # Create an original signal and a reconstructed signal with some noise
    original_signal = np.sin(np.linspace(0, 10, 1000))
    noise = np.random.normal(0, 0.1, 1000)
    reconstructed_signal = original_signal + noise
    
    # Calculate SNR
    snr = calculate_snr(original_signal, reconstructed_signal)
    
    # Get the output directory (root)/testingOutputs/eval_plots
    output_dir = get_output_dir()
    
    # Plot the signals and save the plot
    plot_signals(original_signal, reconstructed_signal, snr=snr, save_path=output_dir, filename='test_plot.png')
    
    print(f"Plot saved to '{output_dir}/test_plot.png'.")
    print("Test complete for plot_signals.\n")


if __name__ == "__main__":
    # Run the tests
    test_calculate_snr()
    test_plot_signals()
