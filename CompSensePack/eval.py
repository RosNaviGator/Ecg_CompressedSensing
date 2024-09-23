"""
eval.py - Functions for Signal-to-Noise Ratio (SNR) calculation and signal plotting.

This module provides utility functions for:
1. Calculating the Signal-to-Noise Ratio (SNR) between an original and a reconstructed signal.
2. Plotting the original and reconstructed signals together, with options to display the SNR and save the plot.

Note:
-----
- Requires `matplotlib` and `numpy`.
- Useful for comparing signals and visualizing differences in various signal processing applications.

Example usage:
--------------
>>> import numpy as np
>>> from eval import calculate_snr, plot_signals

>>> original_signal = np.random.rand(100)
>>> reconstructed_signal = np.random.rand(100)

>>> snr = calculate_snr(original_signal, reconstructed_signal)
>>> plot_signals(original_signal, reconstructed_signal, snr=snr, save_path='./plots')
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def calculate_snr(signal, recovered_signal):
    """
    Calculates the Signal-to-Noise Ratio (SNR) between the original signal and the recovered signal.

    Parameters
    ----------
    signal : numpy.ndarray
        The original signal.
    recovered_signal : numpy.ndarray
        The recovered signal after some processing or recovery algorithm.

    Returns
    -------
    snr : float
        The Signal-to-Noise Ratio (SNR) in decibels (dB).

    Notes
    -----
    - The SNR is calculated as 20 * log10(norm(signal) / norm(signal - recovered_signal)).
    - A higher SNR value indicates a better recovery, with less error relative to the original signal.
    - If the signals are identical, the SNR would be infinite. If the recovered signal has no similarity, the SNR will be very low or negative.
    
    Example
    -------
    >>> original = np.random.rand(100)
    >>> recovered = original + np.random.normal(0, 0.1, 100)
    >>> snr = calculate_snr(original, recovered)
    >>> print(f"SNR: {snr:.2f} dB")
    """
    # Ensure the signals are numpy arrays
    if not isinstance(signal, np.ndarray) or not isinstance(recovered_signal, np.ndarray):
        raise ValueError("Both signal and recovered_signal must be numpy arrays.")
    
    # Calculate the error between the signals
    error = recovered_signal - signal
    
    # Calculate and return the SNR in dB
    snr = 20 * np.log10(np.linalg.norm(signal) / np.linalg.norm(error))
    
    return snr


def plot_signals(original_signal, reconstructed_signal, snr=None, original_name="Original Signal", 
                 reconstructed_name="Reconstructed Signal", save_path=None, filename=None,
                 start_pct=0.0, num_samples=None, show_snr_box=False):
    """
    Plots a section of the original signal and the reconstructed signal on the same plot with the given names,
    displays the Signal-to-Noise Ratio (SNR) in a text box if enabled, and saves the plot to a specified directory.

    Parameters
    ----------
    original_signal : numpy.ndarray
        The original signal to be plotted.
    
    reconstructed_signal : numpy.ndarray
        The reconstructed signal to be plotted.
    
    snr : float, optional (default=None)
        The Signal-to-Noise Ratio to display. If None, it will be computed using the original and reconstructed signals.
    
    original_name : str, optional (default="Original Signal")
        The name to display for the original signal in the plot.
    
    reconstructed_name : str, optional (default="Reconstructed Signal")
        The name to display for the reconstructed signal in the plot.
    
    save_path : str, optional
        The directory path where the plot should be saved. If None, the plot will not be saved.
    
    filename : str, optional
        The name of the file to save the plot as. If None and save_path is provided, a default name will be generated.
    
    start_pct : float, optional (default=0.0)
        The percentage (between 0 and 1) of the way through the signal to start plotting. For example, 0.5 means start 
        from the halfway point of the signals.
    
    num_samples : int, optional (default=None)
        The number of samples to plot from the start point. If None, it will plot to the end of the signals.
    
    show_snr_box : bool, optional (default=False)
        Whether to display the SNR value in a text box on the plot.

    Returns
    -------
    None
        This function does not return any value. It either displays or saves the plot.

    Notes
    -----
    - Ensure the original and reconstructed signals have the same length; otherwise, a ValueError will be raised.
    - The plot shows a section of the signals starting at `start_pct` and plots `num_samples` samples.
    - The SNR can be displayed in a text box if `show_snr_box=True` and the SNR value is provided or calculated.
    
    Example
    -------
    >>> original = np.sin(np.linspace(0, 10, 100))
    >>> reconstructed = original + np.random.normal(0, 0.1, 100)
    >>> plot_signals(original, reconstructed, snr=20, save_path='./plots')
    """
    
    # Ensure the signals have the same length
    if len(original_signal) != len(reconstructed_signal):
        raise ValueError("The original signal and the reconstructed signal must have the same length.")
    
    # Calculate the start index based on percentage
    start_idx = int(start_pct * len(original_signal))
    
    # Determine the end index based on num_samples
    if num_samples is not None:
        end_idx = start_idx + num_samples
    else:
        end_idx = len(original_signal)
    
    # Ensure that the end index does not exceed the signal length
    end_idx = min(end_idx, len(original_signal))

    # Slice the signals to the selected section
    original_signal_section = original_signal[start_idx:end_idx]
    reconstructed_signal_section = reconstructed_signal[start_idx:end_idx]
    
    # Calculate SNR if not provided
    if snr is None and show_snr_box:
        snr = calculate_snr(original_signal_section, reconstructed_signal_section)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(original_signal_section, label=original_name, color='#1f77b4', linewidth=1.5)
    plt.plot(reconstructed_signal_section, label=reconstructed_name, color='#ff7f0e', linestyle='--', linewidth=1.5)
    
    # Title and labels
    plt.title(f"{original_name} vs {reconstructed_name} (Section: {start_pct*100:.1f}% - {num_samples} samples)")
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    
    # Add a legend in the upper-right corner with a white background
    plt.legend(loc='upper right', frameon=True, facecolor='white')
    
    # Display SNR in a text box in the top-left corner if show_snr_box is True
    if show_snr_box and snr is not None:
        plt.text(0.05, 0.95, f'SNR: {snr:.2f} dB', transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Grid and show plot
    plt.grid(True)
    
    # Save the plot if a save path is provided
    if save_path is not None:
        # Ensure the save directory exists
        os.makedirs(save_path, exist_ok=True)
        
        # Use provided filename or generate a default one
        if filename is None:
            filename = f"{original_name}_vs_{reconstructed_name}_section.png"
        
        # Define the file path to save the plot
        file_path = os.path.join(save_path, filename)
        plt.savefig(file_path)
        print(f"Plot saved to {file_path}")
    
    # Display the plot
    plt.show()
