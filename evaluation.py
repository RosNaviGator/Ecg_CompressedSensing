import matplotlib.pyplot as plt
import numpy as np
import os

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
    - The SNR is calculated as 20 * log10(norm(original_signal) / norm(original_signal - recovered_signal)).
    - A higher SNR value indicates a better recovery, with less error relative to the original signal.
    """
    error = recovered_signal - signal
    snr = 20 * np.log10(np.linalg.norm(signal) / np.linalg.norm(error))
    
    return snr



def plot_signals(original_signal, reconstructed_signal, snr=None, original_name="Original Signal", 
                 reconstructed_name="Reconstructed Signal", save_path=None, filename=None):
    """
    Plots the original signal and the reconstructed signal on the same plot with the given names,
    displays the Signal-to-Noise Ratio (SNR) in a text box, and saves the plot to a specified directory.

    Parameters
    ----------
    original_signal : numpy.ndarray
        The original signal to be plotted.
    
    reconstructed_signal : numpy.ndarray
        The reconstructed signal to be plotted.
    
    original_name : str, optional (default="Original Signal")
        The name to display for the original signal in the plot.
    
    reconstructed_name : str, optional (default="Reconstructed Signal")
        The name to display for the reconstructed signal in the plot.
    
    save_path : str, optional
        The directory path where the plot should be saved. If None, the plot will not be saved.
    
    filename : str, optional
        The name of the file to save the plot as. If None and save_path is provided, a default name will be generated.

    snr : float, optional (default=None)
        The Signal-to-Noise Ratio to display. If None, it will be computed using the original and reconstructed signals.
    """
    
    # Ensure the signals have the same length
    if len(original_signal) != len(reconstructed_signal):
        raise ValueError("The original signal and the reconstructed signal must have the same length.")
    
    # Calculate SNR if not provided
    if snr is None:
        snr = calculate_snr(original_signal, reconstructed_signal)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(original_signal, label=original_name, color='blue', linewidth=1.5)
    plt.plot(reconstructed_signal, label=reconstructed_name, color='red', linestyle='--', linewidth=1.5)
    
    # Title and labels
    plt.title(f"{original_name} vs {reconstructed_name}")
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    
    # Add a legend
    plt.legend()
    
    # Display SNR in a text box
    plt.text(0.05, 0.95, f'SNR: {snr:.2f} dB', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Grid and show plot
    plt.grid(True)
    
    # Save the plot if a save path is provided
    if save_path is not None:
        # Ensure the save directory exists
        os.makedirs(save_path, exist_ok=True)
        
        # Use provided filename or generate a default one
        if filename is None:
            filename = f"{original_name}_vs_{reconstructed_name}.png"
        
        # Define the file path to save the plot
        file_path = os.path.join(save_path, filename)
        plt.savefig(file_path)
        print(f"Plot saved to {file_path}")
    
    # Display the plot
    plt.show()