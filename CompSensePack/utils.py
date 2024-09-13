"""
utils.py - Utility functions for formatted printing, saving to CSV, and loading ECG signals.

This module provides utility functions for:
1. Printing NumPy matrices in a formatted way.
2. Saving matrices as CSV files to the project root's Outputs directory.
3. Loading ECG signals from the MIT-BIH Arrhythmia Database using the `wfdb` package.

All output files from this module will be stored in the Outputs directory located in the root of the project:
    (root)/Outputs/py_test_csv/ for saving CSV files.

Note:
-----
- Requires the following third-party libraries: `numpy`, `wfdb`.
- Make sure `wfdb` is properly installed and configured to access PhysioNet databases.

Example usage:
--------------
>>> import numpy as np
>>> from utils import printFormatted, py_test_csv, load_signal_from_wfdb

>>> matrix = np.array([[1.234567, 123.456789], [0.0001234, 1.2345]])
>>> printFormatted(matrix, decimals=4)
    1.2346  123.4568
    0.0001    1.2345

>>> signal, record_name = load_signal_from_wfdb('100', duration_minutes=1)
>>> print(f"ECG signal for record {record_name}: {signal[:10]}")  # Display first 10 samples
"""

# System imports
import os
from pathlib import Path

# Third-party imports
import numpy as np
import wfdb


def printFormatted(matrix, decimals=4):
    """
    Prints the matrix with formatted elements aligned in columns for improved readability.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to be printed. Should be a 2D numpy array.
    decimals : int, optional
        The number of decimal places to display for each element (default is 4).

    Returns
    -------
    None
        This function does not return any value; it prints the formatted matrix directly to the console.

    Example
    -------
    >>> import numpy as np
    >>> matrix = np.array([[1.234567, 123.456789], [0.0001234, 1.2345]])
    >>> printFormatted(matrix, decimals=4)
         1.2346  123.4568
         0.0001    1.2345

    Notes
    -----
    - This function is useful for visual inspection of numerical matrices, especially when the values
      have varying magnitudes.
    """
    # Check if the input is a 2D numpy array
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Input should be a 2D numpy array.")

    # Determine the maximum width needed to keep alignment
    max_width = max(len(f'{value:.{decimals}f}') for row in matrix for value in row)

    # Create a formatted string for each element in the matrix, ensuring alignment
    formatted_matrix = '\n'.join([' '.join([f'{value:>{max_width}.{decimals}f}' for value in row]) for row in matrix])

    # Print the formatted matrix
    print(formatted_matrix)

def py_test_csv(array, output_dir):
    """
    Save a numpy array as a CSV file in the specified output directory.

    Parameters
    ----------
    array : np.ndarray
        The input array to be saved as a CSV file.
    output_dir : str or Path
        The path to the directory where the CSV file should be saved. If the directory does not exist,
        it will be created.

    Returns
    -------
    None
        This function saves the array to a CSV file but does not return any value.

    Example
    -------
    >>> import numpy as np
    >>> array = np.random.rand(5, 3)
    >>> py_test_csv(array, './Outputs/py_test_csv')

    Notes
    -----
    - The output file will be saved in the specified directory as 'py_test.csv'.
    - The directory will be created if it doesn't exist.
    """
    # Convert the output directory to a Path object if it's a string
    output_dir = Path(output_dir)
    
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the output file path
    py_dict_path = output_dir / 'py_test.csv'
    
    # Save the array to the CSV file
    np.savetxt(py_dict_path, array, delimiter=',', fmt='%.6f')

    print(f"CSV file written to {py_dict_path}")


def load_signal_from_wfdb(record_name, duration_minutes=None):
    """
    Load an ECG signal from the MIT-BIH Arrhythmia Database using the wfdb package.

    Parameters
    ----------
    record_name : str
        The record name from the MIT-BIH Arrhythmia Database (e.g., '100' for record 100).
    duration_minutes : int, optional
        The duration of the signal to load in minutes. If None, loads the entire signal.

    Returns
    -------
    signal : np.ndarray
        The ECG signal as a numpy array (units in microvolts).
    record_name : str
        The name of the record loaded.

    Example
    -------
    >>> signal, record_name = load_signal_from_wfdb('100', duration_minutes=1)
    >>> print(f"ECG signal for record {record_name}: {signal[:10]}")  # Display first 10 samples

    Notes
    -----
    - The function uses the PhysioNet's MIT-BIH Arrhythmia Database to load ECG signals.
    - It multiplies the signal values by 1000 to convert from millivolts to microvolts.
    """
    # Load the record from the PhysioNet MIT-BIH dataset online
    record = wfdb.rdrecord(f'{record_name}', pn_dir='mitdb', channels=[0])  # Load channel 0 (MLII)
    fs = record.fs  # Get the sampling frequency

    print(f"Sampling frequency: {fs} Hz")
    print(f"Units: {record.units}")

    # Convert from millivolts to microvolts
    record.p_signal[:, 0] *= 1000

    # Convert the number of minutes to samples
    if duration_minutes is not None:
        num_samples = int(fs * duration_minutes)
        signal = record.p_signal[:num_samples, 0]  # Load specified duration
    else:
        signal = record.p_signal[:, 0]  # Load entire signal

    return signal, record_name
