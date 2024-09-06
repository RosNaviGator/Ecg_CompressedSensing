# system imports
import os

# third party imports
import numpy as np


def printFormatted(matrix, decimals=4):
    """
    Prints the matrix with formatted elements aligned in columns for improved readability.

    Parameters:
    ----------
    matrix : numpy array
        The matrix to be printed.
    decimals : int, optional (default=4)
        The number of decimal places for formatting the elements.

    Returns:
    -------
    None
        This function does not return any value; it prints the formatted matrix directly to the console.

    Notes:
    -----
    - The function aligns columns based on the maximum width needed for the formatted elements, ensuring the matrix is displayed neatly.
    - This function is useful for visual inspection of numerical matrices, especially those with varying magnitudes.
    
    Example:
    --------
    >>> import numpy as np
    >>> matrix = np.array([[1.234567, 123.456789], [0.0001234, 1.2345]])
    >>> print('Classic print:')
    >>> print(matrix)
    Classic print:
    [[1.2345670e+00 1.2345679e+02]
     [1.2340000e-04 1.2345000e+00]]
     
    >>> print('\nFormatted print:')
    >>> printFormatted(matrix, decimals=4)
         1.2346  123.4568
         0.0001    1.2345
    
    """

    # Determine the maximum width needed to keep alignment
    max_width = max(len(f'{value:.{decimals}f}') for row in matrix for value in row)

    # Create a formatted string for each element in the matrix, ensuring alignment
    formatted_matrix = '\n'.join([' '.join([f'{value:>{max_width}.{decimals}f}' for value in row]) for row in matrix])

    # Print the formatted matrix
    print(formatted_matrix)


def py_test_csv(array):
    """
    Save a numpy array as a CSV file in ./debugCsvPy/py_test.csv

    Parameters:
    array (numpy.ndarray): The input array to be saved as a CSV file.
    Returns:
    None
    """
    
    output_dir = 'debugCsvPy'  # Directory where CSV files will be stored
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    py_dict_path = os.path.join(output_dir, 'py_test.csv')
    np.savetxt(py_dict_path, array, delimiter=',', fmt='%.6f')

