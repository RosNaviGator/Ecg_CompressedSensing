"""
MOD.py - Method of Optimal Directions (MOD) algorithm for dictionary learning.

This module implements the MOD algorithm, which is used for learning a dictionary for sparse coding.
It includes utility functions to measure the distance between dictionaries and to learn a dictionary
that can sparsely represent a set of signals using the OMP algorithm.

Example usage:
--------------
>>> from MOD import MOD, I_findDistanceBetweenDictionaries
>>> data = np.random.randn(100, 200)  # Random signals
>>> params = {'K': 50, 'num_iterations': 10, 'initialization_method': 'DataElements', 'L': 5}
>>> dictionary, coefficients = MOD(data, params)

>>> # Distance between original and updated dictionary
>>> ratio, total_distance = I_findDistanceBetweenDictionaries(dictionary, new_dictionary)
"""

# system imports
import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve
from .OMP import OMP  # Import OMP algorithm from the OMP module

def I_findDistanceBetweenDictionaries(original, new):
    """
    Calculates the distance between two dictionaries.

    This function compares two dictionaries by calculating the number of columns 
    that have a very low error between the original and the new dictionary.

    Parameters
    ----------
    original : numpy.ndarray
        The original dictionary.
    
    new : numpy.ndarray
        The new dictionary.

    Returns
    -------
    ratio : float
        The ratio of dictionary elements with a small error (i.e., those that 
        satisfy the condition `errorOfElement < 0.01`).
    
    totalDistances : float
        The sum of all error values between the original and new dictionary elements.
    
    Notes
    -----
    - The function modifies the new dictionary to ensure that each column starts 
      with a positive value (consistent with the original dictionary).
    
    Example
    -------
    >>> original_dict = np.random.randn(100, 50)
    >>> new_dict = np.random.randn(100, 50)
    >>> ratio, total_distance = I_findDistanceBetweenDictionaries(original_dict, new_dict)
    """
    
    catchCounter = 0
    totalDistances = 0

    # Ensure columns in new dictionary start with positive values
    for i in range(new.shape[1]):
        new[:, i] = np.sign(new[0, i]) * new[:, i]

    for i in range(original.shape[1]):
        d = np.sign(original[0, i]) * original[:, i]
        distances = np.sum(new - np.tile(d, (1, new.shape[1])), axis=0)
        index = np.argmin(distances)
        errorOfElement = 1 - np.abs(new[:, index].T @ d)
        totalDistances += errorOfElement
        catchCounter += errorOfElement < 0.01

    ratio = catchCounter / original.shape[1]
    return ratio, totalDistances


def MOD(data, parameters):
    """
    Method of Optimal Directions (MOD) algorithm for dictionary learning.

    The MOD algorithm is a method for learning a dictionary for sparse representation of signals.
    It iteratively updates the dictionary to best represent the input data with sparse coefficients
    using the Orthogonal Matching Pursuit (OMP) algorithm.

    Parameters
    ----------
    data : numpy.ndarray
        An (n x N) matrix containing N signals, each of dimension n.
    
    parameters : dict
        A dictionary containing the parameters for the MOD algorithm:
            - K : int
                The number of dictionary elements (columns) to train.
            
            - num_iterations : int
                The number of iterations to perform for dictionary learning.
            
            - initialization_method : str
                Method to initialize the dictionary. Options are:
                * 'DataElements' - Initializes the dictionary using the first K data signals.
                * 'GivenMatrix' - Initializes the dictionary using a provided matrix 
                  (requires 'initial_dictionary' key).

            - initial_dictionary : numpy.ndarray, optional
                The initial dictionary matrix to use if 'initialization_method' is 
                set to 'GivenMatrix'. It should be of size (n x K).

            - L : int
                The number of non-zero coefficients to use in OMP for sparse
                representation of each signal.

    Returns
    -------
    dictionary : numpy.ndarray
        The trained dictionary of size (n x K), where each column is a dictionary element.

    coef_matrix : numpy.ndarray
        The coefficient matrix of size (K x N), representing the sparse representation
        of the input data using the trained dictionary.
    
    Notes
    -----
    - The dictionary is initialized either from the data signals or from a provided matrix.
    - The OMP algorithm is used to solve the sparse coding step at each iteration.
    - The algorithm iteratively refines the dictionary by solving a least squares problem.
    - The columns of the dictionary are normalized to unit norm and positive first elements.
    
    Example
    -------
    >>> data = np.random.randn(100, 200)
    >>> params = {'K': 50, 'num_iterations': 10, 'initialization_method': 'DataElements', 'L': 5}
    >>> dictionary, coefficients = MOD(data, params)
    """
    
    # Check if the number of signals is smaller than the dictionary size
    if data.shape[1] < parameters['K']:
        print("MOD: number of training signals is smaller than the dictionary size. Returning trivial solution...")
        dictionary = data[:, :data.shape[1]]
        coef_matrix = np.eye(data.shape[1])  # Trivial coefficients
        return dictionary, coef_matrix

    # Initialize dictionary based on the specified method
    if parameters['initialization_method'] == 'DataElements':
        dictionary = data[:, :parameters['K']]
    elif parameters['initialization_method'] == 'GivenMatrix':
        if 'initial_dictionary' not in parameters:
            raise ValueError("initial_dictionary parameter is required when "
                             "initialization_method is set to 'GivenMatrix'.")
        dictionary = parameters['initial_dictionary']
    else:
        raise ValueError(
            "Invalid value for initialization_method. Choose 'DataElements' or 'GivenMatrix'.")

    # Convert to float64 for precision
    dictionary = dictionary.astype(np.float64)

    # Normalize dictionary columns and avoid division by zero
    column_norms = np.linalg.norm(dictionary, axis=0)
    column_norms[column_norms < 1e-10] = 1  # Prevent division by zero
    dictionary /= column_norms

    # Ensure positive first elements
    dictionary *= np.sign(dictionary[0, :])

    prev_dictionary = dictionary.copy()

    # Run MOD algorithm
    for iter_num in range(parameters['num_iterations']):
        # Step 1: Sparse coding using OMP
        coef_matrix = OMP(dictionary, data, parameters['L'])

        # Step 2: Update the dictionary
        regularization_term = 1e-7 * sp.eye(coef_matrix.shape[0])
        matrix_a = coef_matrix @ coef_matrix.T + regularization_term.toarray()

        # Use pinv for numerical stability: "lstsq" or "regularizations" could also work
        dictionary = data @ coef_matrix.T @ np.linalg.pinv(matrix_a)

        # Normalize dictionary columns and avoid division by zero
        column_norms = np.linalg.norm(dictionary, axis=0)
        column_norms[column_norms < 1e-10] = 1  # Prevent division by zero
        dictionary /= column_norms

        # Ensure positive first elements
        dictionary *= np.sign(dictionary[0, :])

        # Convergence check
        if np.linalg.norm(dictionary - prev_dictionary) < 1e-5:
            print(f"MOD converged after {iter_num + 1} iterations.")
            break

        prev_dictionary = dictionary.copy()

    return dictionary, coef_matrix
