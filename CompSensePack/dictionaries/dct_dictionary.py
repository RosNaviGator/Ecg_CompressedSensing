"""
dct_dictionary.py - Utility functions for generating a Discrete Cosine Transform (DCT) 
orthonormal basis matrix and testing matrix properties.

This module provides functionality to:
1. Generate a Discrete Cosine Transform (DCT) orthonormal basis matrix.
2. Use the resulting matrix for orthogonal transformations in signal processing.

The DCT basis is widely used in data compression (such as JPEG image compression) 
and signal processing due to its energy compaction properties.

Example usage:
--------------
>>> from dct_dictionary import dct_dictionary
>>> dct_matrix = dct_dictionary(4)

>>> print(dct_matrix)
array([[ 0.5       ,  0.5       ,  0.5       ,  0.5       ],
       [ 0.65328148,  0.27059805, -0.27059805, -0.65328148],
       [ 0.5       , -0.5       , -0.5       ,  0.5       ],
       [ 0.27059805, -0.65328148,  0.65328148, -0.27059805]])

"""

import numpy as np
import scipy.fftpack as fftpack

def dct_dictionary(N):
    """
    Generates a Discrete Cosine Transform (DCT) orthonormal basis matrix.

    The DCT basis transforms a signal into a sum of cosine functions oscillating at different frequencies.
    This matrix is useful in signal processing, data compression, and other areas where an orthogonal transformation 
    is needed to represent signals in terms of their frequency components.

    Parameters
    ----------
    N : int
        The size of the dictionary (i.e., the length of the signal).

    Returns
    -------
    dict_matrix : numpy.ndarray
        The generated DCT dictionary matrix of shape (N, N), where each column represents 
        a DCT basis vector.

    Raises
    ------
    ValueError
        If the input size N is not positive.

    Notes
    -----
    - The DCT basis is energy compaction-efficient, meaning most of the signal's energy is concentrated 
      in the first few coefficients, making it useful for compression.
    - The DCT basis is orthonormal, meaning its columns are orthogonal vectors of unit length.
    
    Example
    -------
    >>> dct_dictionary(4)
    array([[ 0.5       ,  0.5       ,  0.5       ,  0.5       ],
           [ 0.65328148,  0.27059805, -0.27059805, -0.65328148],
           [ 0.5       , -0.5       , -0.5       ,  0.5       ],
           [ 0.27059805, -0.65328148,  0.65328148, -0.27059805]])
    """
    
    # Input validation
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    
    # Generate the DCT orthonormal basis matrix
    dict_matrix = fftpack.dct(np.eye(N), norm='ortho')
    
    return dict_matrix
