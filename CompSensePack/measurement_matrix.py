"""
measurement_matrix.py - Functions for generating deterministic and random measurement matrices.

This module provides functionality to generate:
1. Deterministic Diagonally Blocked Block Diagonal (DBBD) matrices.
2. Random matrices based on different types, such as Gaussian and binary matrices.

These matrices are commonly used in compressed sensing and other signal processing applications.

Example usage:
--------------
>>> from measurement_matrix import generate_DBBD_matrix, generate_random_matrix

>>> # Generate a DBBD matrix
>>> dbbd_matrix = generate_DBBD_matrix(3, 9)

>>> # Generate a Gaussian random matrix
>>> random_matrix = generate_random_matrix(3, 9, matrix_type='gaussian')
"""

import numpy as np

def generate_DBBD_matrix(M, N):
    """
    Generates a deterministic Diagonally Blocked Block Diagonal (DBBD) matrix.

    A DBBD matrix is a type of block diagonal matrix where each block is a square diagonal matrix.

    Parameters
    ----------
    M : int
        Number of rows in the matrix.
    N : int
        Number of columns in the matrix. Should be a multiple of M.

    Returns
    -------
    A : numpy.ndarray
        The generated DBBD matrix of shape (M, N).

    Raises
    ------
    ValueError
        If `N` is not a multiple of `M`.

    Example
    -------
    >>> generate_DBBD_matrix(3, 9)
    array([[1., 1., 1., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 1., 1., 1., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 1., 1., 1.]])
    """
    if N % M != 0:
        raise ValueError("N should be a multiple of M.")
    
    A = np.zeros((M, N))
    m = N // M
    
    for i in range(M):
        A[i, i*m:(i+1)*m] = 1

    return A


def generate_random_matrix(M, N, matrix_type='gaussian'):
    """
    Generates a random matrix based on the specified type.

    Parameters
    ----------
    M : int
        Number of rows in the matrix.
    N : int
        Number of columns in the matrix.
    matrix_type : str, optional (default='gaussian')
        The type of random matrix to generate. Options are:
        - 'gaussian': A matrix with entries drawn from a normal distribution scaled by 1/M.
        - 'scaled_binary': A matrix with binary entries (±0.5), scaled by 1/sqrt(M).
        - 'unscaled_binary': A matrix with binary entries (±1), with no scaling.

    Returns
    -------
    A : numpy.ndarray
        The generated random matrix of shape (M, N).

    Raises
    ------
    ValueError
        If `matrix_type` is not one of the supported types.

    Example
    -------
    >>> generate_random_matrix(2, 3, matrix_type='gaussian')
    array([[ 0.01, -0.02,  0.03],
           [-0.04,  0.05, -0.06]])

    >>> generate_random_matrix(2, 3, matrix_type='scaled_binary')
    array([[-0.5,  0. , -0.5],
           [ 0.5, -0.5,  0. ]])
    
    >>> generate_random_matrix(2, 3, matrix_type='unscaled_binary')
    array([[ 1., -1.,  1.],
           [-1.,  1., -1.]])
    """
    if matrix_type == 'gaussian':
        A = (1 / M) * np.random.randn(M, N)

    elif matrix_type == 'scaled_binary':
        A = np.random.binomial(1, 0.5, size=(M, N)) - 0.5
        A = (1/np.sqrt(M)) * A

    elif matrix_type == 'unscaled_binary':
        A = np.random.binomial(1, 0.5, size=(M, N)) * 2 - 1

    else:
        raise ValueError("Unsupported matrix type. Choose either 'gaussian', 'scaled_binary', or 'unscaled_binary'.")

    return A
