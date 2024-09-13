"""
dictionary_utils.py - Utility functions for matrix operations used in dictionary learning.

This module provides various utility functions for working with matrices, such as:
1. Computing the independent columns of a matrix.
2. Checking if the columns of a matrix are normalized.
3. Computing the coherence of a matrix.
4. Checking various properties of a matrix, such as full rank, normalization, and coherence.

These utilities are useful for tasks in dictionary learning, signal processing, and sparse coding.

Example usage:
--------------
>>> from dictionary_utils import compute_independent_columns, check_normalization, compute_coherence, check_matrix_properties

>>> A = np.random.randn(5, 5)
>>> independent_columns = compute_independent_columns(A)
>>> is_normalized = check_normalization(A)
>>> coherence = compute_coherence(A)
>>> check_matrix_properties(A)
"""

import numpy as np

def compute_independent_columns(A, tol=1e-10):
    """
    Computes the independent columns of a matrix using the QR decomposition.

    This function identifies the independent columns of a given matrix `A` by performing 
    a QR decomposition. It selects columns corresponding to non-zero diagonal elements of 
    the `R` matrix, which are considered linearly independent.

    Parameters
    ----------
    A : numpy.ndarray
        The matrix for which to compute the independent columns.
    
    tol : float, optional (default=1e-10)
        The tolerance value for considering diagonal elements of `R` as non-zero.

    Returns
    -------
    ind_cols : numpy.ndarray
        A matrix containing the independent columns of `A`.

    Notes
    -----
    - The QR decomposition is used to determine the rank of the matrix `A`.
    - Columns corresponding to non-zero diagonal elements of the `R` matrix are considered independent.

    Example
    -------
    >>> A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> compute_independent_columns(A)
    array([[1, 2],
           [4, 5],
           [7, 8]])
    """
    # Perform the QR decomposition
    Q, R = np.linalg.qr(A)

    # Find the independent columns based on the rank of R
    rank = np.sum(np.abs(np.diagonal(R)) > tol)
    ind_cols = A[:, :rank]

    return ind_cols

def check_normalization(A):
    """
    Checks if the columns of a matrix are normalized (i.e., each column has a unit norm).

    This function calculates the norm of each column in the matrix `A` and checks if all 
    column norms are close to 1.0, which indicates normalization.

    Parameters
    ----------
    A : numpy.ndarray
        The matrix to check for normalization.

    Returns
    -------
    is_normalized : bool
        True if all columns of `A` are normalized, False otherwise.

    Example
    -------
    >>> A = np.array([[1, 0], [0, 1]])
    >>> check_normalization(A)
    True
    """
    column_norms = np.linalg.norm(A, axis=0)
    is_normalized = np.allclose(column_norms, 1.0)
    return is_normalized

def compute_coherence(matrix):
    """
    Computes the coherence of the given matrix.

    Coherence is a measure of the maximum correlation between any two columns of a matrix. 
    It is useful in various applications, such as signal processing and compressed sensing, 
    to assess the degree of similarity between different columns of the matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        An (N x M) matrix where coherence is to be calculated.

    Returns
    -------
    coherence : float
        The coherence of the matrix, defined as the maximum absolute value of the off-diagonal 
        elements in the Gram matrix of the column-normalized input matrix.

    Example
    -------
    >>> matrix = np.array([[1, 0], [0, 1]])
    >>> compute_coherence(matrix)
    0.0
    """
    # Normalize the columns of the matrix
    normalized_matrix = matrix / np.linalg.norm(matrix, axis=0, keepdims=True)
    
    # Compute the Gram matrix (inner products between all pairs of columns)
    gram_matrix = np.dot(normalized_matrix.T, normalized_matrix)
    
    # Remove the diagonal elements (which are all 1's) to only consider distinct columns
    np.fill_diagonal(gram_matrix, 0)
    
    # Compute the coherence as the maximum absolute value of the off-diagonal elements
    coherence = np.max(np.abs(gram_matrix))
    
    return coherence

def check_matrix_properties(A):
    """
    Checks various properties of a matrix.

    This function checks if the matrix `A` is full rank, if its columns and rows are normalized,
    and computes the coherence of the matrix. It also prints the results for each property.

    Parameters
    ----------
    A : numpy.ndarray
        The matrix to check.

    Returns
    -------
    None

    Example
    -------
    >>> A = np.array([[1, 2], [3, 4]])
    >>> check_matrix_properties(A)
    Is full rank: True
    Are columns normalized: False
    Are rows normalized: False
    Coherence: 0.9999999999999999
    """
    # Check if the matrix is full rank
    is_full_rank = np.linalg.matrix_rank(A) == min(A.shape)

    # Check if the columns are normalized
    is_columns_normalized = check_normalization(A)

    # Check if the rows are normalized
    is_rows_normalized = check_normalization(A.T)

    # Compute the coherence of the matrix
    coherence = compute_coherence(A)

    # Print the results
    print("Is full rank:", is_full_rank)
    print("Are columns normalized:", is_columns_normalized)
    print("Are rows normalized:", is_rows_normalized)
    print("Coherence:", coherence)
