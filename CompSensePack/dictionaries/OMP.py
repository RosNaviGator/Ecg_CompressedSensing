"""
OMP.py - Orthogonal Matching Pursuit (OMP) algorithm for sparse coding.

This module implements the OMP algorithm, which is widely used in compressed sensing 
and sparse signal recovery. It provides a method to represent signals as sparse combinations 
of atoms from a given dictionary.

Example usage:
--------------
>>> from OMP import OMP
>>> dictio = np.random.randn(100, 200)  # Random dictionary with 200 atoms
>>> signals = np.random.randn(100, 10)  # 10 random signals
>>> sparse_codes = OMP(dictio, signals, max_coeff=5)
"""

import numpy as np

def OMP(dictio, sig, max_coeff):
    """
    Orthogonal Matching Pursuit (OMP) algorithm for sparse coding.

    This function implements the OMP algorithm, which is used to find the sparse
    representation of a signal over a given dictionary.
    
    Parameters
    ----------
    dictio : numpy.ndarray
        The dictionary to use for sparse coding. It should be a matrix of size (n x K), 
        where n is the signal dimension and K is the number of atoms in the dictionary.
        The columns of the dictionary must be normalized (i.e., each column should have a unit norm).
    
    sig : numpy.ndarray
        The signals to represent using the dictionary. 
        It should be a matrix of size (n x N), where N is the number of signals, and n is the signal dimension.
    
    max_coeff : int
        The maximum number of coefficients (non-zero entries) to use for representing each signal.
    
    Returns
    -------
    s : numpy.ndarray
        The sparse representation of the signals over the dictionary.
        It will be a matrix of size (K x N), where K is the number of atoms in the dictionary 
        and N is the number of signals.
    
    Notes
    -----
    - This algorithm iteratively selects atoms from the dictionary that are most correlated 
      with the current residual and updates the residual at each iteration.
    - The process stops when the norm of the residual is sufficiently small or when the maximum 
      number of coefficients (`max_coeff`) has been reached.
    - The dictionary's columns **must be normalized** before using this algorithm, as the algorithm 
      relies on the assumption that the atoms are unit-norm.

    Example
    -------
    >>> dictio = np.random.randn(100, 200)  # Random dictionary with 200 atoms
    >>> signals = np.random.randn(100, 10)  # 10 random signals
    >>> sparse_codes = OMP(dictio, signals, max_coeff=5)
    """
    
    # Get dimensions of signal and dictionary
    n, p = sig.shape
    _, key = dictio.shape

    # Initialize the sparse code matrix (K x N)
    s = np.zeros((key, p))

    # Loop over each signal
    for k in range(p):
        x = sig[:, k]          # Current signal
        residual = x.copy()     # Initialize the residual
        indx = np.array([], dtype=int)  # Indices of selected atoms
        current_atoms = np.empty((n, 0))  # Matrix to store selected atoms
        norm_x = np.linalg.norm(x)  # Norm of the original signal

        # Perform OMP for `max_coeff` iterations
        for j in range(max_coeff):
            # Compute the projection of the residual onto the dictionary
            proj = dictio.T @ residual
            pos = np.argmax(np.abs(proj))  # Select the atom with the highest correlation
            indx = np.append(indx, pos)    # Add index of selected atom
            
            # Update the selected atoms matrix
            current_atoms = np.column_stack((current_atoms, dictio[:, pos]))

            # Solve least squares problem using QR decomposition
            q, r = np.linalg.qr(current_atoms)
            a = np.linalg.solve(r, q.T @ x)  # Compute the coefficients

            # Update the residual
            residual = x - current_atoms @ a

            # Break if residual norm is small relative to original signal
            if np.linalg.norm(residual) < 1e-6 * norm_x:
                break

        # Store the sparse coefficients
        temp = np.zeros((key,))
        temp[indx] = a
        s[:, k] = temp  # Store sparse code for the current signal

    return s
