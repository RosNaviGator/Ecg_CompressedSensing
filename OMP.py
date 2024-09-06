"""
Orthogonal Matching Pursuit (OMP) algorithm for sparse coding.
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
        (its columns MUST be normalized).
    
    sig : numpy.ndarray
        The signals to represent using the dictionary. 
        It should be a matrix of size (n x N), where N is the number of signals.
    
    max_coeff : int
        The maximum number of coefficients to use for representing each signal.
    
    Returns
    -------
    s : numpy.ndarray
        The sparse representation of the signals over the dictionary.
        It should be a matrix of size (K x N).
    """

    [n, p] = sig.shape
    [_, key] = dictio.shape
    s = np.zeros((key, p))
    for k in range(p):
        x = sig[:, k]
        residual = x.copy()
        indx = np.array([], dtype=int)
        current_atoms = np.empty((n, 0))
        norm_x = np.linalg.norm(x)
        for j in range(max_coeff):
            proj = dictio.T @ residual
            pos = np.argmax(np.abs(proj))
            indx = np.append(indx, pos)
            # Update selected atoms matrix
            current_atoms = np.column_stack((current_atoms, dictio[:, pos]))
            # Solve least squares problem using QR decomposition for stability
            q, r = np.linalg.qr(current_atoms)
            a = np.linalg.solve(r, q.T @ x)
            residual = x - current_atoms @ a
            # Break if norm of residual is suff small (relative to original signal)
            if np.linalg.norm(residual) < 1e-6 * norm_x:
                break
        temp = np.zeros((key,))
        temp[indx] = a
        s[:, k] = temp

    return s
