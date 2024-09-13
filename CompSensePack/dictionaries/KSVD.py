"""
KSVD.py - K-SVD algorithm for dictionary learning and sparse coding using Orthogonal Matching Pursuit (OMP).

This module implements the K-SVD algorithm, which is widely used for dictionary learning. The algorithm
iteratively updates dictionary elements to best represent a set of input signals. This module also includes
utility functions to update dictionary elements using Singular Value Decomposition (SVD) and handle redundant
dictionary elements.

Example usage:
--------------
>>> from KSVD import KSVD
>>> data = np.random.randn(100, 200)  # Random signals
>>> params = {'K': 50, 'num_iterations': 10, 'initialization_method': 'DataElements', 'L': 5, 'preserve_dc_atom': 0}
>>> dictionary, coefficients = KSVD(data, params)
"""

import numpy as np
from scipy.sparse.linalg import svds
from .OMP import OMP  # Import OMP algorithm from the OMP module

def svds_vector(v):
    """
    Handle SVD for a vector or a 2D matrix with one dimension equal to 1.

    Parameters
    ----------
    v : numpy.ndarray
        Input vector or 2D matrix with one dimension equal to 1.
    
    Returns
    -------
    u : numpy.ndarray
        The left singular vector (normalized).
    
    s : float
        The singular value (the norm of the vector).
    
    vt : numpy.ndarray
        The right singular vector, which is always [[1]] for vectors.
    
    Raises
    ------
    ValueError
        If the input is not a vector or a 2D array with one dimension equal to 1.
    """
    v = np.asarray(v)
    
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    elif v.ndim == 2 and (v.shape[0] == 1 or v.shape[1] == 1):
        pass
    else:
        raise ValueError("Input must be a vector or a 2D array with one dimension equal to 1.")
    
    s = np.linalg.norm(v)
    if s > 0:
        u = v / s
    else:
        u = np.zeros_like(v)
    
    vt = np.array([[1]])

    return u, s, vt

def I_findBetterDictionaryElement(data, dictionary, j, coeff_matrix, numCoefUsed=1):
    """
    Update the j-th dictionary element using the current sparse representation.

    Parameters
    ----------
    data : numpy.ndarray
        The data matrix (n x N), where n is the signal dimension and N is the number of signals.
    
    dictionary : numpy.ndarray
        The current dictionary matrix (n x K), where K is the number of dictionary atoms.
    
    j : int
        The index of the dictionary element to be updated.
    
    coeff_matrix : numpy.ndarray
        The sparse coefficient matrix (K x N), representing the sparse codes for the signals.
    
    numCoefUsed : int, optional (default=1)
        The number of coefficients used in the sparse representation.

    Returns
    -------
    betterDictionaryElement : numpy.ndarray
        The updated dictionary element (vector).
    
    coeff_matrix : numpy.ndarray
        The updated coefficient matrix with the new coefficients for the updated dictionary element.
    
    newVectAdded : int
        A flag indicating if a new vector was added to the dictionary.
    """
    relevantDataIndices = np.nonzero(coeff_matrix[j, :])[0]
    if relevantDataIndices.size == 0:
        # Find the signal with the largest residual
        errorMat = data - dictionary @ coeff_matrix
        errorNormVec = np.sum(errorMat ** 2, axis=0)
        i = np.argmax(errorNormVec)
        betterDictionaryElement = data[:, i] / np.linalg.norm(data[:, i])
        betterDictionaryElement *= np.sign(betterDictionaryElement[0])
        coeff_matrix[j, :] = 0
        newVectAdded = 1
        return betterDictionaryElement, coeff_matrix, newVectAdded
    
    newVectAdded = 0
    tmpCoefMatrix = coeff_matrix[:, relevantDataIndices]
    tmpCoefMatrix[j, :] = 0
    errors = data[:, relevantDataIndices] - dictionary @ tmpCoefMatrix

    if np.min(errors.shape) <= 1:
        u, s, vt = svds_vector(errors)
        betterDictionaryElement = u
        singularValue = s
        betaVector = vt
    else:
        u, s, vt = svds(errors, k=1)
        betterDictionaryElement = u[:, 0]
        singularValue = s[0]
        betaVector = vt[0, :]

    coeff_matrix[j, relevantDataIndices] = singularValue * betaVector.T

    return betterDictionaryElement, coeff_matrix, newVectAdded

def I_clearDictionary(dictionary, coeff_matrix, data):
    """
    Clear or replace redundant dictionary elements.

    Parameters
    ----------
    dictionary : numpy.ndarray
        The dictionary matrix to be updated.
    
    coeff_matrix : numpy.ndarray
        The coefficient matrix representing the sparse codes for the data.
    
    data : numpy.ndarray
        The original data matrix.

    Returns
    -------
    dictionary : numpy.ndarray
        The updated dictionary with redundant elements replaced.
    """
    T2 = 0.99  # Coherence threshold
    T1 = 3     # Minimum number of non-zero coefficients
    K = dictionary.shape[1]
    Er = np.sum((data - dictionary @ coeff_matrix) ** 2, axis=0)
    G = dictionary.T @ dictionary
    G -= np.diag(np.diag(G))
    
    for jj in range(K):
        if np.max(G[jj, :]) > T2 or np.count_nonzero(np.abs(coeff_matrix[jj, :]) > 1e-7) <= T1:
            pos = np.argmax(Er)
            Er[pos] = 0
            dictionary[:, jj] = data[:, pos] / np.linalg.norm(data[:, pos])
            G = dictionary.T @ dictionary
            G -= np.diag(np.diag(G))
    
    return dictionary

def KSVD(data, param):
    """
    K-SVD algorithm for dictionary learning.

    The K-SVD algorithm is an iterative method for updating the dictionary and sparse coefficients
    to best represent the input signals. The dictionary is updated using the singular value decomposition (SVD)
    of the data approximation error.

    Parameters
    ----------
    data : numpy.ndarray
        The data matrix (n x N) containing N signals of dimension n.
    
    param : dict
        A dictionary containing parameters for the K-SVD algorithm:
        - 'K': int, the number of dictionary atoms.
        - 'num_iterations': int, the number of iterations to run the K-SVD algorithm.
        - 'initialization_method': str, how to initialize the dictionary ('DataElements' or 'GivenMatrix').
        - 'initial_dictionary': numpy.ndarray, the initial dictionary (if 'GivenMatrix' is used).
        - 'L': int, the number of non-zero coefficients for sparse coding (used in OMP).
        - 'preserve_dc_atom': int, flag to preserve a DC atom (default is 0).
    
    Returns
    -------
    dictionary : numpy.ndarray
        The learned dictionary of size (n x K).
    
    coef_matrix : numpy.ndarray
        The sparse coefficient matrix (K x N), representing the sparse representation of the data.
    """
    # Check if the number of data samples is smaller than the dictionary size
    if data.shape[1] < param['K']:
        print('KSVD: number of training data is smaller than the dictionary size. Trivial solution...')
        dictionary = data[:, :data.shape[1]]
        coef_matrix = np.eye(data.shape[1])
        return dictionary, coef_matrix

    # Initialize the dictionary
    dictionary = np.zeros((data.shape[0], param['K']), dtype=np.float64)    
    if param['initialization_method'] == 'DataElements':
        dictionary[:, :param['K'] - param['preserve_dc_atom']] = \
            data[:, :param['K'] - param['preserve_dc_atom']]
    elif param['initialization_method'] == 'GivenMatrix':
        dictionary[:, :param['K'] - param['preserve_dc_atom']] = \
            param['initial_dictionary'][:, :param['K'] - param['preserve_dc_atom']]

    # Iterate to update the dictionary and coefficients
    for iterNum in range(param['num_iterations']):
        coef_matrix = OMP(dictionary, data, param['L'])
        
        rand_perm = np.random.permutation(dictionary.shape[1])
        for j in rand_perm:
            betterDictElem, coef_matrix, newVectAdded = I_findBetterDictionaryElement(
                data,
                dictionary,
                j,
                coef_matrix,
                param['L']
            )
            dictionary[:, j] = betterDictElem.ravel()

        dictionary = I_clearDictionary(dictionary, coef_matrix, data)

    return dictionary, coef_matrix
