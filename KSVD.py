import numpy as np
from scipy.sparse.linalg import svds
from OMP import OMP


def svds_vector(v):
    """
    Handle SVD for a vector or a 2D matrix with one dimension equal to 1.
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
    Update the j-th dictionary element.
    """
    relevantDataIndices = np.nonzero(coeff_matrix[j, :])[0]
    if relevantDataIndices.size == 0:
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
    """
    T2 = 0.99
    T1 = 3
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
    """
    if param['preserve_dc_atom'] > 0:
        fixedDictElem = np.zeros((data.shape[0], 1))  
        fixedDictElem[:data.shape[0], 0] = 1 / np.sqrt(data.shape[0])
    else:
        fixedDictElem = np.empty((0, 0))

    if data.shape[1] < param['K']:
        print('KSVD: number of training data is smaller than the dictionary size. Trivial solution...')
        dictionary = data[:, :data.shape[1]]
        coef_matrix = np.eye(data.shape[1])
        return dictionary, coef_matrix
    
    dictionary = np.zeros((data.shape[0], param['K']), dtype=np.float64)    
    if param['initialization_method'] == 'DataElements':
        dictionary[:, :param['K'] - param['preserve_dc_atom']] = \
            data[:, :param['K'] - param['preserve_dc_atom']]
    elif param['initialization_method'] == 'GivenMatrix':
        dictionary[:, :param['K'] - param['preserve_dc_atom']] = \
            param['initial_dictionary'][:, :param['K'] - param['preserve_dc_atom']]

    if param['preserve_dc_atom']:
        tmpMat = np.linalg.lstsq(dictionary + 1e-7 * np.eye(dictionary.shape[1]), fixedDictElem, rcond=None)[0]
        dictionary -= fixedDictElem @ tmpMat

    column_norms = np.sqrt(np.sum(dictionary ** 2, axis=0))
    column_norms[column_norms < 1e-10] = 1
    dictionary /= column_norms
    dictionary *= np.sign(dictionary[0, :])

    for iterNum in range(param['num_iterations']):
        coef_matrix = OMP(
            np.hstack((fixedDictElem, dictionary)) if fixedDictElem.size > 0 else dictionary,
            data,
            param['L']
        )
        
        rand_perm = np.random.permutation(dictionary.shape[1])
        for j in rand_perm:
            betterDictElem, coef_matrix, newVectAdded = I_findBetterDictionaryElement(
                data,
                np.hstack((fixedDictElem, dictionary)) if fixedDictElem.size > 0 else dictionary,
                j + fixedDictElem.shape[1],
                coef_matrix,
                param['L']
            )

            dictionary[:, j] = betterDictElem.ravel()
            if param['preserve_dc_atom']:
                tmpCoeff = np.linalg.lstsq(betterDictElem + 1e-7, fixedDictElem, rcond=None)[0]
                dictionary[:, j] -= fixedDictElem @ tmpCoeff
                dictionary[:, j] /= np.linalg.norm(dictionary[:, j])

        dictionary = I_clearDictionary(dictionary, coef_matrix[fixedDictElem.shape[1]:, :], data)

    dictionary = np.hstack((fixedDictElem, dictionary)) if fixedDictElem.size > 0 else dictionary
    
    return dictionary, coef_matrix
