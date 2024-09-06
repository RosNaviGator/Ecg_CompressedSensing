import numpy as np
import scipy.fftpack as fftpack
import pywt

def dct_dictionary(N):
    """
    Generates a Discrete Cosine Transform (DCT) orthonormal basis matrix.

    The DCT basis is commonly used in signal processing and data compression. 
    It transforms a signal into a sum of cosine functions oscillating at different frequencies. 
    The resulting matrix can be used for orthogonal transformations of signals.
    
    DCT basis is sparifying.
    

    Parameters
    ----------
    N : int
        The size of the dictionary (i.e., the length of the signal).

    Returns
    -------
    dict_matrix : numpy.ndarray
        The generated DCT dictionary matrix of shape (N, N), where each column represents 
        a DCT basis vector.

    Example
    -------
    >>> generate_dct_dictionary(4)
    array([[ 0.5       ,  0.5       ,  0.5       ,  0.5       ],
           [ 0.65328148,  0.27059805, -0.27059805, -0.65328148],
           [ 0.5       , -0.5       , -0.5       ,  0.5       ],
           [ 0.27059805, -0.65328148,  0.65328148, -0.27059805]])
    """
    
    # Generate a DCT basis dictionary
    dict_matrix = fftpack.dct(np.eye(N), norm='ortho')    
    return dict_matrix





def generate_DWT_basis(dim, wavelet='db5', mode='per', level=None):
    """

    BEWARE: This function is not working properly!
    - Result matrix is orthonormal, but doesn't work for compressed sensing! So it's not computing
       the correct dictionary ...



    Generates a wavelet orthonormal basis matrix using the Discrete Wavelet Transform (DWT).

    The wavelet basis matrix is useful in signal processing, where it is used for analyzing
    and reconstructing signals. Each column of the matrix represents a wavelet basis vector 
    for the specified wavelet type and decomposition level.

    Parameters
    ----------
    dim : int
        The dimension of the basis (i.e., the length of the signal and the size of the matrix).
    
    wavelet : str, optional (default='db5')
        The name of the wavelet to use. Options include Daubechies wavelets ('db1' to 'db20') and
        other wavelet families available in PyWavelets.

    mode : str, optional (default='per')
        The signal extension mode to use when applying the wavelet transform. The default is
        'per' for periodic extension. Other options include 'zero', 'symmetric', etc.

    level : int, optional
        The level of decomposition to perform. If not provided, the function defaults
        to the maximum level, which is log2(dim). If a value is provided, it must be 
        an integer greater than 0 and not exceed log2(dim).

    Returns
    -------
    basis_matrix : numpy.ndarray
        The generated wavelet basis matrix of shape (dim, dim). Each column of this matrix 
        represents a wavelet basis vector for the given wavelet and decomposition level.

    Raises
    ------
    ValueError
        If `dim` is not a power of 2, or if `level` is not an integer greater than 0.

    Notes
    -----
    - The function assumes that `dim` is a power of 2. If not, a ValueError is raised.
    - If `level` is not provided, the function sets it to log2(dim) for a full decomposition.
    - The basis matrix is constructed by applying the wavelet transform to each standard 
      basis vector (i.e., the columns of the identity matrix) and concatenating the resulting 
      coefficients into the final basis matrix.

    Example
    -------
    >>> generate_dwt_basis(8, wavelet='db1', mode='symmetric')
    array([[ 0.35355339,  0.35355339,  0.35355339,  0.35355339,  0.35355339,  0.35355339,  0.35355339,  0.35355339],
           [ 0.35355339,  0.35355339, -0.35355339, -0.35355339,  0.35355339,  0.35355339, -0.35355339, -0.35355339],
           [ 0.5       , -0.5       ,  0.        ,  0.        ,  0.5       , -0.5       ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.5       , -0.5       ,  0.        ,  0.        ,  0.5       , -0.5       ],
           [ 0.70710678, -0.70710678,  0.        ,  0.        , -0.70710678,  0.70710678,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.70710678, -0.70710678,  0.        ,  0.        , -0.70710678,  0.70710678],
           [ 0.70710678,  0.70710678,  0.        ,  0.        , -0.70710678, -0.70710678,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.70710678,  0.70710678,  0.        ,  0.        , -0.70710678, -0.70710678]])
    """

    # Check if the dimension is a power of 2
    if not np.log2(dim).is_integer():
        raise ValueError("Dimension must be a power of 2.")

    # Check if level is provided and valid
    if level is None:
        print("Level is not provided, setting level to log2(dim)")
        level = int(np.log2(dim))
    elif level < 1:
        raise ValueError("Level must be an integer greater than 0.")
    elif level > int(np.log2(dim)):
        print("Level provided is greater than max_level=log2(dim); setting level to log2(dim).")
        level = int(np.log2(dim))

    # Initialize the basis matrix
    basis_matrix = np.zeros((dim, dim))
    
    for i in range(dim):
        # Apply wavelet transform to each basis vector
        coeffs = pywt.wavedec(data=np.eye(dim)[:, i], wavelet=wavelet, mode=mode, level=level, axis=0)
        
        # Flatten the coefficients and assign to the corresponding column in the basis matrix
        basis_matrix[:, i] = np.hstack(coeffs)
    
    return basis_matrix





## ------------------------------------------------------------------------------------------------
## REST OF THE FUNCTIONS ARE FOR TESTING PURPOSES
## ------------------------------------------------------------------------------------------------



def compute_independent_columns(A, tol=1e-10):
    """
    Computes the independent columns of a matrix using the QR decomposition.

    The function identifies independent columns of a given matrix `A` by performing a QR 
    decomposition. It selects columns corresponding to non-zero diagonal elements of the 
    `R` matrix, which are considered linearly independent.

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

    The function calculates the norm of each column in the matrix `A` and checks if all 
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
        An N x M matrix where coherence is to be calculated.

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

    The function checks if the matrix `A` is full rank, if its columns and rows are normalized,
    and computes the coherence of the matrix. It also prints the results.

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
