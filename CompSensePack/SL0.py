"""
SL0.py - Smoothed L0 (SL0) Algorithm for Sparse Signal Recovery

This module contains an implementation of the Smoothed L0 (SL0) algorithm, a popular approach 
for sparse signal recovery from an underdetermined system of linear equations.

SL0 is a fast algorithm for solving the sparse coding problem by minimizing the L0-norm of the 
solution. It achieves this by iteratively smoothing the objective function and reducing sigma 
to approximate the L0-norm.

References:
-----------
- Original MATLAB authors: Massoud Babaie-Zadeh, Hossein Mohimani, August 4, 2008.
- Web-page: http://ee.sharif.ir/~SLzero
- Ported to Python by RosNaviGator in 2024.
"""

import numpy as np

def SL0(y, A, sigma_min, sigma_decrease_factor=0.5, mu_0=2, L=3, A_pinv=None, showProgress=False):
    """
    Solves the underdetermined system `A @ s = y` for the sparsest vector `s` using the Smoothed L0 (SL0) algorithm.

    Parameters
    ----------
    y : numpy.ndarray
        The observed vector (Mx1), where M is the number of rows in `A`.
    
    A : numpy.ndarray
        The measurement matrix (MxN), which should be 'wide', meaning N > M (more columns than rows).
    
    sigma_min : float
        The minimum value of `sigma`, controlling when the algorithm terminates.
    
    sigma_decrease_factor : float, optional (default=0.5)
        Factor by which `sigma` is reduced in each iteration.
    
    mu_0 : float, optional (default=2)
        Scaling factor for `mu`, where `mu = mu_0 * sigma^2`. Controls convergence rate.
    
    L : int, optional (default=3)
        Number of iterations for the inner loop (steepest descent).
    
    A_pinv : numpy.ndarray, optional
        Precomputed pseudoinverse of `A` (MxN). If not provided, it will be computed internally.
    
    showProgress : bool, optional (default=False)
        If True, prints the value of `sigma` during iterations.

    Returns
    -------
    s : numpy.ndarray
        The estimated sparse signal (Nx1) that best satisfies `A @ s = y`.

    Notes
    -----
    - The algorithm begins by initializing `s` using the pseudoinverse of `A`. 
    - It iteratively adjusts `s` to reduce the L0-norm by applying a Gaussian smoothing kernel.
    - The process continues until `sigma` reaches `sigma_min`, with `sigma` being reduced geometrically.

    Example
    -------
    >>> A = np.random.randn(50, 200)  # Random wide matrix
    >>> y = np.random.randn(50)       # Random observation vector
    >>> s = SL0(y, A, sigma_min=0.01)
    >>> print("Recovered sparse signal:", s)

    """
    
    if A_pinv is None:
        A_pinv = np.linalg.pinv(A)
        
    # Initialize the estimated sparse signal
    s = A_pinv @ y
    sigma = 2 * max(np.abs(s))

    # Define the delta function (Gaussian kernel applied to `s`)
    OurDelta = lambda s, sigma: s * np.exp(-s**2 / sigma**2)

    # Main loop: reduce sigma iteratively
    while sigma > sigma_min:
        for _ in range(L):
            delta = OurDelta(s, sigma)
            s = s - mu_0 * delta
            s = s - A_pinv @ (A @ s - y)
        
        if showProgress:
            print(f'sigma: {sigma}')
        
        # Decrease sigma geometrically
        sigma *= sigma_decrease_factor

    return s
