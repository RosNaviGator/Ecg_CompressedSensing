# testSL0.py - Testing script for the SL0 algorithm in CompSensePack

import numpy as np
from CompSensePack import SL0

def test_SL0(sigma_min=0.01, sigma_decrease_factor=0.5, mu_0=2, L=3, showProgress=True):
    """
    Test the SL0 algorithm with tweakable parameters.

    Parameters
    ----------
    sigma_min : float, optional
        The minimum value of sigma for the SL0 algorithm (default: 0.01).
    
    sigma_decrease_factor : float, optional
        The factor by which sigma is reduced in each iteration (default: 0.5).
    
    mu_0 : float, optional
        Scaling factor for mu (default: 2).
    
    L : int, optional
        Number of iterations in the inner loop of SL0 (default: 3).
    
    showProgress : bool, optional
        If True, prints the value of sigma during iterations (default: True).
    """
    # Create a random measurement matrix A (50 rows, 200 columns - underdetermined system)
    A = np.random.randn(50, 200)
    
    # Generate a sparse signal s (200 elements, only a few non-zero)
    s_true = np.zeros(200)
    non_zero_indices = np.random.choice(200, size=5, replace=False)  # Randomly choose 5 indices to be non-zero
    s_true[non_zero_indices] = np.random.randn(5)  # Set those indices with random values
    
    # Generate the observation vector y
    y = A @ s_true
    
    # Run the SL0 algorithm to recover the sparse signal
    s_est = SL0(y, A, sigma_min=sigma_min, sigma_decrease_factor=sigma_decrease_factor, 
                mu_0=mu_0, L=L, showProgress=showProgress)
    
    # Output results
    print("\nTrue sparse signal (non-zero elements):\n", s_true[non_zero_indices])
    print("\nRecovered sparse signal (non-zero elements):\n", s_est[non_zero_indices])
    
    # Test how close the estimated sparse signal is to the true sparse signal
    error = np.linalg.norm(s_true - s_est)
    print(f"\nL2-norm error between true and estimated signal: {error}\n")
    
    assert error < 1e-2, "Test failed! The SL0 algorithm did not recover the signal accurately."
    print("SL0 test passed successfully!")


if __name__ == "__main__":
    # You can change these parameters to test different configurations of SL0
    test_SL0(sigma_min=0.001, sigma_decrease_factor=0.7, mu_0=2, L=3, showProgress=True)
