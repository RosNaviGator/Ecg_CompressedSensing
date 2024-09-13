# testMeasurementMatrix.py - Testing script for the measurement_matrix module of CompSensePack

import numpy as np
from CompSensePack import generate_DBBD_matrix, generate_random_matrix
from pathlib import Path

# Test for generate_DBBD_matrix function
def test_generate_DBBD_matrix():
    print("Testing generate_DBBD_matrix...")
    
    # Test 1: Generate a DBBD matrix with M = 3, N = 9
    M, N = 3, 9
    dbbd_matrix = generate_DBBD_matrix(M, N)
    print(f"DBBD matrix (M={M}, N={N}):\n{dbbd_matrix}")
    
    # Test 2: Error case where N is not a multiple of M
    try:
        generate_DBBD_matrix(3, 10)
    except ValueError as e:
        print(f"Expected error: {e}")
    
    print("Test complete for generate_DBBD_matrix.\n")

# Test for generate_random_matrix function
def test_generate_random_matrix():
    print("Testing generate_random_matrix...")
    
    M, N = 2, 3
    
    # Gaussian matrix
    gaussian_matrix = generate_random_matrix(M, N, matrix_type='gaussian')
    print(f"Gaussian random matrix (M={M}, N={N}):\n{gaussian_matrix}\n")
    
    # Scaled binary matrix
    scaled_binary_matrix = generate_random_matrix(M, N, matrix_type='scaled_binary')
    print(f"Scaled binary random matrix (M={M}, N={N}):\n{scaled_binary_matrix}\n")
    
    # Unscaled binary matrix
    unscaled_binary_matrix = generate_random_matrix(M, N, matrix_type='unscaled_binary')
    print(f"Unscaled binary random matrix (M={M}, N={N}):\n{unscaled_binary_matrix}\n")
    
    print("Test complete for generate_random_matrix.\n")


if __name__ == "__main__":
    test_generate_DBBD_matrix()
    test_generate_random_matrix()
