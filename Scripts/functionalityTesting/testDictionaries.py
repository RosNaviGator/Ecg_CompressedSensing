# testDictionaries.py - Testing script for the dictionaries module of CompSensePack

import numpy as np
from CompSensePack.dictionaries import (
    dct_dictionary, OMP, MOD, KSVD, I_findDistanceBetweenDictionaries, 
    compute_independent_columns, check_normalization, compute_coherence, 
    check_matrix_properties
)


def test_dct_dictionary():
    print("Testing dct_dictionary...")
    N = 4
    dct_matrix = dct_dictionary(N)
    print(f"DCT dictionary of size {N}x{N}:\n{dct_matrix}\n")
    print("Test complete for dct_dictionary.\n")


def test_OMP():
    print("Testing OMP...")
    dictio = np.random.randn(10, 20)  # Random dictionary with 20 atoms
    signals = np.random.randn(10, 5)  # 5 random signals
    sparse_codes = OMP(dictio, signals, max_coeff=5)
    print(f"OMP sparse codes:\n{sparse_codes}\n")
    print("Test complete for OMP.\n")


def test_MOD():
    print("Testing MOD...")
    data = np.random.randn(10, 100)  # Random signals
    params = {'K': 20, 'num_iterations': 5, 'initialization_method': 'DataElements', 'L': 3}
    dictionary, coefficients = MOD(data, params)
    print(f"MOD dictionary:\n{dictionary}\nMOD coefficients:\n{coefficients}\n")
    print("Test complete for MOD.\n")


def test_KSVD():
    print("Testing KSVD...")
    data = np.random.randn(10, 100)  # Random signals
    params = {'K': 20, 'num_iterations': 5, 'initialization_method': 'DataElements', 'L': 3, 'preserve_dc_atom': 0}
    dictionary, coefficients = KSVD(data, params)
    print(f"KSVD dictionary:\n{dictionary}\nKSVD coefficients:\n{coefficients}\n")
    print("Test complete for KSVD.\n")


def test_dictionary_utils():
    print("Testing dictionary_utils functions...")
    
    # Test for compute_independent_columns
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    ind_cols = compute_independent_columns(A)
    print(f"Independent columns of A:\n{ind_cols}\n")
    
    # Test for check_normalization
    B = np.array([[1, 0], [0, 1]])
    is_normalized = check_normalization(B)
    print(f"Is matrix B normalized?: {is_normalized}\n")
    
    # Test for compute_coherence
    coherence = compute_coherence(B)
    print(f"Coherence of matrix B: {coherence}\n")
    
    # Test for check_matrix_properties
    check_matrix_properties(A)
    
    print("Test complete for dictionary_utils.\n")


if __name__ == "__main__":
    # Run all tests
    test_dct_dictionary()
    test_OMP()
    test_MOD()
    test_KSVD()
    test_dictionary_utils()
