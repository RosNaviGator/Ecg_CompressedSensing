"""
CompSensePack.dictionaries package

This module contains submodules for dictionary learning algorithms like OMP, MOD, and KSVD.

Submodules:
- OMP: Orthogonal Matching Pursuit algorithm.
- MOD: Method of Optimal Directions for dictionary learning.
- KSVD: K-SVD algorithm for dictionary learning.
"""

from .OMP import OMP
from .MOD import MOD, I_findDistanceBetweenDictionaries
from .KSVD import KSVD, svds_vector, I_findBetterDictionaryElement, I_clearDictionary
from .dct_dictionary import dct_dictionary
from .dictionary_utils import compute_independent_columns, check_normalization, compute_coherence, check_matrix_properties

__all__ = [
    'OMP',
    'MOD',
    'I_findDistanceBetweenDictionaries',
    'KSVD',
    'svds_vector',
    'I_findBetterDictionaryElement',
    'I_clearDictionary',
    'dct_dictionary',
    'compute_independent_columns',
    'check_normalization',
    'compute_coherence',
    'check_matrix_properties'
]
