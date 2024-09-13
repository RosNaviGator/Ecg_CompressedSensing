# __init__.py for the dictionaries module in CompSensePack

from .dct_dictionary import dct_dictionary
from .OMP import OMP
from .MOD import MOD, I_findDistanceBetweenDictionaries
from .KSVD import KSVD, svds_vector, I_findBetterDictionaryElement, I_clearDictionary
from .dictionary_utils import compute_independent_columns, check_normalization, compute_coherence, check_matrix_properties

__all__ = [
    'dct_dictionary',
    'OMP',
    'MOD',
    'I_findDistanceBetweenDictionaries',
    'KSVD',
    'svds_vector',
    'I_findBetterDictionaryElement',
    'I_clearDictionary',
    'compute_independent_columns',
    'check_normalization',
    'compute_coherence',
    'check_matrix_properties'
]
