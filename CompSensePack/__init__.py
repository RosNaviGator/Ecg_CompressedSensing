# __init__.py for the main CompSensePack package

# Import functions from utils
from .utils import printFormatted, py_test_csv, load_signal_from_wfdb

# Import functions from eval
from .eval import calculate_snr, plot_signals

# Import functions from measurement_matrix
from .measurement_matrix import generate_DBBD_matrix, generate_random_matrix

# Import functions from dictionaries
from .dictionaries import (
    dct_dictionary,
    OMP,
    MOD,
    I_findDistanceBetweenDictionaries,
    KSVD,
    svds_vector,
    I_findBetterDictionaryElement,
    I_clearDictionary,
    compute_independent_columns,
    check_normalization,
    compute_coherence,
    check_matrix_properties
)

# Import SL0 from SL0.py
from .SL0 import SL0


__all__ = [
    'printFormatted',
    'py_test_csv',
    'load_signal_from_wfdb',
    'calculate_snr',
    'plot_signals',
    'generate_DBBD_matrix',
    'generate_random_matrix',
    # Functions from dictionaries module
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
    'check_matrix_properties',
    # SL0 function
    'SL0',
    # compressedSensing class
    'compressedSensing'
]
