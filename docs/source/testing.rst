Testing Documentation
=====================

This section provides an overview of the testing scripts located in `Scripts/functionalityTesting`.

testUtils.py
------------

The `testUtils.py` script is designed to test the utilities provided by the `CompSensePack` package. It covers:

- Matrix printing functionality (`printFormatted`)
- CSV saving for debugging purposes (`py_test_csv`)
- ECG signal loading from the MIT-BIH Arrhythmia Database (`load_signal_from_wfdb`)


testEval.py
-----------

The `testEval.py` script tests the SNR calculation and signal plotting functions in the `CompSensePack.eval` module:

- Signal-to-Noise Ratio (SNR) calculation (`calculate_snr`)
- Signal plotting and SNR visualization (`plot_signals`)


testMeasurementMatrix.py
------------------------

The `testMeasurementMatrix.py` script tests the matrix generation functions in the `CompSensePack.measurement_matrix` module:

- Deterministic Diagonally Blocked Block Diagonal (DBBD) matrix generation (`generate_DBBD_matrix`)
- Random matrix generation (`generate_random_matrix`)
