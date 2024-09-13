# testUtils.py - Testing script for the utils module of CompSensePack
# This script is supposed to stay in (root)/Scripts/functionalityTesting and outputs are always
# saved to (root)/testingOutputs/py_test_csv, regardless of where the script is launched from.

import numpy as np
from CompSensePack import printFormatted, py_test_csv, load_signal_from_wfdb
from pathlib import Path

# Function to determine the absolute path for (root)/testingOutputs/py_test_csv
def get_output_dir():
    # Determine the directory where this script is located (i.e., (root)/Scripts/functionalityTesting)
    script_dir = Path(__file__).resolve().parent
    
    # Navigate up to the root of the project and define the testingOutputs path
    root_dir = script_dir.parents[1]  # Go up to the project root (2 levels up from this script)
    output_dir = root_dir / 'testingOutputs' / 'py_test_csv'
    
    return output_dir

# Test for printFormatted function
def test_print_formatted():
    print("Testing printFormatted...")
    matrix = np.array([[1.234567, 123.456789], [0.0001234, 1.2345]])
    printFormatted(matrix, decimals=4)
    print("\nTest complete for printFormatted.\n")


# Test for py_test_csv function
def test_py_test_csv():
    print("Testing py_test_csv...")
    
    # Create a random 5x3 matrix
    array = np.random.rand(5, 3)
    
    # Get the output directory (root)/testingOutputs/py_test_csv
    output_dir = get_output_dir()
    
    # Call the py_test_csv function, specifying the output directory
    py_test_csv(array, output_dir)
    
    print(f"CSV file written to '{output_dir}/py_test.csv'.\n")


# Test for load_signal_from_wfdb function
def test_load_signal_from_wfdb():
    print("Testing load_signal_from_wfdb...")
    try:
        # Load 1 minute of record '100' from MIT-BIH Arrhythmia Database (if wfdb and dataset are configured)
        signal, record_name = load_signal_from_wfdb('100', duration_minutes=1)
        print(f"Successfully loaded signal for record {record_name}. First 10 samples:\n{signal[:10]}")
    except Exception as e:
        print(f"Error loading signal: {e}")
    print("\nTest complete for load_signal_from_wfdb.\n")


if __name__ == "__main__":
    # Run all tests
    test_print_formatted()
    test_py_test_csv()
    test_load_signal_from_wfdb()
