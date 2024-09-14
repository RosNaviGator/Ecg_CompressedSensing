"""
comp_sense_class.py - Compressed Sensing Class

This module provides a `compressedSensing` class that implements a framework for performing
compressed sensing using various measurement matrices, dictionary learning methods, and 
sparse signal recovery algorithms.

The class includes methods for signal division, compression, dictionary generation, signal
recovery, and plotting of reconstructed signals. The SL0 algorithm is used for sparse signal
recovery, and dictionary learning can be performed using DCT, MOD, or K-SVD.

Dependencies:
    - NumPy
    - CompSensePack modules (measurement_matrix, dictionaries, SL0, eval)
"""


import warnings
import numpy as np

# Importing from other modules in the CompSensePack package
from .measurement_matrix import generate_DBBD_matrix, generate_random_matrix
from .dictionaries import MOD, KSVD, dct_dictionary
from .SL0 import SL0
from .eval import calculate_snr, plot_signals


class compressedSensing:
    """
    A class to perform compressed sensing, including signal compression, dictionary learning, and signal recovery.

    The `compressedSensing` class provides an interface to apply compressed sensing on a given signal using
    a variety of measurement matrices and dictionary learning techniques, followed by signal recovery using
    the SL0 algorithm.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal, must be a valid array of real numbers.
    BLOCK_LEN : int, optional (default=16)
        The number of rows in the measurement matrix `Phi`.
    CR : int, optional (default=4)
        Compression ratio (controls the number of rows in `Phi`). Must be a positive integer and BLOCK_LEN / CR > 1.
    matrix_type : str, optional (default='gaussian')
        Type of the matrix to generate ('gaussian', 'DBBD', etc.).

    Attributes
    ----------
    signal : numpy.ndarray
        The original input signal.
    BLOCK_LEN : int
        The number of rows in the measurement matrix.
    CR : int
        Compression ratio.
    COMP_LEN : int
        Compression length (number of rows in `Phi`).
    Phi : numpy.ndarray
        The measurement matrix used for compression.
    original_phi : numpy.ndarray
        The original measurement matrix used for compression.
    training_set : numpy.ndarray or None
        The training set extracted from the original signal.
    training_matrix : numpy.ndarray or None
        The reshaped matrix for training.
    test_set : numpy.ndarray or None
        The test set extracted from the original signal.
    Y : numpy.ndarray or None
        The compressed version of the test set.
    dictionary : numpy.ndarray or None
        The generated dictionary for signal recovery.
    coeff_matrix : numpy.ndarray or None
        Coefficient matrix generated from MOD or K-SVD.
    reconstructed_signal : numpy.ndarray or None
        The reconstructed signal after SL0 recovery.
    is_kron : bool
        Whether the Kronecker compression has been activated.
    """

    def __init__(self, signal, BLOCK_LEN=16, CR=4, matrix_type='gaussian'):
        """
        Initializes the `compressedSensing` class with the provided signal, block length, and compression ratio.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal, must be a valid array of real numbers.
        BLOCK_LEN : int, optional (default=16)
            The number of rows in the measurement matrix `Phi`.
        CR : int, optional (default=4)
            Compression ratio (controls the number of rows in `Phi`). Must be a positive integer and BLOCK_LEN / CR > 1.
        matrix_type : str, optional (default='gaussian')
            Type of the matrix to generate ('gaussian', 'DBBD', etc.).
        """
        # Save original parameters
        self.ORIGINAL_BLOCK_LEN = BLOCK_LEN
        self.CR = CR
        self.matrix_type = matrix_type

        # Check if signal is valid
        if signal is None:
            raise ValueError("A signal must be provided.")
        
        # Ensure signal is a vector-like structure (array or list of real numbers)
        if not (isinstance(signal, (list, np.ndarray)) and np.issubdtype(np.array(signal).dtype, np.number)):
            raise ValueError("The signal must be a valid array or list of numerical values.")
        
        self.signal = np.array(signal)  # Convert to numpy array if it isn't already

        # Check that BLOCK_LEN and CR are valid
        if not isinstance(BLOCK_LEN, int) or BLOCK_LEN <= 0:
            raise ValueError("BLOCK_LEN must be a positive integer.")
        if not isinstance(CR, int) or CR <= 2:
            raise ValueError("CR must be a positive integer greater than 2.")
        if BLOCK_LEN % CR != 0 or BLOCK_LEN // CR <= 1:
            raise ValueError("BLOCK_LEN must be divisible by CR, and BLOCK_LEN / CR must be greater than 1.")
        
        self.BLOCK_LEN = BLOCK_LEN
        self.COMP_LEN = BLOCK_LEN // CR  # Compression length (number of rows in Phi)

        # Generate measurement matrix Phi based on the specified type
        if matrix_type == 'DBBD':
            self.Phi = generate_DBBD_matrix(self.COMP_LEN, self.BLOCK_LEN)
        else:
            self.Phi = generate_random_matrix(self.COMP_LEN, self.BLOCK_LEN, matrix_type=matrix_type)

        # Save original Phi and block length for later resets
        self.original_phi = self.Phi

        # Initialize other attributes
        self.clear()  # Initialize/reset all other attributes to their original state


    def clear(self):
        """
        Resets the class to its state after instantiation.
        
        This method resets attributes like `BLOCK_LEN`, `COMP_LEN`, `Phi`, and clears any training 
        set, coefficient matrix, and reconstructed signal from the object.
        """
        # Clear attributes obtained after the constructor
        self.BLOCK_LEN = self.ORIGINAL_BLOCK_LEN
        self.COMP_LEN = self.BLOCK_LEN // self.CR
        self.Phi = self.original_phi
        self.training_set = None
        self.training_matrix = None
        self.reconstructed_signal = None
        self.Y = None
        self.theta = None
        self.theta_pinv = None
        self.coeff_matrix = None
        self.is_kron = False



    def divide_signal(self, training_percentage):
        """
        Divides the signal into a training set and a test set based on the given percentage.

        This method splits the original signal into a training set and a test set. The training set
        is then reshaped into a matrix to be used for dictionary learning.

        Parameters
        ----------
        training_percentage : float
            The percentage of the signal to be used for training (between 0 and 1).

        Raises
        ------
        ValueError
            If the size of the training set is smaller than the block length.
        """
        training_size = int(training_percentage * len(self.signal))
        
        # Calculate the time duration in hours and minutes for both training and testing sets
        training_minutes = training_size / 360
        testing_minutes = (len(self.signal) - training_size) / 360
        
        training_hours = int(training_minutes // 60)
        training_minutes = int(training_minutes % 60)
        
        testing_hours = int(testing_minutes // 60)
        testing_minutes = int(testing_minutes % 60)

        # Print the duration for training and testing sets in hours and minutes
        print(f"Training set duration: {training_hours} hour(s) and {training_minutes} minute(s)")
        print(f"Testing set duration: {testing_hours} hour(s) and {testing_minutes} minute(s)")
        
        # Define the training and test sets
        self.training_set = self.signal[:training_size]
        self.test_set = self.signal[training_size:]

        # Ensure the test set size is a multiple of BLOCK_LEN by truncating the test set
        test_size = len(self.test_set)
        test_size_multiple = (test_size // self.BLOCK_LEN) * self.BLOCK_LEN
        self.test_set = self.test_set[:test_size_multiple]

        # Ensure the training set size is a multiple of BLOCK_LEN
        num_cols = training_size // self.BLOCK_LEN
        if num_cols < self.BLOCK_LEN:
            warnings.warn("The number of samples (columns) in the training matrix is shorter than "
                        "the number of rows, which can cause issues with dictionary learning.")

        # Reshape the training set using Fortran-style ordering ('F')
        self.training_matrix = self.training_set[:num_cols * self.BLOCK_LEN].reshape(self.BLOCK_LEN, num_cols, order='F')

        # print training matrix shape
        print(f"Training matrix shape: {self.training_matrix.shape}")


    def compress_test_set(self):
        """
        Compresses the test set using the original measurement matrix (`original_phi`).

        This method applies the measurement matrix `Phi` to the test set to generate the compressed signal `Y`.

        Raises
        ------
        RuntimeError
            If the test set has already been compressed.
        ValueError
            If the test set is not defined (i.e., if `divide_signal()` hasn't been called).
        """
        if self.Y is not None:
            raise RuntimeError("Test set has already been compressed. Recompression is not allowed.")
        
        if self.test_set is None:
            raise ValueError("Test set not defined. Please divide the signal before compressing.")

        M, N = self.original_phi.shape  # Use original_phi instead of Phi
        SIGNAL_BLOCKS = len(self.test_set) // N
        self.Y = np.zeros((M, SIGNAL_BLOCKS))

        # Sampling phase: Compress signal block-wise
        for i in range(SIGNAL_BLOCKS):
            self.Y[:, i] = self.original_phi @ self.test_set[i * N: (i + 1) * N]  # Use original_phi here



    def Y_kron(self):
        """
        Reshapes the compressed signal `Y` into its Kronecker form by concatenating multiple columns.

        Raises
        ------
        ValueError
            If the Kronecker factor (`KRON_FACT`) is not set or if the signal hasn't been compressed.
        """
        if not hasattr(self, 'KRON_FACT') or self.KRON_FACT is None:
            raise ValueError("KRON_FACT has not been set. Please activate the Kronecker method before reshaping Y.")
        
        # Generate Y_kron from Y by concatenating KRON_FACT consecutive columns
        M, SIGNAL_BLOCKS = self.Y.shape
        SIGNAL_BLOCKS_KRON = len(self.test_set) // self.BLOCK_LEN  # BLOCK_LEN must be already "the kronecker one"
        temp_y = np.zeros((M * self.KRON_FACT, SIGNAL_BLOCKS_KRON))

        for i in range(SIGNAL_BLOCKS_KRON):
            temp_y[:, i] = self.Y[:, i * self.KRON_FACT: (i + 1) * self.KRON_FACT].flatten(order='F')
        
        self.Y = temp_y


    def kronecker_activate(self, KRON_FACT):
        """
        Activates Kronecker compression, adjusting `BLOCK_LEN` and reprocessing `Phi` and `Y`.

        Parameters
        ----------
        KRON_FACT : int
            The Kronecker factor to use for compression.
        
        Raises
        ------
        ValueError
            If Kronecker compression has already been activated or if the signal hasn't been compressed.
        """
        if hasattr(self, 'is_kron') and self.is_kron:
            raise ValueError("Kronecker compression has already been activated. Cannot activate again.")

        if self.Y is None:
            raise ValueError("Y has not been computed. Please compress the signal before activating the kronecker method.")

        self.dictionary, self.coeff_matrix = None, None  # Clear dictionary and coefficients
        
        self.KRON_FACT = KRON_FACT
        self.BLOCK_LEN = self.BLOCK_LEN * self.KRON_FACT

        # Compute Kronecker product for Phi
        self.Phi = np.kron(np.eye(self.KRON_FACT), self.Phi)

        # Reprocess the training set if it exists
        if self.training_set is not None:
            training_size = len(self.training_set)
            num_cols = training_size // self.BLOCK_LEN
            if num_cols < self.BLOCK_LEN:
                warnings.warn("The number of samples (columns) in the training matrix is shorter than "
                            "the number of rows, which can cause issues with dictionary learning.")
            
            self.training_matrix = self.training_set[:num_cols * self.BLOCK_LEN].reshape(self.BLOCK_LEN, num_cols, order='F')

            # print training matrix shape
            print(f"KRONECKER ACTIVATE Training matrix shape: {self.training_matrix.shape}")
        
        self.Y_kron()

        # Set the is_kron flag to True
        self.is_kron = True


    def generate_dictionary(self, dictionary_type='dct', mod_params=None, ksvd_params=None):
        """
        Generates the dictionary based on the specified type (DCT, MOD, or K-SVD).

        Parameters
        ----------
        dictionary_type : str
            The type of dictionary to generate ('dct', 'mod', 'ksvd').
        mod_params : dict, optional
            Dictionary of parameters for the MOD algorithm, if using MOD.
        ksvd_params : dict, optional
            Dictionary of parameters for the K-SVD algorithm, if using K-SVD.
        
        Raises
        ------
        ValueError
            If the training matrix is not defined or if the necessary parameters for MOD/K-SVD are not provided.
        """
        if dictionary_type == 'dct':
            self.dictionary = dct_dictionary(self.BLOCK_LEN)
        elif dictionary_type == 'mod':
            if self.training_matrix is None:
                raise ValueError("Training matrix not defined. Please divide the signal before running MOD.")
            if mod_params is None:
                raise ValueError("MOD parameters not provided.")
            
            # Compute 'K' using redundancy factor and BLOCK_LEN
            mod_params['K'] = mod_params['redundancy'] * self.BLOCK_LEN
            
            # Run MOD algorithm with training matrix and mod_params
            self.dictionary, self.coeff_matrix = MOD(self.training_matrix, mod_params)
        elif dictionary_type == 'ksvd':
            if self.training_matrix is None:
                raise ValueError("Training matrix not defined. Please divide the signal before running K-SVD.")
            if ksvd_params is None:
                raise ValueError("K-SVD parameters not provided.")
            
            # Compute 'K' using redundancy factor and BLOCK_LEN
            ksvd_params['K'] = ksvd_params['redundancy'] * self.BLOCK_LEN
            
            # Run K-SVD algorithm with training matrix and ksvd_params
            self.dictionary, self.coeff_matrix = KSVD(self.training_matrix, ksvd_params)
        else:
            raise ValueError("Unsupported dictionary type. Use 'dct', 'mod', or 'ksvd'.")

    def recover_signal(self, sl0_params=None):
        """
        Recovers the original signal using the SL0 algorithm after compressing the test set.

        Parameters
        ----------
        sl0_params : dict, optional
            Parameters for the SL0 algorithm, including 'sigma_min', 'sigma_decrease_factor', 'mu_0', 'L', and 'showProgress'.

        Raises
        ------
        ValueError
            If the test set has not been compressed or if the dictionary has not been generated.
        """
        if self.Y is None:
            raise ValueError("Test set has not been compressed. Please compress the signal first.")
        if self.dictionary is None:
            raise ValueError("Dictionary has not been generated. Please generate a dictionary before recovery.")
        
        M, N = self.Phi.shape
        SIGNAL_BLOCKS = self.Y.shape[1]
        reconstructed_signal = np.zeros(N * SIGNAL_BLOCKS)

        # Precompute theta and theta_pinv
        self.theta = self.Phi @ self.dictionary
        self.theta_pinv = np.linalg.pinv(self.theta)

        # Set default SL0 parameters and update with user-provided values
        default_sl0_params = {
            'sigma_min': 1e-4,
            'sigma_decrease_factor': 0.5,
            'mu_0': 2,
            'L': 3,
            'showProgress': False
        }
        
        if sl0_params is not None:
            default_sl0_params.update(sl0_params)

        # SL0 recovery for each block
        for i in range(SIGNAL_BLOCKS):
            y = self.Y[:, i]

            # SL0: Sparse reconstruction using the parameters
            xp = SL0(
                y, self.theta,
                sigma_min=default_sl0_params['sigma_min'],
                sigma_decrease_factor=default_sl0_params['sigma_decrease_factor'],
                mu_0=default_sl0_params['mu_0'],
                L=default_sl0_params['L'],
                A_pinv=self.theta_pinv,
                showProgress=default_sl0_params['showProgress']
            )

            # Recovery Phase: Reconstruct the original signal
            reconstructed_signal[i * N : (i + 1) * N] = self.dictionary @ xp

        # Store the reconstructed signal as an attribute
        self.reconstructed_signal = reconstructed_signal



    def plot_reconstructed_vs_original(self, save_path=None, filename=None, start_pct=0.0, num_samples=None, 
                                   reconstructed_label="Reconstructed Signal", show_snr_box=False):
        """
        Plots the original test set against the reconstructed signal, with an option to display the SNR.

        Parameters
        ----------
        save_path : str, optional
            Directory path where the plot should be saved.
        filename : str, optional
            Name of the file to save the plot as.
        start_pct : float, optional (default=0.0)
            The percentage of the way through the signal to start plotting.
        num_samples : int, optional
            The number of samples to plot from the start point.
        reconstructed_label : str, optional (default="Reconstructed Signal")
            Label for the reconstructed signal in the plot.
        show_snr_box : bool, optional (default=False)
            Whether to display the SNR value in a text box on the plot.

        Raises
        ------
        ValueError
            If the reconstructed signal or test set is not found.
        """
        if self.reconstructed_signal is None:
            raise ValueError("Reconstructed signal not found. Please call recover_signal() first.")
        
        if self.test_set is None:
            raise ValueError("Test set not found. Please divide the signal before plotting.")

        # Check if the lengths of the signals match
        if len(self.test_set) != len(self.reconstructed_signal):
            warnings.warn("The original and reconstructed signals have different lengths. "
                        "They will both be plotted up to the length of the shorter signal.")

        # Calculate the start index based on percentage
        total_samples = min(len(self.test_set), len(self.reconstructed_signal))
        start_idx = int(start_pct * total_samples)

        # Determine the end index based on num_samples or plot till the end if num_samples is None
        if num_samples is None:
            end_idx = total_samples
        else:
            end_idx = min(start_idx + num_samples, total_samples)

        # Slice the signals for plotting
        original_signal_section = self.test_set[start_idx:end_idx]
        reconstructed_signal_section = self.reconstructed_signal[start_idx:end_idx]

        # Calculate SNR between the original test set and the reconstructed signals
        snr = calculate_snr(original_signal_section, reconstructed_signal_section) if show_snr_box else None

        # Plot the selected section of the signals and display SNR
        plot_signals(
            original_signal_section, 
            reconstructed_signal_section, 
            snr=snr, 
            original_name="Original Signal",  # Fixed label for original signal
            reconstructed_name=reconstructed_label,  # Custom label for reconstructed signal
            save_path=save_path, 
            filename=filename,
            start_pct=start_pct,
            num_samples=len(original_signal_section),  # Update plot title with actual number of samples being plotted
            show_snr_box=show_snr_box  # Use the SNR box toggle
        )


    def get_measurement_matrix(self):
        """Retrieves the measurement matrix Phi."""
        return self.Phi

    def get_compressed_signal(self):
        """Retrieves the compressed signal Y."""
        return self.Y

    def get_dictionary(self):
        """Retrieves the generated dictionary."""
        return self.dictionary

    def get_coeff_matrix(self):
        """
        Retrieves the coefficient matrix generated by MOD or K-SVD.
        
        Raises
        ------
        ValueError
            If the coefficient matrix has not been generated.
        """

        if self.coeff_matrix is None:
            raise ValueError("The coefficient matrix has not been generated yet. Call generate_dictionary() with MOD or K-SVD first.")
        return self.coeff_matrix

    def get_reconstructed_signal(self):
        """
        Retrieves the reconstructed signal after SL0 recovery.
        
        Raises
        ------
        ValueError
            If the signal has not been reconstructed.
        """
        if self.reconstructed_signal is None:
            raise ValueError("The signal has not been reconstructed yet. Call recover_signal() first.")
        return self.reconstructed_signal

    def get_original_signal(self):
        """Retrieves the original test signal that was passed to the class."""
        return self.test_set

    def get_theta(self):
        """
        Retrieves theta (Phi @ dictionary).
        
        Raises
        ------
        ValueError
            If theta has not been computed yet.
        """
        if self.theta is None:
            raise ValueError("theta has not been computed yet. Call recover_signal() first.")
        return self.theta

    def get_theta_pinv(self):
        """
        Retrieves theta_pinv (pseudoinverse of Phi @ dictionary).
        
        Raises
        ------
        ValueError
            If theta_pinv has not been computed yet.
        """
        if self.theta_pinv is None:
            raise ValueError("theta_pinv has not been computed yet. Call recover_signal() first.")
        return self.theta_pinv
    
    def get_test_set(self):
        """
        Retrieves the test set.
        
        Raises
        ------
        ValueError
            If the test set is not defined.
        """
        if self.test_set is None:
            raise ValueError("Test set not defined. Please divide the signal before retrieving.")
        return self.test_set

    def get_training_set(self):
        """
        Retrieves the training set.
        
        Raises
        ------
        ValueError
            If the training set is not defined.
        """
        if self.training_set is None:
            raise ValueError("Training set not defined. Please divide the signal before retrieving.")
        return self.training_set

    def get_snr(self):
        """
        Computes and returns the Signal-to-Noise Ratio (SNR) between the original test set and the reconstructed signal.

        Raises
        ------
        ValueError
            If the test set or reconstructed signal is not found.
        """
        if self.test_set is None:
            raise ValueError("Test set not found. Please divide the signal before computing SNR.")
        
        if self.reconstructed_signal is None:
            raise ValueError("Reconstructed signal not found. Please call recover_signal() first.")
        
        # Ensure both signals have the same length by truncating to the shorter one
        total_samples = min(len(self.test_set), len(self.reconstructed_signal))
        
        original_signal_section = self.test_set[:total_samples]
        reconstructed_signal_section = self.reconstructed_signal[:total_samples]
        
        return calculate_snr(original_signal_section, reconstructed_signal_section)

    def extract_model(self):
        """
        Extracts the current state of the model for future recovery.

        This method creates a snapshot of the important model parameters, including
        the measurement matrix (`Phi`), dictionary, Kronecker compression status, and
        the original measurement matrix (`original_phi`).

        Returns
        -------
        model : dict
            A dictionary containing the following keys:
            - 'phi' : numpy.ndarray
                The current measurement matrix being used (`self.Phi`).
            - 'dict' : numpy.ndarray or None
                The generated dictionary used for signal recovery (`self.dictionary`).
            - 'is_kron' : bool
                A flag indicating whether Kronecker compression is activated.
            - 'original_phi' : numpy.ndarray
                The original measurement matrix before any Kronecker adjustments were made.
        """
        return {
            'phi': self.Phi,
            'dict': self.dictionary,
            'is_kron': self.is_kron,  # Use the is_kron flag directly
            'original_phi': self.original_phi  # Always return the original version of Phi
        }

