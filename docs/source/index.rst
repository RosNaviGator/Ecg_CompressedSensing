Welcome to CompSensePack’s Documentation!
==========================================

Author: `Francesco Rosnati <https://github.com/RosNaviGator>`_
Date: September 2024
Version: |release|

Repository: `GitHub Repo <https://github.com/RosNaviGator/NAML_ECG_compressor>`_

Project Overview
----------------

`CompSensePack` is a Python package built to solve problems in compressed sensing, a powerful technique in signal processing. Compressed sensing exploits the fact that many real-world signals can be represented with far fewer samples than what traditional methods require, as long as the signal has an underlying sparse structure. Instead of acquiring a full set of data and then compressing it, compressed sensing allows us to directly acquire the compressed version of the data, leading to faster and more efficient data acquisition, which is particularly valuable in fields like medical imaging, wireless communication, and audio signal processing.

`CompSensePack` at the moment is focalized on `Ecg signals`, it was developed working on the `MIT-BIH Arrhythmia Database`.

At the heart of compressed sensing is the principle that if a signal is sparse in some basis (for example, in the dct or wavelet domains), it can be reconstructed from a much smaller number of measurements than what traditional Nyquist sampling theory suggests. This is achieved by designing measurement matrices that capture enough information about the signal and applying algorithms to recover the original signal from these compressed measurements.


CompSensePack
-------------

The library `CompSensePack` attemps to execute the entire 'pipeline' of `Compressed Sensing (CS)`: *compressed measurement* followed by a *reconstruction phase*.

The library can simulate a `CS` measurement, it offers various measurement matrices to experiment: deterministic *DBBD* but also randomly generated *gaussian* or *binary* (*scaled* or *unscaled*).

The idea of compress sensing is to re-build the *sparse* version of the original signal from the *compressed measurement* 

Recovery phase of `CS` has two fundamental components, the first one is the *sparsifying dictionary*: given that the original natural signal is *sparse* in some domain, the dictionary is the base of such domain. It's possible to exploit both *fixed dictionaries*, like *DCT* or *DWT* based ones, or *adaptive dictionary learning* where the dictionary is learned from a *training set* to better fit the specific data. The `CompSensePack` offer uses a *DCT*-based dictionary as a benchmark to test in which case *MOD* and *KSVD* adaptive dictionary learning algorithms can (or can't) perform better than the first.

The second component of is a *recovery method*: in order to re-build the sparse solution we need to solve the *L0* minimization problem, which literally means to find the sparsest solution that verify a given condition. Usually the problem is not possible to solve directly, it is though possible to solve *L1* minimization problem with same condition, it can be demonstrated that the solutions of the two are approximately the same given the correct conditions. The `CompSensePack` however uses teh `SL0` algorithm that directly approximates the solution of the *L0* problem. 


Contents
--------

.. toctree::
   :maxdepth: 3

   CompSensePack
