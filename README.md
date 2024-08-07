# ECG Data Compression and Reconstruction Project

**Date:** August 7, 2024

## Project Overview

This project aims to develop a Python software for ECG data processing, compression, and reconstruction using both non-CS-based and CS-based methods. The software will include modules for data reading, visualization, structuring, compression, reconstruction, and evaluation. Currently, the project is in its conceptual stage, and the initial planning documents have been created.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Planned Features](#planned-features)
   - [Preprocessing](#preprocessing)
   - [Processing/Compression](#processingcompression)
   - [Reconstruction](#reconstruction)
   - [Evaluation of Results](#evaluation-of-results)
3. [Implementation Plan](#implementation-plan)
4. [Future Enhancements](#future-enhancements)
5. [References](#references)

## Planned Features

### Preprocessing

1. **Data Reader**
   - Purpose: To read ECG data from various formats such as CSV, TXT, etc.
   - Future Implementation: Develop functionality to read ECG data files and ensure compatibility with different data formats.

2. **Data Visualizer**
   - Purpose: To visualize ECG data, specifically the PQRST curves.
   - Future Implementation: Develop plotting functionality for ECG data visualization.

3. **Data Structurer**
   - Purpose: To organize ECG data into a structured format with defined signal lengths.
   - Future Implementation: Transform the original data into a long series of contiguous memory cells and define parameters for signal length.

### Processing/Compression

1. **Non CS-Based Compressor**
   - Purpose: To compress ECG data using traditional methods such as DWT and DCT.
   - Future Implementation: Implement transformations and apply sparsification with thresholding.

2. **CS-Based Compressor with Fixed Dictionary**
   - Purpose: To compress ECG data using compressed sensing with a fixed dictionary.
   - Future Implementation: Implement compression using CS and utilize the Kronecker technique for improved compression.

3. **CS-Based Compressor with Adaptive Dictionary Learning**
   - Purpose: To compress ECG data using compressed sensing with adaptive dictionary learning.
   - Future Implementation: Implement dictionary learning methods and apply CS with the Kronecker technique.

### Reconstruction

1. **Reconstruction for Non CS-Based Compression**
   - Purpose: To reconstruct data from non-CS-based compressed measurements.
   - Future Implementation: Develop reconstruction methods specific to non-CS-based compression.

2. **Reconstruction for CS-Based Compression**
   - Purpose: To reconstruct data from CS-based compressed measurements.
   - Future Implementation: Implement methods such as Basis Pursuit, Greedy algorithm, and smooth-L0 for reconstruction.

### Evaluation of Results

1. **Qualitative Assessment**
   - Purpose: To evaluate the quality of reconstructed ECG data.
   - Future Implementation: Use visualization tools to compare original and reconstructed signals.

2. **Quantitative Assessment**
   - Purpose: To measure the performance of compression and reconstruction.
   - Future Implementation: Develop metrics to evaluate actual compression rate, algorithm complexity, processing speed, PRD, and SNR.

## Implementation Plan

### Initial Pipeline Development

1. Develop Data Reader
2. Develop Data Visualizer
3. Develop Data Structurer
4. Implement Non CS-Based Compressor
5. Implement Reconstruction for Non CS-Based Compression
6. Implement qualitative and quantitative assessment methods
7. Ensure compatibility between all modules

### Future Enhancements

1. Add CS-Based Compressor with Fixed Dictionary
2. Add CS-Based Compressor with Adaptive Dictionary Learning
3. Implement Reconstruction for CS-Based Compression
4. Evaluate and compare with the initial pipeline

## References

- Izadi, V., Shahri, P. K., & Ahani, H. (2020). A compressed-sensing-based compressor for ECG. Biomedical Engineering Letters, 10, 299-307. [DOI:10.1007/s13534-020-00148-7](https://doi.org/10.1007/s13534-020-00148-7)
