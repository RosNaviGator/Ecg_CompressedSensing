# ECG Compressed Sensing with Kronecker Technique & Adaptive Dictionary Learning

__Date:__ September 10, 2024

__Author:__ Francesco Rosnati, HPC Engineering Master's student at Politecnico di Milano

__Course:__ Numerical Analysis for Machine Learning at Politecnico di Milano -- Professor Edie Miglio

<div style="text-align: center;">
    <img src="./.img/ECG_wave.jpg" alt="ECG graph" width="600">
</div>

## Overview

This project explores the application of **Compressed Sensing (CS)** techniques to electrocardiogram (ECG) signal processing, specifically leveraging adaptive dictionary learning methods and the Kronecker technique for sparse signal recovery. In particular the goal is to explore soultions that would fit on _portable remote_ Ecg machines which have low computational power and limited storage capacity.

### Compressed Sensing

Compressed Sensing is a method that exploits the sparsity of signals to enable recovery from fewer measurements than what classical Nyquist sampling theory requires. In essence, CS leverages the fact that many real-world signals ($ x $) are sparse when represented in certain domains (e.g., frequency domain). Sparsity means that in such domain the signal can be represented with only a few non-zero coefficients, the _dictionary_ $\Psi$ can be seen as the tranformation that maps the real signal to it's sparse counterpart: $ x = \Psi s $, where $ s $ is the sparse version.

Given a sparse signal $ x $, we can acquire compressed measurements $ y $ by multiplying $ x $ with a measurement matrix: $ y = \Phi x $

Compressed sensing aims to recover $ x $ from the compressed measurements $ y $. The goal is to find the __sparsest__ vector $s$ that is consistent with:

$$
y = \Phi x = \Phi \Psi s
$$

- $x \in \mathbb{R}^n$ _real_ signal coming from sensors
- $y \in \mathbb{R}^m$ _compressed measurement_
- $\Psi \in \mathbb{R}^{n \times n}$ is the _dictionary_ (same as explained in previous section)
- $\Phi \in \mathbb{R}^{m \times n}$ with $m \ll n$ is the _measurement matrix_.
- $s \in \mathbb{R}^n$ is the _sparse representation_ of $x$ in $\Psi$

Such system of equations is __under-determined__ since there are infinitely many consistent solution $s$. The __sparsest solution__ is the one that satisfies:

$$
\hat{s} = \arg_{s} \min \|s\|_0 \text{ subject to } y = \Phi \Psi \alpha
$$

where $\min \|s\|_0$ denotes the $\ell_0$-pseudo-norm, given by the _non-zero entries_, also referred as the _cardinality_ of $s$.

The optimization is non-convex, and in general, the solution can only be found with a brute-force search that is combinatorial in $n$ and $K$. In particular, all possible $K$-sparse vectors in $\mathbb{R}^n$ must be checked; if the exact level of sparsity $K$ is unknown, the search is even broader. Because this search is combinatorial, solving such minimization is intractable for even moderately large $n$ and $K$, and the prospect of solving larger problems does not improve with Moore’s law of exponentially increasing computational power.

### Measurement matrix
In the present project different _measurement matrices_ are explored: deterministic _DBBD_ (deterministic binary block diagonal) and _randomic_ gaussian or binary (binary both normalized and not). The idea is to test which dictionaries work best on a given the given measurement matrices.

### Dictionaries
The success of CS hinges on the sparsity of the signal. Sparsity refers to how efficiently a signal can be represented using only a few non-zero coefficients in a dictionary. A **dictionary** is a set of basis functions used to represent the signal in a sparse form. 

There are two types of dictionaries:
- **Fixed Dictionaries**: Predefined, like the **Discrete Cosine Transform (DCT)**, used as a baseline in this project.
- **Adaptive Dictionary Learning**: Methods that adapt the dictionary to the data, aiming for better representation. Two key adaptive methods explored here are:
  - **MOD (Method of Optimal Directions)**
  - **K-SVD (K-Singular Value Decomposition)**

These adaptive methods aim to outperform fixed dictionaries by tailoring the dictionary to the specific signal.

### Solve the minimization problem: SL0 Algorithm
The **Smoothed L0 (SL0)** algorithm is employed to solve the sparse recovery problem. SL0 approximates the L0 norm, which counts the number of non-zero coefficients, with a smooth function. It minimizes this approximation while maintaining accuracy in signal recovery.

### Kronecker Technique
The **Kronecker technique** is used in this project to exploit the sparsity structure of the signal. It involves constructing a Kronecker product of smaller measurement matrices, which leads to computational efficiency and better recovery performance in certain scenarios. The project evaluates the performance of the Kronecker technique in conjunction with fixed and adaptive dictionaries.

### Goal
The goal of this project is to apply compressed sensing to ECG data and investigate the performance of adaptive dictionary learning (MOD and K-SVD) compared to fixed dictionaries (DCT), with and without the Kronecker technique. We aim to determine whether adaptive methods can outperform fixed dictionaries in terms of signal reconstruction quality. In particular it will be tested trying to simulate what would be possible to achieve with a _remote portable_ Ecg machine, such apparatus limits the computational power in measurement phase and has linited torage capabilites. (Recovery is not limited because it doesn't have to happen on-chip, it can be done on more powerful machines in a second moment),


## Code Overview

### CompSensePack

The **CompSensePack** is the core Python package developed for compressive sensing and dictionary learning. It provides functionality for:
- **Signal Processing**: Tools for dividing, compressing, and reconstructing signals.
- **Dictionary Learning**: Implements fixed dictionaries (DCT) and adaptive dictionary learning methods (MOD, K-SVD).
- **SL0 Algorithm**: Used for sparse signal recovery.
- **Kronecker Technique**: Enables more efficient compression and recovery.

Full documentation is available [here](#link-to-documentation).

### Scripts

The **Scripts** directory contains Python scripts to run the experiments and test the package functionality:

- **ecgStudy**:
  - `study_100m_signal.py`: Applies the compressed sensing techniques (DCT, MOD, K-SVD) to a specific ECG signal from the MIT-BIH Arrhythmia Database (100m.mat), comparing the performance of the different methods.
  - `visualize_wfdb_signals.py`: Allows users to visualize the compressed sensing results on a signal from the MIT-BIH database using different dictionaries and measurement matrices.

- **functionalityTesting**:
  - `testDictionaries.py`: Tests dictionary learning methods like DCT, MOD, and K-SVD.
  - `testEval.py`: Tests evaluation methods for comparing reconstructed signals.
  - `testMeasurementMatrix.py`: Tests different measurement matrices used in compressed sensing.
  - `testSL0.py`: Verifies the implementation and performance of the SL0 algorithm.
  - `testUtils.py`: Tests utility functions within the package.

## How to Use

### 1. Set Up a Virtual Environment

Before running the code, create and activate a virtual environment:

# Add code to create and activate the virtual environment

### 2. Install Required Dependencies

Install the necessary Python packages by running:

# Add code to install the required dependencies

Make sure all dependencies, including `numpy`, `matplotlib`, `scipy`, and `wfdb`, are installed.

### 3. Run the Scripts

You can run the scripts to perform the experiments or test the package's functionality.

#### Example: Running `study_100m_signal.py`

To compare dictionary learning methods on an ECG signal:

# Add the command to run study_100m_signal.py

#### Example: Running `visualize_wfdb_signals.py`

To visualize the compressed sensing performance on a signal from the MIT-BIH Arrhythmia Database:

# Add the command to run visualize_wfdb_signals.py

### 4. Documentation

For more detailed information on the available functions, classes, and modules, you can refer to the full documentation [here](#link-to-documentation).

## Contributing

If you'd like to contribute to this project, feel free to open issues and submit pull requests. Any improvements or additional features are welcome.

---

__Note__: The ECG data used in this project is sourced from the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/), available on PhysioNet.
