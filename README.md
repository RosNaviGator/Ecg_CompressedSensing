# ECG Compressed Sensing with Kronecker Technique & Adaptive Dictionary Learning

__Date:__ September 10, 2024

__Author:__ Francesco Rosnati, HPC Engineering Master's student at Politecnico di Milano

__Course:__ Numerical Analysis for Machine Learning at Politecnico di Milano -- Professor Edie Miglio

<div style="text-align: center;">
    <img src="./.img/ECG_wave.jpg" alt="ECG graph" width="600">
</div>


## Overview

This project explores the application of **Compressed Sensing (CS)** techniques to electrocardiogram (ECG) signal processing, specifically leveraging adaptive dictionary learning methods and the Kronecker technique for sparse signal recovery. In particular the goal is to explore soultions that would fit on _portable remote_ Ecg machines which have low computational power and limited storage capacity.

### Goal
The goal of this project is to apply compressed sensing to ECG data and investigate the performance of adaptive dictionary learning (MOD and K-SVD) compared to fixed dictionaries (DCT), with and without the Kronecker technique. We aim to determine whether adaptive methods can outperform fixed dictionaries in terms of signal reconstruction quality. In particular it will be tested trying to simulate what would be possible to achieve with a _remote portable_ Ecg machine, such apparatus limits the computational power in measurement phase and has linited torage capabilites. (Recovery is not limited because it doesn't have to happen on-chip, it can be done on more powerful machines in a second moment)

The present project is mainly ispired by the work "A compressed-sensing-based compressor for ECG." by Izadi V, Shahri PK, Ahani H. <a href="#ref1">[1]</a>

### Compressed Sensing

Compressed Sensing is a method that exploits the sparsity of signals to enable recovery from fewer measurements than what classical Nyquist sampling theory requires. In essence, CS leverages the fact that many real-world signals ($`x`$) are sparse when represented in certain domains (e.g., frequency domain). Sparsity means that in such domain the signal can be represented with only a few non-zero coefficients, the _dictionary_ $`\Psi`$ can be seen as the tranformation that maps the real signal to it's sparse counterpart: $`x = \Psi s`$, where $`s`$ is the sparse version.

Given a sparse signal $`x`$, we can acquire compressed measurements $`y`$ by multiplying $`x`$ with a measurement matrix: $`y = \Phi x`$

Compressed sensing aims to recover $`x`$ from the compressed measurements $`y`$. The goal is to find the __sparsest__ vector $`s`$ that is consistent with:

$$
y = \Phi x = \Phi \Psi s
$$

- $`x \in \mathbb{R}^n`$ _real_ signal coming from sensors
- $`y \in \mathbb{R}^m`$ _compressed measurement_
- $`\Psi \in \mathbb{R}^{n \times n}`$ is the _dictionary_ (same as explained in previous section)
- $`\Phi \in \mathbb{R}^{m \times n}`$ with $m \ll n$ is the _measurement matrix_.
- $`s \in \mathbb{R}^n`$ is the _sparse representation_ of $x$ in $\Psi$

Such system of equations is __under-determined__ since there are infinitely many consistent solution $`s`$. The __sparsest solution__ is the one that satisfies:

$$
\hat{s} = \arg_{s} \min \|s\|_0 \text{ subject to } y = \Phi \Psi \alpha
$$

where $`\min \|s\|_0`$ denotes the $`\ell_0`$-pseudo-norm, given by the _non-zero entries_, also referred as the _cardinality_ of $`s`$.

The optimization is non-convex, and in general, the solution can only be found with a brute-force search that is combinatorial in $`n`$ and $`K`$. In particular, all possible $`K`$-sparse vectors in $`\mathbb{R}^n`$ must be checked; if the exact level of sparsity $`K`$ is unknown, the search is even broader. Because this search is combinatorial, solving such minimization is intractable for even moderately large $`n`$ and $`K`$, and the prospect of solving larger problems does not improve with Moore’s law of exponentially increasing computational power. <a href="#ref7">[7]</a>

### Measurement matrix
In the present project different _measurement matrices_ are explored: deterministic _DBBD_ (deterministic binary block diagonal) and _randomic_ gaussian or binary (binary both normalized and not). The idea is to test which dictionaries work best on a given the given measurement matrices.

### Dictionaries
The success of CS hinges on the sparsity of the signal. Sparsity refers to how efficiently a signal can be represented using only a few non-zero coefficients in a dictionary. A **dictionary** is a set of basis functions used to represent the signal in a sparse form. 

There are two types of dictionaries:
- **Fixed Dictionaries**: Predefined, like the **Discrete Cosine Transform (DCT)**, used as a baseline in this project.
- **Adaptive Dictionary Learning**: Methods that adapt the dictionary to the data, aiming for better representation. <a href="#ref2">[2]</a> Two key adaptive methods explored here are:
  - **MOD (Method of Optimal Directions)** <a href="#ref3">[3]</a>
  - **K-SVD (K-Singular Value Decomposition)** <a href="#ref4">[4]</a>

These adaptive methods aim to outperform fixed dictionaries by tailoring the dictionary to the specific signal.

### Solve the minimization problem: SL0 Algorithm
The **Smoothed L0 (SL0)** algorithm is employed to solve the sparse recovery problem. SL0 approximates the L0 norm, which counts the number of non-zero coefficients, with a smooth function. It minimizes this approximation while maintaining accuracy in signal recovery. <a href="#ref6">[6]</a>

### Kronecker Technique
The **Kronecker technique** is used in this project to exploit the sparsity structure of the signal. It involves constructing a Kronecker product of smaller measurement matrices, which leads to computational efficiency and better recovery performance in certain scenarios. The project evaluates the performance of the Kronecker technique in conjunction with fixed and adaptive dictionaries. <a href="#ref5">[5]</a>


## Code Overview

### Refer to the [Official Documentation Website](https://rosnavigator.github.io/NAML_ECG_compressor/) for more in depth description.

### [CompSensePack](./CompSensePack/)

The **CompSensePack** package is designed to handle the core tasks of compressed sensing and dictionary learning. It provides modular tools and high-level abstractions for applying compressive sensing techniques to various signals, with a focus on electrocardiogram (ECG) data in this project. The package is divided into two major components: the main package, which handles signal processing, sparse recovery, and general utilities, and the dictionaries subpackage, which is focused on dictionary learning techniques.

The main package offers a range of modules to perform the following tasks:

- **SL0 Algorithm**
- **Measurement Matrix Generation**
- **Utilities**
- **Evaluation and Plotting**
- **Compressed Sensing Class**

### Dictionaries Subpackage

The **Dictionaries Subpackage** is a specialized part of the library (it's contained in the _CompSensePack_), dedicated to the generation and learning of dictionaries. Dictionaries are the set of basis functions in which the signal is sparsely represented, and they are central to the performance of compressed sensing. The subpackage offers both fixed dictionaries and adaptive dictionary learning algorithms, enabling flexibility in how the signal is represented and reconstructed.

- **KSVD Dictionary Learning**
- **MOD Dictionary Learning**
- **OMP Algorithm**
- **DCT Dictionary**
- **Dictionary Utilities**

### [Scripts](./Scripts/)

The **Scripts** directory contains Python scripts to run the experiments and test the package functionality:

#### ecgStudy
Main experiments runned
  - `study_100m_signal.py`: Applies the compressed sensing techniques (DCT, MOD, K-SVD) to a specific ECG signal from the MIT-BIH Arrhythmia Database (Record 100), comparing the performance of the different methods.
  - `visualize_wfdb_signals.py`: Allows users to visualize the compressed sensing results on a signal from the MIT-BIH database using different dictionaries and measurement matrices.

#### functionalityTesting
_Debug_ scripts to test single modules
  - `testDictionaries.py`: Tests dictionary learning methods like DCT, MOD, and K-SVD.
  - `testEval.py`: Tests evaluation methods for comparing reconstructed signals.
  - `testMeasurementMatrix.py`: Tests different measurement matrices used in compressed sensing.
  - `testSL0.py`: Verifies the implementation and performance of the SL0 algorithm.
  - `testUtils.py`: Tests utility functions within the package.

## How to Use

Two methods are offered:
- Instruction to [run in local environment](#run-in-local-environment)
- Support to easily [run on colab](#jupyter-notebook-to-run-directly-on-google-colab)


### Run in local environment

#### Python3 required

### 1. Set Up a Virtual Environment (optional)
```bash
if [[ "$VIRTUAL_ENV" != "" ]]; then
    # Deactivate any active virtual environment if one exists
    deactivate
    # Remove any previous instance of .venvCsp
rm -r .venvCsp
fi
# Create and activate a new virtual environment, ensure 'python' refers to python3
python -m venv .venvCsp
source .venvCsp/bin/activate
```

### 2. Install CompSensePack (it also installs requirements)
```bash
pip install .
```

### 3. Run the Scripts
__Run from the _root_ directory of the project!__

#### Methods efficiency comparison
- Test the various _measurement matrices_, _dictionaries_, _Kronecker technique_, and so on.
- [Change the parameters at the bottom of the script](./Scripts/ecgStudy/study_100m_signal.py) in the `__main__`, feel free to experiment. It works with a single _record_, which is contained in [data](./data/) directory.
```bash
python Scripts/ecgStudy/study_100m_signal.py
``` 
#### Visualize recontructed signal
- This script will process the signals and plot reconstructed version over the original. It's possible to download any record of the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) available through the [wfdb](https://wfdb.readthedocs.io/en/latest/) library.
- [Feel free to change parameters](./Scripts/ecgStudy/visualize_wfdb_signals.py).
```bash
python Scripts/ecgStudy/visualize_wfdb_signals.py
```


### Jupyter notebook to run directly on [Google Colab](https://colab.google/)
- Just upload on colab the [notebook]. 
- __Do not use this .ipynb on your machine, as it will clone the repository again!__


## Contributing

If you'd like to contribute to this project, feel free to open issues and submit pull requests. Any improvements or additional features are welcome.


## References

1. <a id="ref1"></a> Izadi V, Shahri PK, Ahani H. "[A compressed-sensing-based compressor for ECG](./Others/Papers/ECG-compressedSensing/CS-based_ECGcompressor.pdf)." *Biomed Eng Lett.* 2020 Feb 6;10(2):299-307. doi: [10.1007/s13534-020-00148-7](https://doi.org/10.1007/s13534-020-00148-7). PMID: 32431956; PMCID: PMC7235110.

2. <a id="ref2"></a> Olshausen, B., Field, D. "[Emergence of simple-cell receptive field properties by learning a sparse code for natural images](./Others/Papers/AdaptiveDictionaryLearning/adaptiveDictOriginalArticle.pdf)." *Nature* 381, 607–609 (1996). doi: [10.1038/381607a0](https://doi.org/10.1038/381607a0).

3. <a id="ref3"></a> Engan, K., Aase, S., Husoy, J. "[Method of Optimal Directions for frame design](./Others/Papers/AdaptiveDictionaryLearning/mod.pdf)." *ICASSP, IEEE International Conference on Acoustics, Speech and Signal Processing* - Proceedings. 5. 2443 - 2446 vol.5. 1999. doi: [10.1109/ICASSP.1999.760624](https://doi.org/10.1109/ICASSP.1999.760624).

4. <a id="ref4"></a> M. Aharon, M. Elad and A. Bruckstein. "[K-SVD: An algorithm for designing overcomplete dictionaries for sparse representation](./Others/Papers/AdaptiveDictionaryLearning/ksvd.pdf)." *IEEE Transactions on Signal Processing,* vol. 54, no. 11, pp. 4311-4322, Nov. 2006. doi: [10.1109/TSP.2006.881199](https://doi.org/10.1109/TSP.2006.881199).

5. <a id="ref5"></a> Zanddizari H, Rajan S, Zarrabi H. "[Increasing the quality of reconstructed signal in compressive sensing utilizing Kronecker technique](./Others/Papers/Kronecker%20Technique/kronecker.pdf)." *Biomed Eng Lett.* 2018 Jan 31;8(2):239-247. doi: [10.1007/s13534-018-0057-4](https://doi.org/10.1007/s13534-018-0057-4). PMID: 30603207; PMCID: PMC6208527.

6. <a id="ref6"></a> H. Mohimani, M. Babaie-Zadeh and C. Jutten. "[A Fast Approach for Overcomplete Sparse Decomposition Based on Smoothed  ℓ0  Norm](./Others/Papers/ReconstructionMethods/Smooth-L0/smooth-L0.pdf)." *IEEE Transactions on Signal Processing,* vol. 57, no. 1, pp. 289-301, Jan. 2009. doi: [10.1109/TSP.2008.2007606](https://doi.org/10.1109/TSP.2008.2007606).

7. <a id="ref7"></a> Brunton, S. L., Kutz, J. N. "Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control." *Cambridge University Press,* 2019. ISBN: 9781108422093. Available at [Cambridge University Press](https://www.cambridge.org/highereducation/books/data-driven-science-and-engineering/44AE36BB7D4FD41E241ADBE6118E64BB).



