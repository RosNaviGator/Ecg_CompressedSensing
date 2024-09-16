from setuptools import setup, find_packages

setup(
    name='CompSensePack',
    version='1.0.0',
    author='Francesco Rosnati',
    author_email='francesco.rosnati@mail.polimi.it',
    description='ECG Compressed Sensing with Kronecker Technique & Adaptive Dictionary Learning',
    packages=find_packages(),  # Automatically find all packages and subpackages
    install_requires=[
        'numpy', 
        'matplotlib',
        'pandas',
        'scipy',
        #'PyWavelets',
        'wfdb'
    ],
    include_package_data=True,
    zip_safe=False,
)
