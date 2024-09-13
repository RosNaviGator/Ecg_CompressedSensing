from setuptools import setup, find_packages

setup(
    name='CompSensePack',
    version='0.1.0',
    author='Francesco Rosnati',
    author_email='francesco.rosnati@mail.polimi.it',
    description='ECG Compressed Sensing with Kronecker Technique & Adaptive Dictionary Learning',
    packages=find_packages(),  # Automatically find all packages and subpackages
    install_requires=[
        'numpy',               # List your dependencies here
        'scipy',
        'PyWavelets',
        'wfdb',
        # Add any other dependencies your package needs
    ],
    include_package_data=True,
    zip_safe=False,
)
