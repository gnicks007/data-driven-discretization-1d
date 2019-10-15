# Super-resolution methods for solving PDEs

Repurposed for solving stationary PDEs of interest in materials science.

# Deprecation
This code for Data Driven Discretization was developed for and used in [https://arxiv.org/abs/1808.04930]. The code is fully functional, but is no longer maintained. It was deprecated by a new implementation that can natively handle higher dimensions and is better designed to be generalized. The new code is available [here](https://github.com/google-research/data-driven-pdes). If you want to implement our method on your favorite equation, please contact the authors.

## Installation

Clone this repository and install in-place:

    git clone https://github.com/google/data-driven-discretization-1d.git
    pip install -e pde-superresolution

Note that Python 3 is required. Dependencies for the core library (including
TensorFlow) are specified in setup.py and should be installed automatically as
required.

## Running tests

From the source directory, execute each test file:

    cd data-driven-discretization-1d
    python ./pde_superresolution/integrate_test.py
    python ./pde_superresolution/training_test.py
