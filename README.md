# Grover-Rudolph Algorithm [![CI](../../actions/workflows/ci.yaml/badge.svg?branch=main)](../../actions/workflows/ci.yaml)

Work at LUH on Grover-Rudolph algorithm for state preparation.

## Usage
Add your code in folder `Algorithm/`.

Recommended structure:
```
.
├── requirements.txt       # add all dependencies here
├── data                   # data folder, not committed by default
└── Algorithm
    ├── algorithm.py       # source code for your implementation
    ├── algorithm_test.py  # unittests for above code
    └── algorithm.ipynb    # notebook with executions of the algorithm, example usages, plots, visualizations etc.
```

## Testing

If you want to edit the project or run tests, first install all dependencies using:

```sh
pip install -r requirements.txt
```

Then run `pytest` to run all tests. You can alternatively run `make` to run all tests run by the CI.

## Overview of the Algorithm

- **state_preparation**: main file. It computes the dictionary for standard and optimized Grover Rudolph, it build the permutation for Permutation Grover Rudolph and it counts the gates needed for the algorithms.
- **helping_sp**: collection of helping functions
- **state_preparation_circuit**: builds the circuit
- **state_preparation_test**: test that the algorithm works correctly 
- **Grudolph_get_data_1**: Stores gate counts in a txt as a function of the sparsity d
- **Grudolph_plot_2**: plots the previous data
