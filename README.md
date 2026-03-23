# Grover-Rudolph Algorithm [![CI](../../actions/workflows/ci.yaml/badge.svg?branch=main)](../../actions/workflows/ci.yaml)

Work at LUH on Grover-Rudolph algorithm for state preparation.

## Overview of the Algorithm

The code for the algorithm is in the folder `src/grover_rudolph`.

- **state_preparation**: main file. It computes the dictionary for standard and optimized Grover Rudolph, it builds the permutation for Permutation Grover Rudolph, and it counts the gates needed for the algorithms.
- **helping_sp**: collection of helping functions
- **state_preparation_circuit**: builds the circuit

In the `scripts/` folder, there are two files: one to generate data, and the other to generate the plots
- **Grudolph_get_data_1**: Stores gate counts in a txt as a function of the sparsity d
- **Grudolph_plot_2**: plots the previous data

Finally, the `tests/` folder contains a pytest check:
- **state_preparation_test**: test that the algorithm works correctly 

## Testing

If you want to edit the project or run tests, first install all dependencies using:

```sh
python -m pip install -r requirements.txt & python -m pip install -e .
```

Then run `pytest` to run all tests. You can alternatively run `make` to run all tests run by the CI.

