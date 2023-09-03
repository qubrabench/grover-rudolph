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
