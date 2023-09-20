import pytest
import numpy as np

from helping_sp import generate_sparse_unit_vector, ZERO
from state_preparation import phase_angle_dict
from state_preparation_circuit import main_circuit


@pytest.mark.parametrize("n_qubit", [3, 4, 5])
def test_circuit(n_qubit: int):
    N = 2**n_qubit

    for d in range(1, N):
        full_vec = generate_sparse_unit_vector(n_qubit, d)
        # build the respective density matrix
        vect = full_vec.toarray()
        rho_input = np.outer(vect.conjugate(), vect)

        # Build with improved grover rudolph
        rho_GR = main_circuit(full_vec, n_qubit)

        assert (abs(rho_GR - rho_input) < ZERO).all()


@pytest.mark.parametrize("n_qubit", [3, 4, 5])
def test_optimization(n_qubit: int):
    """
    Checks if the optimization of the dictionary works as expected:
    the state of all ones should be completely merged"""

    vector = np.ones(2**n_qubit, dtype=float) / np.sqrt(2**n_qubit)
    nonzero_loc = np.arange(2**n_qubit)
    dictionary_list = phase_angle_dict(vector, nonzero_loc, n_qubit)

    for i in range(0, len(dictionary_list)):
        expected_keys = {"e" * i}
        assert (
            expected_keys == dictionary_list[i].keys()
        ), "optimization is not working as expected"
