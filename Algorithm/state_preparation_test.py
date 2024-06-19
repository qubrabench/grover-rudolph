import pytest
import numpy as np

from helping_sp import generate_sparse_unit_vector, ZERO
from state_preparation import grover_rudolph
from state_preparation_circuit import permutation_GR_circuit, GR_circuit


@pytest.mark.parametrize("n_qubit", [3, 4, 5])
def test_circuit(n_qubit: int):
    N = 2**n_qubit

    for d in range(1, N):
        vector = generate_sparse_unit_vector(n_qubit, d)
        # build the respective density matrix
        full_vec = vector.toarray()
        rho_input = np.outer(full_vec.conjugate(), full_vec)

        # Build with improved grover rudolph
        rho_GR = permutation_GR_circuit(vector)

        dictionary = grover_rudolph(vector, optimization=False)
        psi_old = GR_circuit(dictionary)
        rho_old = np.outer(psi_old, psi_old)

        # assert (abs(rho_old - rho_input) < ZERO).all()

        assert (abs(rho_GR - rho_input) < ZERO).all()


@pytest.mark.parametrize("n_qubit", [3, 4, 5])
def test_optimization(n_qubit: int):
    """
    Checks if the optimization of the dictionary works as expected:
        the state of all ones should be completely merged
    """

    vector = np.ones(2**n_qubit, dtype=float)
    gate_list = grover_rudolph(vector)

    for i, gates in enumerate(gate_list):
        assert gates.keys() == {"e" * i}, "optimization failed, expected a single key"
