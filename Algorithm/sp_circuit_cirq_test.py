import pytest
import numpy as np
import scipy as sp
import cirq

from helping_sp import (
    StateVector,
    sanitize_sparse_state_vector,
    generate_sparse_unit_vector,
    ZERO,
)
from sp_circuit_cirq import GR_circuit, permutation_GR_circuit


def run_cirq_circuit(
    circuit: cirq.Circuit, n_qubit: int, *, initial_state: StateVector | None = None
) -> StateVector:
    if initial_state:
        initial_state = sanitize_sparse_state_vector(initial_state).toarray()
    return (
        cirq.Simulator()
        .simulate(
            circuit,
            qubit_order=cirq.LineQubit.range(n_qubit),
            initial_state=initial_state,
        )
        .final_state_vector
    )


@pytest.mark.parametrize("n_qubit", [3, 4, 5])
def test_circuit(n_qubit: int):
    N = 2**n_qubit

    for d in range(1, N):
        vec = generate_sparse_unit_vector(n_qubit, d)
        circuit = permutation_GR_circuit(vec)
        out = run_cirq_circuit(circuit, n_qubit + 1)

        inp = sanitize_sparse_state_vector(vec).toarray().reshape((N, 1))
        inp = np.kron(inp.conj().transpose(), np.eye(2))

        # \ket{residue} = (\bra{inp} \otimes I_2) \ket{out}
        residue = inp @ out

        # assert abs(1 - sp.linalg.norm(residue)) < ZERO
