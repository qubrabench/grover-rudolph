import numpy as np
import numpy.typing as npt
import cirq

from state_preparation import grover_rudolph, build_permutation
from helping_sp import (
    ControlledRotationGateMap,
    StateVector,
    number_of_qubits,
    sanitize_sparse_state_vector,
)

__all__ = ["GR_circuit", "permutation_GR_circuit"]


def GR_circuit(gate_maps: list[ControlledRotationGateMap]) -> cirq.Circuit:
    n_qubit = len(gate_maps)

    qubits = cirq.LineQubit.range(n_qubit)
    circuit = cirq.Circuit()

    for i, gate_map in enumerate(gate_maps):
        for controls, (theta, phase) in gate_map.items():
            control_qubits = []
            control_values = []
            for q, v in zip(qubits, controls):
                if v != "e":
                    control_qubits.append(q)
                    control_values.append(int(v))

            # list of controlled gates to be applied on the `i`-th qubit
            gates: list[cirq.Gate] = []
            if theta is not None:
                gates.append(cirq.Ry(rads=theta))

            if phase is not None:
                gates.append(cirq.ZPowGate(exponent=phase))

            # add all the gates to the circuit
            for gate in gates:
                circuit.append(
                    gate.on(qubits[i]).controlled_by(
                        *control_qubits, control_values=control_values
                    )
                )

    return circuit


def cycle_circuit(cycle: list[int], N_qubit: int) -> cirq.Circuit:
    """
    Given a cycle, build a circuit that permutes the corresponding computational basis vectors.
    Uses one extra ancilla at index N_qubit.

    Args:
        cycle: A single permuation cycle
        N_qubit: total number of qubits
    """
    cycle = cycle.copy()

    # Compute the list with the bit difference (xor) between each element and the following one of cycle
    difference_cycle = cycle.copy()
    difference_cycle.append(difference_cycle.pop(0))  # Move first element to the end

    difference_cycle = [
        difference_cycle[i] ^ cycle[i] for i in range(len(cycle))
    ]  # xor and binary
    difference_cycle.append(0)

    cycle.append(cycle[0])

    def to_bin(xs: list[int]) -> list[str]:
        return [bin(x)[2:].zfill(N_qubit) for x in xs]

    cycle_bin = to_bin(cycle)
    diff_bin = to_bin(difference_cycle)

    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(N_qubit)
    ancilla = cirq.LineQubit(N_qubit)

    for nonzero_loc, diff in zip(cycle_bin, diff_bin):
        circuit.append(
            cirq.X(ancilla).controlled_by(*qubits, control_values=map(int, nonzero_loc))
        )
        circuit.append(
            [cirq.X(q).controlled_by(ancilla) for q, d in zip(qubits, diff) if d == "1"]
        )

    return circuit


def permutation_GR_circuit(state: StateVector) -> cirq.Circuit:
    """
    Implement the permutation GR:
    given a list of coefficients, returns the prepared quantum state vector.

    First prepare the dense encoding in the first log(d) qubits,
    and then permute them into the right position over the entire log(N) qubits.

    Args:
        state: A complex vector proportional to the state to be prepared

    Returns:
        Output of the circuit as a density matrix (complex matrix)
    """
    state = sanitize_sparse_state_vector(state)
    dense = state.data

    circuit = GR_circuit(grover_rudolph(dense))

    permutation = build_permutation(state.nonzero()[1])
    N_qubit = number_of_qubits(state.shape[1])
    for cycle in permutation:
        circuit.append(cycle_circuit(cycle, N_qubit))

    return circuit
