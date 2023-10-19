"""
**state_preparation** is a collection of functions to estimate the number of gates needed in the state preparation (in terms of Toffoli, 2-qbits gates and 1-qbit gates) and to build the circuit that prepares the state.
The algorithm used is Grover Rudolph.
"""
import numpy as np

from helping_sp import (
    ZERO,
    hamming_weight,
    x_gate_merging,
    optimize_dict,
    ControlledRotationGateMap,
    number_of_qubits,
    sanitize_sparse_state_vector,
    StateVector,
    RotationGate,
)

__all__ = [
    "grover_rudolph",
    "gate_count",
    "build_permutation",
    "count_cycle",
    "permutation_grover_rudolph",
    "GateCounts",
]

GateCounts = np.ndarray
"""Gate counts stored as a numpy array with three integers: Tofolli, CNOT, 1-qubit"""


def grover_rudolph(
    vector: StateVector, *, optimization: bool = True
) -> list[ControlledRotationGateMap]:
    """
    Generate a list of dictonaries for the angles given the amplitude vector
    Each dictonary is of the form:
        {key = ('0' if apply controlled on the state 0, '1' if controlled on 1, 'e' if apply identy) : value = [angle, phase]
        {'00' : [1.2, 0.]} the gate is a rotation of 1.2 and a phase gate with phase 0, controlled on the state |00>

    You are basically building the cicuit vertically, where each element of the dictionary is one layer of the circuit
    if the dictonary is in position 'i' of the list (starting from 0), its key will be of length 'i', thus the controls act on the fist i qubits

    Args:
        vector: compressed version (only non-zero elements) of the sparse state vector to be prepared
        optimization: decide if optimize the angles or not, defaults to True

    Returns:
        a sequence of controlled gates to be applied.
    """
    vector = sanitize_sparse_state_vector(vector)

    nonzero_values = vector.data
    nonzero_locations = vector.nonzero()[1]
    N_qubit = number_of_qubits(nonzero_values)

    final_gates: list[ControlledRotationGateMap] = []

    for qbit in range(N_qubit):
        new_nonzero_values = []
        new_nonzero_locations = []

        gate_operations: ControlledRotationGateMap = {}
        sparsity = len(nonzero_locations)

        phases: np.ndarray = np.angle(nonzero_values)

        i = 0
        while i in range(sparsity):
            angle: float
            phase: float

            loc = nonzero_locations[i]

            # last step of the while loop
            if i + 1 == sparsity:
                new_nonzero_locations.append(loc // 2)
                if nonzero_locations[i] % 2 == 0:
                    # if the non_zero element is at the very end of the vector
                    angle = 0.0
                    phase = -phases[i]
                    new_nonzero_values.append(nonzero_values[i])
                else:
                    # if the non_zero element is second-last
                    angle = np.pi
                    phase = phases[i]
                    new_nonzero_values.append(abs(nonzero_values[i]))
            else:
                # divide the non_zero locations in pairs
                loc0 = nonzero_locations[i]
                loc1 = nonzero_locations[i + 1]

                # if the non_zero locations are consecutive, with the first one in an even position
                if (loc1 - loc0 == 1) and (loc0 % 2 == 0):
                    new_component = np.exp(1j * phases[i]) * np.sqrt(
                        abs(nonzero_values[i]) ** 2 + abs(nonzero_values[i + 1]) ** 2
                    )
                    new_nonzero_values.append(new_component)
                    new_nonzero_locations.append(loc0 // 2)

                    angle = (
                        2
                        * np.arccos(
                            np.clip(abs(nonzero_values[i] / new_component), -1, 1)
                        )
                        if abs(new_component) > ZERO
                        else 0.0
                    )
                    phase = -phases[i] + phases[i + 1]
                    i += 1
                else:
                    # the non_zero location is on the right of the pair
                    if loc0 % 2 == 0:
                        angle = 0.0
                        phase = -phases[i]
                        new_nonzero_values.append(nonzero_values[i])
                        new_nonzero_locations.append(loc0 // 2)

                    else:
                        angle = np.pi
                        phase = phases[i]
                        new_nonzero_values.append(abs(nonzero_values[i]))
                        new_nonzero_locations.append(loc0 // 2)

            i += 1

            # add in the dictionary gate_operations if they are not zero
            if abs(angle) > ZERO or abs(phase) > ZERO:
                # number of control qubits for the current rotation gates
                num_controls = N_qubit - qbit - 1
                gate: RotationGate = (angle, phase)

                if num_controls == 0:
                    gate_operations = {"": gate}
                else:
                    controls = str(bin(loc // 2)[2:]).zfill(num_controls)
                    gate_operations[controls] = gate

        nonzero_values, nonzero_locations = (new_nonzero_values, new_nonzero_locations)

        if optimization:
            gate_operations = optimize_dict(gate_operations)

        final_gates.append(gate_operations)

    final_gates.reverse()
    return final_gates


def gate_count(total_gate_operations: list[ControlledRotationGateMap]) -> GateCounts:
    """
    Counts how many gates you need to build the circuit for Grover Rudolph (optimized or not optimized, but without permutations) in terms of elemental ones (single rotation gates, one-control-one-target
    gates on the |1⟩ state, refered as 2 qubits gates, and Toffoli gates)

    Args:
        total_gate_operations = the list of dictionaries of the form dict[str] = [float,float], where str is made of '0','1','e'

    Returns:
        A GateCounts (ndarray) object
    """
    N_toffoli = 0
    N_cnot = 0
    N_1_gate = 0

    for gate_operations in total_gate_operations:
        # Build the unitary for each dictonary
        for k in gate_operations:
            count0 = k.count("0")
            count1 = k.count("1")

            if count0 + count1 == 0:
                N_toffoli += 0
                N_cnot += 0
                N_1_gate += 1
            else:
                N_toffoli += (count0 + count1 - 1) * 2
                N_cnot += 2
                N_1_gate += 4 + (2 * count0)

        # Subtract the two x-gates that form an identity from the total count of 1-qubit gates
        N_1_gate -= 2 * x_gate_merging(gate_operations)

    return np.array([N_toffoli, N_cnot, N_1_gate])


def build_permutation(nonzero_locations: list[int]) -> list[list[int]]:
    """
    Given a classical permutation, return its cyclic decomposition
    Construct a permutation unitary that maps |i⟩ → |x_i⟩

    Args:
        nonzero_locations: the locations of nonzero elements of a sparse vector state |b⟩

    Returns:
        a list of permutation cycles
    """
    d = len(nonzero_locations)  # Sparsity
    seen = [False for i in range(d)]
    cycles = []  # list to store the permutation cycles

    # Build the cycles
    for i in range(d):
        j = nonzero_locations[i]

        if seen[i] or j == i:
            continue

        c = [i, j]
        while j < d:
            seen[j] = True
            j = nonzero_locations[j]
            c.append(j)

        # Add elements to the cycle until the end of the cycle is reached
        cycles.append(c)

    return cycles


def count_cycle(cycle: list[int], N_qubit: int) -> GateCounts:
    """
    Counts the number if universal gates in terms of Toffoli, Cnots and 1 qubit gates

    Args:
        cycle: a single permutation cycle
        N_qubit: total number of qubits?

    Returns:
        A GateCounts (ndarray) object
    """
    cycle = cycle.copy()

    length = len(cycle)
    cycle.append(cycle[0])

    # compute the two summation terms for the counting
    sum_neg = 0
    sum_diff = 0
    for i in range(length):
        sum_neg += N_qubit - hamming_weight(cycle[i])
        sum_diff += hamming_weight(cycle[i] ^ cycle[i + 1])

    sum_neg += N_qubit - hamming_weight(cycle[length])

    N_toffoli = 2 * (length + 1) * (N_qubit - 1)
    N_cnot = (2 * (length + 1)) + (2 * sum_diff)
    N_1_gate = (4 * (length + 1)) + (2 * sum_neg) + (4 * sum_diff)

    return np.array([N_toffoli, N_cnot, N_1_gate])


def permutation_grover_rudolph(
    state: StateVector, *, optimization: bool = True
) -> GateCounts:
    """
    Estimation of the number of gates needed to prepare a sparse state using permutation Grover Rudolph

    Args:
        state: the quantum state to be prepared.
        optimization: choose if you want to merge the gate (set to True by default)

    Returns:
        A GateCounts (ndarray) object
    """
    state = sanitize_sparse_state_vector(state)
    N_qubit = number_of_qubits(state.shape[1])

    vector = state.data
    nonzero_locations = state.nonzero()[1]

    # standard Grover Rudolph
    gr_gates = grover_rudolph(vector, optimization=optimization)
    count = gate_count(gr_gates)

    # Permutation algorithm
    permutation = build_permutation(nonzero_locations)

    for cycle in permutation:
        count += count_cycle(cycle, N_qubit)

    return count
