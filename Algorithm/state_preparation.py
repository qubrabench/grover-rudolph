"""
**state_preparation** is a collection of functions to estimate the number of gates needed in the state preparation (in terms of Toffoli, 2-qbits gates and 1-qbit gates) and to build the circuit that prepares the state.
The algorithm used is Grover Rudolph.
"""
import numpy as np
import numpy.typing as npt
import scipy as sp

from helping_sp import (
    ZERO,
    hamming_weight,
    x_gate_merging,
    optimize_dict,
    ControlledRotationGateMap,
    number_of_qubits,
    sanitize_sparse_state_vector,
    StateVector,
)

__all__ = [
    "phase_angle_dict",
    "gate_count",
    "build_permutation",
    "count_cycle",
    "main",
    "GateCounts",
]

GateCounts = np.ndarray
"""Gate counts stored as a numpy array with three integers: Tofolli, CNOT, 1-qubit"""


def phase_angle_dict(
    vector: npt.NDArray[np.complexfloating] | list[float],
    nonzero_locations: npt.NDArray[np.integer] | list[int],
    n_qubit: int,
    *,
    optimization: bool = True,
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
        nonzero_locations: list of non-zero indices in the original vector
        n_qubit: total number of qubits?
        optimization: decide if optimize the angles or not, defaults to True

    Returns:
        a sequence of controlled gates to be applied.
    """

    if abs(sp.linalg.norm(vector) - 1.0) > ZERO:
        raise ValueError("vector should be normalized")

    final_gates: list[dict] = []

    for qbit in range(n_qubit):
        new_nonzero_locations = []
        new_vector = []
        # lenght of the resulting dictionary without optimization and without discarding zero angles/phases
        length_dict = 2 ** (n_qubit - qbit - 1)
        gate_operations: ControlledRotationGateMap = {}
        sparsity = len(nonzero_locations)

        phases: np.ndarray = np.angle(vector)

        i = 0
        while i in range(sparsity):
            # compute angles and phases
            angle: float
            phase: float
            # last step of the while loop
            if i + 1 == sparsity:
                loc = nonzero_locations[i]
                new_nonzero_locations.append(loc // 2)
                # if the non_zero element is at the very end of the vector
                if nonzero_locations[i] % 2 == 0:
                    angle = 0.0
                    phase = -phases[i]
                    new_vector.append(vector[i])
                # if the non_zero element is second-last
                else:
                    angle = np.pi
                    phase = phases[i]
                    new_vector.append(abs(vector[i]))
                # add in the dictionary gate_operations if they are not zero
                if (abs(angle) > ZERO) or (abs(phase) > ZERO):
                    if length_dict == 1:
                        gate_operations = {"": (angle, phase)}
                    else:
                        key = str(bin(loc // 2)[2:]).zfill(n_qubit - qbit - 1)
                        gate_operations[key] = (angle, phase)

                i += 1
            else:
                # divide the non_zero locations in pairs
                loc0 = nonzero_locations[i]
                loc1 = nonzero_locations[i + 1]

                # if the non_zero locations are consecutive, with the first one in an even position
                if (loc1 - loc0 == 1) and (loc0 % 2 == 0):
                    new_component = np.exp(1j * phases[i]) * np.sqrt(
                        abs(vector[i]) ** 2 + abs(vector[i + 1]) ** 2
                    )
                    new_vector.append(new_component)
                    new_nonzero_locations.append(loc0 // 2)

                    angle = (
                        2
                        * np.arccos(
                            np.clip(abs(vector[i] / new_component), -1, 1)
                        )  # TODO(doubt) why clip? the argument should not overflow the range right?
                        # It doesn't but it is to avoid the warning or small precision errors (could be 1.0000001)
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
                        new_vector.append(vector[i])
                        new_nonzero_locations.append(loc0 // 2)

                    else:
                        angle = np.pi
                        phase = phases[i]
                        new_vector.append(abs(vector[i]))
                        new_nonzero_locations.append(loc0 // 2)

                i += 1  # TODO(doubt) should this be inside the above else: branch (starting line 121)
                # No, if the above if happens you should skip one iteration, that is i+=2, you can put this inside the else and += 2 beforehand, maybe it is more readable

                if (abs(angle) > ZERO) or (abs(phase) > ZERO):
                    if length_dict == 1:
                        gate_operations = {"": (angle, phase)}
                    else:
                        key = str(bin(loc0 // 2)[2:]).zfill(n_qubit - qbit - 1)
                        gate_operations[key] = (angle, phase)

        vector, nonzero_locations = new_vector, new_nonzero_locations

        if optimization:
            gate_operations = optimize_dict(gate_operations)

        final_gates.append(gate_operations)

    final_gates.reverse()
    return final_gates


def gate_count(total_gate_operations: list[ControlledRotationGateMap]) -> GateCounts:
    """
    Counts how many gates you need to build  the circuit in terms of elemental ones (single rotation gates, one-control-one-target
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


def main(
    state: StateVector,
    N_qubit: int,
    *,
    optimization: bool = True,
) -> GateCounts:
    """
    Estimation of the number of gates needed to prepare a sparse state using permutation Grover Rudolph

    Args:
        state: the quantum state to be prepared.
        N_qubit: total number of qubits used?
        optimization: choose if you want to merge the gate (set to True by default)

    Returns:
        A GateCounts (ndarray) object
    """
    state = sanitize_sparse_state_vector(state)

    vector = state.data
    nonzero_locations = state.nonzero()[1]

    # standard Grover Rudolph
    d = number_of_qubits(nonzero_locations)  # sparsity
    angle_phase_dict = phase_angle_dict(
        vector, np.arange(len(vector)), d, optimization=optimization
    )
    count = gate_count(angle_phase_dict)

    # Permutation algorithm
    permutation = build_permutation(nonzero_locations)

    for cycle in permutation:
        count += count_cycle(cycle, N_qubit)

    return count
