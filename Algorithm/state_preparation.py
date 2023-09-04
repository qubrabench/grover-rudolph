"""
**state_preparation** is a collection of functions to estimate the number of gates needed in the state preparation (in terms of Toffoli, 2-qbits gates and 1-qbit gates) and to build the circuit that prepares the state.
The algorithm used is Grover Rudolph.
"""

import numpy as np
from helping_sp import ZERO, hamming_weight, pad_to_pow2

__all__ = [
    "phase_angle_dict",
    "gate_count",
    "build_permutation",
    "count_cycle",
    "main",
]


def phase_angle_dict(vector, nonzero_locations, n_qubit, optimization=True):
    """
    Generate a list of dictonaries for the angles given the amplitude vector
    Each dictonary is of the form:
    {key = ('0' if apply controlled on the state 0, '1' if controlled on 1, 'e' if apply identy) : value = [angle, phase]
    {'00' : [1.2, 0.]} the gate is a rotation of 1.2 and a phase gate with phase 0, controlled on the state |00>
    ~you are basically building the cicuit vertically, where each element of the dictionary is one layer of the circuit
    if the dictonary is in position 'i' of the list (starting from 0), its key will be of length 'i', thus the controls act on the fist i qubits

    Args:
        vector = list of complex numbers. Its entries have to be normalized.
        nonzero_locations = list on int numbers indicating the non zero locations
        n_qubit = int
        optimization = Boolean variable, decide if optimize the angles or not
    Returns:
        list of dictionaries
    """

    if abs(np.linalg.norm(vector) - 1.0) > ZERO:
        raise ValueError("vector should be normalized")

    list_dictionaries = []

    for qbit in range(n_qubit):
        new_nonzero_locations = []
        new_vector = []

        length_dict = 2 ** (n_qubit - qbit - 1)
        dictionary = {}
        sparsity = len(nonzero_locations)
        i = 0

        while i in range(sparsity):
            if i + 1 == sparsity:
                loc = nonzero_locations[i]
                new_nonzero_locations.append(int(np.floor(loc / 2)))

                if nonzero_locations[i] % 2 == 0:
                    angle = 0.0
                    phase = -np.angle(vector[i])
                    new_vector.append(vector[i])
                else:
                    angle = np.pi
                    phase = np.angle(vector[i])
                    new_vector.append(abs(vector[i]))

                if (abs(angle) > ZERO) or (abs(phase) > ZERO):
                    if length_dict == 1:
                        dictionary = {"": [angle, phase]}
                    else:
                        key = str(bin(int(np.floor(loc / 2)))[2:]).zfill(
                            n_qubit - qbit - 1
                        )
                        dictionary[key] = [angle, phase]

                i += 1
                continue

            # check consecutives numbers and even position
            loc0 = nonzero_locations[i]
            loc1 = nonzero_locations[i + 1]
            if (loc1 - loc0 == 1) and (loc0 % 2 == 0):
                new_component = np.exp(1j * np.angle(vector[i])) * np.sqrt(
                    abs(vector[i]) ** 2 + abs(vector[i + 1]) ** 2
                )
                new_vector.append(new_component)
                new_nonzero_locations.append(int(np.floor(loc0 / 2)))

                angle = (
                    2 * np.arccos(np.clip(abs(vector[i] / new_component), -1, 1))
                    if abs(new_component) > ZERO
                    else 0.0
                )
                phase = -np.angle(vector[i]) + np.angle(vector[i + 1])
                i += 1
            else:
                if loc0 % 2 == 0:
                    angle = 0.0
                    phase = -np.angle(vector[i])
                    new_vector.append(vector[i])
                    new_nonzero_locations.append(int(np.floor(loc0 / 2)))

                else:
                    angle = np.pi
                    phase = np.angle(vector[i])
                    new_vector.append(abs(vector[i]))
                    new_nonzero_locations.append(int(np.floor(loc0 / 2)))

            i += 1

            if (abs(angle) > ZERO) or (abs(phase) > ZERO):
                if length_dict == 1:
                    dictionary = {"": [angle, phase]}
                else:
                    key = str(bin(int(np.floor(loc0 / 2)))[2:]).zfill(
                        n_qubit - qbit - 1
                    )
                    dictionary[key] = [angle, phase]

        vector = new_vector
        nonzero_locations = new_nonzero_locations

        # if optimization == True:
        #     dictionary_opt = optimize_dict(dictionary)
        list_dictionaries.insert(0, dictionary)

    return list_dictionaries


def gate_count(dict_list):
    """
    Counts how many gates you need to build  the circuit in terms of elemental ones (single rotation gates, one-control-one-target
    gates on the |1⟩ state, refered as 2 qubits gates, and Toffoli gates)

    Args:
                    dict_list = the list of dictionaries
    Rerurns:
                    list of int = [number of Toffoli gates, number of 2 qubits gates, number of single rotation gate]
    """
    N_toffoli = 0
    N_cnot = 0
    N_1_gate = 0

    for dictionary in dict_list:
        # Build the unitary for each dictonary
        for k in dictionary.keys():
            count0 = 0  # count 0 gate
            count1 = 0  # count 1 gate

            for s in k:
                if s == "0":
                    count0 += 1

                elif s == "1":
                    count1 += 1

            if count0 + count1 == 0:
                N_toffoli += 0
                N_cnot += 0
                N_1_gate += 1
            else:
                N_toffoli += (count0 + count1 - 1) * 2
                N_cnot += 2
                N_1_gate += 4 + (2 * count0)

    # Subtract the two x-gates that form an identity from the total count of 1-qubit gates
    N_1_gate -= 2 * x_gate_merging(dictionary)

    count = np.array([N_toffoli, N_cnot, N_1_gate])
    return count


def build_permutation(nonzero_locations):
    """
    Given a classical permutation, return its cyclic decomposition
    Construct a permutation unitary that maps |i⟩ → |x_i⟩

    Args:
        nonzero_locations (list): the locations of nonzero elements of a sparse vector state |b⟩

    Returns:
        cycles (list): permutation cycles

    """
    d = len(nonzero_locations)  # Sparsity
    S = [1 for i in range(d)]
    cycles = []  # list to store the permutation cycles

    # Build the cycles
    for i in range(d):
        j = nonzero_locations[i]
        c = [i, j]

        if (S[i] == 0) or (nonzero_locations[i] == i):
            continue

        while j < d:
            S[j] = 0
            j = nonzero_locations[j]
            c.append(j)

        # Add elements to the cycle until the end of the cycle is reached
        cycles.append(c)

    return cycles


def count_cycle(cycle, N_qubit):
    """
    Counts the number if universal gates in terms of Toffoli, Cnots and 1 qubit gates
    Args:
        cycle = permutation cycle of int
        N_qubit = int
    Returns:
        List of int = [Number of Toffoli, Number of cnots, Number of 1 qubit gates]
    """
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


def main(vector, nonzero_locations, N_qubit):
    if not (np.sort(nonzero_locations) == nonzero_locations).all():
        raise (ValueError("the nonzero_locations location vector must be ordered\n"))

    # add zeros to the vector until it has as length a power of 2
    vector, nonzero_locations = pad_to_pow2(vector, nonzero_locations, N_qubit)

    # standard Grover Rudolph
    d = int(np.ceil(np.log2(len(nonzero_locations))))  # sparsity
    angle_phase_dict = phase_angle_dict(vector, list(np.arange(0, len(vector))), d)
    count = gate_count(angle_phase_dict)

    # Permutation algorithm
    permutation = build_permutation(nonzero_locations)

    for cycle in permutation:
        count += count_cycle(cycle, N_qubit)

    return count
