"""
**state_preparation** is a collection of functions to estimate the number of gates needed in the state preparation (in terms of Toffoli, 2-qbits gates and 1-qbit gates) and to build the circuit that prepares the state.
The algorithm used is Grover Rudolph.
"""

from helping_sp import *


def phase_angle_dict(vector):
    """
    Generate a list of dictonaries for the angles given the amplitude vector
    Each dictonary is of the form:
    {key = ('0' if apply controlled on the state 0, '1' if controlled on 1, 'e' if apply identy) : value = [angle, phase]
    >>>{'00' : [1.2, 0.]} the gate is a rotation of 1.2 and a phase gate with phase 0, controlled on the state |00>
    ~you are basically building the cicuit vertically, where each element of the dictionary is one layer of the circuit
    if the dictonary is in position 'i' of the list (starting from 0), its key will be of length 'i', thus the controls act on the fist i qubits

    Args:
        vector = np.array of complex type. Its entries have to be normalized.
    Returns:
        list of dictionaries
    """

    # if vector is not a power of 2 make it so
    if len(vector) & (len(vector) - 1) != 0:
        extra_zeros = 2 ** (int(np.log2(len(vector))) + 1) - len(vector)
        vector = np.pad(vector, (0, extra_zeros))

    if abs(linalg.norm(vector) - 1.0) > ZERO:
        raise ValueError("vector should be normalized")

    n_qubit = int(np.log2(len(vector)))
    list_dictionaries = []

    for qbit in range(n_qubit):
        # Compute the angles recursively
        phase_vector = [
            np.exp(1j * np.angle(vector[::2][i])) for i in range(len(vector[::2]))
        ]
        new_vector = (
            np.sqrt(abs(vector[::2]) ** 2 + abs(vector[1::2]) ** 2) * phase_vector
        )

        angles = [
            2 * np.arccos(np.clip(abs((vector[::2][i] / new_vector[i])), -1, 1))
            if (abs(new_vector[i]) > ZERO)
            else 0
            for i in range(len(new_vector))
        ]

        phases = -np.angle(vector[::2]) + np.angle(vector[1::2])
        vector = new_vector

        # Assign keys(binary numbers) and values (angles, phases) in  two dictionaries

        # the first gate is not controlled by anything, thus its key is ''
        lenght_dict = 2 ** (n_qubit - qbit - 1)
        if lenght_dict == 1:
            dict_angles = {"": angles[-1]} if abs(angles[-1]) > ZERO else {}
            dict_phases = {"": phases[-1]} if abs(phases[-1]) > ZERO else {}

        # generate the keys: all binary numbers with fixed lenght
        else:
            dict_angles = {}
            dict_phases = {}
            for i in range(lenght_dict - 1, -1, -1):
                k = str(bin(i))[2:].zfill(n_qubit - qbit - 1)
                if abs(angles[i]) > ZERO:
                    dict_angles[k] = angles[i]

                if abs(phases[i]) > ZERO:
                    dict_phases[k] = phases[i]

        dict_angles_opt = optimize_dict(dict_angles)
        dict_phases_opt = optimize_dict(dict_phases)

        dictionary = merge_dict(dict_angles_opt, dict_phases_opt)
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

    for i in range(len(dict_list)):
        dictionary = dict_list[i]

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

    count = [N_toffoli, N_cnot, N_1_gate]

    return count


def build_permutation(nonzero_locations, N_ancilla):
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
    lenght = len(cycle)

    # compute the two summation terms for the counting
    sum_neg = 0
    sum_diff = 0
    for i in range(lenght - 1):
        sum_neg += N_qubit - hamming_weight(cycle[i])
        sum_diff += hamming_weight(cycle[i] ^ cycle[i + 1])

    sum_neg += N_qubit - hamming_weight(cycle[lenght - 1])

    N_toffoli = 2 * (lenght + 1) * (N_qubit - 1)
    N_cnot = (2 * (lenght + 1)) + (2 * sum_diff)
    N_1_gate = (4 * (lenght + 1)) + (2 * sum_neg) + (4 * sum_diff)

    return [N_toffoli, N_cnot, N_1_gate]


def main(vector, nonzero_locations, N_qubit):
    e0 = np.array([float(1), float(0)])  # zero state

    if not (np.sort(nonzero_locations) == nonzero_locations).all():
        raise (ValueError("the nonzero_locations location vector must be ordered\n"))

    # add zeros to the vector until it has as lenght a power of 2
    sparsity = len(nonzero_locations)

    if sparsity & (sparsity - 1) != 0:
        extra_zeros = 2 ** (int(np.log2(sparsity)) + 1) - sparsity
        counter_pad = 0
        for i in range(2**N_qubit):
            if i not in nonzero_locations:
                vector = np.insert(vector, i, 0)
                nonzero_locations = np.insert(nonzero_locations, i, i)
                counter_pad += 1
            if counter_pad == extra_zeros:
                break

    d = int(np.log2(len(nonzero_locations)))  # sparsity

    # standard Grover Rudolph
    angle_phase_dict = phase_angle_dict(vector)
    count = gate_count(angle_phase_dict)

    # Permutation algorithm
    permutation = build_permutation(nonzero_locations, N_qubit - d)

    for cycle in permutation:
        count += count_cycle(cycle, N_qubit)

    return count
