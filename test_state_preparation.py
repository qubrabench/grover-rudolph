import pytest

from helping_sp import *
from state_preparation import *


def circuit_GR(dict_list):
    """

    The same procedure is applied to the phases, and at the end the two dictionaries are merged together, taking into account the commutation rules.
    Build the circuit of the state preparation with as input the list of dictonaries (good to check if the preparation is succesfull)

    Returns:
         The final state and the number of gates needed (Number of Toffoli gates, 2qubits gate and 1qubit gate)
    """

    # Vector to apply the circuit to
    psi = np.zeros(2 ** len(dict_list))
    psi[0] = float(1)

    e0 = np.array([float(1), float(0)])  # zero state
    e1 = np.array([float(0), float(1)])

    P0 = np.outer(e0, e0)  # Projector
    P1 = np.outer(e1, e1)

    Id = np.eye(2)

    for i in range(len(dict_list)):
        dictionary = dict_list[i]

        # Build the unitary for each dictonary
        for k, [theta, phase] in dictionary.items():
            if theta is None:
                R = np.eye(2)
            else:
                R = np.array(
                    [
                        [np.cos(theta / 2), -np.sin(theta / 2)],
                        [np.sin(theta / 2), np.cos(theta / 2)],
                    ]
                )

            if phase is None:
                P_phase = np.eye(2)
            else:
                P_phase = np.array([[1.0, 0.0], [0.0, np.exp(1j * phase)]])

            P = 1  # Projector P which controls R
            for s in k:
                if s == "e":
                    P = np.kron(P, Id)

                elif s == "0":
                    P = np.kron(P, P0)

                elif s == "1":
                    P = np.kron(P, P1)

            U = np.kron(P, P_phase @ R) + np.kron(np.eye(2**i) - P, Id)

            for n in range(len(dict_list) - i - 1):
                U = np.kron(U, Id)

            psi = U @ psi

    return psi


def cycle_circuit(cycle, state):
    """
    Given a cycle, it return the unitary that permutes the vector of the computational basis
    circuit building the permutation

    Args:
        List of cycles

    Returns:
        matrix implementing the permutation
    """

    lenght = len(cycle)
    N_qubit = int(np.log2(len(state)))

    # Compute the list with the bit difference (xor) between each element and the following one of cycle
    difference_cycle = cycle.copy()
    difference_cycle.append(difference_cycle.pop(0))  # Move first element to the end

    difference_cycle = [
        bin(difference_cycle[i] ^ cycle[i])[2:].zfill(N_qubit - 1)
        for i in range(len(cycle))
    ]  # xor and binary

    cycle = [bin(j)[2:].zfill(N_qubit - 1) for j in cycle]
    cycle.append(cycle[0])
    difference_cycle.append(bin(0)[2:].zfill(N_qubit - 1))

    e0 = np.array([float(1), float(0)])  # zero state
    e1 = np.array([float(0), float(1)])

    P0 = np.outer(e0, e0)  # Projector
    P1 = np.outer(e1, e1)

    I = np.eye(2)
    X = np.array([[float(0), float(1)], [float(1), float(0)]])

    for j in range(len(cycle)):
        nonzero_loc = cycle[j]
        diff = difference_cycle[j]
        P = 1  # Projector P which controls the X operator
        X_diff = 1  # X gate when the qbits are different

        for i in range(len(nonzero_loc)):
            if nonzero_loc[i] == "0":
                P = np.kron(P, P0)
            elif nonzero_loc[i] == "1":
                P = np.kron(P, P1)
            if diff[i] == "0":
                X_diff = np.kron(X_diff, I)
            elif diff[i] == "1":
                X_diff = np.kron(X_diff, X)

        state = (np.kron(P, X) + np.kron(np.eye(2 ** (N_qubit - 1)) - P, I)) @ state
        state = (
            np.kron(X_diff, P1) + np.kron(np.eye(2 ** (N_qubit - 1)), I - P1)
        ) @ state

    return state


def main_circuit(vector, nonzero_locations, N_qubit):
    """
    Implement the permutation GR:
    given a sparse state given as input as two vectors, one indicating the value of the non zero components and the other their positions,
    returns the prepared quantum state vector

    Args:
        np complex array, np int array, int
    Returns:
        Output of the circuit as a density matrix (complex matrix)
    """

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

    angle_phase_dict = phase_angle_dict(vector)
    phi = circuit_GR(angle_phase_dict)

    # ancilla qubits
    for i in range(N_qubit - d):
        phi = np.kron(e0, phi)
    phi = np.kron(phi, e0)

    # permutations
    permutation = sparseperm(nonzero_locations, N_qubit - d)

    for cycle in permutation:
        phi = cycle_circuit(cycle, phi)

    # Remove ancilla
    density_matrix = np.outer(phi.conjugate(), phi)
    density_matrix = reduced_density_matrix(density_matrix, 2)

    return density_matrix


def test_circuit():
    n_qubit = 6
    N = 2**n_qubit

    for d in range(1, 2**n_qubit):
        vector, nonzero_loc = sparse_couple_vect(n_qubit, d)

        # build the respective density matrix
        vect = np.zeros(N, dtype=np.complex128)
        for j in range(len(vector)):
            vect[nonzero_loc[j]] = vector[j]
        rho_input = np.outer(vect.conjugate(), vect)

        # Build with improved grover rudolph
        rho_GR = main_circuit(vector, nonzero_loc, n_qubit)

        assert (abs(rho_GR - rho_input) < ZERO).all(), "test failed"
