import numpy as np
import numpy.typing as npt
from functools import reduce

from helping_sp import (
    ControlledRotationGateMap,
    StateVector,
    number_of_qubits,
    sanitize_sparse_state_vector,
)
from state_preparation import grover_rudolph, build_permutation


__all__ = ["permutation_GR_circuit"]


def reduced_density_matrix(rho: np.ndarray, traced_dim: int) -> np.ndarray:
    """
    Computes the partial trace on a second subspace of dimension traced_dimension

    If rho is in a composite Hilbert space H_a x H_b
    and we want to discard the second space H_b,
    the final state will be the reduced density matrix in H_a,
    computed by this function by specifying the dimension of H_b, which is traced over

    E.g. |01><01| -> |0><0|

    Args:
        rho: complex 2D array with as dimensions powers of two
        traced_dim: dimension of the subspace that is traced over

    Returns:
        reduced_rho: Complex 2D array
    """

    total_dim = len(rho[0])
    final_dim = int(total_dim / traced_dim)
    reduced_rho = np.trace(
        rho.reshape(final_dim, traced_dim, final_dim, traced_dim),
        axis1=1,
        axis2=3,
    )

    return reduced_rho


def GR_circuit(dict_list: list[ControlledRotationGateMap]) -> np.ndarray:
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

    Id = np.eye(2)

    control_matrix: dict[str, np.ndarray] = {
        "e": Id,
        "0": np.outer(e0, e0),
        "1": np.outer(e1, e1),
    }

    for i, gates in enumerate(dict_list):
        # Build the unitary for each dictonary
        for k, (theta, phase) in gates.items():
            if theta is None:
                R = Id
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

            # tensor product of all the 2x2 control matrices
            P = reduce(np.kron, [control_matrix[s] for s in k], np.eye(1))

            U = np.kron(P, P_phase @ R) + np.kron(np.eye(2**i) - P, Id)

            extra = len(dict_list) - i - 1
            U = np.kron(U, np.eye(2**extra))

            psi = U @ psi

    return psi


def cycle_circuit(cycle: list[int], state: np.ndarray) -> np.ndarray:
    """
    Given a cycle, it return the unitary that permutes the vector of the computational basis
    circuit building the permutation

    Args:
        List of cycles

    Returns:
        matrix implementing the permutation
    """

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

    Id = np.eye(2)
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
                X_diff = np.kron(X_diff, Id)
            elif diff[i] == "1":
                X_diff = np.kron(X_diff, X)

        state = (np.kron(P, X) + np.kron(np.eye(2 ** (N_qubit - 1)) - P, Id)) @ state
        state = (
            np.kron(X_diff, P1) + np.kron(np.eye(2 ** (N_qubit - 1)), Id - P1)
        ) @ state

    return state


def permutation_GR_circuit(state: StateVector) -> npt.NDArray[np.complexfloating]:
    """
    Implement the permutation GR:
    given a list of coefficients, returns the prepared quantum state vector.

    Args:
        state: A complex vector proportional to the state to be prepared

    Returns:
        Output of the circuit as a density matrix (complex matrix)
    """
    state = sanitize_sparse_state_vector(state)

    vector = state.data
    nonzero_locations = state.nonzero()[1]

    N_qubit = number_of_qubits(state.shape[1])

    e0 = np.array([float(1), float(0)])  # zero state

    # add zeros to the vector until it has as length a power of 2
    d = number_of_qubits(nonzero_locations)  # sparsity
    gr_gates = grover_rudolph(vector)
    phi = GR_circuit(gr_gates)

    # ancilla qubits
    for i in range(N_qubit - d):
        phi = np.kron(e0, phi)
    phi = np.kron(phi, e0)

    # permutations
    permutation = build_permutation(nonzero_locations)

    for cycle in permutation:
        phi = cycle_circuit(cycle, phi)

    # Remove ancilla
    density_matrix = np.outer(phi.conjugate(), phi)
    density_matrix = reduced_density_matrix(density_matrix, 2)

    return density_matrix
