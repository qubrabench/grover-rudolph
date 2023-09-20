import numpy as np
import numpy.typing as npt
from typing import Any
from functools import reduce

from helping_sp import reduced_density_matrix, ControlledRotationGateMap
from state_preparation import phase_angle_dict, build_permutation


__all__ = ["main_circuit"]


def circuit_GR(dict_list: list[ControlledRotationGateMap]) -> Any:
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
        for k, [theta, phase] in gates.items():
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


def cycle_circuit(cycle: Any, state: Any) -> Any:
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


def main_circuit(
    vector: npt.NDArray[np.complexfloating],
    nonzero_locations: npt.NDArray[np.integer],
    N_qubit: int,
) -> npt.NDArray[np.complexfloating]:
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
        raise ValueError("the nonzero_locations location vector must be ordered")

    # add zeros to the vector until it has as length a power of 2

    d = int(np.ceil(np.log2(len(nonzero_locations))))  # sparsity
    angle_phase_dict = phase_angle_dict(vector, list(np.arange(0, len(vector))), d)
    phi = circuit_GR(angle_phase_dict)

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
