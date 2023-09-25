from typing import Sized, Union

import numpy as np
import scipy as sp

__all__ = [
    "RotationGate",
    "ControlledRotationGateMap",
    "StateVector",
    "ZERO",
    "neighbour_dict",
    "optimize_dict",
    "reduced_density_matrix",
    "generate_sparse_unit_vector",
    "hamming_weight",
    "x_gate_merging",
    "number_of_qubits",
    "sanitize_sparse_state_vector",
]

# Some useful type aliases:
RotationGate = tuple[float, float]
"""(phase, angle) pair describing a rotation gate"""

Controls = str
"""a sequence of control bits. each bit is one of {0, 1, e}"""

ControlledRotationGateMap = dict[Controls, RotationGate]
"""keys are control bits, target is a rotation gate description"""

StateVector = Union[np.ndarray, sp.sparse.spmatrix, list[float]]
"""A row vector representing a quantum state"""

ZERO = 1e-8
"""global zero precision"""


def neighbour_dict(controls: Controls) -> dict[Controls, int]:
    """
    Finds the neighbours of a string (ignoring e), i.e. the mergeble strings
    Returns a dictionary with as keys the neighbours and as value the position in which they differ

    >>> assert neighbour_dict("10") == {"00": 0, "11": 1}
    >>> assert neighbour_dict("1e") == {'0e': 0}

    Args:
        controls: string made of '0', '1', 'e'
    Returns:
        A dictionary {control-string: swapped-index}
    """
    neighbours = {}
    for i, c in enumerate(controls):
        if c == "e":
            continue

        c_opposite = "1" if c == "0" else "0"
        key = controls[:i] + c_opposite + controls[i + 1 :]
        neighbours[key] = i

    return neighbours


def optimize_dict(
    gate_dictionary: ControlledRotationGateMap,
) -> ControlledRotationGateMap:
    """
    Optimize the dictionary by merging some gates in one:
    if the two values are the same and they only differ in one control (one char of the key  is 0 and the other is 1) they can be merged
    >> {'11':[3.14,0] ; '10':[3.14,0]} becomes {'1e':[3.14,0]} where 'e' means no control (identity)

    >>> assert optimize_dict({"11": (3.14, 0), "10": (3.14, 0)}) == {"1e": (3.14, 0)}

    Args:
        gate_dictionary: collection of controlled gates to be applied
    Returns:
        optimized collection of controlled gates
    """
    while run_one_merge_step(gate_dictionary):
        pass
    return gate_dictionary


def run_one_merge_step(
    gate_dictionary: ControlledRotationGateMap,
) -> bool:
    """
    Run a single merging step, modifying the input dictionary.

    Args:
        gate_dictionary: collection of controlled gates to be applied
    Returns:
        True if some merge happened
    """
    if len(gate_dictionary) <= 1:
        return False

    for k1, v1 in gate_dictionary.items():
        neighbours = neighbour_dict(k1)

        for k2, position in neighbours.items():
            if k2 not in gate_dictionary:
                continue

            v2 = gate_dictionary[k2]

            # Consider only different items with same angle and phase
            if (abs(v1[0] - v2[0]) > ZERO) or (abs(v1[1] - v2[1]) > ZERO):
                continue

            # Replace the different char with 'e' and remove the old items
            gate_dictionary.pop(k1)
            gate_dictionary.pop(k2)
            gate_dictionary[k1[:position] + "e" + k1[position + 1 :]] = v1
            return True

    return False


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


def x_gate_merging(dictionary: ControlledRotationGateMap) -> int:
    """
    Counts the number of x-gates that can be merged given an optimized dictionary.

    For each consecutive pairs of keys, if they have matching '0's at some index, then then X gates can be dropped.

    Parameters:
        dictionary: A dictionary containing quantum gates represented as keys.

    Returns:
        The count of x-gates that can be merged.
    """
    keys = list(dictionary.keys())
    return sum(list(zip(k1, k2)).count(("0", "0")) for k1, k2 in zip(keys, keys[1:]))


def generate_sparse_unit_vector(
    n_qubit: int, d: int, *, dtype: str = "complex"
) -> sp.sparse.spmatrix:
    """
    Generate random complex amplitudes vector of N qubits (length 2^N) with sparsity d
    as couples: position and value of the i-th non zero element
    The sign of the first entry  is  always real positive to fix the overall phase

    Args:
        n_qubit: number of qubits
        d: number of non-zero entries required in the output state vector
        dtype: datatype of the generated entries. defaults to "complex"

    Returns:
        A state vector stored as a scipy.sparse.spmatrix object with shape (1, 2**n_qubit), having exactly d non-zero elements.
    """
    N = 2**n_qubit

    if d > N:
        raise ValueError(
            "Sparsity must be less or equal than the dimension of the vector"
        )

    sparse_v = sp.sparse.random(1, N, density=d / N, format="csr", dtype=dtype)
    sparse_v /= sp.linalg.norm(sparse_v.data)

    return sparse_v


def hamming_weight(n: int) -> int:
    h_weight = 0
    while n:
        h_weight += 1
        n &= n - 1
    return h_weight


def number_of_qubits(vec: Sized) -> int:
    return int(np.ceil(np.log2(len(vec))))


def sanitize_sparse_state_vector(
    vec: StateVector, *, copy=False
) -> sp.sparse.csr_matrix:
    vec = sp.sparse.csr_matrix(vec)
    if copy:
        vec = vec.copy()
    vec.sort_indices()
    return vec
