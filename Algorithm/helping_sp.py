from typing import Any

import numpy as np
import scipy as sp
import scipy.sparse

__all__ = [
    "ZERO",
    "neighbour_dict",
    "optimize_dict",
    "reduced_density_matrix",
    "generate_sparse_vect",
    "hamming_weight",
    "x_gate_merging",
    "RotationGate",
    "ControlledRotationGateMap",
]

# Some useful type aliases:
RotationGate = tuple[float, float]
"""(phase, angle) pair describing a rotation gate"""

Controls = str
"""a sequence of control bits
each bit is one of "0", "1" or "e"
"""

ControlledRotationGateMap = dict[Controls, RotationGate]
"""keys are control bits, target is a rotation gate description"""

ZERO = 1e-8
"""global zero precision"""


def neighbour_dict(string1: Controls) -> dict[Controls, int]:
    """
    Finds the neighbours of a string (ignoring e), i.e. the mergeble strings
    Returns a dictionary with as keys the neighbours and as value the position in which they differ

    >>> assert neighbour_dict("10") == {"00": 0, "11": 1}
    >>> assert neighbour_dict("1e") == {'0e': 0}

    Args:
        string1: string made of '0', '1', 'e'
    Returns:
        dict = {string: int}
    """
    neighbours = {}
    for i, c in enumerate(string1):
        if c == "e":
            continue

        c_opposite = "1" if c == "0" else "0"
        key = string1[:i] + c_opposite + string1[i + 1 :]
        neighbours[key] = i

    return neighbours


def optimize_dict(
    gate_dictionary: ControlledRotationGateMap,
) -> ControlledRotationGateMap:
    """
    Optimize the dictionary by merging some gates in one:
    if the two values are the same and they only differ in one control (one char of the key  is 0 and the other is 1) they can be merged
    >> {'11':[3.14,0] ; '10':[3.14,0]} becomes {'1e':[3.14,0]} where 'e' means no control (identity)

    >>> assert optimize_dict({"11": [3.14,0] "10": [3.14,0]}) == {"1e": [3.14,0]}

    Args:
        dictionary: {key = (string of '0', '1') : value = [float,float]}
    Returns:
        dictionary = {key = (string of '0', '1', 'e') : value = float}
    """
    merging_success = True

    # Continue until everything that can be merged is merged
    while merging_success and len(gate_dictionary) > 1:
        merging_success = meargeable(gate_dictionary)

    return gate_dictionary


def meargeable(
    gate_dictionary: ControlledRotationGateMap,
) -> bool:
    """
    Returns True if a merging happens, False otherwise.
    It modifies the dictionary by doing the merging.

    Args:
        dictionary: {key = (string of '0', '1') : value = [float,float]}
    Returns:
        bool
    """

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


def reduced_density_matrix(rho: Any, traced_dim: int) -> Any:
    """
    Computes the partial trace on a second subspace of dimension traced_dimension

    Args:
        rho: Complex 2D array
        traced_dim: int

    Returns:
        Complex 2D array
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

    Parameters:
        dictionary: A dictionary containing quantum gates represented as keys.

    Returns:
        The count of x-gates that can be merged.
    """
    keys = list(dictionary.keys())

    # Iterate through consecutive pairs of keys
    # Check if x-gate merging condition is met: if two consecutive bit strings have a '0' at the same position
    # Increment the counter if the condition is met
    return [
        any(c1 == "0" and c2 == "0" for c1, c2 in zip(key1, key2))
        for key1, key2 in zip(keys, keys[1:])
    ].count(True)


def generate_sparse_vect(
    n_qubit: int, d: int, vec_type="complex"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate random complex amplitudes vector of N qubits (length 2^N) with sparsity d
    as couples: position and value of the i-th non zero element
    The sign of the first entry  is  always real positive to fix the overall phase

    Args:
        Number of qubits N, sparsity d

    Returns:
        vector = complex array of length d, with the values of the amplitudes
        nonzero_locations = int array of length d (ordered) with the position of the non-zero element
    """
    N = 2**n_qubit

    if d > N:
        raise ValueError(
            "Sparsity must be less or equal than the dimension of the vector"
        )

    sparse_v = sp.sparse.random(1, N, density=d / N, format="csr", dtype=vec_type)
    sparse_v.sort_indices()
    nonzero_loc = sparse_v.nonzero()[1]
    values = sparse_v.data
    values = values / np.linalg.norm(values)
    return values, nonzero_loc


def hamming_weight(n: int) -> int:
    h_weight = 0
    while n:
        h_weight += 1
        n &= n - 1
    return h_weight
