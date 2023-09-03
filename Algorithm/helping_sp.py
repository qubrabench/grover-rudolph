import numpy as np
from itertools import combinations
import scipy as sp

__all__ = [
    "ZERO",
    "where_diff_one",
    "optimize_dict",
    "reduced_density_matrix",
    "generate_sparse_vect",
    "hamming_weight",
    "pad_to_pow2",
]

# global zero precision
ZERO = 1e-8


def where_diff_one(string_1, string_2) -> int | None:
    """
    Checks if two string differ by ONLY one char that is not 'e', and it finds its position
    It is checking if two controlled gates can be merged: if they differ in only one control
    >>> assert where_diff_one('010', '011') == 2
    >>> assert where_diff_one('011', '100') is None
    >>> assert where_diff_one('e10', 'e10') is None

    Args:
        string1, string2 = string made of '0', '1', 'e'
    Returns:
        The position (int), or None if the conidtions are not met
    """

    differ = 0  # difference count
    for i in range(len(string_1)):
        if string_1[i] != string_2[i]:
            differ += 1
            position = i
            # if they differ with the char 'e' we can't merge
            if string_1[position] == "e" or string_2[position] == "e":
                return None
        if differ > 1:
            return None
    if differ == 0:
        return None
    return position


def optimize_dict(dictionary):
    """
    Optimize the dictionary by merging some gates in one:
    if the two values are the same and they only differ in one control (one char of the key  is 0 and the other is 1) they can be merged
    >> {'11':3.14, ; '10':3.14} becomes {'1e':3.14} where 'e' means no control (identity)
    Args:
        dictionary = {key = (string of '0', '1') : value = float}
    Returns:
        dictionary = {key = (string of '0', '1', 'e') : value = float}
    """
    Merging_success = True  # Initialization value

    # Continue until everything that can be merged is merged
    while Merging_success and len(dictionary) > 1:
        for k1, k2 in combinations(dictionary.keys(), 2):
            v1 = dictionary[k1]
            v2 = dictionary[k2]
            Merging_success = False

            # Consider only different items with same angle and phase
            if (abs(v1[0] - v2[0]) > ZERO) or (abs(v1[1] - v2[1]) > ZERO):
                continue

            position = where_diff_one(k1, k2)

            if position is None:
                continue

            # Replace the different char with 'e' and remove the old items
            k1_list = list(k1)
            k1_list[position] = "e"

            dictionary.pop(k1)
            dictionary.pop(k2)
            dictionary.update({"".join(k1_list): v1})
            Merging_success = True
            break

    return dictionary


def reduced_density_matrix(rho, traced_dim):
    """
    Computes the partial trace on a second subspace of dimension traced_dimension

    Args:
        Complex array, int
    Returns:
        Complex matrix
    """

    total_dim = len(rho[0])
    final_dim = int(total_dim / traced_dim)
    reduced_rho = np.trace(
        rho.reshape(final_dim, traced_dim, final_dim, traced_dim),
        axis1=1,
        axis2=3,
    )

    return reduced_rho


def generate_sparse_vect(n_qubit, d):
    """
    Generate random complex amplitudes vector of N qubits (length 2^N) with sparsity d
    as couples: position and value of the i-th non zero element
    The sign of the first entry  is  always real positive to fix the overall phase

        Args:
            Number of qubits N, sparsity d

        Returns:
            vector = complex array of lenght d, with the values of the amplitudes
            nonzero_locations = int array of lenght d (ordered) with the position of the non-zero element
    """
    N = 2**n_qubit

    if d > N:
        raise (
            ValueError(
                "Sparsity must be less or equal than the dimension of the vector\n"
            )
        )

    sparse_v = sp.sparse.random(1, N, density=d / N, format="csr", dtype="complex")
    sparse_v.sort_indices()
    nonzero_loc = sparse_v.nonzero()[1]
    values = sparse_v.data
    values = values / np.linalg.norm(values)
    return values, nonzero_loc


def hamming_weight(n: int):
    h_weight = 0
    while n:
        h_weight += 1
        n &= n - 1
    return h_weight


def pad_to_pow2(vector, nonzero_locations, N_qubit):
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
    return vector, nonzero_locations