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
]

# global zero precision
ZERO = 1e-8


def neighbour_dict(string1):
    """
    Finds the neighbours of a string (ignoring e), i.e. the mergeble strings
    Returns a dictionary with as keys the neighbours and as value the position in which they differ
    - 10 -> {'11': 1, '00': 0}
    - 1e -> {'0e': 0}

    Args:
        string1 = string made of '0', '1', 'e'
    Returns:
        dict = {string: int}
    """
    neighbours = {}
    list1 = list(string1)
    for i in range(len(string1)):
        list2 = list1.copy()

        if list1[i] == "e":
            continue

        if list1[i] == "0":
            list2[i] = "1"
            neighbours["".join(list2)] = i
            continue

        if list1[i] == "1":
            list2[i] = "0"
            neighbours["".join(list2)] = i

    return neighbours


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
        for k1 in dictionary.keys():
            Merging_success = False
            v1 = dictionary[k1]
            neighbours = neighbour_dict(k1)

            for k2 in neighbours.keys():
                if k2 not in dictionary:
                    continue

                v2 = dictionary[k2]
                position = neighbours[k2]

                # Consider only different items with same angle and phase
                if (abs(v1[0] - v2[0]) > ZERO) or (abs(v1[1] - v2[1]) > ZERO):
                    continue

                # Replace the different char with 'e' and remove the old items
                k1_list = list(k1)
                k1_list[position] = "e"

                dictionary.pop(k1)
                dictionary.pop(k2)
                dictionary.update({"".join(k1_list): v1})
                Merging_success = True
                break
            else:
                continue
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


def x_gate_merging(dictionary):
    """
    Counts the number of x-gates that can be merged given an optimized dictionary.

    Parameters:
        dictionary (dict): A dictionary containing quantum gates represented as keys.

    Returns:
        int (count): The count of x-gates that can be merged.
    """

    # Extract the keys from the dictionary
    keys = list(dictionary.keys())
    # Initialize the x-gate count
    x_gates = 0

    # Iterate through consecutive pairs of keys
    for i in range(len(keys) - 1):
        key1 = keys[i]
        key2 = keys[i + 1]

        # Flag to indicate if x-gate merge occurs
        is_equal = False

        # Iterate through characters at each position
        for position in range(min(len(key1), len(key2))):
            char1 = key1[position]
            char2 = key2[position]

            # Check if x-gate merging condition is met: if two consecutive bit strings have a '0' at the same position
            if char1 == "0" and char2 == "0":
                is_equal = True
                break
            else:
                continue

        # Increment the counter if the condition is met
        if is_equal:
            x_gates += 1

    return x_gates


def generate_sparse_vect(n_qubit, d, vec_type="complex"):
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
        raise (
            ValueError(
                "Sparsity must be less or equal than the dimension of the vector\n"
            )
        )

    sparse_v = sp.sparse.random(1, N, density=d / N, format="csr", dtype=vec_type)
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
