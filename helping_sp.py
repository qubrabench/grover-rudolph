import numpy as np
from numpy import linalg
from itertools import combinations
import random

# global zero precision
ZERO = 1e-8


def merge_dict(dict1, dict2):
    """
    Merges the angle dictionary and the phase dictionary by joining their values in a list
    Note that the order of the values in the dictionary is important for later, since the gates don't commute
    Thus, we do first all the rotations (phase == None), then the phase together with the rotations, in the end only the phases (angle == None)

    Args:
        dict1 = dictionary of angles
        dict2 = dictionary of phases

    Returns:
        A dictionary with elements of the form {key : [dict1[key], dict2[key]]}.
        If dict1[key] doesn't exist its value is set to None. Same for dict2.
    """

    dict3 = {}

    for k1, v1 in dict1.items():
        if k1 not in dict2:
            dict3[k1] = [v1, None]

    for k1, v1 in dict1.items():
        if k1 in dict2:
            dict3[k1] = [v1, dict2[k1]]

    for k2, v2 in dict2.items():
        if k2 not in dict1:
            dict3[k2] = [None, v2]

    return dict3


def where_diff_one(string_1, string_2) -> int | None:
    """
    Checks if two string differ by ONLY one char that is not 'e', and it finds its position
    It is checking if two controlled gates can be merged: if they differ in only one control
    >>> '010', '011' -> 2
    >>> '011', '100' -> None
    >>> 'e10', 'e10' -> None

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
    >>>{'11':3.14, ; '10':3.14} becomes {'1e':3.14} where 'e' means no control (identity)
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

            # Consider only different items with same value (angle)
            if abs(v1 - v2) > ZERO:
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


def sparse_couple_vect(N, d):
    """
    Generate random complex amplitudes vector of N qubits (length 2^N) with sparsity d
    as couples: position and value of the i-th non zero element
    The sign of the first entry  is  always real positive to fix the overall phase

        Args:
            Number of qubits N, sparsity d

        Returns:
            int array of lenght d, complex array of lenght d: the first one with the values and the second one (ordered) with the position of the non-zero element
    """
    vector = np.empty(d, dtype=np.complex128)
    nonzero_locations = np.empty(d, dtype=int)
    i = 0

    if d > 2**N:
        raise (
            ValueError(
                "Sparsity must be less or equal than the dimension of the vector\n"
            )
        )

    while i in range(d):
        position = random.randint(0, 2**N - 1)

        if position in nonzero_locations:
            continue

        nonzero_locations[i] = position
        vector[i] = np.random.uniform(-1, 1) + 1.0j * np.random.uniform(-1, 1)

        i += 1

    # increasing order of the locations, and binary representation
    nonzero_locations = np.sort(nonzero_locations)

    return vector / linalg.norm(vector), nonzero_locations
