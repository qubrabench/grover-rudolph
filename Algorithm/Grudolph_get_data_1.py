from pathlib import Path
import numpy as np
from helping_sp import generate_sparse_vect
from state_preparation import (
    phase_angle_dict,
    gate_count,
)

data_folder = Path(__file__).parent / "data"


def generate_data(n_qubit, repeat=1, percentage=100, step=1):
    """
    Create a txt file with the data: sparsity, gate count
    Creates 2^N * repeat data
    Notice that each time you run the program more data are added to the file referring to n_qubit, if you want to avoid this open the file with 'w' instead of 'a'
    """

    N = 2**n_qubit

    with open(data_folder / f"Count_{n_qubit}.npy", "w") as f:
        for d in range(1, int(N * percentage / 100), step):
            opt_old = np.empty((repeat, 3), dtype=float)
            perm_opt = np.empty((repeat, 3), dtype=float)
            oldcount = np.empty((repeat, 3), dtype=float)
            optcount = np.empty((repeat, 3), dtype=float)
            perm = np.empty((repeat, 3), dtype=float)

            for i in range(repeat):
                # Permutation GR
                vector, nonzero_loc = generate_sparse_vect(n_qubit, d)
                perm[i][:] = main(vector, nonzero_loc, n_qubit)

                # compare with ver 1.0 GR
                angle_phase_dict = phase_angle_dict(
                    vector, nonzero_loc, n_qubit, optimization=False
                )
                oldcount[i][:] = gate_count(angle_phase_dict)

                # Compare with optimized GR
                for index, dictionary in enumerate(angle_phase_dict):
                    angle_phase_dict[index] = optimize_dict(dictionary)
                optcount[i][:] = gate_count(angle_phase_dict)

                # update values
                opt_old[i][:] = [
                    (optcount[i][j] / oldcount[i][j]) if oldcount[i][j] > ZERO else 0.0
                    for j in range(3)
                ]

                perm_opt[i][:] = [
                    (perm[i][j] / optcount[i][j]) if optcount[i][j] > ZERO else 0.0
                    for j in range(3)
                ]

            # Write in a file mean and standard deviation
            data = np.concatenate(
                [
                    d,
                    np.mean(opt_old, axis=0),
                    np.std(opt_old, axis=0),
                    np.mean(perm_opt, axis=0),
                    np.std(perm_opt, axis=0),
                    np.mean(oldcount, axis=0),
                    np.std(oldcount, axis=0),
                    np.mean(optcount, axis=0),
                    np.std(optcount, axis=0),
                    np.mean(perm, axis=0),
                    np.std(perm, axis=0),
                ],
                axis=None,
            )
            np.savetxt(f, data.reshape(1, -1), delimiter="\t")
            print(
                int(d / step),
                "/",
                int(N * percentage / (step * 100)),
                "---------------------------------------------------------------------",
            )  # check status


if __name__ == "__main__":
    generate_data()
