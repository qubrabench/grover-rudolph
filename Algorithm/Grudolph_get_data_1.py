from pathlib import Path
import numpy as np
from helping_sp import generate_sparse_vect
from state_preparation import (
    phase_angle_dict,
    not_op_phase_angle_dict,
    gate_count,
)

data_folder = Path(__file__).parent / "data"


def generate_data():
    """
    Create a txt file with the data: sparsity, gate count
    Creates 2^N * repeat data
    Notice that each time you run the program more data are added to the file referring to n_qubit, if you want to avoid this open the file with 'w' instead of 'a'
    """

    n_qubit = 10
    N = 2**n_qubit
    repeat = 10
    percentage = 100  # sparsity percentage
    step = 10  # every step in sparsity get one data

    with open(data_folder / f"Gate_count_{n_qubit}.npy", "w") as f:
        for d in range(1, int(N * percentage / 100), step):
            for i in range(repeat):
                vector, nonzero_loc = generate_sparse_vect(n_qubit, d)
                # count = main(vector, nonzero_loc, n_qubit)

                # compare with ver 1.0 GR
                vect = np.zeros(N, dtype=np.complex128)
                for j in range(len(vector)):
                    vect[nonzero_loc[j]] = vector[j]

                op_angle_phase_dict = phase_angle_dict(vect)
                op_count = gate_count(op_angle_phase_dict)

                angle_phase_dict = not_op_phase_angle_dict(vect)
                count = gate_count(angle_phase_dict)

                f.write(
                    f"{d}\t{count[0]}\t{count[1]}\t{count[2]}\t{op_count[0]}\t{op_count[1]}\t{op_count[2]}\n"
                )
            print(
                int(d / step),
                "/",
                int(N * percentage / (step * 100)),
                "---------------------------------------------------------------------",
            )  # check status


if __name__ == "__main__":
    generate_data()
