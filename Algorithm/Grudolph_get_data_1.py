from state_preparation import *

# Create a txt file with the data: sparsity, gate count
# Creates 2^N * repeat data
# Notice that each time you run the program more data are added to the file referring to n_qubit, if you want to avoid this open the file with 'w' instead of 'a'

n_qubit = 10
N = 2**n_qubit
repeat = 1
total_steps = repeat * (N - 1)

with open(f"test/Gate_count_{n_qubit}.npy", "w") as f:
    for i in range(repeat):
        for d in range(1, 2**n_qubit):
            vector, nonzero_loc = sparse_couple_vect(n_qubit, d)
            count = main(vector, nonzero_loc, n_qubit)

            # compare with ver 1.0 GR
            vect = np.zeros(N, dtype=np.complex128)
            for j in range(len(vector)):
                vect[nonzero_loc[j]] = vector[j]

            angle_phase_dict = phase_angle_dict(vect)
            oldcount = gate_count(angle_phase_dict)

            f.write(
                f"{d}\t{count[0]}\t{count[1]}\t{count[2]}\t{oldcount[0]}\t{oldcount[1]}\t{oldcount[2]}\n"
            )
            print(
                d + (i * (N - 1)),
                "/",
                total_steps,
                "---------------------------------------------------------------------",
            )  # check status
