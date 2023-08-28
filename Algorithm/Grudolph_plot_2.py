import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

n_qubit = 10

d, c_t, c2, c1, old_ct, old_c2, old_c1 = np.loadtxt(
    f"data/Gate_count_{n_qubit}.npy", unpack=True
)

fig, axs = plt.subplots(2)

axs[0].scatter(d, c_t, color="r", label="Toffoli")
axs[0].scatter(d, c2, color="g", label="CNOT")
axs[0].scatter(d, c1, color="b", label="1-qubit gates")

axs[1].scatter(d, old_ct, color="darkred", label="old Toffoli")
axs[1].scatter(d, old_c2, color="darkgreen", label="old CNOT")
axs[1].scatter(d, old_c1, color="darkblue", label="old 1-qubit gates")

axs[1].set_xlabel("Sparsity d")
axs[0].set_ylabel("Permutation Grover Rudolph")
axs[1].set_ylabel("Improved Grover Rudolph/Permutation")


axs[0].legend()
axs[1].legend()

plt.show()
# FIT
slope, intercept, r_value, p_value, std_err = stats.linregress(d, c_t)
print(f"Fit Number of Toffoli gates:\nslope: {slope}", f"\t intercept: {intercept}\n")

slope, intercept, r_value, p_value, std_err = stats.linregress(d, c2)
print(f"Fit Number of 2-qubits gates:\nslope: {slope}", f"\t intercept: {intercept}\n")

slope, intercept, r_value, p_value, std_err = stats.linregress(d, c1)
print(f"Fit Number of 1-qubits gates:\nslope: {slope}", f"\t intercept: {intercept}\n")

plt.show()
