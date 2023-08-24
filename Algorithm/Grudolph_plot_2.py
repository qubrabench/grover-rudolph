import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

n_qubit = 10

d, c_t, c2, c1, old_ct, old_c2, old_c1 = np.loadtxt(
    f"test/Gate_count_{n_qubit}.npy", unpack=True
)

fig, axs = plt.subplots(2)

axs[0].scatter(d, c_t, color="r", label="Toffoli")
axs[0].scatter(d, c2, color="g", label="CNOT")
axs[0].scatter(d, c1, color="b", label="1-qubit gates")

axs[1].scatter(d, old_ct / c_t, color="darkred", label="old Toffoli")
axs[1].scatter(d, old_c2 / c2, color="darkgreen", label="old CNOT")
axs[1].scatter(d, old_c1 / c1, color="darkblue", label="old 1-qubit gates")

plt.xlabel("Sparsity d")
plt.ylabel("Gate count")
plt.title("")

plt.legend()

# To load the display window
plt.show()
# FIT
slope, intercept, r_value, p_value, std_err = stats.linregress(d, c_t)
print(f"Fit Number of Toffoli gates:\nslope: {slope}", f"\t intercept: {intercept}\n")

slope, intercept, r_value, p_value, std_err = stats.linregress(d, c2)
print(f"Fit Number of 2-qubits gates:\nslope: {slope}", f"\t intercept: {intercept}\n")

slope, intercept, r_value, p_value, std_err = stats.linregress(d, c1)
print(f"Fit Number of 1-qubits gates:\nslope: {slope}", f"\t intercept: {intercept}\n")

plt.show()
