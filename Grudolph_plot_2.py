import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

n_qubit = 10

d, c_t, c2, c1 = np.loadtxt(f"test/Gate_count_{n_qubit}.npy", unpack=True)

fig = plt.figure()

plt.title("Gate Counting")

ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)

ax1.scatter(d, c_t, marker=".")
ax2.scatter(d, c2, marker=".")
ax3.scatter(d, c1, marker=".")

ax1.get_shared_x_axes().join(ax1, ax3)
ax2.get_shared_x_axes().join(ax2, ax3)
ax1.set_xticklabels([])
ax2.set_xticklabels([])

n = 15

ax3.set_xlabel("Sparsity", fontsize=n)
ax1.set_ylabel("Toffoli Gates", fontsize=n)
ax2.set_ylabel("2-qubit Gates", fontsize=n)
ax3.set_ylabel("1-qubit Gates", fontsize=n)

# FIT
slope, intercept, r_value, p_value, std_err = stats.linregress(d, c_t)
print(f"Fit Number of Toffoli gates:\nslope: {slope}", f"\t intercept: {intercept}\n")

slope, intercept, r_value, p_value, std_err = stats.linregress(d, c2)
print(f"Fit Number of 2-qubits gates:\nslope: {slope}", f"\t intercept: {intercept}\n")

slope, intercept, r_value, p_value, std_err = stats.linregress(d, c1)
print(f"Fit Number of 1-qubits gates:\nslope: {slope}", f"\t intercept: {intercept}\n")

plt.show()
