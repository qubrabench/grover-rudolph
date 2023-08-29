import numpy as np
import matplotlib.pyplot as plt

# from scipy import stats


def generate_plots():
    n_qubit = 10

    d, c_t, c2, c1, op_ct, op_c2, op_c1 = np.loadtxt(
        f"data/Gate_count_{n_qubit}.npy", unpack=True
    )

    fig, axs = plt.subplots(2)

    axs[0].scatter(d, c_t, color="r", label="Toffoli")
    axs[0].scatter(d, c2, color="g", label="CNOT")
    axs[0].scatter(d, c1, color="b", label="1-qubit gates")

    axs[1].scatter(d, op_ct, color="darkred", label="opt Toffoli")
    axs[1].scatter(d, op_c2, color="darkgreen", label="opt CNOT")
    axs[1].scatter(d, op_c1, color="darkblue", label="opt 1-qubit gates")

    axs[1].set_xlabel("Sparsity d")
    axs[0].set_ylabel("Permutation Grover Rudolph")
    axs[1].set_ylabel("Improved Grover Rudolph/Permutation")

    axs[0].legend()
    axs[1].legend()

    plt.show()

    plt.figure()

    plt.scatter(d, op_ct / c_t, color="r", label="Toffoli opt/not opt")
    plt.scatter(d, op_c2 / c2, color="g", label="CNOT opt/ not opt")
    plt.scatter(d, op_c1 / c1, color="b", label="1-qbt gates opt/ not opt")

    plt.plot(d, d * 0 + 1.0, color="k", linestyle="dashed")

    plt.legend()

    plt.show()
    # FIT

    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(d, c_t)
    print(f"Fit Number of Toffoli gates:\nslope: {slope}", f"\t intercept: {intercept}\n")

    slope, intercept, r_value, p_value, std_err = stats.linregress(d, c2)
    print(f"Fit Number of 2-qubits gates:\nslope: {slope}", f"\t intercept: {intercept}\n")

    slope, intercept, r_value, p_value, std_err = stats.linregress(d, c1)
    print(f"Fit Number of 1-qubits gates:\nslope: {slope}", f"\t intercept: {intercept}\n")
    """


if __name__ == "__main__":
    generate_plots()
