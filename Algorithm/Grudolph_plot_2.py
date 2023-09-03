from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

data_folder = Path(__file__).parent.parent / "data"  # ../data
data_folder.mkdir(parents=True, exist_ok=True)  # create it if it does not already exist


def generate_plots(n_qubit):
    (
        d,
        opt_old0,
        opt_old1,
        opt_old2,
        opt_old_err0,
        opt_old_err1,
        opt_old_err2,
        perm_opt0,
        perm_opt1,
        perm_opt2,
        perm_opt_err0,
        perm_opt_err1,
        perm_opt_err2,
        oldcount0,
        oldcount1,
        oldcount2,
        oldcount_err0,
        oldcount_err1,
        oldcount_err2,
        optcount0,
        optcount1,
        optcount2,
        optcount_err0,
        optcount_err1,
        optcount_err2,
        perm0,
        perm1,
        perm2,
        perm_err0,
        perm_err1,
        perm_err2,
    ) = np.loadtxt(data_folder / f"Count_{n_qubit}.npy", unpack=True)

    plt.errorbar(d, opt_old0, yerr=opt_old_err0, color="r", label="Toffoli opt/not opt")
    plt.errorbar(d, opt_old1, yerr=opt_old_err1, color="g", label="CNOT opt/ not opt")
    plt.errorbar(
        d, opt_old2, yerr=opt_old_err2, color="b", label="1-qbt gates opt/ not opt"
    )
    """
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

    slope, intercept, r_value, p_value, std_err = stats.linregress(d, c_t)
    print(f"Fit Number of Toffoli gates:\nslope: {slope}", f"\t intercept: {intercept}\n")

    slope, intercept, r_value, p_value, std_err = stats.linregress(d, c2)
    print(f"Fit Number of 2-qubits gates:\nslope: {slope}", f"\t intercept: {intercept}\n")

    slope, intercept, r_value, p_value, std_err = stats.linregress(d, c1)
    print(f"Fit Number of 1-qubits gates:\nslope: {slope}", f"\t intercept: {intercept}\n")
    """


if __name__ == "__main__":
    for n in [2, 3, 4, 5]:
        generate_plots(n)
