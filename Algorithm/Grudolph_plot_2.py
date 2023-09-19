from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

data_folder = Path(__file__).parent.parent / "data"  # ../data
data_folder.mkdir(parents=True, exist_ok=True)  # create it if it does not already exist


def generate_plots(n_qubit, *, show_plots=True):
    # TODO organize data better for easier debugging.
    # - could use JSON, with dictionary keys to keep track of the variables
    # - or a python pickle (disadvantage: not human readable)
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

    plt.errorbar(d, opt_old0, yerr=opt_old_err0, color="r", label="Toffoli")
    plt.errorbar(d, opt_old1, yerr=opt_old_err1, color="g", label="CNOT")
    plt.errorbar(d, opt_old2, yerr=opt_old_err2, color="b", label="1-qbt gates")
    plt.axhline(
        y=1.0,
        color="k",
        linestyle="dashed",
    )
    plt.xlabel("d")
    plt.title("Number of gates with optimized GR / Number of gates with standard GR")
    plt.legend()

    plt.figure()

    plt.errorbar(d, perm_opt0, yerr=perm_opt_err0, color="r", label="Toffoli")
    # TODO(bug) should these below be 1 and 2?
    plt.errorbar(d, perm_opt0, yerr=perm_opt_err0, color="g", label="CNOT")
    plt.errorbar(d, perm_opt0, yerr=perm_opt_err0, color="b", label="1-qbt gates")
    plt.axhline(
        y=1.0,
        color="k",
        linestyle="dashed",
    )
    plt.title("Number of gates with permutation GR / Number of gates with optimized GR")
    plt.legend()
    plt.xlabel("d")

    if show_plots:
        plt.show()
    """
    # FIT

    slope, intercept, r_value, p_value, std_err = stats.linregress(d, c_t)
    print(f"Fit Number of Toffoli gates:\nslope: {slope}", f"\t intercept: {intercept}\n")

    slope, intercept, r_value, p_value, std_err = stats.linregress(d, c2)
    print(f"Fit Number of 2-qubits gates:\nslope: {slope}", f"\t intercept: {intercept}\n")

    slope, intercept, r_value, p_value, std_err = stats.linregress(d, c1)
    print(f"Fit Number of 1-qubits gates:\nslope: {slope}", f"\t intercept: {intercept}\n")
    """


if __name__ == "__main__":
    generate_plots(16)
