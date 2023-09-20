from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_folder = Path(__file__).parent.parent / "data"  # ../data
data_folder.mkdir(parents=True, exist_ok=True)  # create it if it does not already exist


def make_plot(data: pd.DataFrame, *, title: str = ""):
    colors = {"Toffoli": "r", "CNOT": "g", "1-qubit": "b"}

    data = data.groupby(["d"])
    means = data.mean(numeric_only=True)
    errors = data.sem(numeric_only=True)

    for gate_type, color in colors.items():
        plt.errorbar(
            means.index,
            means[gate_type],
            yerr=errors[gate_type],
            color=color,
            label=gate_type,
        )
    plt.axhline(
        y=1.0,
        color="k",
        linestyle="dashed",
    )
    plt.xlabel("d")
    plt.title(title)
    plt.legend()


def generate_plots(n_qubit, *, show_plots=True):
    fulldata = pd.read_csv(data_folder / f"Count_{n_qubit}.csv", index_col=False)

    data = {group[0]: data for group, data in fulldata.groupby(["name"])}

    make_plot(
        data["opt_old"],
        title="Number of gates with optimized GR / Number of gates with standard GR",
    )
    plt.figure()

    make_plot(
        data["perm_opt"],
        title="Number of gates with permutation GR / Number of gates with optimized GR",
    )

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
