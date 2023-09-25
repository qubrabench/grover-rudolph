from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

data_folder = Path(__file__).parent.parent / "data"  # ../data
data_folder.mkdir(parents=True, exist_ok=True)  # create it if it does not already exist


def make_plot(full_data: pd.DataFrame, *, title: str = ""):
    colors = {"Toffoli": "r", "CNOT": "g", "1-qubit": "b"}

    data = full_data.groupby(["d"])
    means = data.mean(numeric_only=True)  # type: ignore
    errors = data.sem(numeric_only=True)  # type: ignore

    for gate_type, color in colors.items():
        plt.errorbar(  # type: ignore
            means.index,
            means[gate_type],
            yerr=errors[gate_type],
            color=color,
            label=gate_type,
        )
    plt.axhline(
        y=1.0,
        color="k",
        linestyle="dashed",  # type: ignore
    )
    plt.xlabel("d")
    plt.title(title)
    plt.legend()


def generate_plots(n_qubit: int, *, show_plots: bool = True):
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


if __name__ == "__main__":
    generate_plots(16)
