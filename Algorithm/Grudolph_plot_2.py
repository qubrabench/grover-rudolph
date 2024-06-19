from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_folder = Path(__file__).parent.parent / "data"  # ../data
data_folder.mkdir(parents=True, exist_ok=True)  # create it if it does not already exist

plot_folder = Path(__file__).parent.parent / "plots"  # ../plots
plot_folder.mkdir(parents=True, exist_ok=True)


def make_plot(
    z_label: str,
    z_values: list,
    data_name: str,
    *,
    title: str = "",
    x_axis: str = "",
    ratio_plot: bool = False,
    log_x_axis: bool = False,
    log_y_axis: bool = False,
    font_size: int = 15,
):
    for z in z_values:
        fulldata = pd.read_csv(
            data_folder / f"Count_{z_label}_{z}.csv", index_col=False
        )
        data = {group[0]: data for group, data in fulldata.groupby(["name"])}
        data = data[data_name]

        data = data.groupby([x_axis])
        means = data.mean(numeric_only=True)  # type: ignore
        errors = data.sem(numeric_only=True)  # type: ignore

        x_label = x_axis

        # When plotting ration plot as a function of d/N
        x = means.index
        y = means["Toffoli"]
        yerr = errors["Toffoli"]

        if log_x_axis:
            mask = y != 0
            y = y[mask]
            x = x[mask]
            yerr = yerr[mask]

        if ratio_plot:
            x_label = "d/N"

            if z_label == "n":
                x = x / 2**z
            else:
                x = z / 2**x

        plt.errorbar(x, y, yerr=yerr, label=f"{z_label} = {z}")

    if log_y_axis:
        plt.yscale("log")
    if log_x_axis:
        plt.xscale("log")

    if ratio_plot:
        plt.axhline(
            y=1.0,
            color="k",
            linestyle="dashed",  # type: ignore
        )

    plt.xlabel(x_label, fontsize=font_size)
    plt.title(title, fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.savefig(plot_folder / f"{data_name}_{z_label}.png")


def generate_plots_as_a_function_of_d(n_values: list, *, show_plots: bool = True):
    make_plot(
        "n",
        n_values,
        "perm",
        title="#Toffoli with Alg. 5",
        x_axis="d",
        log_x_axis=True,
        log_y_axis=True,
    )
    plt.figure()

    make_plot(
        "n",
        n_values,
        "optcount",
        title="# Toffoli with Alg. 6",
        x_axis="d",
        log_x_axis=True,
        log_y_axis=True,
    )
    plt.figure()

    make_plot(
        "n",
        n_values,
        "oldcount",
        title="# Toffoli with Alg. 1",
        x_axis="d",
        log_x_axis=True,
        log_y_axis=True,
    )
    plt.figure()

    make_plot(
        "n",
        n_values,
        "opt_old",
        title="#Toffoli with Alg. 6 / Alg. 1",
        x_axis="d",
        ratio_plot=True,
        log_x_axis=True,
    )
    plt.figure()

    make_plot(
        "n",
        n_values,
        "perm_opt",
        title="#Toffoli with Alg. 5 / Alg. 6",
        x_axis="d",
        ratio_plot=True,
        log_x_axis=True,
    )

    if show_plots:
        plt.show()


def generate_plots_as_a_function_of_n(d_values: list, *, show_plots: bool = True):
    make_plot(
        "d",
        d_values,
        "perm",
        title="#Toffoli with Alg. 5",
        x_axis="n",
    )
    plt.figure()

    make_plot(
        "d",
        d_values,
        "optcount",
        title="#Toffoli with Alg. 6",
        x_axis="n",
    )
    plt.figure()

    make_plot(
        "d",
        d_values,
        "oldcount",
        title="#Toffoli with Alg. 1",
        x_axis="n",
    )
    plt.figure()

    make_plot(
        "d",
        d_values,
        "opt_old",
        title="#Toffoli with Alg. 6 / Alg. 1",
        x_axis="n",
        ratio_plot=True,
        log_x_axis=True,
    )
    plt.figure()

    make_plot(
        "d",
        d_values,
        "perm_opt",
        title="#Toffoli with Alg. 5 / Alg. 6",
        x_axis="n",
        ratio_plot=True,
        log_x_axis=True,
    )

    if show_plots:
        plt.show()


if __name__ == "__main__":
    generate_plots_as_a_function_of_d([12, 16, 20])
    generate_plots_as_a_function_of_n([10, 100, 200])
