from pathlib import Path
import pandas as pd

from helping_sp import generate_sparse_unit_vector, ZERO, optimize_dict
from state_preparation import phase_angle_dict, gate_count, main, GateCounts

data_folder = Path(__file__).parent.parent / "data"  # ../data
data_folder.mkdir(parents=True, exist_ok=True)  # create it if it does not already exist


class Stats:
    def __init__(self):
        self.data = []

    def add_row(self, name: str, gate_counts: GateCounts | list[int]):
        self.data.append([name] + list(gate_counts))


def generate_data(
    n_qubit: int, repeat: int = 1, percentage: float = 100, step: int = 1
):
    """
    Create a txt file with the data: sparsity, gate count
    Creates 2^N * repeat data
    Notice that each time you run the program more data are added to the file referring to n_qubit, if you want to avoid this open the file with 'w' instead of 'a'

    Args:
        n_qubit: number of qubits
        repeat: TODO add description
        percentage: TODO add description
        step: TODO add description
    """

    N = 2**n_qubit
    d_range = range(1, int(N * percentage / 100), step)

    data: list[pd.DataFrame] = []
    for ix, d in enumerate(d_range):
        stats = Stats()
        for _ in range(repeat):
            # Permutation GR
            full_vec = generate_sparse_unit_vector(n_qubit, d)

            vector = full_vec.data
            nonzero_loc = full_vec.nonzero()[1]

            perm = main(full_vec, n_qubit)
            stats.add_row("perm", perm)

            # compare with ver 1.0 GR
            angle_phase_dict = phase_angle_dict(
                vector, nonzero_loc, n_qubit, optimization=False
            )
            oldcount = gate_count(angle_phase_dict)
            stats.add_row("oldcount", oldcount)

            # Compare with optimized GR
            angle_phase_dict_opt = [optimize_dict(gates) for gates in angle_phase_dict]
            optcount = gate_count(angle_phase_dict_opt)
            stats.add_row("optcount", optcount)

            # update values
            opt_old = [
                (optcount[j] / oldcount[j]) if oldcount[j] > ZERO else 0.0
                for j in range(3)
            ]
            stats.add_row("opt_old", opt_old)

            perm_opt = [
                (perm[j] / optcount[j]) if optcount[j] > ZERO else 0.0 for j in range(3)
            ]
            stats.add_row("perm_opt", perm_opt)

        data_d = pd.DataFrame(
            stats.data, columns=["name", "Toffoli", "CNOT", "1-qubit"]
        )
        data_d.insert(1, "d", d)  # type: ignore

        data.append(data_d)

        # check status
        print(f"{ix + 1} / {len(d_range)} " + "-" * 40)

    pd.concat(data).to_csv(data_folder / f"Count_{n_qubit}.csv", mode="w", index=False)


if __name__ == "__main__":
    generate_data(16, percentage=1, step=100, repeat=10)
