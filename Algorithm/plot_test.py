import pytest
from Grudolph_get_data_1 import generate_data
from Grudolph_plot_2 import generate_plots


@pytest.mark.parametrize("n_qubit", [3, 4, 5, 6])
def test_generate_data_and_plots(n_qubit):
    generate_data(n_qubit, repeat=10)
    generate_plots(n_qubit, show_plots=False)
