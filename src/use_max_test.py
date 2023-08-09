from .use_max import max_value_of_function
import numpy as np
import pytest


def test_use_max_quadratic():
    def f(x: float):
        return 4 * x - x**2

    domain = np.linspace(-10, 10, num=10000)
    result, _ = max_value_of_function(domain, f)
    assert result == pytest.approx(2, rel=1e-3)
