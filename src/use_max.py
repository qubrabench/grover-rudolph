from qubrabench.algorithms.max import max
from qubrabench.stats import QueryStats


def max_value_of_function(it, f):
    stats = QueryStats()
    result = max(it, key=f, error=1e-5, stats=stats)

    return result, stats
