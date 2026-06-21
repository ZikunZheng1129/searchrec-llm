import numpy as np

from src.utils.seed import set_seed


def test_set_seed_makes_numpy_deterministic() -> None:
    set_seed(123)
    first = np.random.rand(5)

    set_seed(123)
    second = np.random.rand(5)

    assert np.array_equal(first, second)
