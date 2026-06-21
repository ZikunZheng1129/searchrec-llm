"""Random seed helpers."""

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set lightweight random seeds for reproducible local experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
