import numpy as np
from scipy import stats


def generate_data(size: int = 10, mean: float = 0):
    """
    generate random data from normal distribution
    """

    return np.random.randn(size) + mean
