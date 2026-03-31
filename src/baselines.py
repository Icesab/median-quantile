import numpy as np
from .utils import local_mean, local_median


def noisy_passthrough(counts: np.ndarray) -> np.ndarray:
    return counts.astype(float)


def mean_filter_3x3(counts: np.ndarray) -> np.ndarray:
    return local_mean(counts, win=3)


def median_filter_3x3(counts: np.ndarray) -> np.ndarray:
    return local_median(counts, win=3)
