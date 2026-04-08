import numpy as np

try:
    from .utils import local_mean, local_median
except ImportError:  # pragma: no cover - fallback for direct local imports
    from utils import local_mean, local_median


def noisy_passthrough(counts: np.ndarray) -> np.ndarray:
    return counts.astype(float)


def mean_filter_3x3(counts: np.ndarray) -> np.ndarray:
    return local_mean(counts, win=3)


def median_filter_3x3(counts: np.ndarray) -> np.ndarray:
    return local_median(counts, win=3)
