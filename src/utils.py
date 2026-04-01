import numpy as np
from scipy.ndimage import uniform_filter, median_filter
from scipy.stats import poisson


def local_mean(counts: np.ndarray, win: int = 3) -> np.ndarray:
    """Compute local mean with reflect padding semantics."""
    return uniform_filter(counts.astype(float), size=win, mode="reflect")


def local_median(counts: np.ndarray, win: int = 3) -> np.ndarray:
    """Compute local median with reflect padding semantics."""
    return median_filter(counts.astype(float), size=win, mode="reflect")


def poisson_prob_strict_less(lam, threshold):
    """
    Compute P(S < threshold), S ~ Poisson(lam).
    Uses CDF at threshold - 1 and returns 0 when threshold <= 0.
    """
    lam_arr = np.asarray(lam, dtype=float)
    thr_arr = np.asarray(threshold, dtype=float)
    result = np.zeros(np.broadcast(lam_arr, thr_arr).shape, dtype=float)
    valid = thr_arr > 0
    if np.any(valid):
        result[valid] = poisson.cdf(thr_arr[valid] - 1, lam_arr[valid])
    return result


def weighted_quantile(values, weights=None, q: float = 0.5) -> float:
    """
    Weighted empirical quantile.
    Returns the first value whose cumulative normalized weight >= q.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("values cannot be empty")

    q = float(np.clip(q, 0.0, 1.0))

    if weights is None:
        weights = np.ones_like(values, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.shape != values.shape:
            raise ValueError("weights must have same shape as values")

    weights = np.maximum(weights, 0.0)
    total = weights.sum()
    if total <= 0:
        weights = np.ones_like(values, dtype=float)
        total = weights.sum()
    weights = weights / total

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumsum = np.cumsum(sorted_weights)

    idx = int(np.searchsorted(cumsum, q, side="left"))
    idx = min(idx, sorted_values.size - 1)
    return float(sorted_values[idx])


def extract_patch(img: np.ndarray, center, patch_size: int, padding: str = "reflect") -> np.ndarray:
    """Extract patch centered at (r, c) with reflect padding."""
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd")

    r, c = center
    rad = patch_size // 2
    padded = np.pad(img, pad_width=rad, mode=padding)
    rp, cp = r + rad, c + rad
    return padded[rp - rad : rp + rad + 1, cp - rad : cp + rad + 1]


def glr_patch_distance(patch_a: np.ndarray, patch_b: np.ndarray) -> float:
    """GLR patch distance for nonnegative count patches."""
    a = np.asarray(patch_a, dtype=float)
    b = np.asarray(patch_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("patches must have same shape")
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("patch values must be nonnegative")

    m = (a + b) / 2.0

    def xlogx(x):
        out = np.zeros_like(x, dtype=float)
        nz = x > 0
        out[nz] = x[nz] * np.log(x[nz])
        return out

    dist = np.sum(xlogx(a) + xlogx(b) - (a + b) * np.where(m > 0, np.log(m), 0.0))
    return float(dist)
