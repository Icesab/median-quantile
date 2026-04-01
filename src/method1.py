import numpy as np
from .utils import local_mean, local_median, poisson_prob_strict_less, weighted_quantile


def method1_denoise(counts, peak, loc_win=3, fit_window=3, p_clip=(0.05, 0.95)):
    counts = np.asarray(counts, dtype=float)
    h, w = counts.shape

    mu_loc_map = local_mean(counts, win=loc_win)
    m_loc_map = local_median(counts, win=loc_win)

    p_raw = poisson_prob_strict_less(9.0 * mu_loc_map, 9.0 * m_loc_map)
    p_map = np.clip(p_raw, p_clip[0], p_clip[1])

    rad = fit_window // 2
    padded = np.pad(counts, pad_width=rad, mode="reflect")
    denoised_counts = np.zeros_like(counts, dtype=float)

    for r in range(h):
        for c in range(w):
            patch = padded[r : r + fit_window, c : c + fit_window].reshape(-1)
            denoised_counts[r, c] = weighted_quantile(patch, None, p_map[r, c])

    denoised_intensity = denoised_counts / float(peak)
    return denoised_counts, denoised_intensity, p_map, mu_loc_map, m_loc_map
