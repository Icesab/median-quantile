import numpy as np

try:
    from .utils import (
        clip_mean_to_median_band,
        local_mean,
        local_median,
        poisson_prob_self_leq,
        poisson_prob_strict_less,
        weighted_quantile,
    )
except ImportError:  # pragma: no cover - fallback for direct local imports
    from utils import (
        clip_mean_to_median_band,
        local_mean,
        local_median,
        poisson_prob_self_leq,
        poisson_prob_strict_less,
        weighted_quantile,
    )


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


def methodA_local_denoise(
    counts,
    peak,
    loc_win=3,
    fit_window=5,
    alpha=0.2,
    clip_ratio=(0.7, 1.3),
    median_floor=0.0,
    q_clip=None,
):
    if fit_window % 2 == 0:
        raise ValueError("fit_window must be odd")

    counts = np.asarray(counts, dtype=float)
    h, w = counts.shape

    mu_loc_map = local_mean(counts, win=loc_win)
    m_loc_map = local_median(counts, win=loc_win)
    mu_clip_map = clip_mean_to_median_band(
        mu_loc_map,
        m_loc_map,
        lower=clip_ratio[0],
        upper=clip_ratio[1],
        median_floor=median_floor,
    )
    lam_bar_map = float(alpha) * m_loc_map + (1.0 - float(alpha)) * mu_clip_map
    q_map = poisson_prob_self_leq(lam_bar_map)
    if q_clip is not None:
        q_map = np.clip(q_map, q_clip[0], q_clip[1])

    rad = fit_window // 2
    padded = np.pad(counts, pad_width=rad, mode="reflect")
    denoised_counts = np.zeros_like(counts, dtype=float)

    for r in range(h):
        for c in range(w):
            patch = padded[r : r + fit_window, c : c + fit_window].reshape(-1)
            denoised_counts[r, c] = weighted_quantile(patch, None, q_map[r, c])

    denoised_counts = np.maximum(denoised_counts, 0.0)
    denoised_intensity = denoised_counts / float(peak)
    return (
        denoised_counts,
        denoised_intensity,
        q_map,
        mu_loc_map,
        m_loc_map,
        mu_clip_map,
        lam_bar_map,
    )
