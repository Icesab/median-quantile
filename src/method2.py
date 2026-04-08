import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:
    from .utils import local_mean, poisson_prob_strict_less, weighted_quantile
except ImportError:  # pragma: no cover - fallback for direct local imports
    from utils import local_mean, poisson_prob_strict_less, weighted_quantile


def _xlogx(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    log_values = np.zeros_like(values, dtype=float)
    np.log(values, out=log_values, where=values > 0)
    return values * log_values


def _build_patch_bank(counts: np.ndarray, patch_size: int) -> np.ndarray:
    patch_rad = patch_size // 2
    padded = np.pad(counts, pad_width=patch_rad, mode="reflect")
    windows = sliding_window_view(padded, (patch_size, patch_size))
    return windows.reshape(counts.shape[0], counts.shape[1], patch_size * patch_size)


def method2_denoise(
    counts,
    peak,
    loc_win=3,
    patch_size=5,
    search_size=21,
    top_k=24,
    p_clip=(0.05, 0.95),
):
    counts = np.asarray(counts, dtype=float)
    h, w = counts.shape
    search_rad = search_size // 2

    mu_loc_map = local_mean(counts, win=loc_win)
    denoised_counts = np.zeros_like(counts, dtype=float)
    p_map = np.zeros_like(counts, dtype=float)
    m_nl_map = np.zeros_like(counts, dtype=float)
    patches = _build_patch_bank(counts, patch_size)
    patch_xlogx_sum = _xlogx(patches).sum(axis=-1)

    for r in range(h):
        for c in range(w):
            r0, r1 = max(0, r - search_rad), min(h, r + search_rad + 1)
            c0, c1 = max(0, c - search_rad), min(w, c + search_rad + 1)

            ref_patch = patches[r, c]
            ref_sum = patch_xlogx_sum[r, c]

            window_patches = patches[r0:r1, c0:c1].reshape(-1, ref_patch.size)
            window_patch_sums = patch_xlogx_sum[r0:r1, c0:c1].reshape(-1)
            window_values = counts[r0:r1, c0:c1].reshape(-1)

            row_coords = np.repeat(np.arange(r0, r1), c1 - c0)
            col_coords = np.tile(np.arange(c0, c1), r1 - r0)
            center_mask = (row_coords != r) | (col_coords != c)

            candidate_patches = window_patches[center_mask]
            candidate_patch_sums = window_patch_sums[center_mask]
            candidate_values = window_values[center_mask]
            candidate_rows = row_coords[center_mask]
            candidate_cols = col_coords[center_mask]

            if candidate_patches.size == 0:
                denoised_counts[r, c] = counts[r, c]
                p_map[r, c] = 0.5
                m_nl_map[r, c] = counts[r, c]
                continue

            m = 0.5 * (candidate_patches + ref_patch)
            log_m = np.zeros_like(m, dtype=float)
            np.log(m, out=log_m, where=m > 0)
            dists = ref_sum + candidate_patch_sums - np.sum((candidate_patches + ref_patch) * log_m, axis=1)

            k = min(top_k, dists.size)
            if k < dists.size:
                selected_idx = np.argpartition(dists, k - 1)[:k]
            else:
                selected_idx = np.arange(dists.size)

            selected_dists = dists[selected_idx]
            selected_rows = candidate_rows[selected_idx]
            selected_cols = candidate_cols[selected_idx]
            stable_order = np.lexsort((selected_cols, selected_rows, selected_dists))
            selected_idx = selected_idx[stable_order]

            dists = dists[selected_idx]
            values = candidate_values[selected_idx]

            h_i = np.median(dists) + 1e-8
            weights = np.exp(-dists / h_i)
            weights = weights / (weights.sum() + 1e-12)

            m_nl = weighted_quantile(values=values, weights=weights, q=0.5)
            m_nl_map[r, c] = m_nl

            p_raw = poisson_prob_strict_less(9.0 * mu_loc_map[r, c], 9.0 * m_nl)
            p_val = float(np.clip(p_raw, p_clip[0], p_clip[1]))
            p_map[r, c] = p_val

            denoised_counts[r, c] = weighted_quantile(values=values, weights=weights, q=p_val)

    denoised_intensity = denoised_counts / float(peak)
    return denoised_counts, denoised_intensity, p_map, mu_loc_map, m_nl_map
