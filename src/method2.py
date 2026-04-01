import numpy as np
from .utils import (
    local_mean,
    poisson_prob_strict_less,
    weighted_quantile,
    extract_patch,
    glr_patch_distance,
)


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

    for r in range(h):
        for c in range(w):
            ref_patch = extract_patch(counts, (r, c), patch_size)
            candidates = []

            r0, r1 = max(0, r - search_rad), min(h, r + search_rad + 1)
            c0, c1 = max(0, c - search_rad), min(w, c + search_rad + 1)

            for rr in range(r0, r1):
                for cc in range(c0, c1):
                    if rr == r and cc == c:
                        continue
                    cand_patch = extract_patch(counts, (rr, cc), patch_size)
                    d = glr_patch_distance(ref_patch, cand_patch)
                    candidates.append((d, rr, cc))

            if not candidates:
                denoised_counts[r, c] = counts[r, c]
                p_map[r, c] = 0.5
                m_nl_map[r, c] = counts[r, c]
                continue

            candidates.sort(key=lambda x: x[0])
            selected = candidates[: min(top_k, len(candidates))]
            dists = np.array([d for d, _, _ in selected], dtype=float)
            values = np.array([counts[rr, cc] for _, rr, cc in selected], dtype=float)

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
