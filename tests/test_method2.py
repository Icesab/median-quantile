import unittest

import numpy as np

from src.data import generate_poisson_counts, load_clean_image
from src.method2 import method2_denoise
from src.utils import extract_patch, glr_patch_distance, local_mean, poisson_prob_strict_less, weighted_quantile


def method2_reference(
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
                    dist = glr_patch_distance(ref_patch, cand_patch)
                    candidates.append((dist, rr, cc))

            if not candidates:
                denoised_counts[r, c] = counts[r, c]
                p_map[r, c] = 0.5
                m_nl_map[r, c] = counts[r, c]
                continue

            candidates.sort(key=lambda item: item[0])
            selected = candidates[: min(top_k, len(candidates))]
            dists = np.array([dist for dist, _, _ in selected], dtype=float)
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


class Method2Tests(unittest.TestCase):
    def test_optimized_method2_matches_reference(self):
        clean = load_clean_image("camera", size=(12, 12))
        counts, _ = generate_poisson_counts(clean, peak=8, seed=0)

        actual = method2_denoise(counts, peak=8, patch_size=3, search_size=5, top_k=4)
        expected = method2_reference(counts, peak=8, patch_size=3, search_size=5, top_k=4)

        for actual_arr, expected_arr in zip(actual, expected):
            self.assertEqual(actual_arr.shape, expected_arr.shape)
            np.testing.assert_allclose(actual_arr, expected_arr, atol=1e-10, rtol=1e-10)

        self.assertGreaterEqual(actual[2].min(), 0.05)
        self.assertLessEqual(actual[2].max(), 0.95)


if __name__ == "__main__":
    unittest.main()
