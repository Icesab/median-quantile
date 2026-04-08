import unittest

import numpy as np

from src.data import generate_centered_additive_poisson, generate_poisson_counts, load_clean_image
from src.method1 import methodA_local_denoise
from src.utils import (
    clip_mean_to_median_band,
    local_mean,
    local_median,
    poisson_prob_self_leq,
    weighted_quantile,
)


def methodA_reference(
    counts,
    peak,
    loc_win=3,
    fit_window=5,
    alpha=0.2,
    clip_ratio=(0.7, 1.3),
    median_floor=0.0,
    q_clip=None,
):
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
    lam_bar_map = alpha * m_loc_map + (1.0 - alpha) * mu_clip_map
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


class Method1Tests(unittest.TestCase):
    def test_methodA_matches_reference(self):
        clean = load_clean_image("camera", size=(10, 10))
        counts, _ = generate_poisson_counts(clean, peak=8, seed=0)

        actual = methodA_local_denoise(
            counts,
            peak=8,
            loc_win=3,
            fit_window=5,
            alpha=0.2,
            clip_ratio=(0.7, 1.3),
        )
        expected = methodA_reference(
            counts,
            peak=8,
            loc_win=3,
            fit_window=5,
            alpha=0.2,
            clip_ratio=(0.7, 1.3),
        )

        for actual_arr, expected_arr in zip(actual, expected):
            self.assertEqual(actual_arr.shape, expected_arr.shape)
            np.testing.assert_allclose(actual_arr, expected_arr, atol=1e-10, rtol=1e-10)

        self.assertGreaterEqual(actual[0].min(), 0.0)
        self.assertGreaterEqual(actual[2].min(), 0.0)
        self.assertLessEqual(actual[2].max(), 1.0)

    def test_clip_mean_to_median_band_keeps_mean_when_median_is_zero(self):
        mean = np.array([[0.25, 5.0], [4.0, 1.0]])
        median = np.array([[0.0, 2.0], [10.0, 1.0]])

        clipped = clip_mean_to_median_band(mean, median, lower=0.7, upper=1.3, median_floor=0.0)

        expected = np.array([[0.25, 2.6], [7.0, 1.0]])
        np.testing.assert_allclose(clipped, expected, atol=1e-12, rtol=0.0)

    def test_generate_centered_additive_poisson_matches_definition(self):
        clean = np.array([[0.2, 0.5], [0.8, 0.1]], dtype=float)
        baseline_counts, noisy = generate_centered_additive_poisson(clean, lambda0=2.0, seed=7, clip=False)

        rng = np.random.default_rng(7)
        expected_counts = rng.poisson(2.0, size=clean.shape).astype(np.int32)
        expected_noisy = clean + expected_counts.astype(float) - 2.0

        np.testing.assert_array_equal(baseline_counts, expected_counts)
        np.testing.assert_allclose(noisy, expected_noisy, atol=1e-12, rtol=0.0)

    def test_generate_centered_additive_poisson_clip_enforces_nonnegative_output(self):
        clean = np.zeros((4, 4), dtype=float)
        _, noisy = generate_centered_additive_poisson(clean, lambda0=3.0, seed=1, clip=True)
        self.assertGreaterEqual(noisy.min(), 0.0)


if __name__ == "__main__":
    unittest.main()
