import time
import matplotlib.pyplot as plt

from src.data import load_clean_image, generate_poisson_counts
from src.baselines import noisy_passthrough, mean_filter_3x3, median_filter_3x3
from src.method1 import method1_denoise
from src.method2 import method2_denoise
from src.metrics import psnr, ssim, mae

import os
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/figs", exist_ok=True)

def evaluate(name, clean, pred, runtime):
    return {
        "method": name,
        "psnr": psnr(clean, pred),
        "ssim": ssim(clean, pred),
        "mae": mae(clean, pred),
        "runtime_sec": runtime,
    }


def main():
    image_name = "camera"
    peak = 8
    seed = 0

    clean = load_clean_image(image_name, size=(128, 128))
    counts, noisy = generate_poisson_counts(clean, peak=peak, seed=seed)

    rows = []

    t0 = time.perf_counter()
    noisy_counts = noisy_passthrough(counts)
    noisy_pred = noisy_counts / peak
    rows.append(evaluate("noisy", clean, noisy_pred, time.perf_counter() - t0))

    t0 = time.perf_counter()
    mean_counts = mean_filter_3x3(counts)
    mean_pred = mean_counts / peak
    rows.append(evaluate("mean3x3", clean, mean_pred, time.perf_counter() - t0))

    t0 = time.perf_counter()
    med_counts = median_filter_3x3(counts)
    med_pred = med_counts / peak
    rows.append(evaluate("median3x3", clean, med_pred, time.perf_counter() - t0))

    t0 = time.perf_counter()
    m1_counts, m1_pred, p1, _, _ = method1_denoise(counts, peak=peak)
    rows.append(evaluate("method1", clean, m1_pred, time.perf_counter() - t0))

    t0 = time.perf_counter()
    m2_counts, m2_pred, p2, _, _ = method2_denoise(counts, peak=peak)
    rows.append(evaluate("method2", clean, m2_pred, time.perf_counter() - t0))

    print("Demo metrics:")
    for row in rows:
        print(row)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    panels = [
        (clean, "clean"),
        (noisy, "noisy"),
        (mean_pred, "mean3x3"),
        (med_pred, "median3x3"),
        (m1_pred, "method1"),
        (m2_pred, "method2"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("outputs/figs/camera_peak8_seed0_comparison.png", dpi=150)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(p1, cmap="viridis")
    plt.title("p-map Method1")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 2, 2)
    plt.imshow(p2, cmap="viridis")
    plt.title("p-map Method2")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("outputs/figs/camera_peak8_seed0_pmps.png", dpi=150)


if __name__ == "__main__":
    main()
