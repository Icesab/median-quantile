import csv
import time
from src.data import load_clean_image, generate_poisson_counts
from src.baselines import noisy_passthrough, mean_filter_3x3, median_filter_3x3
from src.method1 import method1_denoise
from src.method2 import method2_denoise
from src.metrics import psnr, ssim, mae


def run_once(clean, counts, peak, method_name):
    t0 = time.perf_counter()
    if method_name == "noisy":
        out_counts = noisy_passthrough(counts)
        pred = out_counts / peak
    elif method_name == "mean3x3":
        out_counts = mean_filter_3x3(counts)
        pred = out_counts / peak
    elif method_name == "median3x3":
        out_counts = median_filter_3x3(counts)
        pred = out_counts / peak
    elif method_name == "method1":
        _, pred, _, _, _ = method1_denoise(counts, peak=peak)
    elif method_name == "method2":
        _, pred, _, _, _ = method2_denoise(counts, peak=peak)
    else:
        raise ValueError(method_name)
    dt = time.perf_counter() - t0
    return {
        "psnr": psnr(clean, pred),
        "ssim": ssim(clean, pred),
        "mae": mae(clean, pred),
        "runtime_sec": dt,
    }


def main():
    images = ["camera", "moon", "shepp_logan"]
    peaks = [1, 2, 4, 8, 16, 32]
    seeds = [0, 1, 2, 3, 4]
    methods = ["noisy", "mean3x3", "median3x3", "method1", "method2"]

    rows = []
    for image_name in images:
        clean = load_clean_image(image_name, size=(128, 128))
        for peak in peaks:
            for seed in seeds:
                counts, _ = generate_poisson_counts(clean, peak=peak, seed=seed)
                for method in methods:
                    metrics = run_once(clean, counts, peak, method)
                    row = {
                        "image": image_name,
                        "peak": peak,
                        "seed": seed,
                        "method": method,
                        **metrics,
                    }
                    rows.append(row)
                    print(row)

    fieldnames = ["image", "peak", "seed", "method", "psnr", "ssim", "mae", "runtime_sec"]
    with open("outputs/metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
