import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines import noisy_passthrough, mean_filter_3x3, median_filter_3x3
from data import generate_poisson_counts, load_clean_image
from method1 import method1_denoise, methodA_local_denoise
from method2 import method2_denoise
from metrics import mae, psnr, ssim

DEFAULT_IMAGES = ["camera", "moon"] #"shepp_logan"
DEFAULT_PEAKS = [5, 32]
DEFAULT_SEEDS = [3]
DEFAULT_METHODS = ["noisy", "mean3x3", "median3x3", "method1", "method2"]


def run_method(clean, counts, peak, method_name):
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
    elif method_name == "methodA_local":
        _, pred, _, _, _, _, _ = methodA_local_denoise(counts, peak=peak)
    elif method_name == "method2":
        _, pred, _, _, _ = method2_denoise(counts, peak=peak)
    else:
        raise ValueError(method_name)
    dt = time.perf_counter() - t0
    metrics = {
        "psnr": psnr(clean, pred),
        "ssim": ssim(clean, pred),
        "mae": mae(clean, pred),
        "runtime_sec": dt,
    }
    return pred, metrics


def save_gray_png(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img_u8 = np.rint(255.0 * np.clip(np.asarray(img, dtype=float), 0.0, 1.0)).astype(np.uint8)
    Image.fromarray(img_u8).save(path)


def run_experiments(
    images=None,
    peaks=None,
    seeds=None,
    methods=None,
    image_size=(128, 128),
    output_dir="outputs",
):
    images = list(DEFAULT_IMAGES if images is None else images)
    peaks = list(DEFAULT_PEAKS if peaks is None else peaks)
    seeds = list(DEFAULT_SEEDS if seeds is None else seeds)
    methods = list(DEFAULT_METHODS if methods is None else methods)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_name in images:
        clean = load_clean_image(image_name, size=image_size)
        for peak in peaks:
            for seed in seeds:
                counts, noisy = generate_poisson_counts(clean, peak=peak, seed=seed)
                case_dir = images_dir / image_name / f"peak_{peak}" / f"seed_{seed}"
                clean_path = case_dir / "clean.png"
                noisy_path = case_dir / "noisy.png"
                save_gray_png(clean, clean_path)
                save_gray_png(noisy, noisy_path)
                for method in methods:
                    pred, metrics = run_method(clean, counts, peak, method)
                    if method == "noisy":
                        denoised_path = noisy_path
                    else:
                        denoised_path = case_dir / f"{method}.png"
                        save_gray_png(pred, denoised_path)
                    row = {
                        "image": image_name,
                        "peak": peak,
                        "seed": seed,
                        "method": method,
                        "clean_path": str(clean_path),
                        "noisy_path": str(noisy_path),
                        "denoised_path": str(denoised_path),
                        **metrics,
                    }
                    rows.append(row)
                    print(row)

    fieldnames = [
        "image",
        "peak",
        "seed",
        "method",
        "psnr",
        "ssim",
        "mae",
        "runtime_sec",
        "clean_path",
        "noisy_path",
        "denoised_path",
    ]
    with open(output_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    run_experiments()


if __name__ == "__main__":
    main()
