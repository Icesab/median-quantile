# median-quantile

Minimal runnable Poisson denoising V1 framework based on `plan.md`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy scipy scikit-image matplotlib pillow
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate denoise
```

## Run demo

```bash
python run_demo.py
```

This saves:
- `outputs/figs/camera_peak8_seed0_comparison.png`
- `outputs/figs/camera_peak8_seed0_pmps.png`

## Run full experiments

```bash
python run_experiments.py
```

This writes `outputs/metrics.csv` with columns:
`image, peak, seed, method, psnr, ssim, mae, runtime_sec, clean_path, noisy_path, denoised_path`.

It also saves per-case images under `outputs/images/<image>/peak_<peak>/seed_<seed>/`:
- `clean.png`
- `noisy.png`
- `mean3x3.png`
- `median3x3.png`
- `method1.png`
- `methodA_local.png` when `methodA_local` is included in the run
- `method2.png`

## Implemented methods

- Baselines: noisy, 3x3 mean, 3x3 median
- Method 1: local adaptive quantile filter
- Method A local: purely local quantile filter for Poisson counts
- Method 2: GLR non-local adaptive quantile filter
