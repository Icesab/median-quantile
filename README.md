# median-quantile

Minimal runnable Poisson denoising V1 framework based on `plan.md`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy scipy scikit-image matplotlib
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
`image, peak, seed, method, psnr, ssim, mae, runtime_sec`.

## Implemented methods

- Baselines: noisy, 3x3 mean, 3x3 median
- Method 1: local adaptive quantile filter
- Method 2: GLR non-local adaptive quantile filter
