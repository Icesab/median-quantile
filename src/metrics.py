import numpy as np
from skimage.metrics import structural_similarity


def psnr(x_true: np.ndarray, x_pred: np.ndarray, data_range: float = 1.0) -> float:
    mse = np.mean((x_true - x_pred) ** 2)
    if mse <= 0:
        return float("inf")
    return float(10 * np.log10((data_range**2) / mse))


def ssim(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    return float(structural_similarity(x_true, x_pred, data_range=1.0))


def mae(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(x_true - x_pred)))
