import numpy as np
from skimage.data import camera, moon, shepp_logan_phantom
from skimage.transform import resize


def load_clean_image(name: str, size=(128, 128)) -> np.ndarray:
    name = name.lower()
    if name == "camera":
        img = camera()
    elif name == "moon":
        img = moon()
    elif name in {"shepp_logan", "shepp_logan_phantom", "shepp-logan"}:
        img = shepp_logan_phantom()
    else:
        raise ValueError(f"unknown image name: {name}")

    img = resize(img, size, anti_aliasing=True, preserve_range=True)
    img = img.astype(float)
    if img.max() > 1:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)
    return img


def generate_poisson_counts(clean: np.ndarray, peak: float, seed: int):
    rng = np.random.default_rng(seed)
    lam = peak * np.clip(clean, 0.0, 1.0)
    counts = rng.poisson(lam).astype(np.int32)
    noisy_intensity = counts.astype(float) / float(peak)
    return counts, noisy_intensity


def generate_centered_additive_poisson(
    clean: np.ndarray,
    lambda0: float,
    seed: int,
    clip: bool = False,
):
    rng = np.random.default_rng(seed)
    clean = np.asarray(clean, dtype=float)
    baseline_counts = rng.poisson(float(lambda0), size=clean.shape).astype(np.int32)
    noisy_intensity = clean + baseline_counts.astype(float) - float(lambda0)
    if clip:
        noisy_intensity = np.maximum(noisy_intensity, 0.0)
    return baseline_counts, noisy_intensity
