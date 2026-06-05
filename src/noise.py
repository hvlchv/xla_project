"""
noise.py — Impulsive noise models for simulation and testing.
"""

import numpy as np
from typing import Optional


def add_impulse_noise(
    image: np.ndarray,
    noise_ratio: float = 0.15,
    mode: str = "gaussian",
    mean: float = 0.0,
    std: float = 50.0,
    salt_pepper: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Add impulsive noise to an image.

    Args:
        image:       Clean input image (any dtype; internally cast to float64).
        noise_ratio: Fraction of pixels to corrupt  (0 < p < 1).
        mode:        'gaussian' — noise amplitude is Gaussian distributed.
                     'uniform'  — noise amplitude is uniformly distributed.
                     'salt_pepper' — classic salt-and-pepper noise (overrides mode).
        mean:        Mean of Gaussian noise amplitude.
        std:         Standard deviation of Gaussian noise amplitude.
        salt_pepper: Shortcut to force salt-and-pepper mode.
        seed:        Random seed for reproducibility.

    Returns:
        Noisy image as float64.
    """
    rng = np.random.default_rng(seed)
    X = image.astype(np.float64)

    if salt_pepper:
        mode = "salt_pepper"

    mask = rng.random(X.shape) < noise_ratio

    if mode == "gaussian":
        noise_amplitude = rng.normal(mean, std, X.shape)
        X[mask] += noise_amplitude[mask]
    elif mode == "uniform":
        lo, hi = mean - std, mean + std
        noise_amplitude = rng.uniform(lo, hi, X.shape)
        X[mask] += noise_amplitude[mask]
    elif mode == "salt_pepper":
        vmin, vmax = X.min(), X.max()
        salt_mask = mask & (rng.random(X.shape) < 0.5)
        pepper_mask = mask & ~salt_mask
        X[salt_mask] = vmax
        X[pepper_mask] = vmin
    else:
        raise ValueError(f"Unknown noise mode: {mode!r}")

    return X


def estimate_noise_ratio(
    clean: np.ndarray,
    noisy: np.ndarray,
    threshold: float = 1.0,
) -> float:
    """Compute the empirical noise ratio given the clean reference.

    Args:
        clean:     Original clean image.
        noisy:     Corrupted image.
        threshold: Pixel difference magnitude considered as noise.

    Returns:
        Fraction of corrupted pixels.
    """
    diff = np.abs(clean.astype(np.float64) - noisy.astype(np.float64))
    return float((diff > threshold).mean())
