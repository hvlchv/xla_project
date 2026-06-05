"""
metrics.py — Image quality metrics for filter evaluation.
"""

import numpy as np
from typing import Dict


def mse(reference: np.ndarray, restored: np.ndarray) -> float:
    """Normalised Mean Squared Error (Eq. 11 in the paper)."""
    ref = reference.astype(np.float64)
    res = restored.astype(np.float64)
    return float(np.sum((res - ref) ** 2) / ref.size)


def mae(reference: np.ndarray, restored: np.ndarray) -> float:
    """Normalised Mean Absolute Error (Eq. 11 in the paper)."""
    ref = reference.astype(np.float64)
    res = restored.astype(np.float64)
    return float(np.sum(np.abs(res - ref)) / ref.size)


def psnr(reference: np.ndarray, restored: np.ndarray, max_val: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    err = mse(reference, restored)
    if err == 0:
        return float("inf")
    return 10.0 * np.log10(max_val ** 2 / err)


def ssim(reference: np.ndarray, restored: np.ndarray, max_val: float = 255.0) -> float:
    """Structural Similarity Index (Wang et al. 2004) — simplified single-scale."""
    ref = reference.astype(np.float64)
    res = restored.astype(np.float64)

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    mu_r, mu_s = ref.mean(), res.mean()
    sigma_r2 = ref.var()
    sigma_s2 = res.var()
    sigma_rs = np.mean((ref - mu_r) * (res - mu_s))

    numerator = (2 * mu_r * mu_s + C1) * (2 * sigma_rs + C2)
    denominator = (mu_r ** 2 + mu_s ** 2 + C1) * (sigma_r2 + sigma_s2 + C2)
    return float(numerator / denominator)


def all_metrics(reference: np.ndarray, restored: np.ndarray) -> Dict[str, float]:
    """Compute MSE, MAE, PSNR, and SSIM in one call."""
    return {
        "MSE":  mse(reference, restored),
        "MAE":  mae(reference, restored),
        "PSNR": psnr(reference, restored),
        "SSIM": ssim(reference, restored),
    }
