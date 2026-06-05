"""
benchmark.py — Automated benchmark across noise ratios and images.
"""

import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm

from .filters import (
    standard_median_filter,
    two_pass_median_filter,
    AdaptiveTwoPassMedianFilter,
    EnhancedAdaptiveMedianFilter,
)
from .noise import add_impulse_noise
from .metrics import all_metrics


def run_benchmark(
    images: Dict[str, np.ndarray],
    noise_ratios: Optional[List[float]] = None,
    noise_mode: str = "gaussian",
    n_trials: int = 3,
    seed: int = 42,
    window_size: int = 3,
    verbose: bool = True,
) -> Dict:
    """Run all four filters across multiple noise ratios and images.

    Args:
        images:       {image_name: 2-D array}
        noise_ratios: List of noise fractions to test.
        noise_mode:   Noise type passed to add_impulse_noise.
        n_trials:     Number of random trials per (image, ratio) pair.
        seed:         Base random seed.
        window_size:  Median filter window size for baselines.
        verbose:      Print progress bar.

    Returns:
        results dict with structure:
          results[image_name][filter_name][metric_name] = [value, ...]
          indexed by noise_ratios.
    """
    if noise_ratios is None:
        noise_ratios = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

    filter_fns = {
        "standard": lambda x: standard_median_filter(x, window_size),
        "two_pass": lambda x: two_pass_median_filter(x, window_size, window_size),
        "adaptive": lambda x: AdaptiveTwoPassMedianFilter(window_size, window_size).filter(x),
        "enhanced": lambda x: EnhancedAdaptiveMedianFilter().filter(x),
    }

    results = {}

    iter_ = tqdm(images.items(), desc="Images") if verbose else images.items()
    for img_name, clean in iter_:
        results[img_name] = {fname: {m: [] for m in ("MSE", "MAE", "PSNR", "SSIM")}
                             for fname in filter_fns}

        for r_idx, ratio in enumerate(noise_ratios):
            trial_metrics = {fname: {m: [] for m in ("MSE", "MAE", "PSNR", "SSIM")}
                             for fname in filter_fns}

            for trial in range(n_trials):
                noisy = add_impulse_noise(
                    clean, noise_ratio=ratio, mode=noise_mode,
                    seed=seed + r_idx * 100 + trial,
                )
                for fname, fn in filter_fns.items():
                    try:
                        restored = fn(noisy)
                        m = all_metrics(clean, restored)
                    except Exception:
                        m = {"MSE": np.nan, "MAE": np.nan, "PSNR": np.nan, "SSIM": np.nan}
                    for k, v in m.items():
                        trial_metrics[fname][k].append(v)

            for fname in filter_fns:
                for k in ("MSE", "MAE", "PSNR", "SSIM"):
                    vals = [v for v in trial_metrics[fname][k] if not np.isnan(v)]
                    results[img_name][fname][k].append(
                        float(np.mean(vals)) if vals else np.nan
                    )

    return results, noise_ratios
