"""
visualize.py — Plotting helpers for filter comparison and analysis.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple


# ── colour palette ──────────────────────────────────────────────────────────
PALETTE = {
    "standard":  ("#D62728", "--",  "Standard MF"),
    "two_pass":  ("#1F77B4", "-.",  "Two-pass MF"),
    "adaptive":  ("#2CA02C", "-",   "Adaptive 2-pass MF"),
    "enhanced":  ("#FF7F0E", "-",   "Enhanced Adaptive MF"),
}


def show_comparison(
    images: Dict[str, np.ndarray],
    title: str = "Filter Comparison",
    save_path: Optional[str] = None,
) -> None:
    """Display multiple images side-by-side.

    Args:
        images:    Ordered dict of label → image.
        title:     Figure title.
        save_path: If given, save to this path instead of showing.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (label, img) in zip(axes, images.items()):
        ax.imshow(np.clip(img, 0, 255).astype(np.uint8), cmap="gray")
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_metrics_vs_noise(
    noise_ratios: List[float],
    metrics_by_filter: Dict[str, Dict[str, List[float]]],
    metric_name: str = "MSE",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot a quality metric vs noise ratio for several filters.

    Args:
        noise_ratios:      X-axis values.
        metrics_by_filter: {filter_name: {metric_name: [values]}}
        metric_name:       Which metric to plot ('MSE', 'MAE', 'PSNR', 'SSIM').
        title:             Figure title.
        save_path:         Save path.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    for name, metrics in metrics_by_filter.items():
        color, ls, label = PALETTE.get(name, ("#888888", "-", name))
        ax.plot(
            [r * 100 for r in noise_ratios],
            metrics[metric_name],
            linestyle=ls,
            color=color,
            linewidth=2,
            label=label,
        )

    ax.set_xlabel("Noise Ratio (%)", fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(title or f"{metric_name} vs Noise Ratio", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_all_metrics(
    noise_ratios: List[float],
    metrics_by_filter: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None,
) -> None:
    """2×2 grid: MSE, MAE, PSNR, SSIM — all in one figure."""
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    metric_names = ["MSE", "MAE", "PSNR", "SSIM"]

    for idx, metric_name in enumerate(metric_names):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        for name, metrics in metrics_by_filter.items():
            if metric_name not in metrics:
                continue
            color, ls, label = PALETTE.get(name, ("#888888", "-", name))
            ax.plot(
                [r * 100 for r in noise_ratios],
                metrics[metric_name],
                linestyle=ls,
                color=color,
                linewidth=2,
                label=label,
            )
        ax.set_xlabel("Noise Ratio (%)", fontsize=9)
        ax.set_ylabel(metric_name, fontsize=9)
        ax.set_title(metric_name, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Filter Performance Metrics", fontsize=13, fontweight="bold")
    _save_or_show(fig, save_path)


def plot_noise_distribution(
    true_noise_map: np.ndarray,
    detected_noise_map: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Normal probability plot  (Fig. 3 in the paper) for column noise counts."""
    import scipy.stats as stats

    true_col = true_noise_map.sum(axis=0)
    det_col = detected_noise_map.sum(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, lbl in zip(axes, [true_col, det_col],
                              ["True Noise", "Detected Noise (after MF)"]):
        (osm, osr), (slope, intercept, _) = stats.probplot(data, dist="norm")
        ax.scatter(osm, osr, s=8, alpha=0.6, color="#1F77B4")
        fit_x = np.array([osm.min(), osm.max()])
        ax.plot(fit_x, slope * fit_x + intercept, "r--", linewidth=1.5)
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
        ax.set_title(f"Normal Probability Plot\n{lbl}", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ── helpers ──────────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, path: Optional[str]) -> None:
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    else:
        plt.show()
    plt.close(fig)
