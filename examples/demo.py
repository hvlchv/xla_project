"""
examples/demo.py — End-to-end demonstration.

Reproduces the experiments in Xu & Miller (ICIP 2002) and shows
improvements from the enhanced filter.

Usage:
    python examples/demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")

from src import (
    standard_median_filter,
    two_pass_median_filter,
    AdaptiveTwoPassMedianFilter,
    EnhancedAdaptiveMedianFilter,
    add_impulse_noise,
    all_metrics,
    show_comparison,
    plot_all_metrics,
    run_benchmark,
)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# ── Synthetic test images (since we can't load real ones here) ───────────────

def make_gradient(size=128):
    """Simple gradient image."""
    x = np.linspace(0, 255, size)
    return np.tile(x, (size, 1))


def make_checkerboard(size=128, block=16):
    """Checkerboard pattern."""
    img = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if ((i // block) + (j // block)) % 2 == 0:
                img[i, j] = 200
            else:
                img[i, j] = 50
    return img


def make_circles(size=128):
    """Concentric circles."""
    cx, cy = size // 2, size // 2
    y_idx, x_idx = np.ogrid[:size, :size]
    dist = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
    return (np.sin(dist / 6) * 100 + 128).clip(0, 255)


# ── Demo 1: Visual comparison at 20% noise ───────────────────────────────────

def demo_visual(clean, name="gradient", noise_ratio=0.20):
    print(f"\n[Demo 1] Visual comparison — {name}, noise={noise_ratio*100:.0f}%")
    noisy = add_impulse_noise(clean, noise_ratio=noise_ratio, seed=42)

    std   = standard_median_filter(noisy)
    tp    = two_pass_median_filter(noisy)
    adp   = AdaptiveTwoPassMedianFilter().filter(noisy)
    enh   = EnhancedAdaptiveMedianFilter().filter(noisy)

    images = {
        "Original":        clean,
        "Noisy":           noisy,
        "Standard MF":     std,
        "Two-pass MF":     tp,
        "Adaptive 2-pass": adp,
        "Enhanced":        enh,
    }

    save = os.path.join(RESULTS, f"comparison_{name}.png")
    show_comparison(images, title=f"Filter Comparison — {name} ({noise_ratio*100:.0f}% noise)",
                    save_path=save)

    print("  Metrics:")
    for label, img in [("Noisy", noisy), ("Standard", std),
                        ("Two-pass", tp), ("Adaptive", adp), ("Enhanced", enh)]:
        m = all_metrics(clean, img)
        print(f"    {label:12s}  MSE={m['MSE']:7.2f}  MAE={m['MAE']:6.2f}"
              f"  PSNR={m['PSNR']:6.2f}dB  SSIM={m['SSIM']:.4f}")


# ── Demo 2: Metrics vs noise ratio (replicating Fig. 5 of the paper) ─────────

def demo_benchmark(clean, name="gradient"):
    print(f"\n[Demo 2] Benchmark — {name}")
    images = {name: clean}
    results, noise_ratios = run_benchmark(
        images,
        noise_ratios=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        n_trials=3,
        verbose=True,
    )

    save = os.path.join(RESULTS, f"metrics_{name}.png")
    plot_all_metrics(
        noise_ratios,
        results[name],
        save_path=save,
    )
    print(f"  Plot saved → {save}")


# ── Demo 3: Different analysis axes of the enhanced filter ───────────────────

def demo_axes(clean, name="checkerboard"):
    print(f"\n[Demo 3] Analysis-axis comparison — {name}")
    noisy = add_impulse_noise(clean, noise_ratio=0.25, seed=7)

    configs = {
        "column": EnhancedAdaptiveMedianFilter(analysis_axis="column"),
        "row":    EnhancedAdaptiveMedianFilter(analysis_axis="row"),
        "both":   EnhancedAdaptiveMedianFilter(analysis_axis="both"),
        "block":  EnhancedAdaptiveMedianFilter(analysis_axis="block"),
    }

    images = {"Original": clean, "Noisy (25%)": noisy}
    for label, flt in configs.items():
        restored = flt.filter(noisy)
        m = all_metrics(clean, restored)
        images[f"{label}\nPSNR={m['PSNR']:.1f}dB"] = restored

    save = os.path.join(RESULTS, f"axes_{name}.png")
    show_comparison(images,
                    title=f"Enhanced Filter — Analysis Axis Comparison ({name})",
                    save_path=save)


# ── Demo 4: Multi-pass generalisation ────────────────────────────────────────

def demo_multi_pass(clean, name="circles"):
    print(f"\n[Demo 4] Multi-pass comparison — {name}")
    noisy = add_impulse_noise(clean, noise_ratio=0.35, seed=13)

    images = {"Original": clean, "Noisy (35%)": noisy}
    for n_passes in [2, 3, 4]:
        flt = EnhancedAdaptiveMedianFilter(n_passes=n_passes)
        restored = flt.filter(noisy)
        m = all_metrics(clean, restored)
        images[f"{n_passes}-pass\nPSNR={m['PSNR']:.1f}dB"] = restored

    save = os.path.join(RESULTS, f"multi_pass_{name}.png")
    show_comparison(images,
                    title=f"Multi-pass Enhanced Filter ({name})",
                    save_path=save)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(0)

    gradient     = make_gradient(128)
    checkerboard = make_checkerboard(128)
    circles      = make_circles(128)

    # Demo 1 — visual comparison
    demo_visual(gradient, "gradient", noise_ratio=0.20)
    demo_visual(checkerboard, "checkerboard", noise_ratio=0.20)

    # Demo 2 — benchmark plots (Fig. 5 replica)
    demo_benchmark(gradient, "gradient")
    demo_benchmark(checkerboard, "checkerboard")

    # Demo 3 — analysis axes
    demo_axes(checkerboard, "checkerboard")

    # Demo 4 — multi-pass
    demo_multi_pass(circles, "circles")

    print("\n✓  All results saved to ./results/")
