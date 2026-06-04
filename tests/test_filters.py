"""
tests/test_filters.py — Unit tests for all filters and utilities.
"""

import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.filters import (
    standard_median_filter,
    two_pass_median_filter,
    AdaptiveTwoPassMedianFilter,
    EnhancedAdaptiveMedianFilter,
)
from src.noise import add_impulse_noise, estimate_noise_ratio
from src.metrics import mse, mae, psnr, ssim, all_metrics


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_image():
    rng = np.random.default_rng(0)
    return rng.integers(50, 200, size=(64, 64), dtype=np.uint8).astype(np.float64)


@pytest.fixture
def noisy_image(clean_image):
    return add_impulse_noise(clean_image, noise_ratio=0.15, seed=1)


# ── Noise tests ─────────────────────────────────────────────────────────────

class TestNoise:
    def test_ratio_approximate(self, clean_image):
        for ratio in [0.05, 0.15, 0.30]:
            noisy = add_impulse_noise(clean_image, noise_ratio=ratio, seed=42)
            estimated = estimate_noise_ratio(clean_image, noisy, threshold=0.5)
            assert abs(estimated - ratio) < 0.05

    def test_salt_pepper(self, clean_image):
        noisy = add_impulse_noise(clean_image, noise_ratio=0.1, salt_pepper=True, seed=7)
        assert noisy.shape == clean_image.shape

    def test_uniform_noise(self, clean_image):
        noisy = add_impulse_noise(clean_image, noise_ratio=0.1, mode="uniform", seed=7)
        assert noisy.shape == clean_image.shape


# ── Filter output shape & dtype ─────────────────────────────────────────────

class TestFilterShapes:
    def test_standard_mf(self, noisy_image):
        out = standard_median_filter(noisy_image)
        assert out.shape == noisy_image.shape

    def test_two_pass_mf(self, noisy_image):
        out = two_pass_median_filter(noisy_image)
        assert out.shape == noisy_image.shape

    def test_adaptive_mf(self, noisy_image):
        flt = AdaptiveTwoPassMedianFilter()
        out = flt.filter(noisy_image)
        assert out.shape == noisy_image.shape

    def test_enhanced_mf(self, noisy_image):
        flt = EnhancedAdaptiveMedianFilter()
        out = flt.filter(noisy_image)
        assert out.shape == noisy_image.shape


# ── Filter quality (adaptive ≥ standard on MSE) ─────────────────────────────

class TestFilterQuality:
    def _run(self, clean, noise_ratio=0.20):
        noisy = add_impulse_noise(clean, noise_ratio=noise_ratio, seed=99)
        std = standard_median_filter(noisy)
        adp = AdaptiveTwoPassMedianFilter().filter(noisy)
        enh = EnhancedAdaptiveMedianFilter().filter(noisy)
        return noisy, std, adp, enh

    def test_adaptive_better_than_standard(self, clean_image):
        noisy, std, adp, _ = self._run(clean_image)
        assert mse(clean_image, adp) <= mse(clean_image, std) * 1.05  # at most 5% worse

    def test_enhanced_better_than_standard(self, clean_image):
        noisy, std, _, enh = self._run(clean_image)
        assert mse(clean_image, enh) <= mse(clean_image, std) * 1.05


# ── Metrics ─────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_mse_zero_for_identical(self, clean_image):
        assert mse(clean_image, clean_image) == pytest.approx(0.0)

    def test_psnr_infinite_for_identical(self, clean_image):
        assert psnr(clean_image, clean_image) == float("inf")

    def test_ssim_one_for_identical(self, clean_image):
        assert ssim(clean_image, clean_image) == pytest.approx(1.0, abs=1e-6)

    def test_all_metrics_keys(self, clean_image, noisy_image):
        m = all_metrics(clean_image, noisy_image)
        for key in ("MSE", "MAE", "PSNR", "SSIM"):
            assert key in m
            assert np.isfinite(m[key])


# ── AdaptiveTwoPassMedianFilter parameter sensitivity ───────────────────────

class TestAdaptiveParams:
    @pytest.mark.parametrize("window", [3, 5, 7])
    def test_window_sizes(self, clean_image, window):
        noisy = add_impulse_noise(clean_image, 0.15, seed=0)
        flt = AdaptiveTwoPassMedianFilter(window, window)
        out = flt.filter(noisy)
        assert out.shape == clean_image.shape

    @pytest.mark.parametrize("a,b", [(0.5, 0.5), (1.0, 1.0), (2.0, 2.0)])
    def test_ab_params(self, clean_image, a, b):
        noisy = add_impulse_noise(clean_image, 0.20, seed=0)
        flt = AdaptiveTwoPassMedianFilter(a=a, b=b)
        out = flt.filter(noisy)
        assert out.shape == clean_image.shape


# ── EnhancedAdaptiveMedianFilter analysis axes ──────────────────────────────

class TestEnhancedAxes:
    @pytest.mark.parametrize("axis", ["column", "row", "both", "block"])
    def test_all_axes(self, clean_image, axis):
        noisy = add_impulse_noise(clean_image, 0.20, seed=0)
        flt = EnhancedAdaptiveMedianFilter(analysis_axis=axis)
        out = flt.filter(noisy)
        assert out.shape == clean_image.shape

    def test_multi_pass(self, clean_image):
        noisy = add_impulse_noise(clean_image, 0.25, seed=0)
        flt = EnhancedAdaptiveMedianFilter(n_passes=3)
        out = flt.filter(noisy)
        assert out.shape == clean_image.shape
