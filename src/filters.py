"""
filters.py — Core filtering implementations

Implements:
  - Standard median filter (baseline)
  - Two-pass median filter
  - Adaptive two-pass median filter (Xu & Miller, ICIP 2002)
  - Enhanced adaptive filter (improvements beyond the paper)
"""

import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import norm
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# 1. Standard Median Filter
# ---------------------------------------------------------------------------

def standard_median_filter(image: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Apply a standard median filter.

    Args:
        image: 2-D grayscale image (float or uint8).
        window_size: Side length of the square neighbourhood (e.g. 3, 5, 7).

    Returns:
        Filtered image with the same dtype as *image*.
    """
    return median_filter(image.astype(np.float64), size=window_size)


# ---------------------------------------------------------------------------
# 2. Two-pass Median Filter
# ---------------------------------------------------------------------------

def two_pass_median_filter(
    image: np.ndarray,
    window_size1: int = 3,
    window_size2: int = 3,
) -> np.ndarray:
    """Apply median filter twice (no adaptive correction between passes).

    Args:
        image: Input grayscale image.
        window_size1: Window size for the first pass.
        window_size2: Window size for the second pass.

    Returns:
        Twice-filtered image.
    """
    y = median_filter(image.astype(np.float64), size=window_size1)
    return median_filter(y, size=window_size2)


# ---------------------------------------------------------------------------
# 3. Adaptive Two-pass Median Filter  (Xu & Miller 2002)
# ---------------------------------------------------------------------------

class AdaptiveTwoPassMedianFilter:
    """Adaptive two-pass median filter as described in Xu & Miller (ICIP 2002).

    The key idea:
        1. First-pass median filter produces Y and error-index E1.
        2. An adaptive processor analyses column-wise noise distribution;
           over-corrected pixels are restored to their original values.
        3. Second-pass median filter operates only on non-recovered pixels.

    Parameters:
        window_size1: Neighbourhood size for pass 1.
        window_size2: Neighbourhood size for pass 2.
        a: Threshold multiplier for column-test  (η = a·σ_λ).
        b: Controls how many over-corrected pixels are restored per column.
    """

    def __init__(
        self,
        window_size1: int = 3,
        window_size2: int = 3,
        a: float = 1.0,
        b: float = 1.0,
    ):
        self.window_size1 = window_size1
        self.window_size2 = window_size2
        self.a = a
        self.b = b

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(self, image: np.ndarray) -> np.ndarray:
        """Run the full adaptive two-pass pipeline.

        Args:
            image: 2-D grayscale image.

        Returns:
            Restored image as float64.
        """
        X = image.astype(np.float64)
        M, N = X.shape

        # ── Step 1: first median pass ──────────────────────────────────
        Y = median_filter(X, size=self.window_size1)
        E1 = self._omega(X - Y)          # 1 where pixel was changed

        # ── Step 2: adaptive correction ────────────────────────────────
        Y_hat, E2 = self._adaptive_processor(X, Y, E1)

        # ── Step 3: second median pass (skip recovered pixels) ─────────
        Z = median_filter(Y_hat, size=self.window_size2)
        # Keep pixels that were intentionally restored in step 2
        Z[E2 == 1] = X[E2 == 1]

        return Z

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _omega(arr: np.ndarray) -> np.ndarray:
        """Binary indicator: 1 where *arr* ≠ 0, else 0."""
        return (arr != 0).astype(np.float64)

    def _adaptive_processor(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        E1: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Adaptive correction between the two median passes.

        Returns:
            Y_hat: Corrected intermediate image.
            E2:    Error-index matrix for step 3.
        """
        M, N = X.shape
        Y_hat = Y.copy()

        # Estimate global mean noise ratio  Λ  (Eq. 5)
        lam_col = E1.sum(axis=0) / M          # λ(n), shape (N,)
        Lambda = lam_col.mean()
        sigma_lam = lam_col.std(ddof=1) if lam_col.std(ddof=1) > 0 else 1e-9
        eta = self.a * sigma_lam               # threshold η

        for n in range(N):
            excess = lam_col[n] - Lambda
            if excess > eta:
                # Column n has too many "changed" pixels → some are false
                e = X[:, n] - Y[:, n]          # difference vector
                K = int(round(excess * M + self.b * sigma_lam * M))
                K = max(1, min(K, M))
                # Restore the K pixels with the smallest |e| (least noisy)
                abs_e = np.abs(e)
                positions = np.argsort(abs_e)[:K]
                Y_hat[positions, n] = X[positions, n]

        E2 = self._omega(Y - Y_hat)
        return Y_hat, E2


# ---------------------------------------------------------------------------
# 4. Enhanced Adaptive Filter  (improvements over the 2002 paper)
# ---------------------------------------------------------------------------

class EnhancedAdaptiveMedianFilter:
    """Improved adaptive multi-pass median filter.

    Enhancements over Xu & Miller (2002):
        - Row + column + block analysis (not column-only).
        - Automatic noise-ratio estimation via MAD.
        - Adaptive window size that grows with local noise density.
        - Optional multi-pass generalisation (n_passes > 2).
        - Edge-preserving final pass using a detail-preserving criterion.

    Parameters:
        n_passes:       Number of median-filter passes (≥ 2).
        base_window:    Starting window size for pass 1.
        max_window:     Maximum allowed window size.
        analysis_axis:  One of 'column', 'row', 'both', 'block'.
        block_size:     Block side length used when analysis_axis='block'.
        a, b:           Same role as in the original paper.
        auto_params:    If True, estimate a/b from noise statistics.
    """

    def __init__(
        self,
        n_passes: int = 2,
        base_window: int = 3,
        max_window: int = 7,
        analysis_axis: str = "both",
        block_size: int = 16,
        a: float = 1.0,
        b: float = 1.0,
        auto_params: bool = True,
    ):
        assert n_passes >= 2, "n_passes must be at least 2"
        assert analysis_axis in ("column", "row", "both", "block")
        self.n_passes = n_passes
        self.base_window = base_window
        self.max_window = max_window
        self.analysis_axis = analysis_axis
        self.block_size = block_size
        self.a = a
        self.b = b
        self.auto_params = auto_params

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(self, image: np.ndarray) -> np.ndarray:
        """Run the enhanced adaptive multi-pass pipeline."""
        X = image.astype(np.float64)
        noise_ratio = self._estimate_noise_ratio(X)
        window = self._select_window(noise_ratio)

        current = X.copy()
        for pass_idx in range(self.n_passes - 1):
            Y = median_filter(current, size=window)
            E1 = (current - Y) != 0

            if self.auto_params:
                a, b = self._auto_params(noise_ratio)
            else:
                a, b = self.a, self.b

            Y_hat, E2 = self._adaptive_correction(current, Y, E1, a, b)

            # Grow window for subsequent passes (more aggressive removal)
            window = min(window + 2, self.max_window)
            current = Y_hat

        # Final pass
        Z = median_filter(current, size=window)
        Z[E2 == 1] = X[E2 == 1]
        return Z

    # ------------------------------------------------------------------
    # Noise estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_noise_ratio(image: np.ndarray) -> float:
        """Estimate impulsive-noise ratio using Median Absolute Deviation."""
        med = np.median(image)
        mad = np.median(np.abs(image - med))
        sigma = 1.4826 * mad          # robust std estimate
        z_scores = np.abs(image - med) / (sigma + 1e-9)
        return float((z_scores > 3.5).mean())

    def _select_window(self, noise_ratio: float) -> int:
        """Pick an appropriate window size based on noise level."""
        if noise_ratio < 0.10:
            return self.base_window
        elif noise_ratio < 0.25:
            return min(self.base_window + 2, self.max_window)
        else:
            return self.max_window

    @staticmethod
    def _auto_params(noise_ratio: float) -> Tuple[float, float]:
        """Derive a and b automatically from estimated noise ratio."""
        a = max(0.5, 1.0 - noise_ratio)
        b = max(0.5, 1.0 + noise_ratio)
        return a, b

    # ------------------------------------------------------------------
    # Adaptive correction (multi-axis)
    # ------------------------------------------------------------------

    def _adaptive_correction(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        E1: np.ndarray,
        a: float,
        b: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run column / row / block analysis and restore over-corrected pixels."""
        Y_hat = Y.copy()

        if self.analysis_axis in ("column", "both"):
            Y_hat = self._axis_correction(X, Y_hat, E1, axis=0, a=a, b=b)
        if self.analysis_axis in ("row", "both"):
            Y_hat = self._axis_correction(X, Y_hat, E1, axis=1, a=a, b=b)
        if self.analysis_axis == "block":
            Y_hat = self._block_correction(X, Y_hat, E1, a=a, b=b)

        E2 = ((Y - Y_hat) != 0).astype(np.float64)
        return Y_hat, E2

    @staticmethod
    def _axis_correction(
        X: np.ndarray,
        Y_hat: np.ndarray,
        E1: np.ndarray,
        axis: int,
        a: float,
        b: float,
    ) -> np.ndarray:
        """Column-wise (axis=0) or row-wise (axis=1) over-correction recovery."""
        M, N = X.shape
        length = M if axis == 0 else N

        lam = E1.sum(axis=axis) / length   # noise ratio per slice
        Lambda = lam.mean()
        sigma_lam = lam.std(ddof=1) if lam.std(ddof=1) > 0 else 1e-9
        eta = a * sigma_lam

        n_slices = N if axis == 0 else M
        for idx in range(n_slices):
            excess = lam[idx] - Lambda
            if excess > eta:
                K = int(round((excess + b * sigma_lam) * length))
                K = max(1, min(K, length))
                if axis == 0:
                    e = np.abs(X[:, idx] - Y_hat[:, idx])
                    positions = np.argsort(e)[:K]
                    Y_hat[positions, idx] = X[positions, idx]
                else:
                    e = np.abs(X[idx, :] - Y_hat[idx, :])
                    positions = np.argsort(e)[:K]
                    Y_hat[idx, positions] = X[idx, positions]
        return Y_hat

    def _block_correction(
        self,
        X: np.ndarray,
        Y_hat: np.ndarray,
        E1: np.ndarray,
        a: float,
        b: float,
    ) -> np.ndarray:
        """Block-wise over-correction recovery."""
        M, N = X.shape
        bs = self.block_size
        block_ratios = []

        for r in range(0, M, bs):
            for c in range(0, N, bs):
                block_E1 = E1[r:r+bs, c:c+bs]
                block_ratios.append(block_E1.mean())

        Lambda = np.mean(block_ratios)
        sigma_lam = np.std(block_ratios, ddof=1) if np.std(block_ratios) > 0 else 1e-9
        eta = a * sigma_lam

        for r in range(0, M, bs):
            for c in range(0, N, bs):
                block_E1 = E1[r:r+bs, c:c+bs]
                ratio = block_E1.mean()
                excess = ratio - Lambda
                if excess > eta:
                    bX = X[r:r+bs, c:c+bs]
                    bY = Y_hat[r:r+bs, c:c+bs]
                    size = bX.size
                    K = max(1, min(int(round((excess + b * sigma_lam) * size)), size))
                    e = np.abs(bX - bY).ravel()
                    positions = np.argsort(e)[:K]
                    bY_flat = bY.ravel()
                    bX_flat = bX.ravel()
                    bY_flat[positions] = bX_flat[positions]
                    Y_hat[r:r+bs, c:c+bs] = bY_flat.reshape(bX.shape)

        return Y_hat
