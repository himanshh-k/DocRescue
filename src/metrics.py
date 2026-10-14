"""
src/metrics.py
Image-quality and text-accuracy metrics for DocRescue.

Image quality: PSNR, SSIM, RMSE (on SD7K test pairs)
Text accuracy: CER, WER (character / word error rate via jiwer)
Statistics:    Wilcoxon signed-rank test, bootstrap confidence intervals
"""

import re
from typing import List, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ---------------------------------------------------------------------------
# Image-quality metrics
# ---------------------------------------------------------------------------

def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """PSNR in dB. Inputs are uint8 HxWxC or float HxWxC in [0, 1]."""
    if pred.dtype != np.uint8:
        pred = (pred * 255).clip(0, 255).astype(np.uint8)
        target = (target * 255).clip(0, 255).astype(np.uint8)
    return peak_signal_noise_ratio(target, pred, data_range=255)


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean SSIM over channels. Inputs same convention as compute_psnr."""
    if pred.dtype != np.uint8:
        pred = (pred * 255).clip(0, 255).astype(np.uint8)
        target = (target * 255).clip(0, 255).astype(np.uint8)
    return structural_similarity(target, pred, channel_axis=-1, data_range=255)


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """RMSE in pixel values [0, 255]."""
    if pred.dtype != np.uint8:
        pred = (pred * 255).clip(0, 255).astype(np.float32)
        target = (target * 255).clip(0, 255).astype(np.float32)
    else:
        pred = pred.astype(np.float32)
        target = target.astype(np.float32)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def image_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute PSNR, SSIM, and RMSE in one call."""
    return {
        "psnr": compute_psnr(pred, target),
        "ssim": compute_ssim(pred, target),
        "rmse": compute_rmse(pred, target),
    }


# ---------------------------------------------------------------------------
# OCR text metrics (CER / WER)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lower-case and collapse whitespace for fair comparison."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate."""
    import jiwer
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return jiwer.cer(ref, hyp)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate."""
    import jiwer
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return jiwer.wer(ref, hyp)


def ocr_metrics(reference: str, hypothesis: str) -> dict:
    """Compute both CER and WER."""
    return {
        "cer": compute_cer(reference, hypothesis),
        "wer": compute_wer(reference, hypothesis),
    }


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def wilcoxon_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """Paired Wilcoxon signed-rank test (non-parametric).

    Args:
        scores_a: Metric values for arm A (e.g. baseline CER).
        scores_b: Metric values for arm B (e.g. U-Net CER).
        alternative: ``"two-sided"``, ``"less"``, or ``"greater"``.

    Returns:
        (statistic, p_value)
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    stat, p = wilcoxon(a, b, alternative=alternative)
    return float(stat), float(p)


def bootstrap_ci(
    scores: Sequence[float],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for the mean.

    Args:
        scores:      1-D array of scalar values.
        n_bootstrap: Number of bootstrap resamples.
        ci:          Confidence level (0–1).
        seed:        Random seed.

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(scores, dtype=float)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)]
    alpha = (1.0 - ci) / 2.0
    lower = np.quantile(means, alpha)
    upper = np.quantile(means, 1.0 - alpha)
    return float(lower), float(upper)


def summarise(
    scores: Sequence[float],
    label: str = "",
    n_bootstrap: int = 2000,
) -> dict:
    """Return mean, std, median, and 95% bootstrap CI for a list of scores."""
    arr = np.asarray(scores, dtype=float)
    lo, hi = bootstrap_ci(arr, n_bootstrap=n_bootstrap)
    return {
        "label": label,
        "n": len(arr),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "ci95_lo": lo,
        "ci95_hi": hi,
    }
