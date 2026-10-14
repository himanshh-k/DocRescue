"""
src/classical.py
Classical illumination / shadow removal baselines.

Three methods, increasing in sophistication:
1. background_division  — large median-blur background estimation → divide
2. clahe_enhance        — adaptive histogram equalisation (CLAHE)
3. adaptive_threshold   — Gaussian adaptive thresholding (binarises)

These are the baselines the U-Net must beat on PSNR/SSIM and OCR metrics.
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Background division
# ---------------------------------------------------------------------------

def background_division(
    image: np.ndarray,
    blur_ksize: int = 91,
    use_morphology: bool = False,
    morph_ksize: int = 51,
) -> np.ndarray:
    """Remove uneven illumination by dividing by an estimated background.

    Args:
        image:          RGB uint8 image.
        blur_ksize:     Kernel size for median/Gaussian blur background estimate
                        (must be odd).
        use_morphology: If True, use morphological closing instead of blur to
                        estimate the background (better for hard shadows).
        morph_ksize:    Morphological closing kernel size (odd).

    Returns:
        Illumination-corrected RGB uint8 image.
    """
    # Ensure ksize is odd
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    img_f = image.astype(np.float32) / 255.0

    if use_morphology:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
        background = cv2.morphologyEx(image, cv2.MORPH_CLOSE, k).astype(np.float32) / 255.0
    else:
        background = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0).astype(np.float32) / 255.0

    # Divide: suppress shadow while boosting over-lit areas
    corrected = img_f / (background + 1e-6)
    corrected = np.clip(corrected, 0.0, 1.0)

    # Stretch contrast to [0, 1]
    corrected = (corrected - corrected.min()) / (corrected.max() - corrected.min() + 1e-8)
    return (corrected * 255).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# CLAHE
# ---------------------------------------------------------------------------

def clahe_enhance(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Operates in LAB colour space on the L channel only.

    Args:
        image:          RGB uint8 image.
        clip_limit:     CLAHE clip limit. Higher → more contrast.
        tile_grid_size: Tile grid size for local histogram computation.

    Returns:
        Contrast-enhanced RGB uint8 image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


# ---------------------------------------------------------------------------
# Adaptive threshold (binarisation)
# ---------------------------------------------------------------------------

def adaptive_threshold(
    image: np.ndarray,
    block_size: int = 51,
    C: int = 10,
) -> np.ndarray:
    """Adaptive Gaussian thresholding to produce a binarised document.

    Note: This discards colour information. Useful for text-heavy documents
    but not for colour-critical applications.

    Args:
        image:      RGB uint8 image.
        block_size: Size of the neighbourhood for threshold computation (odd).
        C:          Constant subtracted from the mean.

    Returns:
        RGB uint8 image (greyscale binarised, replicated to 3 channels).
    """
    if block_size % 2 == 0:
        block_size += 1
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


# ---------------------------------------------------------------------------
# Full classical pipeline
# ---------------------------------------------------------------------------

def classical_pipeline(
    image: np.ndarray,
    use_adaptive_thresh: bool = False,
) -> np.ndarray:
    """Apply the full classical baseline: background division → CLAHE.

    Optionally adds adaptive thresholding (binarisation) for ablation.

    Args:
        image:                RGB uint8 image.
        use_adaptive_thresh:  If True, also apply adaptive thresholding.

    Returns:
        Processed RGB uint8 image.
    """
    result = background_division(image, blur_ksize=91, use_morphology=False)
    result = clahe_enhance(result, clip_limit=2.0, tile_grid_size=(8, 8))
    if use_adaptive_thresh:
        result = adaptive_threshold(result)
    return result
