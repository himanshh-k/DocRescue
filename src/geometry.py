"""
src/geometry.py
Classical perspective correction using OpenCV.

Pipeline:
    1. Convert to grayscale
    2. Gaussian blur (reduce noise)
    3. Canny edge detection
    4. Morphological close (join broken edges)
    5. Find contours → largest 4-point approximation via approxPolyDP
    6. Order the four corners (TL, TR, BR, BL)
    7. getPerspectiveTransform → warpPerspective

Falls back to the original image if no valid quadrilateral is found.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Corner ordering
# ---------------------------------------------------------------------------

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four corner points as (top-left, top-right, bottom-right, bottom-left).

    Args:
        pts: Shape (4, 2) float32 array of corner coordinates.

    Returns:
        Ordered (4, 2) float32 array.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]   # top-left:     smallest x+y
    ordered[2] = pts[np.argmax(s)]   # bottom-right: largest x+y

    diff = np.diff(pts, axis=1).ravel()
    ordered[1] = pts[np.argmin(diff)]  # top-right:   smallest y-x
    ordered[3] = pts[np.argmax(diff)]  # bottom-left: largest y-x

    return ordered


# ---------------------------------------------------------------------------
# Perspective correction
# ---------------------------------------------------------------------------

def detect_document_corners(
    image: np.ndarray,
    blur_ksize: int = 5,
    canny_lo: int = 75,
    canny_hi: int = 200,
    morph_ksize: int = 5,
    poly_epsilon_frac: float = 0.02,
    min_area_frac: float = 0.1,
) -> Optional[np.ndarray]:
    """Detect the four corners of a document in a photo.

    Args:
        image:            RGB uint8 image.
        blur_ksize:       Gaussian blur kernel size (must be odd).
        canny_lo:         Lower Canny threshold.
        canny_hi:         Upper Canny threshold.
        morph_ksize:      Morphological close kernel size.
        poly_epsilon_frac: approxPolyDP epsilon as a fraction of arc length.
        min_area_frac:    Minimum contour area as a fraction of image area.

    Returns:
        Ordered (4, 2) float32 corner array, or ``None`` if not found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(blurred, canny_lo, canny_hi)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_ksize, morph_ksize)
    )
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    h, w = image.shape[:2]
    min_area = min_area_frac * h * w

    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)
        if area < min_area:
            break
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, poly_epsilon_frac * peri, True)
        if len(approx) == 4:
            return _order_corners(approx.reshape(4, 2))

    return None


def warp_document(
    image: np.ndarray,
    corners: np.ndarray,
    output_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Apply perspective transform to rectify a detected document.

    Args:
        image:       RGB uint8 image.
        corners:     Ordered (4, 2) float32 corner array (TL, TR, BR, BL).
        output_size: (width, height) of the output. If None, computed from
                     the corner distances.

    Returns:
        Warped RGB image.
    """
    tl, tr, br, bl = corners

    if output_size is None:
        w_top = np.linalg.norm(tr - tl)
        w_bot = np.linalg.norm(br - bl)
        h_left = np.linalg.norm(bl - tl)
        h_right = np.linalg.norm(br - tr)
        W = int(max(w_top, w_bot))
        H = int(max(h_left, h_right))
        output_size = (W, H)

    dst = np.array(
        [
            [0, 0],
            [output_size[0] - 1, 0],
            [output_size[0] - 1, output_size[1] - 1],
            [0, output_size[1] - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, M, output_size)


def correct_perspective(
    image: np.ndarray,
    output_size: Optional[Tuple[int, int]] = None,
    **detect_kwargs,
) -> Tuple[np.ndarray, bool]:
    """Full pipeline: detect + warp, with fallback.

    Returns:
        (corrected_image, success_flag)
        If detection fails, returns (original_image, False).
    """
    corners = detect_document_corners(image, **detect_kwargs)
    if corners is None:
        logger.warning("No document quadrilateral found; returning original image.")
        return image, False
    warped = warp_document(image, corners, output_size=output_size)
    return warped, True
