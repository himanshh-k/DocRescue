"""
src/ocr.py
OCR wrapper supporting Tesseract and EasyOCR.

Key features:
- Text normalisation for fair CER/WER comparison
- Parallel batch OCR via joblib
- Engine-agnostic interface returning plain strings
"""

from __future__ import annotations

import re
import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Lower-case, collapse whitespace for fair CER/WER comparison."""
    return re.sub(r"\s+", " ", text.lower()).strip()


# ---------------------------------------------------------------------------
# Tesseract
# ---------------------------------------------------------------------------

def ocr_tesseract(
    image: np.ndarray,
    lang: str = "eng",
    config: str = "--oem 1 --psm 3",
) -> str:
    """Run Tesseract on a single RGB uint8 image.

    Args:
        image:  RGB uint8 numpy array (H x W x 3).
        lang:   Tesseract language string.
        config: Tesseract config string.
              ``--oem 1`` = LSTM engine only.
              ``--psm 3`` = fully automatic page segmentation.

    Returns:
        Raw OCR string.
    """
    try:
        import pytesseract
        from PIL import Image
        pil_img = Image.fromarray(image)
        return pytesseract.image_to_string(pil_img, lang=lang, config=config)
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# EasyOCR
# ---------------------------------------------------------------------------

_easyocr_reader = None


def _get_easyocr_reader(lang: List[str] = None):
    """Lazy singleton EasyOCR reader (GPU if available)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _lang = lang or ["en"]
        logger.info(f"Loading EasyOCR reader for languages: {_lang}")
        _easyocr_reader = easyocr.Reader(_lang, gpu=_is_gpu_available())
    return _easyocr_reader


def _is_gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def ocr_easyocr(
    image: np.ndarray,
    lang: Optional[List[str]] = None,
) -> str:
    """Run EasyOCR on a single RGB uint8 image.

    Args:
        image: RGB uint8 numpy array.
        lang:  Language list. Default ``["en"]``.

    Returns:
        Raw OCR string (bounding-box results concatenated).
    """
    try:
        reader = _get_easyocr_reader(lang or ["en"])
        results = reader.readtext(image, detail=0, paragraph=True)
        return " ".join(results)
    except Exception as e:
        logger.error(f"EasyOCR failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Engine-agnostic interface
# ---------------------------------------------------------------------------

def run_ocr(
    image: np.ndarray,
    engine: str = "tesseract",
    tesseract_lang: str = "eng",
    tesseract_config: str = "--oem 1 --psm 3",
    easyocr_lang: Optional[List[str]] = None,
) -> str:
    """Run OCR with the selected engine.

    Args:
        image:   RGB uint8 numpy array.
        engine:  ``"tesseract"``, ``"easyocr"``, or ``"both"`` (returns
                 tesseract result; easyocr result stored under ``_easyocr``
                 key if you need it separately).

    Returns:
        OCR string.
    """
    if engine == "tesseract":
        return ocr_tesseract(image, lang=tesseract_lang, config=tesseract_config)
    elif engine == "easyocr":
        return ocr_easyocr(image, lang=easyocr_lang)
    elif engine == "both":
        # Return both; caller decides which to use for stats
        tess = ocr_tesseract(image, lang=tesseract_lang, config=tesseract_config)
        easy = ocr_easyocr(image, lang=easyocr_lang)
        return {"tesseract": tess, "easyocr": easy}
    else:
        raise ValueError(f"Unknown OCR engine: {engine!r}. Choose 'tesseract', 'easyocr', or 'both'.")


# ---------------------------------------------------------------------------
# Parallel batch OCR
# ---------------------------------------------------------------------------

def batch_ocr(
    images: List[np.ndarray],
    engine: str = "tesseract",
    n_jobs: int = 4,
    **kwargs,
) -> List[str]:
    """Parallel OCR over a list of images.

    Args:
        images:  List of RGB uint8 arrays.
        engine:  OCR engine. Same as ``run_ocr``.
        n_jobs:  Number of parallel workers (``joblib``).
        **kwargs: Forwarded to ``run_ocr``.

    Returns:
        List of OCR strings in the same order as inputs.
    """
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(run_ocr)(img, engine=engine, **kwargs) for img in images
    )
