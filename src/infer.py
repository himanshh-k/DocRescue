"""
src/infer.py
Full-resolution gain-map inference for DocRescue.

Key idea:
    The U-Net is trained at 512×512. Running it directly on full-res
    images would distort text (output pixel ≠ input pixel location after
    resize). Instead, we:
        1. Downscale to 512×512 and run the network.
        2. Compute the gain map: pred / input (smoothed with a large Gaussian).
        3. Upsample the gain map back to the original resolution.
        4. Multiply the original by the gain map.

    This preserves sharp text while correcting global illumination.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gain-map inference
# ---------------------------------------------------------------------------

def gain_map_rescue(
    image: np.ndarray,
    model: torch.nn.Module,
    device: Union[str, torch.device] = "cpu",
    input_size: int = 512,
    blur_ksize: int = 51,
    gain_min_clamp: float = 0.05,
) -> np.ndarray:
    """Apply the gain-map trick for full-resolution shadow removal.

    Args:
        image:          RGB uint8 image (H x W x 3).
        model:          Trained ``DocRescueUNet`` in eval mode.
        device:         PyTorch device.
        input_size:     Size to which the image is downscaled for the network.
        blur_ksize:     Gaussian blur kernel for gain-map smoothing (must be odd).
        gain_min_clamp: Minimum value clamped in the small image before dividing
                        (avoids division by near-zero from dark pixels).

    Returns:
        Corrected RGB uint8 image at the original resolution.
    """
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    h, w = image.shape[:2]
    img_f = image.astype(np.float32) / 255.0

    # --- 1. Downscale and run network ---
    small = cv2.resize(img_f, (input_size, input_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(small.transpose(2, 0, 1)).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(tensor)  # (1, 3, 512, 512) in [0, 1]

    pred_np = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (512, 512, 3)

    # --- 2. Compute gain map ---
    small_clamped = np.clip(small, gain_min_clamp, 1.0)
    gain = pred_np / small_clamped  # element-wise gain

    # Smooth to follow illumination, not text edges
    for c in range(3):
        gain[..., c] = cv2.GaussianBlur(gain[..., c], (blur_ksize, blur_ksize), 0)

    # --- 3. Upsample gain map ---
    gain_full = cv2.resize(gain, (w, h), interpolation=cv2.INTER_LINEAR)

    # --- 4. Apply to full-resolution image ---
    result = np.clip(img_f * gain_full, 0.0, 1.0)
    return (result * 255).clip(0, 255).astype(np.uint8)


def direct_512_rescue(
    image: np.ndarray,
    model: torch.nn.Module,
    device: Union[str, torch.device] = "cpu",
    input_size: int = 512,
) -> np.ndarray:
    """Baseline: use the 512 network output directly (no gain map).

    This is the ablation counterpart of ``gain_map_rescue``.
    """
    img_f = image.astype(np.float32) / 255.0
    small = cv2.resize(img_f, (input_size, input_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(small.transpose(2, 0, 1)).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(tensor)

    pred_np = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    # Resize back to original
    result = cv2.resize(pred_np, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return (result * 255).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(
    model: torch.nn.Module,
    ckpt_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> dict:
    """Load a model checkpoint.

    Args:
        model:     Model instance (weights will be updated in-place).
        ckpt_path: Path to the ``.pth`` checkpoint file.
        device:    Device to map checkpoint tensors to.
        strict:    Whether to require exact key matching.

    Returns:
        The full checkpoint dict (includes epoch, metrics, etc.).
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=strict)
    logger.info(f"Loaded checkpoint from {ckpt_path}")
    return ckpt
