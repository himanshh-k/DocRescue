"""
src/losses.py
Training losses for the shadow-removal U-Net.

Composite loss:
    L = w_l1 * L1 + w_ssim * (1 - SSIM) + w_perceptual * L_perceptual

The perceptual term (VGG feature matching) is disabled by default and
turned on as an ablation experiment via config.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


# ---------------------------------------------------------------------------
# VGG Perceptual Loss (lazy-import: only loaded when weight > 0)
# ---------------------------------------------------------------------------

class _VGGFeatures(nn.Module):
    """Extract intermediate VGG-16 features for perceptual loss."""

    def __init__(self) -> None:
        super().__init__()
        import torchvision.models as tvm
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT)
        # Use relu3_3 features (index up to layer 16)
        self.features = nn.Sequential(*list(vgg.features.children())[:16])
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


# ---------------------------------------------------------------------------
# Main composite loss
# ---------------------------------------------------------------------------

class DocRescueLoss(nn.Module):
    """Composite L1 + SSIM (+ optional VGG perceptual) loss.

    Args:
        l1_weight:          Weight for the L1 term.
        ssim_weight:        Weight for the (1 - SSIM) term.
        perceptual_weight:  Weight for the VGG perceptual term. Set to 0 to
                            disable (default). The VGG network is only loaded
                            when this is > 0.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.2,
        perceptual_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.w_l1 = l1_weight
        self.w_ssim = ssim_weight
        self.w_perc = perceptual_weight

        if perceptual_weight > 0:
            self.vgg = _VGGFeatures()
        else:
            self.vgg = None

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Compute the composite loss.

        Args:
            pred:   Predicted image, shape ``(B, C, H, W)``, values in ``[0, 1]``.
            target: Ground-truth image, same shape.

        Returns:
            (total_loss, component_dict) where component_dict contains
            ``l1``, ``ssim``, and (optionally) ``perceptual`` scalar values.
        """
        components = {}

        # L1 term
        l1 = F.l1_loss(pred, target)
        components["l1"] = l1.item()

        # SSIM term (pytorch-msssim; higher is better, so we use 1 - ssim)
        ssim_val = ssim(pred, target, data_range=1.0, size_average=True)
        ssim_loss = 1.0 - ssim_val
        components["ssim"] = ssim_loss.item()

        total = self.w_l1 * l1 + self.w_ssim * ssim_loss

        # Optional VGG perceptual term
        if self.vgg is not None and self.w_perc > 0:
            feat_pred = self.vgg(pred)
            feat_tgt = self.vgg(target)
            perc = F.l1_loss(feat_pred, feat_tgt)
            components["perceptual"] = perc.item()
            total = total + self.w_perc * perc

        components["total"] = total.item()
        return total, components


def build_loss(cfg: dict) -> DocRescueLoss:
    """Instantiate the loss from a config dict."""
    lc = cfg["training"]["loss"]
    return DocRescueLoss(
        l1_weight=lc.get("l1_weight", 1.0),
        ssim_weight=lc.get("ssim_weight", 0.2),
        perceptual_weight=lc.get("perceptual_weight", 0.0),
    )
