"""
src/models/unet.py
Residual U-Net for document shadow / illumination removal.

Architecture:
    - Encoder: ResNet-34 pre-trained on ImageNet (via segmentation_models_pytorch)
    - Decoder: standard U-Net decoder with skip connections
    - Output:  clamp(x + net(x), 0, 1)  — residual connection so the network
               learns the *correction*, not the full clean image.

The residual formulation stabilises training and biases the network toward
the identity mapping when the input is already clean.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DocRescueUNet(nn.Module):
    """Residual U-Net for shadow / illumination removal.

    Args:
        encoder_name:    Encoder backbone. Default ``"resnet34"``.
        encoder_weights: Pre-trained weights. Default ``"imagenet"``.
        in_channels:     Number of input channels. Default ``3`` (RGB).
        out_channels:    Number of output channels. Default ``3`` (RGB).
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,  # raw logits; we apply sigmoid in forward
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, C, H, W)`` with values in ``[0, 1]``.

        Returns:
            Corrected tensor of the same shape, values clamped to ``[0, 1]``.
        """
        residual = torch.tanh(self.unet(x))   # tanh: correction ∈ (-1, 1)
        return (x + residual).clamp(0.0, 1.0)


def build_model(cfg: dict) -> DocRescueUNet:
    """Instantiate the model from a config dict."""
    return DocRescueUNet(
        encoder_name=cfg["model"]["encoder"],
        encoder_weights=cfg["model"]["encoder_weights"],
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
    )
