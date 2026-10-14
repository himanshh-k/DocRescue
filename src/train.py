"""
src/train.py
Training script for the DocRescue U-Net.

Usage (local or Colab):
    python src/train.py --config configs/config.yaml

Usage (Kaggle):
    Import this file in notebook 03_train_unet.ipynb.

Features:
- Mixed precision (AMP) training
- Cosine LR scheduling
- TensorBoard logging (+ optional W&B)
- Best-checkpoint saving by PSNR
- Resume from checkpoint
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import SD7KDataset, get_train_transforms, get_val_transforms
from src.models import build_model
from src.losses import build_loss
from src.metrics import compute_psnr, compute_ssim

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    scaler: GradScaler,
    device: torch.device,
    amp: bool,
    epoch: int,
    writer: SummaryWriter,
) -> dict:
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for shadow, clean in pbar:
        shadow = shadow.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(enabled=amp):
            pred = model(shadow)
            loss, components = criterion(pred, clean)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += components["total"]
        n_batches += 1
        pbar.set_postfix(loss=f"{components['total']:.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    return {"loss": avg_loss}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    amp: bool,
    epoch: int,
    writer: SummaryWriter,
) -> dict:
    model.eval()
    total_loss = 0.0
    psnr_list, ssim_list = [], []
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]", leave=False)
    for shadow, clean in pbar:
        shadow = shadow.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        with autocast(enabled=amp):
            pred = model(shadow)
            loss, components = criterion(pred, clean)

        total_loss += components["total"]
        n_batches += 1

        # Compute PSNR/SSIM on CPU numpy
        for p, c in zip(pred, clean):
            p_np = p.cpu().numpy().transpose(1, 2, 0)
            c_np = c.cpu().numpy().transpose(1, 2, 0)
            psnr_list.append(compute_psnr(p_np, c_np))
            ssim_list.append(compute_ssim(p_np, c_np))

    avg_loss = total_loss / max(n_batches, 1)
    avg_psnr = float(np.mean(psnr_list))
    avg_ssim = float(np.mean(ssim_list))

    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("PSNR/val", avg_psnr, epoch)
    writer.add_scalar("SSIM/val", avg_ssim, epoch)

    return {"loss": avg_loss, "psnr": avg_psnr, "ssim": avg_ssim}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, resume_ckpt: Optional[str] = None) -> None:
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    # Directories
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    log_dir = Path(cfg["training"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Data
    img_size = cfg["preprocessing"]["image_size"]
    train_ds = SD7KDataset(
        cfg["data"]["sd7k_root"], split="train",
        transform=get_train_transforms(img_size)
    )
    val_ds = SD7KDataset(
        cfg["data"]["sd7k_root"], split="val",
        transform=get_val_transforms(img_size)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
    )

    # Model, loss, optimiser
    model = build_model(cfg).to(device)
    criterion = build_loss(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    n_epochs = cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=cfg["training"]["scheduler"]["eta_min"],
    )

    scaler = GradScaler(enabled=cfg["training"]["amp"])
    writer = SummaryWriter(log_dir=str(log_dir))

    start_epoch = 0
    best_metric = -float("inf")

    # Optionally resume
    if resume_ckpt and Path(resume_ckpt).exists():
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric = ckpt.get("best_metric", -float("inf"))
        logger.info(f"Resumed from {resume_ckpt}, starting at epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            cfg["training"]["amp"], epoch, writer
        )
        val_metrics = validate(
            model, val_loader, criterion, device,
            cfg["training"]["amp"], epoch, writer
        )
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"PSNR={val_metrics['psnr']:.2f} | "
            f"SSIM={val_metrics['ssim']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"time={elapsed:.0f}s"
        )

        # Save checkpoint every epoch
        ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "best_metric": best_metric,
            },
            ckpt_path,
        )

        # Save best
        metric_key = cfg["training"]["save_best_metric"]
        if val_metrics[metric_key] > best_metric:
            best_metric = val_metrics[metric_key]
            best_path = ckpt_dir / "best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "best_metric": best_metric,
                },
                best_path,
            )
            logger.info(f"  ✓ New best {metric_key}={best_metric:.4f} → saved to {best_path}")

    writer.close()
    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DocRescue U-Net")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, resume_ckpt=args.resume)
