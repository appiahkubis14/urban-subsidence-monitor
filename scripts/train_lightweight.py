#!/usr/bin/env python3
"""
scripts/train_lightweight.py
=============================
Train Tiny U-Net for subsidence prediction from the 8-channel feature stack.

Architecture: Encoder 8→32→64→128→256, decoder with skip connections.
Loss: 0.7 × HuberLoss + 0.3 × SpatialSmoothnessLoss
Monte Carlo Dropout (p=0.2) kept active at inference time.

Author : Samuel Appiah Kubi
Licence: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from scripts.utils import (
    checkpoint_exists,
    load_config,
    mark_completed,
    setup_logging,
)

log = logging.getLogger(__name__)

SENTINEL_FILE = "data/models/training_done.txt"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _get_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch not installed. Install with: pip install torch torchvision"
        )


def build_tiny_unet(in_channels: int = 8, enc_channels: list = None, dropout: float = 0.2):
    """Build Tiny U-Net model.

    Args:
        in_channels:  Number of input feature channels.
        enc_channels: Encoder channel progression list.
        dropout:      MC Dropout probability.

    Returns:
        TinyUNet nn.Module.
    """
    torch = _get_torch()
    import torch.nn as nn

    if enc_channels is None:
        enc_channels = [32, 64, 128, 256]

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch, drop_p):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(drop_p),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class TinyUNet(nn.Module):
        """Tiny U-Net with MC Dropout for subsidence prediction."""

        def __init__(self):
            super().__init__()
            ch = enc_channels
            self.enc1 = ConvBlock(in_channels, ch[0], dropout)
            self.enc2 = ConvBlock(ch[0], ch[1], dropout)
            self.enc3 = ConvBlock(ch[1], ch[2], dropout)
            self.bottleneck = ConvBlock(ch[2], ch[3], dropout)
            self.pool = nn.MaxPool2d(2)

            self.up3 = nn.ConvTranspose2d(ch[3], ch[2], 2, stride=2)
            self.dec3 = ConvBlock(ch[2] * 2, ch[2], dropout)
            self.up2 = nn.ConvTranspose2d(ch[2], ch[1], 2, stride=2)
            self.dec2 = ConvBlock(ch[1] * 2, ch[1], dropout)
            self.up1 = nn.ConvTranspose2d(ch[1], ch[0], 2, stride=2)
            self.dec1 = ConvBlock(ch[0] * 2, ch[0], dropout)
            self.head = nn.Conv2d(ch[0], 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.head(d1)

        def enable_mc_dropout(self):
            """Ensure Dropout layers stay active during inference."""
            for m in self.modules():
                if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                    m.train()

    return TinyUNet()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def combined_loss(pred, target, huber_w: float = 0.7, smooth_w: float = 0.3,
                  delta: float = 1.0):
    """Combined Huber + spatial smoothness loss.

    Args:
        pred:     Predicted tensor (B, 1, H, W).
        target:   Target tensor (B, 1, H, W).
        huber_w:  Weight for Huber loss term.
        smooth_w: Weight for spatial smoothness term.
        delta:    Huber threshold.

    Returns:
        Scalar loss tensor.
    """
    torch = _get_torch()
    import torch.nn.functional as F

    huber = F.huber_loss(pred, target, delta=delta)
    dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    smoothness = (dy**2).mean() + (dx**2).mean()
    return huber_w * huber + smooth_w * smoothness


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_patches(
    stack: np.ndarray,
    rate_map: np.ndarray,
    patch_size: int = 64,
    stride: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract overlapping patches from feature stack and rate map.

    Args:
        stack:      (8, H, W) feature stack.
        rate_map:   (H, W) subsidence rate map [mm/yr].
        patch_size: Spatial patch size.
        stride:     Patch stride.

    Returns:
        Tuple of (X patches [N, 8, P, P], y patches [N, 1, P, P]).
    """
    _, H, W = stack.shape
    X_patches, y_patches = [], []
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            X_patches.append(stack[:, r:r+patch_size, c:c+patch_size])
            y_patches.append(rate_map[r:r+patch_size, c:c+patch_size][np.newaxis])
    return np.stack(X_patches), np.stack(y_patches)


class PatchDataset:
    """PyTorch-compatible patch dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        import torch
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, resume: bool = False) -> None:
    """Full training loop with validation and checkpoint saving.

    Args:
        cfg:    Full project configuration.
        resume: Resume from last saved checkpoint if True.
    """
    torch = _get_torch()
    import torch
    from torch.utils.data import DataLoader, random_split

    model_cfg = cfg["model"]
    train_cfg = model_cfg["training"]
    paths = model_cfg["paths"]
    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(train_cfg["seed"])
    device = torch.device("cpu")

    # Load data
    stack_path = Path(cfg["features"]["stack_path"])
    rate_map_path = Path(cfg["subsidence"]["rate_map"])

    if stack_path.exists():
        stack = np.load(str(stack_path))
    else:
        log.warning("Feature stack not found — generating synthetic data.")
        rng = np.random.default_rng(0)
        stack = rng.uniform(0, 1, (8, 256, 256)).astype(np.float32)

    if rate_map_path.exists():
        try:
            import rasterio
            with rasterio.open(rate_map_path) as src:
                rate_map = src.read(1).astype(np.float32)
        except Exception:
            rate_map = np.load(str(rate_map_path).replace(".tif", ".npy")) \
                if Path(str(rate_map_path).replace(".tif", ".npy")).exists() \
                else np.random.default_rng(1).uniform(-20, 5, (256, 256)).astype(np.float32)
    else:
        log.warning("Rate map not found — generating synthetic target.")
        rng = np.random.default_rng(1)
        rate_map = rng.uniform(-20, 5, (stack.shape[1], stack.shape[2])).astype(np.float32)

    # Align shapes
    target_shape = (stack.shape[1], stack.shape[2])
    if rate_map.shape != target_shape:
        from scipy.ndimage import zoom
        factors = (target_shape[0] / rate_map.shape[0], target_shape[1] / rate_map.shape[1])
        rate_map = zoom(rate_map, factors, order=1).astype(np.float32)

    patch_size = cfg["features"]["patch_size"]
    stride = cfg["features"]["stride"]
    X, y = make_patches(stack, rate_map, patch_size, stride)
    log.info("Dataset: %d patches of size %d×%d", len(X), patch_size, patch_size)

    dataset = PatchDataset(X, y)
    n_val = max(1, int(len(dataset) * train_cfg["val_split"]))
    n_test = max(1, int(len(dataset) * train_cfg["test_split"]))
    n_train = len(dataset) - n_val - n_test
    train_ds, val_ds, _ = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"])

    model = build_tiny_unet(
        in_channels=model_cfg["in_channels"],
        enc_channels=model_cfg["encoder_channels"],
        dropout=model_cfg["dropout_rate"],
    ).to(device)

    # Resume
    start_epoch = 0
    ckpt_path = output_dir / "checkpoint_last.pt"
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        log.info("Resumed from epoch %d", start_epoch)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"]
    )

    loss_cfg = model_cfg["loss"]
    history = {"train_loss": [], "val_loss": [], "epoch_time_s": []}
    best_val = float("inf")
    patience_count = 0

    tb_dir = Path(paths.get("tensorboard", "data/logs/tensorboard"))
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(tb_dir))
    except ImportError:
        writer = None

    for epoch in range(start_epoch, train_cfg["epochs"]):
        t0 = time.time()
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = combined_loss(
                pred, y_batch,
                huber_w=loss_cfg["huber_weight"],
                smooth_w=loss_cfg["smoothness_weight"],
                delta=loss_cfg["huber_delta"],
            )
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss = combined_loss(pred, y_batch)
                val_losses.append(val_loss.item())

        scheduler.step()
        epoch_time = time.time() - t0
        mean_train = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses))

        history["train_loss"].append(mean_train)
        history["val_loss"].append(mean_val)
        history["epoch_time_s"].append(round(epoch_time, 2))

        if writer:
            writer.add_scalar("Loss/train", mean_train, epoch)
            writer.add_scalar("Loss/val", mean_val, epoch)

        log.info(
            "Epoch %3d/%d  train=%.4f  val=%.4f  lr=%.2e  time=%.1fs",
            epoch + 1, train_cfg["epochs"], mean_train, mean_val,
            optimizer.param_groups[0]["lr"], epoch_time,
        )

        # Checkpoint
        torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_path)

        if mean_val < best_val:
            best_val = mean_val
            patience_count = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_count += 1
            if patience_count >= train_cfg["patience"]:
                log.info("Early stopping at epoch %d.", epoch + 1)
                break

    # Save final
    torch.save(model.state_dict(), paths["unquantized"])
    log.info("Model saved: %s", paths["unquantized"])

    with open(paths["history"], "w") as fh:
        json.dump(history, fh, indent=2)
    log.info("Training history saved: %s", paths["history"])

    if writer:
        writer.close()


def run(cfg: dict, force: bool = False, resume: bool = False) -> None:
    """Execute training pipeline step.

    Args:
        cfg:    Full project configuration.
        force:  Re-run even if checkpoint exists.
        resume: Resume from last checkpoint.
    """
    if checkpoint_exists(SENTINEL_FILE) and not force:
        log.info("Training already complete. Use --force to re-run.")
        return

    train(cfg, resume=resume)
    mark_completed(SENTINEL_FILE)
    log.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tiny U-Net.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level, step="train")
    cfg = load_config(args.config)
    run(cfg, force=args.force, resume=args.resume)
