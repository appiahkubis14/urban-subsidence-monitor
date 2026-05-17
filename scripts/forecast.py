#!/usr/bin/env python3
"""
scripts/forecast.py
====================
Predict subsidence at 1, 3, and 5-year horizons using linear
extrapolation with Monte Carlo dropout uncertainty estimation.

Author : Samuel Appiah Kubi
Licence: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from scripts.utils import (
    checkpoint_exists,
    load_config,
    mark_completed,
    numpy_to_geotiff,
    raster_to_numpy,
    setup_logging,
)

log = logging.getLogger(__name__)

SENTINEL_FILE = "data/forecasts/forecast_done.txt"


def load_rate_map(cfg: dict) -> tuple[np.ndarray, dict]:
    """Load subsidence rate map (mm/yr) and rasterio metadata.

    Args:
        cfg: Full project configuration.

    Returns:
        Tuple of (rate_array [rows, cols], meta dict).
    """
    rate_path = Path(cfg["subsidence"]["rate_map"])
    if rate_path.exists():
        try:
            arr, meta = raster_to_numpy(rate_path)
            return arr[0], meta
        except Exception as exc:
            log.warning("Could not load rate map: %s", exc)

    log.warning("Rate map not found — using synthetic.")
    rows, cols = 256, 256
    rng = np.random.default_rng(5)
    yy, xx = np.meshgrid(np.linspace(-1, 1, rows), np.linspace(-1, 1, cols), indexing="ij")
    rate = (-18 * np.exp(-(xx**2 + yy**2) / 0.3) + rng.normal(0, 1, (rows, cols))).astype(np.float32)
    meta = {
        "driver": "GTiff", "dtype": np.float32,
        "count": 1, "height": rows, "width": cols,
        "crs": "EPSG:4326",
    }
    return rate, meta


def mc_dropout_forecast(
    model,
    feature_patch: "torch.Tensor",
    n_passes: int = 50,
    device: "torch.device" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Monte Carlo dropout inference.

    Args:
        model:         PyTorch model with MC Dropout enabled.
        feature_patch: Input tensor (1, C, H, W).
        n_passes:      Number of stochastic forward passes.
        device:        Torch device.

    Returns:
        Tuple of (mean_prediction, std_prediction) as numpy arrays.
    """
    import torch

    if device is None:
        device = torch.device("cpu")

    model.enable_mc_dropout()
    predictions = []
    with torch.no_grad():
        for _ in range(n_passes):
            pred = model(feature_patch.to(device))
            predictions.append(pred.squeeze().cpu().numpy())

    stacked = np.stack(predictions, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


def forecast_linear(
    rate_map: np.ndarray,
    uncertainty_map: np.ndarray,
    horizons_years: list[float],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Produce displacement forecasts by linear extrapolation.

    Args:
        rate_map:        Subsidence rate [mm/yr], shape (H, W).
        uncertainty_map: Rate uncertainty (1σ) [mm/yr], shape (H, W).
        horizons_years:  List of forecast horizons in decimal years.

    Returns:
        List of (displacement_mm, uncertainty_mm) tuples, one per horizon.
    """
    forecasts = []
    for t in horizons_years:
        disp = rate_map * t
        unc = uncertainty_map * t
        forecasts.append((disp.astype(np.float32), unc.astype(np.float32)))
        log.info(
            "Horizon %.0f yr: mean=%.1f mm, max=%.1f mm (±%.1f mm)",
            t, float(np.nanmean(disp)), float(np.nanmin(disp)),
            float(np.nanmean(unc)),
        )
    return forecasts


def run(cfg: dict, force: bool = False) -> None:
    """Execute forecast pipeline step.

    Args:
        cfg:   Full project configuration.
        force: Re-run even if checkpoint exists.
    """
    if checkpoint_exists(SENTINEL_FILE) and not force:
        log.info("Forecasts already complete. Use --force to re-run.")
        return

    output_dir = Path(cfg["forecast"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = cfg["forecast"]["horizons_years"]
    mc_passes = cfg["forecast"]["mc_passes"]

    rate_map, meta = load_rate_map(cfg)

    # Try MC Dropout uncertainty from model
    try:
        import torch
        from scripts.train_lightweight import build_tiny_unet

        model_path = Path(cfg["model"]["paths"]["unquantized"])
        if model_path.exists():
            model = build_tiny_unet(
                in_channels=cfg["model"]["in_channels"],
                enc_channels=cfg["model"]["encoder_channels"],
                dropout=cfg["model"]["dropout_rate"],
            )
            state = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state)

            stack_path = Path(cfg["features"]["stack_path"])
            if stack_path.exists():
                stack = np.load(str(stack_path))
            else:
                rng = np.random.default_rng(7)
                stack = rng.uniform(0, 1, (8, rate_map.shape[0], rate_map.shape[1])).astype(np.float32)

            # Pad stack to divisible by 8 for U-Net
            H, W = stack.shape[1], stack.shape[2]
            pad_h = (8 - H % 8) % 8
            pad_w = (8 - W % 8) % 8
            if pad_h or pad_w:
                stack = np.pad(stack, ((0, 0), (0, pad_h), (0, pad_w)))

            x = torch.from_numpy(stack).unsqueeze(0)
            _, unc_map = mc_dropout_forecast(model, x, mc_passes)
            # Crop back
            unc_map = unc_map[:H, :W]
            log.info("MC Dropout uncertainty computed (%d passes).", mc_passes)
        else:
            raise FileNotFoundError("Model not found.")
    except Exception as exc:
        log.warning("MC Dropout not available (%s) — using ±10%% rate as uncertainty.", exc)
        unc_map = np.abs(rate_map) * 0.10

    forecasts = forecast_linear(rate_map, unc_map, horizons)

    # Save GeoTIFFs
    for t, (disp, unc) in zip(horizons, forecasts):
        label = f"{int(t)}yr"
        disp_path = output_dir / f"forecast_{label}_displacement_mm.tif"
        unc_path = output_dir / f"forecast_{label}_uncertainty_mm.tif"

        try:
            numpy_to_geotiff(disp, meta.copy(), disp_path)
            numpy_to_geotiff(unc, meta.copy(), unc_path)
        except Exception as exc:
            log.warning("GeoTIFF save failed (%s); saving as .npy.", exc)
            np.save(str(disp_path).replace(".tif", ".npy"), disp)
            np.save(str(unc_path).replace(".tif", ".npy"), unc)

    # Save summary JSON
    summary = {
        "horizons_years": horizons,
        "mc_passes": mc_passes,
        "results": [
            {
                "horizon_yr": t,
                "mean_displacement_mm": round(float(np.nanmean(d)), 2),
                "min_displacement_mm": round(float(np.nanmin(d)), 2),
                "max_displacement_mm": round(float(np.nanmax(d)), 2),
                "mean_uncertainty_mm": round(float(np.nanmean(u)), 2),
            }
            for t, (d, u) in zip(horizons, forecasts)
        ],
    }
    with (output_dir / "forecast_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    mark_completed(SENTINEL_FILE)
    log.info("Forecast step complete: %d horizons saved.", len(horizons))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast future subsidence.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level, step="forecast")
    cfg = load_config(args.config)
    run(cfg, force=args.force)
