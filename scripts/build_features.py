#!/usr/bin/env python3
"""
scripts/build_features.py
==========================
Build 8-channel ML feature stack from InSAR and ancillary data.

Channels:
  0 - Mean coherence
  1 - Mean amplitude
  2 - DEM (elevation)
  3 - Slope (derived from DEM)
  4 - Temporal baseline (years)
  5 - Distance to water
  6 - Urban mask
  7 - Preliminary subsidence rate

Author : Samuel Appiah Kubi
Licence: MIT
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np

from scripts.utils import (
    checkpoint_exists,
    load_config,
    mark_completed,
    setup_logging,
)

log = logging.getLogger(__name__)

SENTINEL_FILE = "data/features/stack_done.txt"


# ---------------------------------------------------------------------------
# Channel builders
# ---------------------------------------------------------------------------

def load_or_synthesise_raster(
    path: Path, shape: tuple[int, int], fill: float = 0.0, seed: int = 0
) -> np.ndarray:
    """Load a GeoTIFF band or generate synthetic data if missing.

    Args:
        path:  Path to GeoTIFF.
        shape: (rows, cols) fallback synthetic shape.
        fill:  Constant fill value for trivial fallback.
        seed:  Random seed for synthetic generator.

    Returns:
        2-D float32 array of shape (rows, cols).
    """
    if path.exists():
        try:
            import rasterio

            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
                log.debug("Loaded: %s shape=%s", path.name, arr.shape)
                return arr
        except Exception as exc:
            log.warning("Failed to read %s: %s — using synthetic.", path, exc)

    log.warning("Not found: %s — generating synthetic channel.", path.name)
    rng = np.random.default_rng(seed)
    return (rng.uniform(0, 1, shape) + fill).astype(np.float32)


def build_dem_channel(dem_path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Load DEM and resample to target shape.

    Args:
        dem_path: Path to DEM GeoTIFF.
        shape:    Target (rows, cols).

    Returns:
        Elevation array (rows, cols).
    """
    arr = load_or_synthesise_raster(dem_path, shape, fill=5.0, seed=1)
    return _resize(arr, shape)


def build_slope_channel(dem: np.ndarray, spacing_m: float = 50.0) -> np.ndarray:
    """Compute slope from DEM using central differences.

    Args:
        dem:       2-D elevation array.
        spacing_m: Grid spacing in metres.

    Returns:
        Slope array in degrees (same shape as dem).
    """
    dz_dy, dz_dx = np.gradient(dem, spacing_m)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    return slope.astype(np.float32)


def build_coherence_channel(
    insar_dir: Path, shape: tuple[int, int]
) -> np.ndarray:
    """Compute mean coherence across all interferograms.

    Args:
        insar_dir: InSAR output directory.
        shape:     Target (rows, cols).

    Returns:
        Mean coherence array (rows, cols).
    """
    coh_dir = insar_dir / "coherence"
    tifs = sorted(coh_dir.glob("*_coherence.tif")) if coh_dir.exists() else []
    if not tifs:
        log.warning("No coherence rasters found; using synthetic.")
        rng = np.random.default_rng(2)
        return (0.6 + rng.normal(0, 0.1, shape)).clip(0, 1).astype(np.float32)

    try:
        import rasterio

        stacks = []
        for tif in tifs[:20]:  # limit to 20 for speed
            with rasterio.open(tif) as src:
                stacks.append(src.read(1))
        mean_coh = np.nanmean(np.stack(stacks, axis=0), axis=0).astype(np.float32)
        return _resize(mean_coh, shape)
    except Exception as exc:
        log.warning("Coherence stack error: %s", exc)
        return np.full(shape, 0.6, dtype=np.float32)


def build_amplitude_channel(
    insar_dir: Path, shape: tuple[int, int]
) -> np.ndarray:
    """Compute mean amplitude across interferograms.

    Args:
        insar_dir: InSAR output directory.
        shape:     Target (rows, cols).

    Returns:
        Mean amplitude array (rows, cols), normalised to [0,1].
    """
    coh_dir = insar_dir / "interferograms"
    tifs = sorted(coh_dir.glob("*_phase.tif")) if coh_dir.exists() else []
    if not tifs:
        rng = np.random.default_rng(3)
        return rng.rayleigh(0.5, shape).clip(0, 1).astype(np.float32)

    try:
        import rasterio

        stacks = []
        for tif in tifs[:20]:
            with rasterio.open(tif) as src:
                stacks.append(np.abs(src.read(1)))
        mean_amp = np.nanmean(np.stack(stacks, axis=0), axis=0).astype(np.float32)
        return _resize(mean_amp, shape)
    except Exception as exc:
        log.warning("Amplitude stack error: %s", exc)
        rng = np.random.default_rng(3)
        return rng.rayleigh(0.5, shape).clip(0, 1).astype(np.float32)


def build_temporal_baseline_channel(
    network_path: Path, shape: tuple[int, int]
) -> np.ndarray:
    """Compute mean temporal baseline (years) as a uniform channel.

    Args:
        network_path: Path to network.json.
        shape:        Target (rows, cols).

    Returns:
        Constant array filled with mean temporal baseline.
    """
    import json

    if network_path.exists():
        with network_path.open() as fh:
            net = json.load(fh)
        pairs = net.get("pairs", [])
        if pairs:
            mean_dt = np.mean([p["temporal_baseline_days"] for p in pairs]) / 365.25
        else:
            mean_dt = 0.25
    else:
        mean_dt = 0.25
    return np.full(shape, mean_dt, dtype=np.float32)


def build_distance_to_water(bbox: list[float], shape: tuple[int, int]) -> np.ndarray:
    """Estimate distance to water using a synthetic gradient.

    In production, this would use OSM water body polygons. Here we
    approximate using a north-south gradient (Accra coast is south).

    Args:
        bbox:  [lon_min, lat_min, lon_max, lat_max].
        shape: (rows, cols).

    Returns:
        Distance-to-water array in normalised units [0, 1].
    """
    rows, cols = shape
    # Gradient from south (coast, ~lat 5.5) to north
    dist = np.linspace(0, 1, rows)[:, np.newaxis] * np.ones((rows, cols))
    return dist.astype(np.float32)


def build_urban_mask(shape: tuple[int, int]) -> np.ndarray:
    """Build a synthetic urban mask (1 = urban, 0 = non-urban).

    In production this uses DLR Global Urban Footprint or OSM.
    Here we create a circular urban core centred on Accra CBD.

    Args:
        shape: (rows, cols).

    Returns:
        Binary float32 urban mask.
    """
    rows, cols = shape
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, rows), np.linspace(-1, 1, cols), indexing="ij"
    )
    mask = (xx**2 + yy**2 < 0.5**2).astype(np.float32)
    return mask


def _resize(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2-D array to target shape using bilinear interpolation.

    Args:
        arr:   Input array.
        shape: Target (rows, cols).

    Returns:
        Resized array.
    """
    if arr.shape == shape:
        return arr
    try:
        from scipy.ndimage import zoom

        factors = (shape[0] / arr.shape[0], shape[1] / arr.shape[1])
        return zoom(arr, factors, order=1).astype(np.float32)
    except ImportError:
        # Nearest-neighbour fallback
        row_idx = np.round(
            np.linspace(0, arr.shape[0] - 1, shape[0])
        ).astype(int)
        col_idx = np.round(
            np.linspace(0, arr.shape[1] - 1, shape[1])
        ).astype(int)
        return arr[np.ix_(row_idx, col_idx)].astype(np.float32)


def normalise_channel(
    arr: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Normalise array to [0, 1].

    Args:
        arr:  Input array.
        vmin: Optional minimum (uses arr.min() if None).
        vmax: Optional maximum (uses arr.max() if None).

    Returns:
        Tuple of (normalised_array, vmin_used, vmax_used).
    """
    v0 = float(np.nanmin(arr)) if vmin is None else vmin
    v1 = float(np.nanmax(arr)) if vmax is None else vmax
    if v1 - v0 < 1e-8:
        return np.zeros_like(arr), v0, v1
    normed = (arr - v0) / (v1 - v0)
    return np.clip(normed, 0, 1).astype(np.float32), v0, v1


def build_feature_stack(
    cfg: dict, shape: tuple[int, int] = (256, 256)
) -> tuple[np.ndarray, list[dict]]:
    """Assemble all 8 channels into a feature stack.

    Args:
        cfg:   Full project configuration.
        shape: Target spatial shape (rows, cols).

    Returns:
        Tuple of (stack [8, rows, cols], scaler_params list of dicts).
    """
    insar_dir = Path(cfg["insar"]["output_dir"])
    dem_path = Path(cfg["dem"]["utm_output_path"])
    network_path = Path(cfg["insar"]["network_file"])
    rate_map_path = Path(cfg["subsidence"]["rate_map"])
    bbox = cfg["study_area"]["bbox"]

    log.info("Building %d-channel feature stack at shape=%s ...", 8, shape)

    dem = build_dem_channel(dem_path, shape)
    slope = build_slope_channel(dem)
    coherence = build_coherence_channel(insar_dir, shape)
    amplitude = build_amplitude_channel(insar_dir, shape)
    temporal = build_temporal_baseline_channel(network_path, shape)
    dist_water = build_distance_to_water(bbox, shape)
    urban_mask = build_urban_mask(shape)
    subsidence = load_or_synthesise_raster(rate_map_path, shape, fill=-5.0, seed=9)

    raw_channels = [
        coherence, amplitude, dem, slope, temporal, dist_water, urban_mask, subsidence
    ]
    channel_names = cfg["features"]["channels"]

    stack = np.zeros((8, shape[0], shape[1]), dtype=np.float32)
    scalers = []

    for i, (ch, name) in enumerate(zip(raw_channels, channel_names)):
        normed, vmin, vmax = normalise_channel(ch)
        stack[i] = normed
        scalers.append({"channel": name, "vmin": vmin, "vmax": vmax})
        log.debug("Channel %d (%s): min=%.3f max=%.3f", i, name, vmin, vmax)

    log.info("Feature stack shape: %s", stack.shape)
    return stack, scalers


def run(cfg: dict, force: bool = False) -> None:
    """Execute feature building pipeline step.

    Args:
        cfg:   Full project configuration.
        force: Re-run even if checkpoint exists.
    """
    if checkpoint_exists(SENTINEL_FILE) and not force:
        log.info("Features already built. Use --force to re-run.")
        return

    feat_cfg = cfg["features"]
    output_dir = Path(feat_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    shape = (256, 256)
    stack, scalers = build_feature_stack(cfg, shape)

    # Save stack
    stack_path = Path(feat_cfg["stack_path"])
    stack_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(stack_path), stack)
    log.info("Feature stack saved: %s", stack_path)

    # Save scalers
    scaler_path = Path(feat_cfg["scaler_path"])
    with scaler_path.open("wb") as fh:
        pickle.dump(scalers, fh)
    log.info("Scalers saved: %s", scaler_path)

    mark_completed(SENTINEL_FILE)
    log.info("Feature building complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build 8-channel ML feature stack."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level, step="build_features")
    cfg = load_config(args.config)
    run(cfg, force=args.force)
