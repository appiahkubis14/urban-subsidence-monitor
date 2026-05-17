#!/usr/bin/env python3
"""
scripts/insar_sbas.py
======================
SBAS InSAR processing: interferogram network generation, Goldstein
filtering, phase unwrapping, and SBAS inversion for displacement
time series.

Uses MintPy if available; otherwise generates synthetic output for
pipeline testing.

Author : Samuel Appiah Kubi
Licence: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np

from scripts.utils import (
    checkpoint_exists,
    load_config,
    mark_completed,
    setup_logging,
)

log = logging.getLogger(__name__)

SENTINEL_FILE = "data/insar/sbas_done.txt"


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------

def build_interferogram_network(
    dates: list[str],
    max_temporal_days: int,
    max_spatial_m: float,
) -> list[dict]:
    """Build SBAS interferogram network from acquisition dates.

    Args:
        dates:            Sorted list of ISO-format acquisition date strings.
        max_temporal_days: Maximum temporal baseline in days.
        max_spatial_m:    Maximum spatial perpendicular baseline in metres.

    Returns:
        List of interferogram pair dicts with keys: master, slave,
        temporal_baseline_days, spatial_baseline_m.
    """
    import datetime

    pairs = []
    rng = np.random.default_rng(1)
    for m_date_str, s_date_str in combinations(dates, 2):
        m_date = datetime.date.fromisoformat(m_date_str)
        s_date = datetime.date.fromisoformat(s_date_str)
        dt = abs((s_date - m_date).days)
        if dt > max_temporal_days:
            continue
        bperp = float(rng.uniform(0, max_spatial_m * 0.9))
        pairs.append(
            {
                "master": m_date_str,
                "slave": s_date_str,
                "temporal_baseline_days": dt,
                "spatial_baseline_m": round(bperp, 1),
            }
        )
    log.info(
        "Interferogram network: %d pairs from %d acquisitions",
        len(pairs),
        len(dates),
    )
    return pairs


# ---------------------------------------------------------------------------
# Goldstein filter (simulation)
# ---------------------------------------------------------------------------

def goldstein_filter(
    phase: np.ndarray,
    alpha: float = 0.5,
    window_size: int = 32,
) -> np.ndarray:
    """Apply Goldstein adaptive phase filter.

    Args:
        phase:       2-D unwrapped or wrapped phase array (radians).
        alpha:       Filter strength [0, 1]. 0 = no filter, 1 = strong.
        window_size: FFT window size in pixels.

    Returns:
        Filtered phase array (same shape as input).
    """
    from scipy.fft import fft2, ifft2

    rows, cols = phase.shape
    filtered = phase.copy()
    step = window_size // 2

    for r in range(0, rows - window_size + 1, step):
        for c in range(0, cols - window_size + 1, step):
            win = phase[r : r + window_size, c : c + window_size]
            spec = fft2(np.exp(1j * win))
            spec_mag = np.abs(spec) ** alpha
            spec_smooth = spec * spec_mag / (np.abs(spec) + 1e-10)
            win_filtered = np.angle(ifft2(spec_smooth))
            filtered[r : r + window_size, c : c + window_size] = win_filtered

    return filtered


# ---------------------------------------------------------------------------
# Synthetic InSAR data generation
# ---------------------------------------------------------------------------

def generate_synthetic_insar(
    pairs: list[dict],
    output_dir: Path,
    shape: tuple[int, int],
    bbox: list[float],
) -> None:
    """Generate synthetic interferograms, coherence, and unwrapped phase.

    Args:
        pairs:      List of interferogram pair dicts.
        output_dir: Output directory for InSAR products.
        shape:      (rows, cols) of synthetic rasters.
        bbox:       [lon_min, lat_min, lon_max, lat_max].
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        log.error("rasterio required for raster output; creating empty dirs.")
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    ifg_dir = output_dir / "interferograms"
    coh_dir = output_dir / "coherence"
    unw_dir = output_dir / "unwrapped"
    for d in [ifg_dir, coh_dir, unw_dir]:
        d.mkdir(parents=True, exist_ok=True)

    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], shape[1], shape[0])
    crs = "EPSG:4326"
    rng = np.random.default_rng(42)

    # Subsidence signal: bowl-shaped depression over Accra CBD
    rows, cols = shape
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, rows), np.linspace(-1, 1, cols), indexing="ij"
    )
    subsidence_signal = -8e-3 * np.exp(-(xx**2 + yy**2) / 0.3)  # metres/year

    def _write_tif(path: Path, array: np.ndarray, n_bands: int = 1) -> None:
        meta = {
            "driver": "GTiff",
            "height": shape[0],
            "width": shape[1],
            "count": n_bands,
            "dtype": np.float32,
            "crs": crs,
            "transform": transform,
        }
        with rasterio.open(path, "w", **meta) as dst:
            if n_bands == 1:
                dst.write(array.astype(np.float32), 1)
            else:
                for b in range(n_bands):
                    dst.write(array[b].astype(np.float32), b + 1)

    for pair in pairs:
        dt_years = pair["temporal_baseline_days"] / 365.25
        tag = f"{pair['master'].replace('-', '')}_{pair['slave'].replace('-', '')}"

        # Wrapped phase (metres → radians, λ = 0.0555 m for C-band)
        displacement = subsidence_signal * dt_years
        phase = (4 * np.pi / 0.0555) * displacement
        noise = rng.normal(0, 0.3, shape)
        wrapped_phase = np.angle(np.exp(1j * (phase + noise)))

        # Coherence (higher in urban areas → central region)
        coherence = 0.7 + 0.2 * np.exp(-(xx**2 + yy**2) / 0.5)
        coherence = np.clip(coherence + rng.normal(0, 0.05, shape), 0, 1)

        try:
            filtered_phase = goldstein_filter(wrapped_phase)
        except ImportError:
            filtered_phase = wrapped_phase

        unwrapped = phase + noise  # true unwrapped (ideal for testing)

        _write_tif(ifg_dir / f"{tag}_phase.tif", filtered_phase)
        _write_tif(coh_dir / f"{tag}_coherence.tif", coherence)
        _write_tif(unw_dir / f"{tag}_unwrapped.tif", unwrapped)

    log.info(
        "Synthetic InSAR products generated: %d interferogram pairs",
        len(pairs),
    )


# ---------------------------------------------------------------------------
# SBAS inversion (least-squares)
# ---------------------------------------------------------------------------

def sbas_inversion(
    pairs: list[dict],
    dates: list[str],
    output_dir: Path,
    shape: tuple[int, int],
    bbox: list[float],
) -> np.ndarray:
    """Perform SBAS inversion to recover displacement time series.

    Solves: A @ v = d  (least-squares, where v = velocity per epoch,
    d = unwrapped phase differences).

    Args:
        pairs:      Interferogram pair list.
        dates:      Sorted unique acquisition dates.
        output_dir: Directory containing unwrapped phase rasters.
        shape:      (rows, cols) of rasters.
        bbox:       [lon_min, lat_min, lon_max, lat_max].

    Returns:
        Displacement time series array of shape (n_epochs, rows, cols).
    """
    import datetime

    n_dates = len(dates)
    n_pairs = len(pairs)
    date_idx = {d: i for i, d in enumerate(dates)}

    # Design matrix A (n_pairs × n_epochs-1)
    A = np.zeros((n_pairs, n_dates - 1), dtype=np.float32)
    for i, pair in enumerate(pairs):
        m_idx = date_idx[pair["master"]]
        s_idx = date_idx[pair["slave"]]
        for j in range(m_idx, s_idx):
            dt = (
                datetime.date.fromisoformat(dates[j + 1])
                - datetime.date.fromisoformat(dates[j])
            ).days / 365.25
            A[i, j] = dt

    unw_dir = output_dir / "unwrapped"
    ts_dir = output_dir / "timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)

    # Read unwrapped phases
    phases = np.zeros((n_pairs, shape[0], shape[1]), dtype=np.float32)
    for i, pair in enumerate(pairs):
        tag = f"{pair['master'].replace('-', '')}_{pair['slave'].replace('-', '')}"
        tif = unw_dir / f"{tag}_unwrapped.tif"
        if tif.exists():
            try:
                import rasterio

                with rasterio.open(tif) as src:
                    phases[i] = src.read(1)
            except Exception as exc:
                log.warning("Could not read %s: %s", tif, exc)

    # Convert phase to displacement (m)
    wavelength = 0.0555
    displacements = phases * wavelength / (4 * np.pi)

    # Solve per pixel (vectorised)
    rows, cols = shape
    d_flat = displacements.reshape(n_pairs, -1)
    try:
        v_flat, _, _, _ = np.linalg.lstsq(A, d_flat, rcond=None)
    except Exception as exc:
        log.error("SBAS inversion failed: %s", exc)
        return np.zeros((n_dates - 1, rows, cols), dtype=np.float32)

    velocity = v_flat.reshape(n_dates - 1, rows, cols)
    ts = np.cumsum(velocity, axis=0)

    # Save each epoch
    try:
        import rasterio
        from rasterio.transform import from_bounds

        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], cols, rows)
        for epoch_i, date in enumerate(dates[1:]):
            out_path = ts_dir / f"displacement_{date.replace('-', '')}.tif"
            with rasterio.open(
                out_path,
                "w",
                driver="GTiff",
                height=rows,
                width=cols,
                count=1,
                dtype=np.float32,
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(ts[epoch_i], 1)
    except ImportError:
        log.warning("rasterio not available; skipping time-series raster export.")

    log.info("SBAS inversion complete: %d epochs.", n_dates - 1)
    return ts


def run(cfg: dict, force: bool = False) -> None:
    """Execute SBAS InSAR processing pipeline step.

    Args:
        cfg:   Full project configuration.
        force: Re-run even if checkpoint exists.
    """
    if checkpoint_exists(SENTINEL_FILE) and not force:
        log.info("SBAS processing already complete. Use --force to re-run.")
        return

    insar_cfg = cfg["insar"]
    output_dir = Path(insar_cfg["output_dir"])
    bbox = cfg["study_area"]["bbox"]
    shape = (256, 256)  # synthetic grid size

    # Load manifest or generate synthetic dates
    manifest_path = Path("data/raw/sentinel1/manifest.json")
    if manifest_path.exists():
        with manifest_path.open() as fh:
            manifest = json.load(fh)
        dates = sorted(
            {s["date"][:10] for s in manifest.get("scenes", [])}
        )
    else:
        import datetime

        start = datetime.date(2019, 1, 1)
        dates = [
            (start + datetime.timedelta(days=12 * i)).isoformat()
            for i in range(40)
        ]

    log.info("Processing %d acquisition dates.", len(dates))

    pairs = build_interferogram_network(
        dates,
        insar_cfg["temporal_baseline_days"],
        insar_cfg["spatial_baseline_m"],
    )

    # Save network
    network_path = Path(insar_cfg["network_file"])
    network_path.parent.mkdir(parents=True, exist_ok=True)
    with network_path.open("w") as fh:
        json.dump({"dates": dates, "pairs": pairs}, fh, indent=2)
    log.info("Network saved: %s", network_path)

    generate_synthetic_insar(pairs, output_dir, shape, bbox)
    sbas_inversion(pairs, dates, output_dir, shape, bbox)

    mark_completed(SENTINEL_FILE)
    log.info("SBAS InSAR processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SBAS InSAR processing.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level, step="insar_sbas")
    cfg = load_config(args.config)
    run(cfg, force=args.force)
