#!/usr/bin/env python3
"""
scripts/preprocess_s1.py
=========================
Preprocess Sentinel-1 SLC data: orbit correction, TOPSAR deburst,
multilooking, and coregistration to a reference scene.

Uses SNAP Python API (snappy) if available; falls back to a
GDAL-based lightweight simulation for testing.

Author : Samuel Appiah Kubi
Licence: MIT
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from scripts.utils import (
    checkpoint_exists,
    load_config,
    mark_completed,
    setup_logging,
)

log = logging.getLogger(__name__)

SENTINEL_FILE = "data/processed/s1_preprocess_done.txt"


# ---------------------------------------------------------------------------
# SNAP path
# ---------------------------------------------------------------------------

def _try_snappy():
    """Attempt to import snappy (SNAP Python API).

    Returns:
        snappy module or None.
    """
    try:
        import snappy  # type: ignore
        log.info("SNAP Python API (snappy) found.")
        return snappy
    except ImportError:
        log.warning(
            "snappy (SNAP Python API) not found. "
            "Install ESA SNAP with Python bindings for full InSAR processing. "
            "Falling back to GDAL simulation."
        )
        return None


def apply_orbit_file(snappy, product):
    """Apply precise orbit file to a Sentinel-1 product.

    Args:
        snappy:  snappy module.
        product: SNAP Product object.

    Returns:
        Orbit-corrected SNAP Product.
    """
    params = snappy.HashMap()
    params.put("orbitType", "Sentinel Precise (Auto Download)")
    params.put("polyDegree", 3)
    params.put("continueOnFail", False)
    op = snappy.GPF.createProduct("Apply-Orbit-File", params, product)
    log.debug("Orbit file applied.")
    return op


def topsar_deburst(snappy, product):
    """Perform TOPSAR deburst.

    Args:
        snappy:  snappy module.
        product: Orbit-corrected SNAP Product.

    Returns:
        Deburst SNAP Product.
    """
    params = snappy.HashMap()
    params.put("selectedPolarisations", "VV")
    op = snappy.GPF.createProduct("TOPSAR-Deburst", params, product)
    log.debug("TOPSAR deburst applied.")
    return op


def multilook(snappy, product, range_looks: int, azimuth_looks: int):
    """Apply multilooking.

    Args:
        snappy:         snappy module.
        product:        Deburst SNAP Product.
        range_looks:    Range looks factor.
        azimuth_looks:  Azimuth looks factor.

    Returns:
        Multilooked SNAP Product.
    """
    params = snappy.HashMap()
    params.put("nRgLooks", range_looks)
    params.put("nAzLooks", azimuth_looks)
    op = snappy.GPF.createProduct("Multilook", params, product)
    log.debug("Multilooking applied (%d rg, %d az).", range_looks, azimuth_looks)
    return op


def coregister(snappy, master, slave):
    """Coregister slave to master using Cross-Correlation.

    Args:
        snappy: snappy module.
        master: SNAP master Product.
        slave:  SNAP slave Product.

    Returns:
        Coregistered SNAP Product stack.
    """
    params = snappy.HashMap()
    params.put("masterBandNames", "Intensity_VV")
    op = snappy.GPF.createProduct(
        "CreateStack", params, snappy.jpy.array("org.esa.snap.core.datamodel.Product", [master, slave])
    )
    log.debug("Coregistration applied.")
    return op


def process_with_snappy(
    scene_paths: list[Path],
    output_dir: Path,
    cfg: dict,
) -> None:
    """Full preprocessing pipeline using SNAP via snappy.

    Args:
        scene_paths: Paths to Sentinel-1 SLC zip files.
        output_dir:  Directory to save coregistered products.
        cfg:         Full project configuration.
    """
    snappy = _try_snappy()
    if snappy is None:
        process_fallback(scene_paths, output_dir, cfg)
        return

    pre_cfg = cfg["preprocessing"]
    output_dir.mkdir(parents=True, exist_ok=True)

    products = []
    for sp in scene_paths:
        log.info("Reading: %s", sp.name)
        try:
            prod = snappy.ProductIO.readProduct(str(sp))
            prod = apply_orbit_file(snappy, prod)
            prod = topsar_deburst(snappy, prod)
            prod = multilook(
                snappy,
                prod,
                pre_cfg["multilook_range"],
                pre_cfg["multilook_azimuth"],
            )
            products.append(prod)
        except Exception as exc:
            log.error("Failed to process %s: %s", sp.name, exc)

    if not products:
        log.error("No products successfully preprocessed.")
        return

    master = products[0]
    log.info("Master scene: %s", scene_paths[0].name)

    for i, (slave, sp) in enumerate(zip(products[1:], scene_paths[1:]), start=1):
        log.info("Coregistering slave %d/%d: %s", i, len(products) - 1, sp.name)
        try:
            stack = coregister(snappy, master, slave)
            out_name = sp.stem + "_coreg.dim"
            out_path = output_dir / out_name
            snappy.ProductIO.writeProduct(stack, str(out_path), "BEAM-DIMAP")
            log.info("Saved: %s", out_path)
        except Exception as exc:
            log.error("Coregistration failed for %s: %s", sp.name, exc)


def process_fallback(
    scene_paths: list[Path],
    output_dir: Path,
    cfg: dict,
) -> None:
    """GDAL-based fallback preprocessing (simulation / testing).

    Creates synthetic coregistered placeholder rasters using GDAL or
    numpy when SNAP is not available.

    Args:
        scene_paths: List of input SLC zip paths (may be placeholders).
        output_dir:  Output directory.
        cfg:         Full project configuration.
    """
    log.warning(
        "Running GDAL fallback preprocessing — outputs are synthetic "
        "placeholders for pipeline testing."
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        bbox = cfg["study_area"]["bbox"]
        epsg = cfg["study_area"]["utm_epsg"]
        spacing = cfg["preprocessing"]["pixel_spacing_m"]
        deg_per_m = 1.0 / 111_000
        dx = spacing * deg_per_m
        width = max(1, int((bbox[2] - bbox[0]) / dx))
        height = max(1, int((bbox[3] - bbox[1]) / dx))
        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)

        rng = np.random.default_rng(0)
        for i, sp in enumerate(scene_paths):
            out_path = output_dir / (sp.stem + "_coreg.tif")
            if out_path.exists():
                log.debug("Skip (exists): %s", out_path.name)
                continue
            amplitude = (rng.rayleigh(1000, (height, width))).astype(np.float32)
            phase = (rng.uniform(-np.pi, np.pi, (height, width))).astype(np.float32)
            with rasterio.open(
                out_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=2,
                dtype=np.float32,
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(amplitude, 1)
                dst.write(phase, 2)
            log.info("Synthetic coregistered raster: %s", out_path.name)

    except ImportError:
        log.error("numpy/rasterio not available; creating empty placeholders.")
        for sp in scene_paths:
            (output_dir / (sp.stem + "_coreg.tif")).touch()


def run(cfg: dict, force: bool = False) -> None:
    """Execute preprocessing pipeline step.

    Args:
        cfg:   Full project configuration.
        force: Re-run even if checkpoint exists.
    """
    if checkpoint_exists(SENTINEL_FILE) and not force:
        log.info("Preprocessing already complete. Use --force to re-run.")
        return

    download_dir = Path(cfg["sentinel1"]["download_dir"])
    output_dir = Path(cfg["preprocessing"]["output_dir"])

    scene_paths = sorted(download_dir.glob("*.zip"))
    if not scene_paths:
        log.warning(
            "No SLC zip files found in %s. "
            "Using synthetic placeholders from preprocess fallback.",
            download_dir,
        )
        from pathlib import Path as _P

        scene_paths = [
            _P(cfg["sentinel1"]["download_dir"]) / f"synthetic_{i:04d}.zip"
            for i in range(5)
        ]

    snappy = _try_snappy()
    if snappy is not None:
        process_with_snappy(scene_paths, output_dir, cfg)
    else:
        process_fallback(scene_paths, output_dir, cfg)

    mark_completed(SENTINEL_FILE)
    log.info("Sentinel-1 preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Sentinel-1 SLC data for InSAR."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level, step="preprocess_s1")
    cfg = load_config(args.config)
    run(cfg, force=args.force)
