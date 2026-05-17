#!/usr/bin/env python3
"""
scripts/download_dem.py
========================
Download SRTM 30 m DEM for the study area, clip, and reproject to UTM.

Author : Samuel Appiah Kubi
Licence: MIT
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from scripts.utils import (
    checkpoint_exists,
    load_config,
    mark_completed,
    setup_logging,
)

log = logging.getLogger(__name__)

SENTINEL_FILE = "data/raw/dem/download_done.txt"


def download_srtm_opentopography(
    bbox: list[float],
    output_path: Path,
    api_key: str = "",
) -> bool:
    """Download SRTM DEM via OpenTopography REST API.

    Args:
        bbox:        [lon_min, lat_min, lon_max, lat_max] in WGS84.
        output_path: Destination GeoTIFF path.
        api_key:     Optional OpenTopography API key.

    Returns:
        True on success, False on failure.
    """
    try:
        import requests

        url = "https://portal.opentopography.org/API/globaldem"
        params = {
            "demtype": "SRTMGL1",
            "south": bbox[1],
            "north": bbox[3],
            "west": bbox[0],
            "east": bbox[2],
            "outputFormat": "GTiff",
        }
        if api_key:
            params["API_Key"] = api_key

        log.info("Requesting SRTM DEM from OpenTopography ...")
        resp = requests.get(url, params=params, stream=True, timeout=120)
        resp.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)
        log.info("DEM downloaded: %s", output_path)
        return True

    except Exception as exc:
        log.warning("OpenTopography download failed: %s", exc)
        return False


def download_srtm_elevation_library(
    bbox: list[float],
    output_path: Path,
) -> bool:
    """Download SRTM DEM tiles using the 'elevation' Python library.

    Args:
        bbox:        [lon_min, lat_min, lon_max, lat_max].
        output_path: Destination GeoTIFF path.

    Returns:
        True on success, False on failure.
    """
    try:
        import elevation

        output_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading SRTM via elevation library ...")
        elevation.clip(bounds=tuple(bbox), output=str(output_path))
        log.info("DEM clipped and saved: %s", output_path)
        return True
    except Exception as exc:
        log.warning("elevation library download failed: %s", exc)
        return False


def create_synthetic_dem(
    bbox: list[float],
    output_path: Path,
    resolution_m: int = 30,
) -> None:
    """Create a synthetic DEM for testing when real data is unavailable.

    Args:
        bbox:         [lon_min, lat_min, lon_max, lat_max].
        output_path:  Destination GeoTIFF path.
        resolution_m: Target pixel size in metres.
    """
    try:
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        deg_per_m = 1.0 / 111_000
        dx = resolution_m * deg_per_m
        width = max(1, int((bbox[2] - bbox[0]) / dx))
        height = max(1, int((bbox[3] - bbox[1]) / dx))

        rng = np.random.default_rng(42)
        dem = (
            5.0  # Accra is near sea level
            + rng.normal(0, 2, (height, width)).cumsum(axis=0) * 0.05
        ).clip(0, 50).astype(np.float32)

        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(dem, 1)
        log.info("Synthetic DEM created: %s (%dx%d)", output_path, width, height)

    except ImportError:
        log.error("rasterio not available; cannot create synthetic DEM.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()
        log.warning("Created empty placeholder: %s", output_path)


def reproject_dem(
    input_path: Path,
    output_path: Path,
    target_epsg: int,
) -> None:
    """Reproject DEM to UTM CRS.

    Args:
        input_path:  Input GeoTIFF (WGS84).
        output_path: Reprojected output GeoTIFF.
        target_epsg: EPSG code of target CRS.
    """
    try:
        import numpy as np
        import rasterio
        from rasterio.warp import (
            Resampling,
            calculate_default_transform,
            reproject,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(input_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs,
                f"EPSG:{target_epsg}",
                src.width,
                src.height,
                *src.bounds,
            )
            kwargs = src.meta.copy()
            kwargs.update(
                {
                    "crs": f"EPSG:{target_epsg}",
                    "transform": transform,
                    "width": width,
                    "height": height,
                }
            )
            with rasterio.open(output_path, "w", **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=f"EPSG:{target_epsg}",
                        resampling=Resampling.bilinear,
                    )
        log.info("DEM reprojected → EPSG:%d: %s", target_epsg, output_path)

    except ImportError:
        log.error("rasterio not available; skipping reproject.")


def run(cfg: dict, force: bool = False) -> None:
    """Execute the DEM download pipeline step.

    Args:
        cfg:   Full project configuration.
        force: Re-run even if checkpoint exists.
    """
    if checkpoint_exists(SENTINEL_FILE) and not force:
        log.info("DEM download already complete. Use --force to re-run.")
        return

    bbox = cfg["study_area"]["bbox"]
    raw_path = Path(cfg["dem"]["output_path"])
    utm_path = Path(cfg["dem"]["utm_output_path"])
    epsg = cfg["study_area"]["utm_epsg"]
    api_key = os.environ.get("OPENTOPOGRAPHY_API_KEY", "")

    success = download_srtm_opentopography(bbox, raw_path, api_key)
    if not success:
        success = download_srtm_elevation_library(bbox, raw_path)
    if not success:
        log.warning("Real DEM unavailable — creating synthetic DEM for testing.")
        create_synthetic_dem(bbox, raw_path)

    reproject_dem(raw_path, utm_path, epsg)
    mark_completed(SENTINEL_FILE)
    log.info("DEM download step complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and reproject SRTM DEM."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level, step="download_dem")
    cfg = load_config(args.config)
    run(cfg, force=args.force)
