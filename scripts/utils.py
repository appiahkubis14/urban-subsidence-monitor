#!/usr/bin/env python3
"""
scripts/utils.py
================
Shared utilities for Urban Subsidence Watcher.

Author : Samuel Appiah Kubi
Licence: MIT
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import requests
    from tqdm import tqdm
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ---------------------------------------------------------------------------
# Colour codes (ANSI)
# ---------------------------------------------------------------------------
_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
    "RESET":    "\033[0m",
}


class _ColouredFormatter(logging.Formatter):
    """Logging formatter that adds ANSI colour codes and step names."""

    def __init__(self, step: str = "") -> None:
        super().__init__()
        self._step = step

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        colour = _COLOURS.get(record.levelname, "")
        reset = _COLOURS["RESET"]
        ts = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        step_tag = f"[{self._step}] " if self._step else ""
        return (
            f"{colour}{ts} | {record.levelname:<8} | "
            f"{step_tag}{record.getMessage()}{reset}"
        )


def setup_logging(
    level: str = "INFO",
    step: str = "",
    log_file: str | None = None,
) -> logging.Logger:
    """Configure coloured console logging with optional file sink.

    Args:
        level:    Log level string (DEBUG, INFO, WARNING, ERROR).
        step:     Pipeline step name included in every log line.
        log_file: Optional path to write plain-text log file.

    Returns:
        Configured root logger.
    """
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(_ColouredFormatter(step))
    logger.addHandler(console)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config_path does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(checkpoint_path: str | Path, data: dict) -> None:
    """Persist checkpoint data as JSON.

    Args:
        checkpoint_path: Destination file path.
        data:            Serialisable dictionary.
    """
    cp = Path(checkpoint_path)
    cp.parent.mkdir(parents=True, exist_ok=True)
    with cp.open("w") as fh:
        json.dump(data, fh, indent=2, default=str)
    logging.getLogger(__name__).debug("Checkpoint saved: %s", cp)


def load_checkpoint(checkpoint_path: str | Path) -> dict:
    """Load checkpoint data from JSON.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        Dictionary of checkpoint data, or empty dict if file missing.
    """
    cp = Path(checkpoint_path)
    if not cp.exists():
        return {}
    with cp.open() as fh:
        return json.load(fh)


def checkpoint_exists(sentinel_path: str | Path) -> bool:
    """Return True if the sentinel file exists (step already completed).

    Args:
        sentinel_path: Path to the sentinel/done file.

    Returns:
        Boolean existence flag.
    """
    return Path(sentinel_path).exists()


def mark_completed(sentinel_path: str | Path) -> None:
    """Create a sentinel file to mark a pipeline step as completed.

    Args:
        sentinel_path: Path to create.
    """
    sp = Path(sentinel_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.touch()
    logging.getLogger(__name__).info("Step marked complete: %s", sp)


# ---------------------------------------------------------------------------
# Raster I/O
# ---------------------------------------------------------------------------

def raster_to_numpy(raster_path: str | Path) -> tuple[np.ndarray, dict]:
    """Read a GeoTIFF into a NumPy array plus metadata dict.

    Args:
        raster_path: Path to input GeoTIFF.

    Returns:
        Tuple of (array [bands, rows, cols], rasterio meta dict).

    Raises:
        ImportError: If rasterio is not installed.
        FileNotFoundError: If raster_path does not exist.
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required for raster_to_numpy")
    rp = Path(raster_path)
    if not rp.exists():
        raise FileNotFoundError(f"Raster not found: {rp}")
    with rasterio.open(rp) as src:
        arr = src.read()
        meta = src.meta.copy()
    return arr, meta


def numpy_to_geotiff(
    array: np.ndarray,
    meta: dict,
    output_path: str | Path,
) -> None:
    """Write a NumPy array to GeoTIFF using provided metadata.

    Args:
        array:       Array of shape (bands, rows, cols) or (rows, cols).
        meta:        rasterio metadata dict (crs, transform, dtype, …).
        output_path: Destination path.

    Raises:
        ImportError: If rasterio is not installed.
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required for numpy_to_geotiff")
    op = Path(output_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    meta.update({"count": array.shape[0], "dtype": array.dtype})
    with rasterio.open(op, "w", **meta) as dst:
        dst.write(array)
    logging.getLogger(__name__).info("GeoTIFF written: %s", op)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics between true and predicted arrays.

    Args:
        y_true: Ground-truth values (1-D).
        y_pred: Predicted values (1-D, same shape as y_true).

    Returns:
        Dictionary with keys: mae, rmse, r2, bias.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "bias": np.nan}
    residuals = y_pred - y_true
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    bias = float(np.mean(residuals))
    return {"mae": mae, "rmse": rmse, "r2": r2, "bias": bias}


# ---------------------------------------------------------------------------
# Spatial block split
# ---------------------------------------------------------------------------

def spatial_block_split(
    gdf: Any,
    n_blocks: int = 5,
    buffer_km: float = 2.0,
) -> list[tuple[Any, Any]]:
    """Split a GeoDataFrame into spatial blocks for cross-validation.

    Args:
        gdf:       Input GeoDataFrame with geometry column.
        n_blocks:  Number of CV folds (blocks).
        buffer_km: Exclusion buffer around block boundaries in km.

    Returns:
        List of (train_idx, test_idx) tuples of integer index arrays.
    """
    try:
        import geopandas as gpd  # noqa: F401
        from shapely.geometry import box
    except ImportError:
        logging.getLogger(__name__).warning(
            "geopandas/shapely not available; returning trivial split"
        )
        n = len(gdf)
        indices = np.arange(n)
        fold_size = n // n_blocks
        splits = []
        for i in range(n_blocks):
            test_idx = indices[i * fold_size: (i + 1) * fold_size]
            train_idx = np.setdiff1d(indices, test_idx)
            splits.append((train_idx, test_idx))
        return splits

    bounds = gdf.total_bounds  # xmin, ymin, xmax, ymax
    x_edges = np.linspace(bounds[0], bounds[2], n_blocks + 1)
    splits = []
    buf_deg = buffer_km / 111.0
    for i in range(n_blocks):
        test_box = box(x_edges[i], bounds[1], x_edges[i + 1], bounds[3])
        inner_box = box(
            x_edges[i] + buf_deg,
            bounds[1],
            x_edges[i + 1] - buf_deg,
            bounds[3],
        )
        test_mask = gdf.geometry.intersects(inner_box)
        buffer_mask = gdf.geometry.intersects(test_box) & ~test_mask
        train_mask = ~gdf.geometry.intersects(test_box)
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]
        splits.append((train_idx, test_idx))
    return splits


# ---------------------------------------------------------------------------
# File download
# ---------------------------------------------------------------------------

def download_file(
    url: str,
    dest: str | Path,
    resume: bool = True,
    retries: int = 3,
    expected_sha256: str | None = None,
) -> Path:
    """Download a file with optional resume, retry, and hash verification.

    Args:
        url:             Remote URL.
        dest:            Local destination path.
        resume:          Resume partial downloads if True.
        retries:         Number of retry attempts on failure.
        expected_sha256: Optional SHA-256 hex digest for verification.

    Returns:
        Path to the downloaded file.

    Raises:
        RuntimeError: If download fails after all retries.
        ImportError:  If requests/tqdm are not installed.
    """
    if not HAS_REQUESTS:
        raise ImportError("requests and tqdm are required for download_file")

    log = logging.getLogger(__name__)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            headers: dict[str, str] = {}
            initial_pos = 0
            if resume and dest.exists():
                initial_pos = dest.stat().st_size
                headers["Range"] = f"bytes={initial_pos}-"
                log.debug("Resuming download from byte %d", initial_pos)

            resp = requests.get(url, headers=headers, stream=True, timeout=60)
            if resp.status_code == 416:
                log.info("File already complete (416 Range Not Satisfiable).")
                break
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0)) + initial_pos
            mode = "ab" if initial_pos > 0 else "wb"
            with (
                dest.open(mode) as fh,
                tqdm(
                    total=total,
                    initial=initial_pos,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                ) as bar,
            ):
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)
                    bar.update(len(chunk))
            break

        except Exception as exc:
            log.warning("Attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt == retries:
                raise RuntimeError(
                    f"Download failed after {retries} attempts: {url}"
                ) from exc
            time.sleep(2**attempt)

    if expected_sha256:
        sha = hashlib.sha256()
        with dest.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                sha.update(chunk)
        digest = sha.hexdigest()
        if digest != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {dest}: got {digest}, "
                f"expected {expected_sha256}"
            )
        log.info("SHA-256 verified: %s", dest.name)

    return dest


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Utility self-test")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    logger = setup_logging("DEBUG", step="utils")
    cfg = load_config(args.config)
    logger.info("Config loaded: %s", cfg.get("project", {}).get("name"))
    logger.info("Metrics test: %s", compute_metrics(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.1, 1.9, 3.2]),
    ))
