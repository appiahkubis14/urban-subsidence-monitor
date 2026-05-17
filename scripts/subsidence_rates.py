#!/usr/bin/env python3
"""
scripts/subsidence_rates.py
============================
Estimate linear subsidence rates and seasonal components from SBAS
displacement time series using Huber robust regression.

Model: d(t) = a + b·t + c·sin(2πt) + d·cos(2πt)
               + e·sin(4πt) + f·cos(4πt)

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
    setup_logging,
)

log = logging.getLogger(__name__)

SENTINEL_FILE = "data/subsidence/subsidence_rates_done.txt"


def huber_loss_gradient(
    residuals: np.ndarray, delta: float
) -> tuple[np.ndarray, float]:
    """Compute Huber loss and pseudo-residuals for IRLS.

    Args:
        residuals: Residual vector (n_obs,).
        delta:     Huber transition threshold.

    Returns:
        Tuple of (weights array, scalar loss value).
    """
    abs_r = np.abs(residuals)
    weights = np.where(abs_r <= delta, 1.0, delta / (abs_r + 1e-10))
    loss = np.where(
        abs_r <= delta,
        0.5 * residuals**2,
        delta * (abs_r - 0.5 * delta),
    ).mean()
    return weights.astype(np.float64), float(loss)


def fit_time_series_pixel(
    t: np.ndarray,
    d: np.ndarray,
    n_harmonics: int = 2,
    delta: float = 1.0,
    max_iter: int = 50,
) -> dict[str, float]:
    """Fit secular + seasonal model to a single pixel time series (IRLS).

    Args:
        t:           Time array in decimal years (n_obs,).
        d:           Displacement array in metres (n_obs,).
        n_harmonics: Number of annual harmonics to include.
        delta:       Huber delta parameter.
        max_iter:    Maximum IRLS iterations.

    Returns:
        Dictionary with keys: rate_mm_year, intercept, seasonal_amplitude,
        seasonal_phase, rmse, n_obs.
    """
    mask = np.isfinite(d)
    t_m = t[mask]
    d_m = d[mask]
    n = len(t_m)

    if n < 4:
        return {
            "rate_mm_year": np.nan,
            "intercept": np.nan,
            "seasonal_amplitude": np.nan,
            "seasonal_phase": np.nan,
            "rmse": np.nan,
            "n_obs": n,
        }

    # Design matrix [1, t, sin(2πt), cos(2πt), ...]
    A = [np.ones(n), t_m]
    for h in range(1, n_harmonics + 1):
        A.append(np.sin(2 * np.pi * h * t_m))
        A.append(np.cos(2 * np.pi * h * t_m))
    A = np.column_stack(A)

    w = np.ones(n)
    coeff = np.zeros(A.shape[1])

    for _ in range(max_iter):
        W = np.diag(w)
        try:
            coeff = np.linalg.solve(A.T @ W @ A, A.T @ W @ d_m)
        except np.linalg.LinAlgError:
            coeff, *_ = np.linalg.lstsq(A, d_m, rcond=None)
            break
        residuals = d_m - A @ coeff
        w_new, _ = huber_loss_gradient(residuals, delta)
        if np.max(np.abs(w_new - w)) < 1e-6:
            break
        w = w_new

    rate_mm = coeff[1] * 1000.0  # convert m/yr → mm/yr
    # Seasonal amplitude from first harmonic
    if n_harmonics >= 1:
        amp = np.sqrt(coeff[2] ** 2 + coeff[3] ** 2) * 1000.0
        phase = np.arctan2(coeff[2], coeff[3])
    else:
        amp, phase = 0.0, 0.0

    residuals = d_m - A @ coeff
    rmse = float(np.sqrt(np.mean(residuals**2))) * 1000.0

    return {
        "rate_mm_year": float(rate_mm),
        "intercept": float(coeff[0] * 1000.0),
        "seasonal_amplitude": float(amp),
        "seasonal_phase": float(phase),
        "rmse": rmse,
        "n_obs": n,
    }


def load_timeseries_stack(
    ts_dir: Path,
) -> tuple[np.ndarray, list[str], dict]:
    """Load displacement time series rasters into a 3-D stack.

    Args:
        ts_dir: Directory containing displacement_*.tif files.

    Returns:
        Tuple of (stack [n_epochs, rows, cols], dates list, rasterio meta).
    """
    try:
        import rasterio
    except ImportError:
        log.warning("rasterio not available — generating synthetic time series.")
        return _synthetic_timeseries()

    tifs = sorted(ts_dir.glob("displacement_*.tif"))
    if not tifs:
        log.warning("No displacement rasters found — generating synthetic data.")
        return _synthetic_timeseries()

    stacks = []
    dates = []
    meta = {}
    for tif in tifs:
        with rasterio.open(tif) as src:
            stacks.append(src.read(1))
            meta = src.meta.copy()
            dates.append(tif.stem.replace("displacement_", ""))

    return np.stack(stacks, axis=0), dates, meta


def _synthetic_timeseries(
    shape: tuple[int, int] = (128, 128),
    n_epochs: int = 30,
) -> tuple[np.ndarray, list[str], dict]:
    """Generate synthetic displacement time series for testing.

    Args:
        shape:    (rows, cols).
        n_epochs: Number of epochs.

    Returns:
        Tuple of (stack, dates, dummy meta).
    """
    import datetime

    rng = np.random.default_rng(7)
    rows, cols = shape
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, rows), np.linspace(-1, 1, cols), indexing="ij"
    )
    rate_map = -0.025 * np.exp(-(xx**2 + yy**2) / 0.3)  # m/yr

    start = datetime.date(2019, 1, 1)
    dates = []
    stack = []
    for i in range(n_epochs):
        date = start + datetime.timedelta(days=12 * i)
        dates.append(date.strftime("%Y%m%d"))
        t = i * 12 / 365.25
        signal = rate_map * t + 0.002 * np.sin(2 * np.pi * t)
        noise = rng.normal(0, 0.001, shape)
        stack.append((signal + noise).astype(np.float32))

    meta = {"driver": "GTiff", "count": 1, "dtype": np.float32}
    return np.stack(stack, axis=0), dates, meta


def process_rates(
    stack: np.ndarray,
    t_years: np.ndarray,
    n_harmonics: int,
    delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit model to every pixel in the displacement stack.

    Args:
        stack:       (n_epochs, rows, cols) displacement array [metres].
        t_years:     Time vector in decimal years (n_epochs,).
        n_harmonics: Number of annual harmonics.
        delta:       Huber delta.

    Returns:
        Tuple of (rate_map [mm/yr], rate_error [mm], seasonal_amplitude [mm]).
    """
    _, rows, cols = stack.shape
    rate_map = np.full((rows, cols), np.nan, dtype=np.float32)
    error_map = np.full((rows, cols), np.nan, dtype=np.float32)
    seasonal_map = np.full((rows, cols), np.nan, dtype=np.float32)

    flat = stack.reshape(stack.shape[0], -1)
    total = rows * cols
    log.info("Fitting time series for %d pixels ...", total)

    for pix in range(total):
        result = fit_time_series_pixel(
            t_years, flat[:, pix], n_harmonics=n_harmonics, delta=delta
        )
        r, c = divmod(pix, cols)
        rate_map[r, c] = result["rate_mm_year"]
        error_map[r, c] = result["rmse"]
        seasonal_map[r, c] = result["seasonal_amplitude"]

        if pix % max(1, total // 10) == 0:
            log.info("  Progress: %d/%d pixels (%.0f%%)", pix, total, 100 * pix / total)

    return rate_map, error_map, seasonal_map


def run(cfg: dict, force: bool = False) -> None:
    """Execute subsidence rate estimation pipeline step.

    Args:
        cfg:   Full project configuration.
        force: Re-run even if checkpoint exists.
    """
    if checkpoint_exists(SENTINEL_FILE) and not force:
        log.info("Subsidence rates already computed. Use --force to re-run.")
        return

    sub_cfg = cfg["subsidence"]
    output_dir = Path(sub_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ts_dir = Path(cfg["insar"]["output_dir"]) / "timeseries"
    stack, dates, meta = load_timeseries_stack(ts_dir)

    # Convert date strings to decimal years
    import datetime
    def _to_decimal_year(d: str) -> float:
        dt = datetime.date(int(d[:4]), int(d[4:6] if len(d) >= 6 else 1),
                           int(d[6:8] if len(d) >= 8 else 1))
        return dt.year + (dt.timetuple().tm_yday - 1) / 365.25

    t_years = np.array([_to_decimal_year(d) for d in dates], dtype=np.float64)
    t_years -= t_years[0]  # relative time

    rate_map, error_map, seasonal_map = process_rates(
        stack,
        t_years,
        n_harmonics=sub_cfg["n_harmonics"],
        delta=sub_cfg["huber_delta"],
    )

    # Save rasters
    def _save(path: str, arr: np.ndarray) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            numpy_to_geotiff(arr, meta.copy(), p)
        except Exception as exc:
            log.warning("Could not write GeoTIFF %s: %s", p, exc)
            np.save(str(p).replace(".tif", ".npy"), arr)

    _save(sub_cfg["rate_map"], rate_map)
    _save(sub_cfg["rate_error_map"], error_map)
    _save(sub_cfg["seasonal_amplitude_map"], seasonal_map)

    # Save time series to Parquet
    try:
        import pandas as pd

        records = []
        for i, d in enumerate(dates):
            records.append(
                {
                    "date": d,
                    "t_years": float(t_years[i]),
                    "mean_displacement_mm": float(
                        np.nanmean(stack[i]) * 1000
                    ),
                }
            )
        df = pd.DataFrame(records)
        parquet_path = Path(sub_cfg["timeseries_parquet"])
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        log.info("Time series saved: %s", parquet_path)
    except ImportError:
        log.warning("pandas/pyarrow not available; skipping Parquet export.")

    mark_completed(SENTINEL_FILE)
    log.info(
        "Subsidence rates complete. Mean rate: %.2f mm/yr",
        float(np.nanmean(rate_map)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate subsidence rates from SBAS time series."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level, step="subsidence_rates")
    cfg = load_config(args.config)
    run(cfg, force=args.force)
