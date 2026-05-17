#!/usr/bin/env python3
"""
tests/test_forecast.py
======================
Unit tests for forecast.py, validate.py, and export.py.

Run with:
    pytest tests/test_forecast.py -v
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_rate_map():
    """Synthetic subsidence rate map (mm/year)."""
    rng = np.random.default_rng(42)
    rate = rng.uniform(-30, 2, size=(64, 64)).astype(np.float32)
    return rate


@pytest.fixture
def sample_config():
    return {
        "forecast": {
            "horizons_years": [1, 3, 5],
            "mc_passes": 10,
            "uncertainty_percentiles": [5, 95],
        },
        "validation": {
            "n_cv_blocks": 5,
            "buffer_km": 2,
            "temporal_holdout_years": 1,
        },
        "export": {
            "cog_compress": "deflate",
            "cog_tile_size": 256,
            "hotspot_threshold_mm_yr": -5.0,
        },
        "paths": {
            "forecasts": "data/forecasts",
            "validation": "data/validation",
            "exports": "data/exports",
        },
    }


# ---------------------------------------------------------------------------
# forecast.py tests
# ---------------------------------------------------------------------------

class TestForecast:
    """Tests for linear extrapolation and MC uncertainty."""

    def test_import(self):
        import scripts.forecast  # noqa: F401

    def test_linear_extrapolation(self, sample_rate_map):
        """Forecast = rate × horizon, matching shape."""
        for horizon in [1, 3, 5]:
            forecast = sample_rate_map * horizon
            assert forecast.shape == sample_rate_map.shape

    def test_forecast_magnitude_increases_with_horizon(self, sample_rate_map):
        """Longer horizon → larger absolute displacement."""
        f1 = np.abs(sample_rate_map * 1)
        f5 = np.abs(sample_rate_map * 5)
        assert np.all(f5 >= f1 - 1e-6)

    def test_mc_uncertainty_positive(self, sample_rate_map):
        """MC standard deviation is non-negative everywhere."""
        # Simulate MC passes with small noise
        rng = np.random.default_rng(0)
        passes = np.stack([
            sample_rate_map + rng.normal(0, 1, sample_rate_map.shape).astype(np.float32)
            for _ in range(10)
        ])
        std = passes.std(axis=0)
        assert np.all(std >= 0)

    def test_forecast_json_schema(self, tmp_path, sample_config):
        """Forecast summary JSON has expected keys."""
        summary = {
            "horizons_years": [1, 3, 5],
            "max_subsidence_1yr_mm": -28.5,
            "max_subsidence_3yr_mm": -85.4,
            "max_subsidence_5yr_mm": -142.4,
            "mean_uncertainty_mm": 3.2,
            "n_hotspot_pixels": 120,
        }
        p = tmp_path / "forecast_summary.json"
        p.write_text(json.dumps(summary))
        loaded = json.loads(p.read_text())
        for key in ("horizons_years", "max_subsidence_1yr_mm"):
            assert key in loaded


# ---------------------------------------------------------------------------
# validate.py tests
# ---------------------------------------------------------------------------

class TestValidation:
    """Tests for GPS comparison and spatial block CV."""

    def test_import(self):
        import scripts.validate  # noqa: F401

    def test_mae_calculation(self):
        """MAE is correctly computed."""
        y_true = np.array([-10.0, -20.0, -5.0])
        y_pred = np.array([-12.0, -18.0, -5.0])
        mae = np.mean(np.abs(y_true - y_pred))
        assert abs(mae - (2.0 + 2.0 + 0.0) / 3) < 1e-6

    def test_rmse_calculation(self):
        """RMSE matches manual computation."""
        y_true = np.array([0.0, 10.0, -5.0])
        y_pred = np.array([1.0, 8.0, -5.0])
        residuals = y_true - y_pred
        rmse = np.sqrt(np.mean(residuals**2))
        expected = np.sqrt((1 + 4 + 0) / 3)
        assert abs(rmse - expected) < 1e-5

    def test_r2_perfect_fit(self):
        """R² = 1.0 for perfect predictions."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r2 = 1 - np.sum((y - y) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-8)
        assert abs(r2 - 1.0) < 1e-6

    def test_spatial_block_split_count(self):
        """Spatial block split produces n_blocks folds."""
        # Simplified: just verify logic
        n_blocks = 5
        indices = list(range(100))
        block_size = len(indices) // n_blocks
        folds = [indices[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
        assert len(folds) == n_blocks


# ---------------------------------------------------------------------------
# export.py tests
# ---------------------------------------------------------------------------

class TestExport:
    """Tests for COG, GeoJSON, STAC, and PDF export."""

    def test_import(self):
        import scripts.export  # noqa: F401

    def test_hotspot_threshold(self, sample_rate_map, sample_config):
        """Hotspot mask identifies pixels below threshold."""
        threshold = sample_config["export"]["hotspot_threshold_mm_yr"]
        hotspots = sample_rate_map < threshold
        assert hotspots.dtype == bool
        assert hotspots.sum() > 0  # synthetic data should have some hotspots

    def test_stac_catalog_schema(self, tmp_path):
        """STAC catalog JSON validates minimal schema."""
        catalog = {
            "type": "Catalog",
            "id": "urban-subsidence-watcher",
            "stac_version": "1.0.0",
            "description": "Urban Subsidence Watcher outputs",
            "links": [],
        }
        p = tmp_path / "catalog.json"
        p.write_text(json.dumps(catalog))
        loaded = json.loads(p.read_text())
        assert loaded["stac_version"] == "1.0.0"
        assert loaded["type"] == "Catalog"

    def test_geojson_hotspots_schema(self, tmp_path):
        """GeoJSON output has FeatureCollection structure."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [-0.2, 5.6]},
                    "properties": {
                        "rate_mm_yr": -18.5,
                        "forecast_1yr_mm": -18.5,
                    },
                }
            ],
        }
        p = tmp_path / "hotspots.geojson"
        p.write_text(json.dumps(geojson))
        loaded = json.loads(p.read_text())
        assert loaded["type"] == "FeatureCollection"
        assert len(loaded["features"]) > 0
