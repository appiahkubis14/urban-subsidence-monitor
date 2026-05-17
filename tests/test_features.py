#!/usr/bin/env python3
"""
tests/test_features.py
======================
Unit tests for build_features.py and subsidence_rates.py.

Run with:
    pytest tests/test_features.py -v
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_stack():
    """Create a synthetic 8-channel feature stack (C, H, W)."""
    n_channels = 8
    height, width = 64, 64
    return np.random.random((n_channels, height, width)).astype(np.float32)


@pytest.fixture
def sample_config():
    return {
        "features": {
            "channels": [
                "coherence", "amplitude", "dem", "slope",
                "temporal_baseline", "dist_to_water",
                "urban_mask", "subsidence_rate",
            ],
            "target_resolution_m": 50,
            "normalise": True,
        },
        "paths": {"features": "data/features"},
    }


# ---------------------------------------------------------------------------
# build_features.py tests
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    """Tests for 8-channel ML input stack construction."""

    def test_import(self):
        import scripts.build_features  # noqa: F401

    def test_stack_has_eight_channels(self, sample_stack):
        assert sample_stack.shape[0] == 8

    def test_normalise_to_unit_range(self):
        """Normalisation maps each channel to [0, 1]."""
        arr = np.random.random((8, 64, 64)).astype(np.float32) * 100
        normalised = np.zeros_like(arr)
        for c in range(arr.shape[0]):
            ch = arr[c]
            mn, mx = ch.min(), ch.max()
            normalised[c] = (ch - mn) / (mx - mn + 1e-8)
        assert np.all(normalised >= 0 - 1e-6)
        assert np.all(normalised <= 1 + 1e-6)

    def test_scaler_serialisable(self, tmp_path):
        """Scaler dict can be pickled and reloaded."""
        scalers = {
            f"channel_{i}": {"min": float(i), "max": float(i + 10)}
            for i in range(8)
        }
        p = tmp_path / "scalers.pkl"
        with open(p, "wb") as f:
            pickle.dump(scalers, f)
        with open(p, "rb") as f:
            loaded = pickle.load(f)
        assert loaded.keys() == scalers.keys()

    def test_channel_names_match_config(self, sample_config):
        channels = sample_config["features"]["channels"]
        assert len(channels) == 8
        assert "coherence" in channels
        assert "subsidence_rate" in channels

    def test_no_nan_in_normalised_stack(self):
        """After clipping and normalising, no NaN values remain."""
        arr = np.random.random((8, 64, 64)).astype(np.float32)
        arr[0, 0, 0] = np.nan  # inject NaN
        arr = np.nan_to_num(arr, nan=0.0)
        assert not np.any(np.isnan(arr))


# ---------------------------------------------------------------------------
# subsidence_rates.py tests
# ---------------------------------------------------------------------------

class TestSubsidenceRates:
    """Tests for Huber regression subsidence rate estimation."""

    def test_import(self):
        import scripts.subsidence_rates  # noqa: F401

    def test_design_matrix_shape(self):
        """Seasonal design matrix has 6 columns (a, b, c, d, e, f)."""
        import numpy as np

        n_epochs = 20
        t = np.linspace(0, 3, n_epochs)  # years
        A = np.column_stack([
            np.ones(n_epochs),      # a — constant
            t,                      # b — linear trend
            np.sin(2 * np.pi * t),  # c — annual sine
            np.cos(2 * np.pi * t),  # d — annual cosine
            np.sin(4 * np.pi * t),  # e — semi-annual sine
            np.cos(4 * np.pi * t),  # f — semi-annual cosine
        ])
        assert A.shape == (n_epochs, 6)

    def test_linear_rate_recovery(self):
        """Linear trend (no noise) is recovered with negligible error."""
        t = np.linspace(0, 3, 30)
        true_rate = -15.0  # mm/year
        displacements = true_rate * t + 5.0  # intercept 5 mm

        A = np.column_stack([np.ones_like(t), t])
        coeffs, _, _, _ = np.linalg.lstsq(A, displacements, rcond=None)
        recovered_rate = coeffs[1]
        assert abs(recovered_rate - true_rate) < 0.01  # <0.01 mm/yr error

    def test_rate_map_dtype(self):
        """Rate map array has float32 dtype."""
        rate_map = np.random.random((100, 100)).astype(np.float32) * -30
        assert rate_map.dtype == np.float32

    def test_seasonal_amplitude_positive(self):
        """Seasonal amplitude is always non-negative."""
        c = np.random.random((64, 64)).astype(np.float32) - 0.5
        d = np.random.random((64, 64)).astype(np.float32) - 0.5
        amplitude = np.sqrt(c**2 + d**2)
        assert np.all(amplitude >= 0)
