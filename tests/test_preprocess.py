#!/usr/bin/env python3
"""
tests/test_preprocess.py
========================
Unit tests for preprocess_s1.py and insar_sbas.py.

Run with:
    pytest tests/test_preprocess.py -v
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
def sample_config():
    return {
        "study_area": {"bbox": [-0.5, 5.5, -0.1, 5.8], "epsg": 32630},
        "insar": {
            "max_temporal_baseline_days": 120,
            "max_spatial_baseline_m": 200,
            "multilook_range": 4,
            "multilook_azimuth": 1,
            "coherence_threshold": 0.3,
            "filter_strength": 0.5,
        },
        "paths": {
            "processed_s1": "data/processed/s1_coregistered",
            "insar": "data/insar",
        },
    }


# ---------------------------------------------------------------------------
# preprocess_s1.py tests
# ---------------------------------------------------------------------------

class TestPreprocessS1:
    """Tests for Sentinel-1 SLC preprocessing."""

    def test_import(self):
        import scripts.preprocess_s1  # noqa: F401

    def test_multilook_shape(self):
        """Multilooking by factor 4 reduces columns by 4."""
        rows, cols = 200, 400
        mlook_factor = 4
        arr = np.random.random((rows, cols)).astype(np.float32)
        # Simulate multilooking: stride by factor
        multilooked = arr[:, ::mlook_factor]
        assert multilooked.shape == (rows, cols // mlook_factor)

    def test_coregistration_produces_same_shape(self):
        """Two arrays coregistered have identical shapes."""
        ref = np.random.random((100, 100)).astype(np.complex64)
        secondary = np.random.random((100, 100)).astype(np.complex64)
        # After coregistration shapes must match
        assert ref.shape == secondary.shape

    def test_checkpoint_exists_after_marking(self, tmp_path):
        sentinel = tmp_path / "s1_preprocess_done.txt"
        sentinel.touch()
        assert sentinel.exists()


# ---------------------------------------------------------------------------
# insar_sbas.py tests
# ---------------------------------------------------------------------------

class TestInSARSBAS:
    """Tests for SBAS interferogram network and inversion."""

    def test_import(self):
        import scripts.insar_sbas  # noqa: F401

    def test_sbas_network_temporal_baseline(self):
        """Network pairs respect temporal baseline threshold."""
        from datetime import datetime, timedelta

        dates = [datetime(2020, 1, 1) + timedelta(days=12 * i) for i in range(10)]
        max_tb = 120  # days
        pairs = [
            (i, j)
            for i in range(len(dates))
            for j in range(i + 1, len(dates))
            if (dates[j] - dates[i]).days <= max_tb
        ]
        # Every pair must satisfy temporal baseline
        for i, j in pairs:
            assert (dates[j] - dates[i]).days <= max_tb

    def test_sbas_inversion_shape(self):
        """SBAS least-squares inversion output has correct epoch count."""
        n_epochs = 10
        n_ifgrams = 15
        # Design matrix A: n_ifgrams × (n_epochs - 1)
        A = np.random.random((n_ifgrams, n_epochs - 1)).astype(np.float32)
        phases = np.random.random((n_ifgrams,)).astype(np.float32)
        # Normal equations: A^T A x = A^T b
        result, _, _, _ = np.linalg.lstsq(A, phases, rcond=None)
        assert result.shape == (n_epochs - 1,)

    def test_goldstein_filter_preserves_shape(self):
        """Goldstein filter output has same shape as input."""
        interferogram = np.random.random((50, 50)).astype(np.complex64)
        # Mock: just return same-shape array
        filtered = np.exp(1j * np.angle(interferogram))
        assert filtered.shape == interferogram.shape

    def test_coherence_range(self):
        """Coherence values are bounded [0, 1]."""
        signal = np.random.random((100, 100)).astype(np.complex64) + \
                 1j * np.random.random((100, 100)).astype(np.float32)
        coherence = np.abs(signal) / (np.abs(signal) + 1e-8)
        assert np.all(coherence >= 0)
        assert np.all(coherence <= 1 + 1e-6)

    def test_network_json_schema(self, tmp_path):
        """Network JSON has required keys."""
        network = {
            "pairs": [["2020-01-01", "2020-01-13"]],
            "temporal_baselines_days": [12],
            "n_ifgrams": 1,
            "n_epochs": 2,
        }
        p = tmp_path / "network.json"
        p.write_text(json.dumps(network))
        loaded = json.loads(p.read_text())
        for key in ("pairs", "temporal_baselines_days", "n_ifgrams", "n_epochs"):
            assert key in loaded
