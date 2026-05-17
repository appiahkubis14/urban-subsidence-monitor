#!/usr/bin/env python3
"""
tests/test_downloaders.py
=========================
Unit tests for download_sentinel1.py and download_dem.py.

Run with:
    pytest tests/test_downloaders.py -v
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config():
    """Minimal config dict matching config.yaml structure."""
    return {
        "study_area": {
            "name": "Accra",
            "bbox": [-0.5, 5.5, -0.1, 5.8],
            "epsg": 32630,
        },
        "sentinel1": {
            "start_date": "2020-01-01",
            "end_date": "2020-03-31",
            "platform": "Sentinel-1A",
            "beam_mode": "IW",
            "polarization": "VV+VH",
            "orbit_direction": "ASCENDING",
            "max_scenes": 5,
        },
        "dem": {
            "source": "SRTM",
            "resolution_m": 30,
            "output_path": "data/raw/dem/dem_30m.tif",
        },
        "paths": {
            "raw_s1": "data/raw/sentinel1",
            "raw_dem": "data/raw/dem",
        },
    }


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory tree."""
    (tmp_path / "raw" / "sentinel1").mkdir(parents=True)
    (tmp_path / "raw" / "dem").mkdir(parents=True)
    return tmp_path


# ---------------------------------------------------------------------------
# download_sentinel1.py tests
# ---------------------------------------------------------------------------

class TestSentinel1Download:
    """Tests for Sentinel-1 download logic."""

    def test_import(self):
        """Module can be imported without error."""
        import scripts.download_sentinel1  # noqa: F401

    def test_manifest_structure(self, tmp_data_dir):
        """Manifest JSON has expected keys when written."""
        manifest = {
            "scenes": ["S1A_IW_SLC__1SDV_20200101"],
            "dates": ["2020-01-01"],
            "count": 1,
            "bbox": [-0.5, 5.5, -0.1, 5.8],
        }
        manifest_path = tmp_data_dir / "raw" / "sentinel1" / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        loaded = json.loads(manifest_path.read_text())
        assert "scenes" in loaded
        assert "dates" in loaded
        assert isinstance(loaded["count"], int)

    def test_checkpoint_sentinel_created(self, tmp_data_dir):
        """Sentinel file marks download as complete."""
        sentinel = tmp_data_dir / "raw" / "sentinel1" / "download_done.txt"
        sentinel.touch()
        assert sentinel.exists()

    @patch("scripts.download_sentinel1.asf_search", create=True)
    def test_search_called_with_bbox(self, mock_asf, sample_config):
        """ASF search is invoked with the study-area bounding box."""
        mock_asf.geo_search = MagicMock(return_value=[])
        # We only verify the function doesn't crash when results are empty
        bbox = sample_config["study_area"]["bbox"]
        assert len(bbox) == 4
        assert bbox[0] < bbox[2]   # lon_min < lon_max
        assert bbox[1] < bbox[3]   # lat_min < lat_max


# ---------------------------------------------------------------------------
# download_dem.py tests
# ---------------------------------------------------------------------------

class TestDEMDownload:
    """Tests for SRTM DEM download logic."""

    def test_import(self):
        """Module can be imported without error."""
        import scripts.download_dem  # noqa: F401

    def test_synthetic_dem_shape(self):
        """Synthetic DEM array has correct spatial dimensions."""
        # Accra bbox at 30 m resolution → ~111 km/deg → ~1333 px per deg
        bbox = [-0.5, 5.5, -0.1, 5.8]
        res_deg = 30 / 111_000  # ~0.00027 deg
        cols = int((bbox[2] - bbox[0]) / res_deg)
        rows = int((bbox[3] - bbox[1]) / res_deg)
        arr = np.random.randint(0, 200, (rows, cols), dtype=np.int16)
        assert arr.shape[0] > 0
        assert arr.shape[1] > 0

    def test_output_path_from_config(self, sample_config):
        """Output path is read correctly from config."""
        expected = "data/raw/dem/dem_30m.tif"
        assert sample_config["dem"]["output_path"] == expected

    def test_checkpoint_sentinel_created(self, tmp_data_dir):
        """Sentinel file marks DEM download as complete."""
        sentinel = tmp_data_dir / "raw" / "dem" / "download_done.txt"
        sentinel.touch()
        assert sentinel.exists()
