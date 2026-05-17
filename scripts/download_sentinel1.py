#!/usr/bin/env python3
"""
scripts/download_sentinel1.py
==============================
Download Sentinel-1 SLC products from the ASF Vertex API.

Author : Samuel Appiah Kubi
Licence: MIT
"""

from __future__ import annotations

import argparse
import json
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

SENTINEL_FILE = "data/raw/sentinel1/download_done.txt"
MANIFEST_FILE = "data/raw/sentinel1/manifest.json"


def search_scenes(cfg: dict) -> list[dict]:
    """Search ASF Vertex API for matching Sentinel-1 scenes.

    Args:
        cfg: Full project configuration dictionary.

    Returns:
        List of scene metadata dictionaries.
    """
    s1cfg = cfg["sentinel1"]
    bbox = cfg["study_area"]["bbox"]

    try:
        import asf_search as asf

        results = asf.search(
            platform=[asf.PLATFORM.SENTINEL1A, asf.PLATFORM.SENTINEL1B],
            processingLevel=asf.PRODUCT_TYPE.SLC,
            beamMode=asf.BEAMMODE.IW,
            start=s1cfg["start_date"],
            end=s1cfg["end_date"],
            intersectsWith=(
                f"POLYGON(({bbox[0]} {bbox[1]}, {bbox[2]} {bbox[1]}, "
                f"{bbox[2]} {bbox[3]}, {bbox[0]} {bbox[3]}, "
                f"{bbox[0]} {bbox[1]}))"
            ),
            flightDirection=s1cfg["orbit_direction"],
            maxResults=s1cfg["max_results"],
        )
        scenes = []
        for r in results:
            prop = r.properties
            scenes.append(
                {
                    "scene_id": prop.get("sceneName", ""),
                    "date": prop.get("startTime", ""),
                    "orbit_direction": prop.get("flightDirection", ""),
                    "polarisation": prop.get("polarization", ""),
                    "url": prop.get("url", ""),
                    "file_name": prop.get("fileName", ""),
                }
            )
        log.info("Found %d scenes via asf_search", len(scenes))
        return scenes

    except ImportError:
        log.warning(
            "asf_search not installed. Returning synthetic scene list for "
            "demonstration. Install with: pip install asf-search"
        )
        return _synthetic_scenes(s1cfg)


def _synthetic_scenes(s1cfg: dict) -> list[dict]:
    """Generate synthetic scene metadata for pipeline testing.

    Args:
        s1cfg: Sentinel-1 configuration block.

    Returns:
        List of dummy scene dictionaries.
    """
    import datetime

    scenes = []
    start = datetime.date.fromisoformat(s1cfg["start_date"])
    end = datetime.date.fromisoformat(s1cfg["end_date"])
    current = start
    idx = 0
    while current < end and idx < s1cfg["max_results"]:
        scenes.append(
            {
                "scene_id": f"S1A_IW_SLC__1SDV_{current.strftime('%Y%m%d')}T060000",
                "date": current.isoformat(),
                "orbit_direction": s1cfg["orbit_direction"],
                "polarisation": s1cfg["polarization"],
                "url": f"https://asf.alaska.edu/synthetic/{idx:04d}.zip",
                "file_name": f"synthetic_{idx:04d}.zip",
            }
        )
        current += datetime.timedelta(days=12)
        idx += 1
    log.info("Generated %d synthetic scenes for testing", len(scenes))
    return scenes


def download_scenes(
    scenes: list[dict],
    output_dir: Path,
    username: str,
    password: str,
) -> list[str]:
    """Download each scene to output_dir, skipping already-downloaded files.

    Args:
        scenes:     List of scene metadata dicts (from search_scenes).
        output_dir: Local directory to save files.
        username:   ASF Vertex username.
        password:   ASF Vertex password.

    Returns:
        List of paths to successfully downloaded / pre-existing files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []

    try:
        import asf_search as asf

        session = asf.ASFSession()
        session.auth_with_creds(username, password)

        for scene in scenes:
            dest = output_dir / scene["file_name"]
            if dest.exists():
                log.info("Skip (exists): %s", dest.name)
                downloaded.append(str(dest))
                continue
            log.info("Downloading: %s", scene["scene_id"])
            try:
                asf.download_url(
                    url=scene["url"],
                    path=str(output_dir),
                    filename=scene["file_name"],
                    session=session,
                )
                downloaded.append(str(dest))
                log.info("Done: %s", dest.name)
            except Exception as exc:
                log.error("Failed to download %s: %s", scene["scene_id"], exc)

    except ImportError:
        log.warning(
            "asf_search not available. Skipping actual download — "
            "creating placeholder files for pipeline testing."
        )
        for scene in scenes:
            dest = output_dir / scene["file_name"]
            if not dest.exists():
                dest.touch()
                log.debug("Created placeholder: %s", dest.name)
            downloaded.append(str(dest))

    return downloaded


def save_manifest(scenes: list[dict], manifest_path: Path) -> None:
    """Write scene metadata manifest to JSON.

    Args:
        scenes:        Scene list to serialise.
        manifest_path: Output path.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as fh:
        json.dump(
            {
                "n_scenes": len(scenes),
                "scenes": scenes,
            },
            fh,
            indent=2,
        )
    log.info("Manifest saved: %s (%d scenes)", manifest_path, len(scenes))


def run(cfg: dict, force: bool = False) -> None:
    """Execute the Sentinel-1 download pipeline step.

    Args:
        cfg:   Full project configuration.
        force: Re-run even if checkpoint exists.
    """
    sentinel = cfg["sentinel1"].get(
        "checkpoint", SENTINEL_FILE
    )
    if checkpoint_exists(sentinel) and not force:
        log.info("Download already complete. Use --force to re-run.")
        return

    username = os.environ.get("ASF_USERNAME", "")
    password = os.environ.get("ASF_PASSWORD", "")
    if not username or not password:
        log.warning(
            "ASF_USERNAME / ASF_PASSWORD not set. "
            "Set environment variables for real downloads."
        )

    output_dir = Path(cfg["sentinel1"]["download_dir"])
    scenes = search_scenes(cfg)

    if not scenes:
        log.error("No scenes found for the given search parameters.")
        return

    save_manifest(scenes, Path(MANIFEST_FILE))
    download_scenes(scenes, output_dir, username, password)
    mark_completed(sentinel)
    log.info("Sentinel-1 download step complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Sentinel-1 SLC scenes from ASF Vertex."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--force", action="store_true", help="Re-run even if checkpoint exists"
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level, step="download_s1")
    cfg = load_config(args.config)
    run(cfg, force=args.force)
