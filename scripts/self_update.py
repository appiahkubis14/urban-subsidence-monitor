#!/usr/bin/env python3
"""
scripts/self_update.py
=======================
Stretch Goal: Automatic self-updating digital twin.
Checks ASF API for new Sentinel-1 acquisitions, downloads, reprocesses,
retrains model, and refreshes the dashboard.

Author : Samuel Appiah Kubi
Licence: MIT
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.utils import load_config, setup_logging

log = logging.getLogger(__name__)

LOG_FILE = "data/self_update.log"


def check_new_acquisitions(cfg: dict) -> list[dict]:
    """Check ASF API for new Sentinel-1 scenes since last run.

    Args:
        cfg: Full project configuration.

    Returns:
        List of new scene metadata dicts (empty if none found).
    """
    manifest_path = Path("data/raw/sentinel1/manifest.json")
    last_date = cfg["sentinel1"]["start_date"]

    if manifest_path.exists():
        import json
        with manifest_path.open() as fh:
            manifest = json.load(fh)
        scenes = manifest.get("scenes", [])
        if scenes:
            dates = [s["date"][:10] for s in scenes if "date" in s]
            if dates:
                last_date = max(dates)

    log.info("Checking for new Sentinel-1 acquisitions since %s ...", last_date)

    try:
        import asf_search as asf
        from datetime import date, timedelta

        today = date.today().isoformat()
        bbox = cfg["study_area"]["bbox"]
        results = asf.search(
            platform=[asf.PLATFORM.SENTINEL1A, asf.PLATFORM.SENTINEL1B],
            processingLevel=asf.PRODUCT_TYPE.SLC,
            beamMode=asf.BEAMMODE.IW,
            start=last_date,
            end=today,
            intersectsWith=(
                f"POLYGON(({bbox[0]} {bbox[1]}, {bbox[2]} {bbox[1]}, "
                f"{bbox[2]} {bbox[3]}, {bbox[0]} {bbox[3]}, "
                f"{bbox[0]} {bbox[1]}))"
            ),
            maxResults=10,
        )
        new_scenes = []
        for r in results:
            prop = r.properties
            if prop.get("startTime", "")[:10] > last_date:
                new_scenes.append({
                    "scene_id": prop.get("sceneName", ""),
                    "date": prop.get("startTime", "")[:10],
                    "url": prop.get("url", ""),
                    "file_name": prop.get("fileName", ""),
                })
        log.info("Found %d new acquisitions.", len(new_scenes))
        return new_scenes

    except ImportError:
        log.warning("asf_search not available. Simulating no new scenes.")
        return []
    except Exception as exc:
        log.error("ASF check failed: %s", exc)
        return []


def update_pipeline(cfg: dict, new_scenes: list[dict]) -> None:
    """Run incremental pipeline update for new scenes.

    Args:
        cfg:        Full project configuration.
        new_scenes: List of new scene metadata dicts.
    """
    log.info("Starting incremental pipeline update for %d new scene(s).", len(new_scenes))

    # Download
    from scripts.download_sentinel1 import download_scenes
    import os
    download_dir = Path(cfg["sentinel1"]["download_dir"])
    username = os.environ.get("ASF_USERNAME", "")
    password = os.environ.get("ASF_PASSWORD", "")
    download_scenes(new_scenes, download_dir, username, password)

    # Remove old checkpoints to force re-run
    checkpoints = [
        "data/processed/s1_preprocess_done.txt",
        "data/insar/sbas_done.txt",
        "data/subsidence/subsidence_rates_done.txt",
        "data/features/stack_done.txt",
        "data/models/training_done.txt",
        "data/forecasts/forecast_done.txt",
        "data/exports/export_done.txt",
    ]
    for cp in checkpoints:
        p = Path(cp)
        if p.exists():
            p.unlink()
            log.debug("Removed checkpoint: %s", cp)

    # Re-run pipeline stages
    stages = [
        ("preprocess", lambda: __import__(
            "scripts.preprocess_s1", fromlist=["run"]
        ).run(cfg, force=True)),
        ("insar", lambda: __import__(
            "scripts.insar_sbas", fromlist=["run"]
        ).run(cfg, force=True)),
        ("rates", lambda: __import__(
            "scripts.subsidence_rates", fromlist=["run"]
        ).run(cfg, force=True)),
        ("features", lambda: __import__(
            "scripts.build_features", fromlist=["run"]
        ).run(cfg, force=True)),
        ("train", lambda: __import__(
            "scripts.train_lightweight", fromlist=["run"]
        ).run(cfg, force=True, resume=True)),
        ("forecast", lambda: __import__(
            "scripts.forecast", fromlist=["run"]
        ).run(cfg, force=True)),
        ("dashboard", lambda: __import__(
            "scripts.dashboard", fromlist=["run"]
        ).run(cfg)),
        ("export", lambda: __import__(
            "scripts.export", fromlist=["run"]
        ).run(cfg, force=True)),
    ]

    for name, fn in stages:
        log.info("Running stage: %s", name)
        try:
            fn()
        except Exception as exc:
            log.error("Stage %s failed: %s", name, exc)


def log_update_event(message: str) -> None:
    """Append a timestamped message to the self-update log.

    Args:
        message: Log message string.
    """
    log_path = Path(LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with log_path.open("a") as fh:
        fh.write(f"{ts} | {message}\n")


def run_once(cfg: dict) -> None:
    """Check for new data and update if available (one-shot).

    Args:
        cfg: Full project configuration.
    """
    log_update_event("Self-update check started.")
    new_scenes = check_new_acquisitions(cfg)

    if new_scenes:
        log.info("New acquisitions found — starting pipeline update.")
        log_update_event(f"New scenes found: {[s['scene_id'] for s in new_scenes]}")
        update_pipeline(cfg, new_scenes)
        log_update_event("Pipeline update complete.")
    else:
        log.info("No new acquisitions. Digital twin is up to date.")
        log_update_event("No new acquisitions found.")


def run_daemon(cfg: dict) -> None:
    """Run self-update check on a schedule (daemon mode).

    Args:
        cfg: Full project configuration.
    """
    interval_s = cfg["self_update"]["check_interval_hours"] * 3600
    log.info(
        "Self-update daemon started. Check interval: %d hours.",
        cfg["self_update"]["check_interval_hours"],
    )
    while True:
        try:
            run_once(cfg)
        except Exception as exc:
            log.error("Self-update error: %s", exc)
            log_update_event(f"ERROR: {exc}")
        log.info("Sleeping %.0f hours until next check ...", interval_s / 3600)
        time.sleep(interval_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Self-updating digital twin — check for new Sentinel-1 data."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--check", action="store_true",
        help="Run once (check + update if new data found).",
    )
    parser.add_argument(
        "--daemon", action="store_true",
        help="Run on a schedule (blocking daemon).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level, step="self_update", log_file=LOG_FILE)
    cfg = load_config(args.config)

    if args.daemon:
        run_daemon(cfg)
    else:
        run_once(cfg)
