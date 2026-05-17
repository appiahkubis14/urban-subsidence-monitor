#!/usr/bin/env python3
"""
main.py
========
Pipeline orchestrator for Urban Subsidence Watcher.

Usage:
    python main.py --step all
    python main.py --step download
    python main.py --step preprocess
    python main.py --step insar
    python main.py --step rates
    python main.py --step features
    python main.py --step train
    python main.py --step quantize
    python main.py --step forecast
    python main.py --step validate
    python main.py --step dashboard
    python main.py --step export
    python main.py --step edge

Author : Samuel Appiah Kubi
Licence: MIT
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from scripts.utils import load_config, setup_logging

log = logging.getLogger(__name__)

STEPS = [
    "download",
    "preprocess",
    "insar",
    "rates",
    "features",
    "train",
    "quantize",
    "forecast",
    "validate",
    "dashboard",
    "export",
    "edge",
]

STEP_DESCRIPTIONS = {
    "download":   "Download Sentinel-1 SLC + DEM",
    "preprocess": "Preprocess Sentinel-1 (orbit, deburst, multilook, coregister)",
    "insar":      "SBAS InSAR processing (interferograms, unwrapping, inversion)",
    "rates":      "Estimate subsidence rates (Huber regression, seasonal model)",
    "features":   "Build 8-channel ML feature stack",
    "train":      "Train Tiny U-Net (Huber + smoothness loss, MC Dropout)",
    "quantize":   "INT8 quantisation + 30% structured pruning → ONNX + TFLite",
    "forecast":   "Predict subsidence at 1, 3, 5-year horizons + uncertainty",
    "validate":   "Validate vs GPS, spatial block CV, temporal hold-out",
    "dashboard":  "Build interactive Folium dashboard",
    "export":     "Export COG, GeoJSON, CSV, STAC catalog, PDF report",
    "edge":       "Benchmark on simulated Raspberry Pi 4 edge profile",
}


def _print_banner() -> None:
    """Print ASCII banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║         URBAN SUBSIDENCE WATCHER  v1.0.0                        ║
║  Self-Modelling Digital Twin · Sentinel-1 InSAR · Edge AI       ║
║  Author : Samuel Appiah Kubi                                     ║
║  Programme: Copernicus Master's in Digital Earth (EMJMD)         ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_step(step: str, cfg: dict, force: bool, resume: bool) -> bool:
    """Execute a single pipeline step.

    Args:
        step:   Step identifier string.
        cfg:    Loaded configuration dict.
        force:  Force re-run flag.
        resume: Resume training flag.

    Returns:
        True on success, False on failure.
    """
    t0 = time.time()
    log.info("▶  Starting step: %s — %s", step.upper(), STEP_DESCRIPTIONS.get(step, ""))

    try:
        if step == "download":
            from scripts.download_sentinel1 import run as run_s1
            from scripts.download_dem import run as run_dem
            run_s1(cfg, force=force)
            run_dem(cfg, force=force)

        elif step == "preprocess":
            from scripts.preprocess_s1 import run
            run(cfg, force=force)

        elif step == "insar":
            from scripts.insar_sbas import run
            run(cfg, force=force)

        elif step == "rates":
            from scripts.subsidence_rates import run
            run(cfg, force=force)

        elif step == "features":
            from scripts.build_features import run
            run(cfg, force=force)

        elif step == "train":
            from scripts.train_lightweight import run
            run(cfg, force=force, resume=resume)

        elif step == "quantize":
            from scripts.quantize import run
            run(cfg, force=force)

        elif step == "forecast":
            from scripts.forecast import run
            run(cfg, force=force)

        elif step == "validate":
            from scripts.validate import run
            run(cfg, force=force)

        elif step == "dashboard":
            from scripts.dashboard import run
            run(cfg)

        elif step == "export":
            from scripts.export import run
            run(cfg, force=force)

        elif step == "edge":
            from scripts.edge_inference import run
            run(cfg, force=force)

        else:
            log.error("Unknown step: %s", step)
            return False

        elapsed = time.time() - t0
        log.info("✔  Step %s complete in %.1f s", step.upper(), elapsed)
        return True

    except Exception as exc:
        log.error("✘  Step %s FAILED: %s", step.upper(), exc, exc_info=True)
        return False


def main() -> None:
    """Main entry point for pipeline orchestration."""
    _print_banner()

    parser = argparse.ArgumentParser(
        description="Urban Subsidence Watcher — pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {s:<12} {d}" for s, d in STEP_DESCRIPTIONS.items()
        ),
    )
    parser.add_argument(
        "--step",
        choices=STEPS + ["all"],
        default="all",
        help="Pipeline step to execute (default: all).",
    )
    parser.add_argument("--config", default="config.yaml", help="Config YAML path.")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run step even if checkpoint exists.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint (train step only).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()

    setup_logging(
        level=args.log_level,
        step="main",
        log_file=f"data/logs/pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log",
    )

    # Ensure data directories exist
    for subdir in [
        "data/raw/sentinel1", "data/raw/dem", "data/processed",
        "data/insar", "data/subsidence", "data/features",
        "data/models", "data/forecasts", "data/validation",
        "data/dashboard", "data/exports", "data/edge", "data/logs",
    ]:
        Path(subdir).mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    log.info("Project: %s v%s", cfg["project"]["name"], cfg["project"]["version"])
    log.info("Study area: %s, %s", cfg["study_area"]["name"], cfg["study_area"]["country"])

    pipeline_start = time.time()
    steps_to_run = STEPS if args.step == "all" else [args.step]
    results = {}

    for step in steps_to_run:
        success = run_step(step, cfg, force=args.force, resume=args.resume)
        results[step] = "PASS" if success else "FAIL"

    elapsed_total = time.time() - pipeline_start

    # Summary
    log.info("")
    log.info("═" * 56)
    log.info("PIPELINE SUMMARY  (total: %.1f s)", elapsed_total)
    log.info("═" * 56)
    for step, status in results.items():
        icon = "✔" if status == "PASS" else "✘"
        log.info("  %s  %-12s  %s", icon, step, status)
    log.info("═" * 56)

    n_fail = sum(1 for v in results.values() if v == "FAIL")
    if n_fail > 0:
        log.warning("%d step(s) failed. Check logs above.", n_fail)
        sys.exit(1)
    else:
        log.info("All steps complete. Digital twin ready.")
        log.info("Dashboard: %s", cfg["dashboard"]["output_html"])


if __name__ == "__main__":
    main()
