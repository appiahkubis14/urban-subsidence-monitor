#!/usr/bin/env python3
"""
scripts/edge_inference.py
==========================
Benchmark the quantised model on simulated edge hardware
(single CPU core, limited memory — Raspberry Pi 4 profile).

Author : Samuel Appiah Kubi
Licence: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from scripts.utils import (
    checkpoint_exists,
    load_config,
    mark_completed,
    setup_logging,
)

log = logging.getLogger(__name__)

SENTINEL_FILE = "data/edge/edge_benchmark_done.txt"


def set_edge_constraints(n_threads: int = 1) -> None:
    """Limit CPU threads to simulate edge deployment.

    Args:
        n_threads: Number of CPU threads to allow.
    """
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
    try:
        import torch
        torch.set_num_threads(n_threads)
        log.info("CPU threads limited to %d.", n_threads)
    except ImportError:
        pass


def benchmark_pytorch(
    model_path: Path,
    cfg: dict,
    n_runs: int = 100,
) -> dict:
    """Benchmark PyTorch model inference latency and memory.

    Args:
        model_path: Path to .pt model weights.
        cfg:        Full project configuration.
        n_runs:     Number of timing runs.

    Returns:
        Dict with latency and model size statistics.
    """
    try:
        import torch
        import psutil
        from scripts.train_lightweight import build_tiny_unet

        model = build_tiny_unet(
            in_channels=cfg["model"]["in_channels"],
            enc_channels=cfg["model"]["encoder_channels"],
            dropout=cfg["model"]["dropout_rate"],
        )
        if model_path.exists():
            try:
                state = torch.load(model_path, map_location="cpu")
                model.load_state_dict(state, strict=False)
                log.info("Loaded: %s", model_path.name)
            except Exception as exc:
                log.warning("Could not load weights: %s — using random init.", exc)
        model.eval()

        patch_size = cfg["features"]["patch_size"]
        in_channels = cfg["model"]["in_channels"]
        dummy = torch.randn(1, in_channels, patch_size, patch_size)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                model(dummy)

        # Timed runs
        times = []
        proc = psutil.Process()
        mem_before = proc.memory_info().rss / 1e6

        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                model(dummy)
                times.append((time.perf_counter() - t0) * 1000)

        mem_after = proc.memory_info().rss / 1e6
        model_size_mb = model_path.stat().st_size / 1e6 if model_path.exists() else 0.0

        return {
            "framework": "PyTorch",
            "model_path": str(model_path),
            "model_size_mb": round(model_size_mb, 3),
            "n_runs": n_runs,
            "latency_median_ms": round(float(np.median(times)), 2),
            "latency_mean_ms": round(float(np.mean(times)), 2),
            "latency_p95_ms": round(float(np.percentile(times, 95)), 2),
            "latency_std_ms": round(float(np.std(times)), 2),
            "memory_delta_mb": round(mem_after - mem_before, 2),
            "meets_size_target": model_size_mb <= cfg["edge"]["target_size_mb"],
            "meets_latency_target": float(np.median(times)) <= cfg["edge"]["target_latency_ms"],
        }

    except ImportError as exc:
        log.error("PyTorch / psutil not available: %s", exc)
        return {"framework": "PyTorch", "error": str(exc)}


def benchmark_tflite(
    tflite_path: Path,
    cfg: dict,
    n_runs: int = 100,
) -> dict:
    """Benchmark TFLite model inference.

    Args:
        tflite_path: Path to .tflite model.
        cfg:         Full project configuration.
        n_runs:      Number of timing runs.

    Returns:
        Dict with latency statistics.
    """
    if not tflite_path.exists():
        return {"framework": "TFLite", "error": f"Model not found: {tflite_path}"}

    try:
        import tflite_runtime.interpreter as tflite  # type: ignore

        interp = tflite.Interpreter(
            model_path=str(tflite_path),
            num_threads=cfg["edge"]["cpu_threads"],
        )
        interp.allocate_tensors()
        in_detail = interp.get_input_details()[0]
        out_detail = interp.get_output_details()[0]

        patch_size = cfg["features"]["patch_size"]
        in_channels = cfg["model"]["in_channels"]
        dummy = np.random.randn(1, in_channels, patch_size, patch_size).astype(np.float32)

        times = []
        for _ in range(n_runs):
            interp.set_tensor(in_detail["index"], dummy)
            t0 = time.perf_counter()
            interp.invoke()
            times.append((time.perf_counter() - t0) * 1000)

        model_size_mb = tflite_path.stat().st_size / 1e6
        return {
            "framework": "TFLite",
            "model_path": str(tflite_path),
            "model_size_mb": round(model_size_mb, 3),
            "n_runs": n_runs,
            "latency_median_ms": round(float(np.median(times)), 2),
            "latency_mean_ms": round(float(np.mean(times)), 2),
            "latency_p95_ms": round(float(np.percentile(times, 95)), 2),
            "meets_size_target": model_size_mb <= cfg["edge"]["target_size_mb"],
            "meets_latency_target": float(np.median(times)) <= cfg["edge"]["target_latency_ms"],
        }

    except ImportError:
        log.warning("tflite_runtime not available. Install for TFLite benchmarking.")
        return {"framework": "TFLite", "error": "tflite_runtime not installed"}
    except Exception as exc:
        return {"framework": "TFLite", "error": str(exc)}


def compare_accuracy(
    float_model_path: Path,
    quantized_model_path: Path,
    cfg: dict,
) -> dict:
    """Compare float vs quantised model outputs on sample data.

    Args:
        float_model_path:     Float32 model path.
        quantized_model_path: Quantised model path.
        cfg:                  Full project configuration.

    Returns:
        Dict with MAE, max_error between float and quantised outputs.
    """
    try:
        import torch
        from scripts.train_lightweight import build_tiny_unet
        from scripts.utils import compute_metrics

        def _load(path):
            m = build_tiny_unet(
                in_channels=cfg["model"]["in_channels"],
                enc_channels=cfg["model"]["encoder_channels"],
            )
            if path.exists():
                try:
                    m.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
                except Exception:
                    pass
            m.eval()
            return m

        float_model = _load(float_model_path)
        quant_model = _load(quantized_model_path)

        patch_size = cfg["features"]["patch_size"]
        in_channels = cfg["model"]["in_channels"]
        rng = np.random.default_rng(42)
        samples = [
            torch.from_numpy(rng.uniform(0, 1, (1, in_channels, patch_size, patch_size)).astype(np.float32))
            for _ in range(10)
        ]

        float_preds, quant_preds = [], []
        with torch.no_grad():
            for x in samples:
                float_preds.append(float_model(x).numpy().ravel())
                quant_preds.append(quant_model(x).numpy().ravel())

        fp = np.concatenate(float_preds)
        qp = np.concatenate(quant_preds)
        metrics = compute_metrics(fp, qp)
        log.info(
            "Accuracy comparison — MAE=%.4f  RMSE=%.4f  R²=%.4f",
            metrics["mae"], metrics["rmse"], metrics["r2"],
        )
        return {
            "mae_float_vs_quantized": round(metrics["mae"], 5),
            "rmse_float_vs_quantized": round(metrics["rmse"], 5),
            "max_error": round(float(np.max(np.abs(fp - qp))), 5),
        }

    except Exception as exc:
        log.warning("Accuracy comparison failed: %s", exc)
        return {"error": str(exc)}


def run(cfg: dict, force: bool = False) -> None:
    """Execute edge inference benchmarking step.

    Args:
        cfg:   Full project configuration.
        force: Re-run even if checkpoint exists.
    """
    if checkpoint_exists(SENTINEL_FILE) and not force:
        log.info("Edge benchmark already done. Use --force to re-run.")
        return

    edge_cfg = cfg["edge"]
    output_dir = Path(edge_cfg["output_json"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    set_edge_constraints(n_threads=edge_cfg["cpu_threads"])
    n_runs = edge_cfg["n_benchmark_runs"]

    quant_paths = cfg["quantization"]["paths"]
    float_path = Path(cfg["model"]["paths"]["unquantized"])
    pruned_path = Path(quant_paths["pruned_quantized"])
    tflite_path = Path(quant_paths["tflite_dir"]) / "model.tflite"

    results = {
        "edge_device_profile": "Raspberry Pi 4 (simulated: 1 CPU core)",
        "targets": {
            "max_size_mb": edge_cfg["target_size_mb"],
            "max_latency_ms": edge_cfg["target_latency_ms"],
        },
        "benchmarks": {},
        "accuracy": {},
    }

    log.info("Benchmarking float model ...")
    results["benchmarks"]["float32"] = benchmark_pytorch(float_path, cfg, n_runs)

    log.info("Benchmarking pruned+quantised model ...")
    results["benchmarks"]["pruned_int8"] = benchmark_pytorch(pruned_path, cfg, n_runs)

    log.info("Benchmarking TFLite model ...")
    results["benchmarks"]["tflite"] = benchmark_tflite(tflite_path, cfg, n_runs)

    log.info("Comparing float vs quantised accuracy ...")
    results["accuracy"] = compare_accuracy(float_path, pruned_path, cfg)

    # Summary
    pruned = results["benchmarks"].get("pruned_int8", {})
    float_b = results["benchmarks"].get("float32", {})
    log.info("=" * 50)
    log.info("EDGE BENCHMARK SUMMARY")
    log.info("  Float32 : %.2f MB  %.1f ms",
             float_b.get("model_size_mb", 0), float_b.get("latency_median_ms", 0))
    log.info("  Pruned+INT8: %.2f MB  %.1f ms",
             pruned.get("model_size_mb", 0), pruned.get("latency_median_ms", 0))
    log.info("  Size target (<%.0f MB): %s",
             edge_cfg["target_size_mb"],
             "PASS ✓" if pruned.get("meets_size_target") else "FAIL ✗")
    log.info("  Latency target (<%.0f ms): %s",
             edge_cfg["target_latency_ms"],
             "PASS ✓" if pruned.get("meets_latency_target") else "FAIL ✗")
    log.info("=" * 50)

    output_path = Path(edge_cfg["output_json"])
    with output_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    log.info("Edge benchmark saved: %s", output_path)

    mark_completed(SENTINEL_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge inference benchmark.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level, step="edge")
    cfg = load_config(args.config)
    run(cfg, force=args.force)
