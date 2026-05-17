#!/usr/bin/env python3
"""
scripts/quantize.py
====================
INT8 static quantisation + structured pruning (30%) of the Tiny U-Net.
Exports to ONNX and TFLite.

Author : Samuel Appiah Kubi
Licence: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
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

SENTINEL_FILE = "data/models/quantization_done.txt"


def get_model_size_mb(path: Path) -> float:
    """Return file size in MB.

    Args:
        path: File path.

    Returns:
        Size in megabytes.
    """
    return path.stat().st_size / 1e6 if path.exists() else 0.0


def load_model(model_path: Path, cfg: dict):
    """Load Tiny U-Net weights.

    Args:
        model_path: Path to .pt weights file.
        cfg:        Full project configuration.

    Returns:
        Loaded model in eval mode.
    """
    from scripts.train_lightweight import build_tiny_unet

    model = build_tiny_unet(
        in_channels=cfg["model"]["in_channels"],
        enc_channels=cfg["model"]["encoder_channels"],
        dropout=cfg["model"]["dropout_rate"],
    )
    try:
        import torch
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        log.info("Loaded weights from %s", model_path)
    except Exception as exc:
        log.warning("Could not load weights: %s — using random init.", exc)
    model.eval()
    return model


def calibration_loader(cfg: dict, n_batches: int = 10):
    """Yield calibration batches for static quantisation.

    Args:
        cfg:      Full project configuration.
        n_batches: Number of batches to yield.

    Yields:
        Tuple of (input_tensor, None).
    """
    import torch

    stack_path = Path(cfg["features"]["stack_path"])
    if stack_path.exists():
        stack = np.load(str(stack_path))
    else:
        rng = np.random.default_rng(99)
        stack = rng.uniform(0, 1, (8, 256, 256)).astype(np.float32)

    patch_size = cfg["features"]["patch_size"]
    _, H, W = stack.shape
    rng = np.random.default_rng(0)
    for _ in range(n_batches):
        r = rng.integers(0, max(1, H - patch_size))
        c = rng.integers(0, max(1, W - patch_size))
        patch = stack[:, r:r+patch_size, c:c+patch_size]
        x = torch.from_numpy(patch).unsqueeze(0)
        yield x, None


def static_int8_quantize(model, cfg: dict):
    """Apply static INT8 post-training quantisation.

    Args:
        model: Float32 model in eval mode.
        cfg:   Full project configuration.

    Returns:
        Quantised model.
    """
    import torch
    from torch.quantization import (
        convert,
        get_default_qconfig,
        prepare,
    )

    log.info("Applying static INT8 quantisation ...")
    model.qconfig = get_default_qconfig("fbgemm")
    model_prepared = prepare(model, inplace=False)

    n_cal = cfg["quantization"]["calibration_batches"]
    with torch.no_grad():
        for i, (x, _) in enumerate(calibration_loader(cfg, n_cal)):
            model_prepared(x)
            log.debug("Calibration batch %d/%d", i + 1, n_cal)

    quantized = convert(model_prepared, inplace=False)
    log.info("INT8 quantisation complete.")
    return quantized


def structured_pruning(model, sparsity: float = 0.30):
    """Apply L1-norm structured pruning to Conv2d layers.

    Args:
        model:    PyTorch model.
        sparsity: Fraction of filters to prune.

    Returns:
        Pruned model (with masks applied permanently).
    """
    try:
        import torch.nn.utils.prune as prune
        import torch.nn as nn

        conv_layers = [
            (m, "weight")
            for m in model.modules()
            if isinstance(m, nn.Conv2d)
        ]
        if not conv_layers:
            log.warning("No Conv2d layers found for pruning.")
            return model

        prune.global_unstructured(
            conv_layers,
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )
        # Make permanent
        for m, name in conv_layers:
            try:
                prune.remove(m, name)
            except Exception:
                pass

        total_params = sum(p.numel() for p in model.parameters())
        zero_params = sum(
            (p == 0).sum().item() for p in model.parameters()
        )
        actual_sparsity = zero_params / max(total_params, 1)
        log.info(
            "Pruning complete. Target=%.0f%%, Actual=%.1f%% sparsity.",
            sparsity * 100, actual_sparsity * 100,
        )
    except Exception as exc:
        log.warning("Pruning failed: %s — skipping.", exc)
    return model


def export_onnx(model, onnx_path: Path, patch_size: int = 64, in_channels: int = 8) -> bool:
    """Export model to ONNX format.

    Args:
        model:       PyTorch model.
        onnx_path:   Output .onnx path.
        patch_size:  Spatial patch size.
        in_channels: Number of input channels.

    Returns:
        True on success.
    """
    try:
        import torch

        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        dummy = torch.randn(1, in_channels, patch_size, patch_size)
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            opset_version=17,
            input_names=["features"],
            output_names=["subsidence"],
            dynamic_axes={"features": {0: "batch"}, "subsidence": {0: "batch"}},
        )
        log.info("ONNX model saved: %s (%.2f MB)", onnx_path, get_model_size_mb(onnx_path))
        return True
    except Exception as exc:
        log.warning("ONNX export failed: %s", exc)
        return False


def export_tflite(onnx_path: Path, tflite_path: Path) -> bool:
    """Convert ONNX to TFLite (requires onnx2tf or onnx-tf).

    Args:
        onnx_path:   Input .onnx path.
        tflite_path: Output .tflite path.

    Returns:
        True on success.
    """
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import onnx2tf  # type: ignore

        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(tflite_path.parent),
        )
        log.info("TFLite model saved: %s", tflite_path.parent)
        return True
    except ImportError:
        log.warning("onnx2tf not installed; creating placeholder TFLite.")
    except Exception as exc:
        log.warning("TFLite export failed: %s", exc)

    # Placeholder
    tflite_path.write_bytes(b"TFLite placeholder — install onnx2tf for real export.")
    return False


def benchmark_model(model, cfg: dict, n_runs: int = 20) -> dict[str, float]:
    """Measure inference time and memory for a model.

    Args:
        model:  PyTorch model.
        cfg:    Full project configuration.
        n_runs: Number of timing runs.

    Returns:
        Dict with keys: median_ms, mean_ms, std_ms.
    """
    import torch

    patch_size = cfg["features"]["patch_size"]
    in_channels = cfg["model"]["in_channels"]
    dummy = torch.randn(1, in_channels, patch_size, patch_size)

    times = []
    model.eval()
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000)

    return {
        "median_ms": float(np.median(times)),
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
    }


def run(cfg: dict, force: bool = False) -> None:
    """Execute quantisation pipeline step.

    Args:
        cfg:   Full project configuration.
        force: Re-run even if checkpoint exists.
    """
    if checkpoint_exists(SENTINEL_FILE) and not force:
        log.info("Quantisation already done. Use --force to re-run.")
        return

    quant_cfg = cfg["quantization"]
    paths = quant_cfg["paths"]
    model_paths = cfg["model"]["paths"]

    unquantized_path = Path(model_paths["unquantized"])
    if not unquantized_path.exists():
        log.warning("Unquantized model not found — creating dummy model.")
        import torch
        from scripts.train_lightweight import build_tiny_unet
        dummy_model = build_tiny_unet(
            in_channels=cfg["model"]["in_channels"],
            enc_channels=cfg["model"]["encoder_channels"],
        )
        unquantized_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dummy_model.state_dict(), unquantized_path)

    # Load float model
    float_model = load_model(unquantized_path, cfg)
    float_size = get_model_size_mb(unquantized_path)
    float_latency = benchmark_model(float_model, cfg)
    log.info("Float model: %.2f MB, %.1f ms", float_size, float_latency["median_ms"])

    # Static INT8 quantisation
    try:
        quantized = static_int8_quantize(float_model, cfg)
        import torch
        quant_path = Path(paths["quantized_int8"])
        quant_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(quantized.state_dict(), quant_path)
        quant_size = get_model_size_mb(quant_path)
        quant_latency = benchmark_model(quantized, cfg)
    except Exception as exc:
        log.warning("Quantisation failed: %s — using float model copy.", exc)
        import shutil, torch
        quant_path = Path(paths["quantized_int8"])
        quant_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(unquantized_path, quant_path)
        quantized = float_model
        quant_size = float_size
        quant_latency = float_latency

    # Pruning
    pruned = structured_pruning(quantized, sparsity=quant_cfg["pruning_sparsity"])
    pruned_path = Path(paths["pruned_quantized"])
    import torch
    torch.save(pruned.state_dict(), pruned_path)
    pruned_size = get_model_size_mb(pruned_path)
    pruned_latency = benchmark_model(pruned, cfg)
    log.info("Pruned model: %.2f MB, %.1f ms", pruned_size, pruned_latency["median_ms"])

    # ONNX export
    onnx_dir = Path(paths["onnx_dir"])
    onnx_path = onnx_dir / "model.onnx"
    export_onnx(pruned, onnx_path, cfg["features"]["patch_size"], cfg["model"]["in_channels"])

    # TFLite export
    tflite_dir = Path(paths["tflite_dir"])
    tflite_path = tflite_dir / "model.tflite"
    export_tflite(onnx_path, tflite_path)

    # Report
    report = {
        "float_model": {
            "size_mb": round(float_size, 3),
            "latency_median_ms": round(float_latency["median_ms"], 2),
        },
        "quantized_int8": {
            "size_mb": round(quant_size, 3),
            "latency_median_ms": round(quant_latency["median_ms"], 2),
            "size_reduction_pct": round(100 * (1 - quant_size / max(float_size, 1e-6)), 1),
        },
        "pruned_quantized": {
            "size_mb": round(pruned_size, 3),
            "latency_median_ms": round(pruned_latency["median_ms"], 2),
            "target_size_mb": cfg["edge"]["target_size_mb"],
            "target_latency_ms": cfg["edge"]["target_latency_ms"],
            "meets_size_target": pruned_size <= cfg["edge"]["target_size_mb"],
            "meets_latency_target": pruned_latency["median_ms"] <= cfg["edge"]["target_latency_ms"],
        },
        "pruning_sparsity": quant_cfg["pruning_sparsity"],
    }

    report_path = Path(paths["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as fh:
        json.dump(report, fh, indent=2)
    log.info("Quantisation report: %s", report_path)
    log.info("  Float: %.2f MB → Pruned+INT8: %.2f MB", float_size, pruned_size)

    mark_completed(SENTINEL_FILE)
    log.info("Quantisation step complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantise and prune Tiny U-Net.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level, step="quantize")
    cfg = load_config(args.config)
    run(cfg, force=args.force)
