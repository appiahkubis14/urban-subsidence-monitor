#!/usr/bin/env python3
"""
tests/test_model.py
===================
Unit tests for train_lightweight.py and quantize.py.

Run with:
    pytest tests/test_model.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_unet():
    """Instantiate a minimal TinyUNet for testing."""
    from scripts.train_lightweight import TinyUNet
    return TinyUNet(in_channels=8, base_filters=16)


@pytest.fixture
def dummy_batch():
    """Random batch: (B, C, H, W) = (2, 8, 64, 64)."""
    return torch.randn(2, 8, 64, 64)


# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------

class TestTinyUNet:
    """Structural and forward-pass tests for TinyUNet."""

    def test_import(self):
        from scripts.train_lightweight import TinyUNet  # noqa: F401

    def test_output_shape_matches_input(self, tiny_unet, dummy_batch):
        """Output spatial dims match input spatial dims."""
        tiny_unet.eval()
        with torch.no_grad():
            out = tiny_unet(dummy_batch)
        assert out.shape[-2:] == dummy_batch.shape[-2:]

    def test_output_channels_is_one(self, tiny_unet, dummy_batch):
        """Output has a single regression channel."""
        tiny_unet.eval()
        with torch.no_grad():
            out = tiny_unet(dummy_batch)
        assert out.shape[1] == 1

    def test_mc_dropout_produces_variance(self, tiny_unet, dummy_batch):
        """Monte Carlo dropout gives different outputs per pass."""
        tiny_unet.train()  # keep dropout active
        outputs = []
        for _ in range(5):
            with torch.no_grad():
                outputs.append(tiny_unet(dummy_batch))
        stacked = torch.stack(outputs)
        # Variance should be non-zero due to dropout
        assert stacked.std() > 0

    def test_parameter_count_is_small(self, tiny_unet):
        """Model has fewer than 5 million parameters (lightweight target)."""
        n_params = sum(p.numel() for p in tiny_unet.parameters())
        assert n_params < 5_000_000, f"Model has {n_params:,} params — too large for edge"

    def test_gradient_flows(self, tiny_unet, dummy_batch):
        """Loss backward pass produces non-zero gradients."""
        tiny_unet.train()
        out = tiny_unet(dummy_batch)
        target = torch.randn_like(out)
        loss = torch.nn.functional.huber_loss(out, target)
        loss.backward()
        grad_norms = [
            p.grad.norm().item()
            for p in tiny_unet.parameters()
            if p.grad is not None
        ]
        assert any(g > 0 for g in grad_norms)


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

class TestLossFunctions:
    """Tests for combined Huber + spatial smoothness loss."""

    def test_import(self):
        from scripts.train_lightweight import CombinedLoss  # noqa: F401

    def test_combined_loss_is_positive(self):
        from scripts.train_lightweight import CombinedLoss
        loss_fn = CombinedLoss(huber_weight=0.7, smooth_weight=0.3)
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randn(2, 1, 32, 32)
        loss = loss_fn(pred, target)
        assert loss.item() > 0

    def test_perfect_prediction_gives_low_loss(self):
        from scripts.train_lightweight import CombinedLoss
        loss_fn = CombinedLoss(huber_weight=0.7, smooth_weight=0.3)
        pred = torch.zeros(2, 1, 32, 32)
        target = torch.zeros(2, 1, 32, 32)
        loss = loss_fn(pred, target)
        assert loss.item() < 0.01


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------

class TestCheckpointing:
    """Tests for model save/load checkpointing."""

    def test_model_save_load(self, tiny_unet, tmp_path):
        """Saved model can be reloaded and produces identical output."""
        dummy_input = torch.randn(1, 8, 32, 32)
        tiny_unet.eval()
        with torch.no_grad():
            original_out = tiny_unet(dummy_input)

        ckpt_path = tmp_path / "model.pt"
        torch.save(tiny_unet.state_dict(), ckpt_path)

        from scripts.train_lightweight import TinyUNet
        loaded = TinyUNet(in_channels=8, base_filters=16)
        loaded.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        loaded.eval()
        with torch.no_grad():
            loaded_out = loaded(dummy_input)

        assert torch.allclose(original_out, loaded_out, atol=1e-5)
