"""Unit tests for the FingerprintCNN architecture (Slide 8)."""

import tempfile
from pathlib import Path

import pytest
import torch

from src.constants import NUM_CLASSES
from src.model import ConvBlock, FingerprintCNN


def test_conv_block():
    block = ConvBlock(in_channels=1, out_channels=16)
    x = torch.rand(2, 1, 32, 32)
    out = block(x)
    assert out.shape == (2, 16, 32, 32)


def test_model_forward_pass_shape():
    model = FingerprintCNN(num_classes=NUM_CLASSES)
    x = torch.rand(4, 1, 128, 128)
    logits = model(x)
    assert logits.shape == (4, NUM_CLASSES)


def test_model_predict_probabilities_softmax():
    model = FingerprintCNN(num_classes=NUM_CLASSES)
    x = torch.rand(2, 1, 128, 128)
    probs = model.predict_probabilities(x)
    assert probs.shape == (2, NUM_CLASSES)
    # Probabilities must sum to 1.0 per sample
    sums = probs.sum(dim=1).numpy()
    assert pytest.approx(sums[0], rel=1e-5) == 1.0
    assert pytest.approx(sums[1], rel=1e-5) == 1.0


def test_model_parameter_count():
    model = FingerprintCNN()
    params = model.count_parameters()
    assert "total_parameters" in params
    assert "trainable_parameters" in params
    assert params["trainable_parameters"] > 100_000


def test_model_save_and_load_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_model.pth"
        model = FingerprintCNN(num_classes=NUM_CLASSES)
        model.eval()

        # Forward pass on random tensor
        x = torch.rand(1, 1, 128, 128)
        with torch.no_grad():
            logits1 = model(x)

        model.save_checkpoint(ckpt_path, metadata={"epoch": 5, "accuracy": 0.95})
        assert ckpt_path.exists()

        loaded_model = FingerprintCNN.load_checkpoint(ckpt_path)
        with torch.no_grad():
            logits2 = loaded_model(x)

        # Loaded model should produce identical logits
        torch.testing.assert_close(logits1, logits2)


def test_model_gradient_flow():
    model = FingerprintCNN(num_classes=NUM_CLASSES)
    model.train()
    x = torch.rand(2, 1, 128, 128)
    y = torch.tensor([0, 1], dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()

    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    # Check gradients exist in first and last conv layers
    assert model.stage1_block1.conv.weight.grad is not None
    assert model.fc2.weight.grad is not None
