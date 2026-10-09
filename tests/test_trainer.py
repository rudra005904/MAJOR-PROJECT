"""Unit tests for the FingerprintTrainer (Phase 3 & 4)."""

import tempfile
from pathlib import Path

import pytest
import torch

from src.dataset import FingerprintSample, generate_synthetic_fingerprint
from src.model import FingerprintCNN
from src.trainer import FingerprintTrainer


@pytest.fixture
def mini_train_val_samples():
    train_samples = [
        FingerprintSample(generate_synthetic_fingerprint("arch", size=64, seed=10 + i), "arch")
        for i in range(4)
    ] + [
        FingerprintSample(generate_synthetic_fingerprint("loop", size=64, seed=20 + i), "loop")
        for i in range(4)
    ] + [
        FingerprintSample(generate_synthetic_fingerprint("whorl", size=64, seed=30 + i), "whorl")
        for i in range(4)
    ]

    val_samples = [
        FingerprintSample(generate_synthetic_fingerprint("arch", size=64, seed=100 + i), "arch")
        for i in range(2)
    ] + [
        FingerprintSample(generate_synthetic_fingerprint("loop", size=64, seed=200 + i), "loop")
        for i in range(2)
    ] + [
        FingerprintSample(generate_synthetic_fingerprint("whorl", size=64, seed=300 + i), "whorl")
        for i in range(2)
    ]
    return train_samples, val_samples


def test_trainer_fit_and_checkpoint(mini_train_val_samples):
    train_s, val_s = mini_train_val_samples
    model = FingerprintCNN()
    trainer = FingerprintTrainer(model=model, learning_rate=0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "best.pth"
        result = trainer.fit(
            train_samples=train_s,
            val_samples=val_s,
            epochs=2,
            batch_size=4,
            patience=2,
            save_best_path=save_path,
            verbose=False,
        )

        assert result["epochs_run"] == 2
        assert "history" in result
        assert len(result["history"]["train_loss"]) == 2
        assert save_path.exists()


def test_trainer_evaluate_test_set(mini_train_val_samples):
    _, val_s = mini_train_val_samples
    model = FingerprintCNN()
    trainer = FingerprintTrainer(model=model)

    report = trainer.evaluate_test_set(val_s, batch_size=4)
    assert report.num_samples == len(val_s)
    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.f1_macro <= 1.0
