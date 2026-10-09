"""Unit tests for dataset handling, synthetic generation, augmentation, and stratified splits."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.constants import CLASSES, NUM_CLASSES
from src.dataset import (
    BiometricAugmentation,
    FingerprintDataset,
    FingerprintSample,
    compute_class_weights,
    create_sample_dataset,
    generate_synthetic_fingerprint,
    load_samples_from_directory,
    stratified_split,
)


def test_generate_synthetic_fingerprint_classes():
    for c in CLASSES:
        img = generate_synthetic_fingerprint(c, size=64, seed=42)
        assert img.shape == (64, 64)
        assert img.dtype == np.uint8
        assert img.min() >= 0
        assert img.max() <= 255


def test_generate_synthetic_fingerprint_invalid_class():
    with pytest.raises(ValueError):
        generate_synthetic_fingerprint("invalid_class_foo")


def test_create_sample_dataset_folder_structure():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        counts = create_sample_dataset(tmp_path, samples_per_class=5, seed=1)
        assert len(counts) == NUM_CLASSES
        for c in CLASSES:
            assert counts[c] == 5
            class_folder = tmp_path / c
            assert class_folder.exists()
            assert len(list(class_folder.glob("*.png"))) == 5

        # Test loading back from directory
        samples = load_samples_from_directory(tmp_path)
        assert len(samples) == 15


def test_biometric_augmentation():
    aug = BiometricAugmentation(max_rotation_deg=10.0, max_shift_px=4.0)
    tensor_img = torch.rand(1, 64, 64)
    augmented = aug(tensor_img)
    assert augmented.shape == (1, 64, 64)
    assert augmented.dtype == torch.float32
    assert 0.0 <= augmented.min() <= augmented.max() <= 1.0


def test_fingerprint_sample_aliases():
    s1 = FingerprintSample(image_source=np.zeros((32, 32)), label="plain arch")
    assert s1.label == "arch"

    s2 = FingerprintSample(image_source=np.zeros((32, 32)), label="left loop")
    assert s2.label == "loop"

    s3 = FingerprintSample(image_source=np.zeros((32, 32)), label="whirl")
    assert s3.label == "whorl"


def test_fingerprint_dataset_dataloader():
    samples = [
        FingerprintSample(image_source=generate_synthetic_fingerprint("arch", size=64, seed=i), label="arch", group_id=f"g_{i}")
        for i in range(6)
    ]
    ds = FingerprintDataset(samples)
    assert len(ds) == 6

    img, label = ds[0]
    assert img.shape == (1, 128, 128)
    assert label == 0

    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch_imgs, batch_labels = next(iter(loader))
    assert batch_imgs.shape == (2, 1, 128, 128)
    assert batch_labels.shape == (2,)


def test_stratified_split():
    samples = []
    for c in CLASSES:
        for i in range(12):
            samples.append(FingerprintSample(
                image_source=np.zeros((32, 32)),
                label=c,
                group_id=f"{c}_group_{i // 2}",  # 6 groups per class, paired impressions
            ))

    train_s, val_s, test_s = stratified_split(samples, ratios=(0.70, 0.15, 0.15), seed=42)
    assert len(train_s) + len(val_s) + len(test_s) == len(samples)

    # Check each split contains all 3 classes
    train_labels = {s.label for s in train_s}
    val_labels = {s.label for s in val_s}
    test_labels = {s.label for s in test_s}

    assert train_labels == set(CLASSES)
    assert val_labels == set(CLASSES)
    assert test_labels == set(CLASSES)

    # Verify no group ID leakage between splits
    train_groups = {s.group_id for s in train_s}
    val_groups = {s.group_id for s in val_s}
    test_groups = {s.group_id for s in test_s}

    assert len(train_groups & val_groups) == 0
    assert len(train_groups & test_groups) == 0
    assert len(val_groups & test_groups) == 0


def test_compute_class_weights():
    # Imbalanced sample set: 10 arch, 20 loop, 5 whorl
    samples = (
        [FingerprintSample(np.zeros((10, 10)), "arch") for _ in range(10)]
        + [FingerprintSample(np.zeros((10, 10)), "loop") for _ in range(20)]
        + [FingerprintSample(np.zeros((10, 10)), "whorl") for _ in range(5)]
    )
    weights = compute_class_weights(samples)
    assert len(weights) == NUM_CLASSES
    # Least frequent class (whorl) should receive highest weight
    assert weights[2] > weights[0] > weights[1]
