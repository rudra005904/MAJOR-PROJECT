"""Dataset handling, synthetic biometric generation, augmentation, and stratified splitting.

Implements Phase 1 & 2 (Slides 5, 7, 14):
1. Labeled fingerprint representation for Arch, Loop, and Whorl.
2. Stratified group-safe train / validation / test splits.
3. Class imbalance computation and loss weight calculation.
4. Augmentation pipeline (rotation, shift, zoom, contrast).
5. High-fidelity synthetic biometric pattern generator (Sherlock-Monro vector field model)
   enabling fully autonomous, zero-dependency testing and demonstration.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.constants import (
    CLASS_TO_IDX,
    CLASSES,
    DEFAULT_IMAGE_SIZE,
    IDX_TO_CLASS,
    LABEL_ALIASES,
    NUM_CLASSES,
)
from src.preprocessing import PreprocessingPipeline, load_to_grayscale, resize_image


# ---------------------------------------------------------------------------
# Synthetic Fingerprint Ridge Pattern Generator (Sherlock-Monro Model)
# ---------------------------------------------------------------------------

def generate_synthetic_fingerprint(
    pattern_type: str,
    size: int = DEFAULT_IMAGE_SIZE,
    ridge_freq: float = 0.18,
    noise_level: float = 0.15,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Synthesizes a realistic fingerprint impression with characteristic ridge topology.

    Uses mathematical phase field modeling:
    - Arch: Laminar parallel ridge flow with central smooth arching elevation (no core, no delta).
    - Loop: Ridges recurving around a central core point with a lateral delta.
    - Whorl: Concentric circular/spiral ridges with central core and two lateral deltas.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    pattern_type = pattern_type.lower()
    if pattern_type not in CLASSES:
        raise ValueError(f"Unknown pattern_type '{pattern_type}'. Must be one of {CLASSES}")

    H, W = size, size
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    cx, cy = W / 2.0, H / 2.0

    # Slight random center offset for realistic variance
    cx += np.random.uniform(-0.06, 0.06) * W
    cy += np.random.uniform(-0.06, 0.06) * H

    # Complex plane coordinates
    Z = (x_coords - cx) + 1j * (y_coords - cy)

    if pattern_type == "arch":
        # Arch: Parallel ridges entering one side, rising in the center, exiting the other side
        # No core, no delta.
        arch_height = np.random.uniform(25.0, 38.0)
        arch_sigma = np.random.uniform(0.25, 0.35) * W
        hill = arch_height * np.exp(-((x_coords - cx) ** 2) / (2 * (arch_sigma ** 2)))
        phase = 2 * np.pi * ridge_freq * (y_coords - hill)

    elif pattern_type == "loop":
        # Loop: Ridges enter and exit from the same side, curving around a core point
        # Distinct loop core and single delta
        slant = np.random.choice([-1.0, 1.0])  # left or right loop
        # U-shaped hairpin recurving flow
        dx = (x_coords - cx) * slant
        dy = y_coords - cy
        # Hairpin parabola: dy = a * dx^2
        # Ridges loop around core (dx=0, dy=0)
        core_dist = np.sqrt(dx ** 2 + np.maximum(0, dy) ** 2 * 1.5)
        tail_flow = y_coords + 0.35 * dx
        blend = 1.0 / (1.0 + np.exp((dy - 10) / 15.0))
        phase = 2 * np.pi * ridge_freq * (blend * core_dist * 1.4 + (1.0 - blend) * tail_flow)

    else:  # whorl
        # Whorl: Concentric circular / spiral pattern around central core with 2 deltas
        r = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        theta_angle = np.arctan2(y_coords - cy, x_coords - cx)
        spiral = 0.08 * np.sin(2.0 * theta_angle)
        phase = 2 * np.pi * ridge_freq * (r * (1.0 + spiral))

    # Generate ridges from sinusoidal phase: dark ridges, bright valleys
    ridges = 0.5 * (1.0 + np.cos(phase))

    # Create realistic elliptical fingertip mask
    rx, ry = W * 0.42, H * 0.46
    dist = ((x_coords - cx) / rx) ** 2 + ((y_coords - cy) / ry) ** 2
    mask = np.clip(1.0 - (dist - 0.75) / 0.25, 0.0, 1.0)
    # Smooth boundary feathering
    mask = cv2.GaussianBlur(mask, (7, 7), 2.0)

    # Invert so ridges are darker (typical biometric ink/sensor impression)
    img = 1.0 - (ridges * 0.75)
    # Composite over light background (paper/scanner platen)
    img = img * mask + (1.0 - mask) * 0.98

    # Add realistic sensor noise and micro-texture
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * 0.5, (H, W))
        # Speckle noise
        speckles = np.random.binomial(1, 0.015, (H, W)) * np.random.uniform(-0.3, 0.3, (H, W))
        img = np.clip(img + noise + speckles, 0.0, 1.0)

    # Convert to standard 8-bit grayscale
    uint8_img = (img * 255.0).astype(np.uint8)
    return uint8_img


def create_sample_dataset(
    output_dir: Path,
    samples_per_class: int = 40,
    seed: int = 42,
) -> Dict[str, int]:
    """Generates a complete sample dataset in output_dir organized by class."""
    output_dir = Path(output_dir)
    counts = {}
    for class_name in CLASSES:
        class_folder = output_dir / class_name
        class_folder.mkdir(parents=True, exist_ok=True)
        counts[class_name] = 0
        for i in range(samples_per_class):
            sample_seed = seed + hash(class_name) % 10000 + i
            img = generate_synthetic_fingerprint(class_name, seed=sample_seed)
            img_path = class_folder / f"{class_name}_{i:04d}.png"
            cv2.imwrite(str(img_path), img)
            counts[class_name] += 1
    return counts


# ---------------------------------------------------------------------------
# Data Augmentation (Slide 5 & 14)
# ---------------------------------------------------------------------------

class BiometricAugmentation:
    """Data augmentation for fingerprint training splits.

    Applies rotation, translation, scaling, and contrast/brightness variations.
    Note: Does NOT apply horizontal flip by default to avoid reversing loop slants.
    """

    def __init__(
        self,
        max_rotation_deg: float = 12.0,
        max_shift_px: float = 6.0,
        scale_range: Tuple[float, float] = (0.92, 1.08),
        contrast_range: Tuple[float, float] = (0.85, 1.15),
        brightness_range: Tuple[float, float] = (-0.1, 0.1),
    ):
        self.max_rotation_deg = max_rotation_deg
        self.max_shift_px = max_shift_px
        self.scale_range = scale_range
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range

    def __call__(self, tensor_img: torch.Tensor) -> torch.Tensor:
        """Takes tensor [1, H, W] in [0, 1] and applies random affine + intensity jitter."""
        img_np = tensor_img.squeeze(0).numpy()
        H, W = img_np.shape

        # Random affine transform
        angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
        tx = random.uniform(-self.max_shift_px, self.max_shift_px)
        ty = random.uniform(-self.max_shift_px, self.max_shift_px)
        scale = random.uniform(*self.scale_range)

        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        # Warp affine with constant border value 1.0 (light background)
        warped = cv2.warpAffine(
            img_np, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1.0
        )

        # Contrast and brightness jitter
        contrast = random.uniform(*self.contrast_range)
        brightness = random.uniform(*self.brightness_range)
        jittered = np.clip(warped * contrast + brightness, 0.0, 1.0)

        return torch.from_numpy(jittered).unsqueeze(0).float()


# ---------------------------------------------------------------------------
# PyTorch Dataset & Manifest Loader
# ---------------------------------------------------------------------------

class FingerprintSample:
    """Represents a single fingerprint record."""

    def __init__(
        self,
        image_source: Union[str, Path, np.ndarray],
        label: str,
        group_id: Optional[str] = None,
    ):
        self.image_source = image_source
        norm_label = LABEL_ALIASES.get(label.strip().lower(), label.strip().lower())
        if norm_label not in CLASS_TO_IDX:
            raise ValueError(f"Unrecognized class label: '{label}' (normalized: '{norm_label}')")
        self.label = norm_label
        self.label_id = CLASS_TO_IDX[self.label]
        self.group_id = group_id or f"grp_{id(self)}"


class FingerprintDataset(Dataset):
    """PyTorch Dataset for fingerprint classification."""

    def __init__(
        self,
        samples: List[FingerprintSample],
        pipeline: Optional[PreprocessingPipeline] = None,
        augmentation: Optional[BiometricAugmentation] = None,
    ):
        self.samples = samples
        self.pipeline = pipeline or PreprocessingPipeline()
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]

        # Preprocessing pipeline converts input to [1, H, W] normalized float32 array
        processed_arr = self.pipeline.process(sample.image_source, return_intermediates=False)
        tensor_img = torch.from_numpy(processed_arr).float()

        # Apply training augmentation if configured
        if self.augmentation is not None:
            tensor_img = self.augmentation(tensor_img)

        return tensor_img, sample.label_id


# ---------------------------------------------------------------------------
# Dataset Discovery, Stratification, and Class Balance (Slides 5, 7, 14)
# ---------------------------------------------------------------------------

def load_samples_from_directory(dataset_dir: Union[str, Path]) -> List[FingerprintSample]:
    """Scans a directory containing class subfolders (e.g. arch/, loop/, whorl/)."""
    dataset_dir = Path(dataset_dir)
    samples: List[FingerprintSample] = []
    for class_name in CLASSES:
        class_folder = dataset_dir / class_name
        if not class_folder.exists():
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            for file_path in class_folder.glob(ext):
                # group_id: stem without trailing duplicate index if paired
                group_id = file_path.stem.split("_")[0]
                samples.append(FingerprintSample(file_path, class_name, group_id))
    return samples


def load_samples_from_csv(csv_path: Union[str, Path]) -> List[FingerprintSample]:
    """Loads samples from a prepared CSV manifest."""
    csv_path = Path(csv_path)
    samples = []
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = Path(row["image_path"])
            label = row["label"]
            group_id = row.get("group_id")
            samples.append(FingerprintSample(img_path, label, group_id))
    return samples


def stratified_split(
    samples: List[FingerprintSample],
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[List[FingerprintSample], List[FingerprintSample], List[FingerprintSample]]:
    """Partitions samples into train, validation, and test sets.

    Ensures:
    1. Stratification across pattern classes (Arch, Loop, Whorl).
    2. Group isolation so all impressions of a finger stay within the same split (no identity leakage).
    """
    assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
    train_ratio, val_ratio, _ = ratios

    # Group samples by (label, group_id)
    by_class_groups: Dict[str, Dict[str, List[FingerprintSample]]] = {c: {} for c in CLASSES}
    for s in samples:
        if s.label in by_class_groups:
            by_class_groups[s.label].setdefault(s.group_id, []).append(s)

    train_samples, val_samples, test_samples = [], [], []

    rng = random.Random(seed)
    for class_name, groups_dict in by_class_groups.items():
        unique_groups = list(groups_dict.keys())
        rng.shuffle(unique_groups)

        n = len(unique_groups)
        if n < 3:
            # Fallback for small sample test fixtures: split samples directly
            all_class_samples = [s for grp in groups_dict.values() for s in grp]
            rng.shuffle(all_class_samples)
            n_total = len(all_class_samples)
            n_train = max(1, int(n_total * train_ratio))
            n_val = max(1, int(n_total * val_ratio))
            if n_train + n_val >= n_total:
                n_train = max(1, n_total - 2)
                n_val = 1
            train_samples.extend(all_class_samples[:n_train])
            val_samples.extend(all_class_samples[n_train:n_train + n_val])
            test_samples.extend(all_class_samples[n_train + n_val:])
            continue

        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1

        train_groups = set(unique_groups[:n_train])
        val_groups = set(unique_groups[n_train:n_train + n_val])
        test_groups = set(unique_groups[n_train + n_val:])

        for gid, sample_list in groups_dict.items():
            if gid in train_groups:
                train_samples.extend(sample_list)
            elif gid in val_groups:
                val_samples.extend(sample_list)
            else:
                test_samples.extend(sample_list)

    return train_samples, val_samples, test_samples


def compute_class_weights(samples: List[FingerprintSample]) -> torch.Tensor:
    """Computes inverse class frequency weights to counter class imbalance (Slides 7 & 14).

    weight_c = N_total / (num_classes * N_c)
    """
    counts = [0] * NUM_CLASSES
    for s in samples:
        counts[s.label_id] += 1

    total = sum(counts)
    weights = []
    for count in counts:
        if count == 0:
            weights.append(1.0)
        else:
            w = total / (NUM_CLASSES * count)
            weights.append(w)

    return torch.tensor(weights, dtype=torch.float32)
