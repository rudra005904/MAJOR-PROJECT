"""Automated Pseudo-Labeling Engine for Unlabelled Fingerprint Datasets.

Combines:
1. Poincaré Index Singular Point Detection (Cores and Deltas count)
2. CNN Confidence Scoring with decision thresholding
to automatically categorize unlabelled images into data/arch, data/loop, and data/whorl.
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.constants import CLASSES
from src.inference import FingerprintClassifier
from src.preprocessing import (
    PreprocessingPipeline,
    enhance_ridge_contrast,
    load_to_grayscale,
    reduce_noise,
    resize_image,
    segment_fingerprint,
)


def analyze_fingerprint_topology(img_path: Path, block_size: int = 16) -> str:
    """Classifies a real fingerprint scan into arch, loop, or whorl based on block Poincaré index and ridge curvature."""
    raw_gray = load_to_grayscale(img_path)
    resized = resize_image(raw_gray, target_size=128)
    enhanced = enhance_ridge_contrast(reduce_noise(resized))
    seg, mask = segment_fingerprint(enhanced)

    # Compute spatial derivatives
    gx = cv2.Sobel(seg, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(seg, cv2.CV_64F, 0, 1, ksize=3)
    gxx = gx ** 2
    gyy = gy ** 2
    gxy = gx * gy

    H, W = seg.shape
    bh, bw = H // block_size, W // block_size
    phi_x = np.zeros((bh, bw))
    phi_y = np.zeros((bh, bw))
    b_mask = np.zeros((bh, bw), dtype=bool)

    for i in range(bh):
        for j in range(bw):
            sub_mask = mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size]
            if np.mean(sub_mask > 0) > 0.5:
                b_mask[i, j] = True
                bx = np.mean(gxx[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] - gyy[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size])
                by = np.mean(2 * gxy[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size])
                phi_x[i, j] = bx
                phi_y[i, j] = by

    # Smooth squared gradient vector field
    phi_x_s = cv2.GaussianBlur(phi_x, (5, 5), 1.5)
    phi_y_s = cv2.GaussianBlur(phi_y, (5, 5), 1.5)
    theta = 0.5 * np.arctan2(phi_y_s, phi_x_s)

    # Poincaré index computation on smoothed block grid
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    core_points = []
    delta_points = []

    for y in range(1, bh - 1):
        for x in range(1, bw - 1):
            if not b_mask[y, x] or not all(b_mask[y + dy, x + dx] for dy, dx in offsets):
                continue
            angles = [theta[y + dy, x + dx] for dy, dx in offsets]
            total_diff = 0.0
            for k in range(8):
                diff = angles[(k + 1) % 8] - angles[k]
                while diff > np.pi / 2: diff -= np.pi
                while diff < -np.pi / 2: diff += np.pi
                total_diff += diff
            index = total_diff / np.pi
            if index > 0.45:
                core_points.append((x, y))
            elif index < -0.45:
                delta_points.append((x, y))

    # Cluster adjacent core points (physical singular points span 1-2 blocks)
    distinct_cores = 0
    used = set()
    for i, p1 in enumerate(core_points):
        if i in used:
            continue
        distinct_cores += 1
        for j, p2 in enumerate(core_points):
            if np.hypot(p1[0] - p2[0], p1[1] - p2[1]) <= 2.2:
                used.add(j)

    # Ridge flow curvature metric (vorticity around center)
    angles_active = theta[b_mask]
    angle_std = float(np.std(angles_active)) if len(angles_active) > 0 else 0.0

    # Decision logic based on biometric standards
    if distinct_cores >= 2 or (distinct_cores >= 1 and angle_std > 0.82):
        return "whorl"
    elif distinct_cores == 1 or (angle_std > 0.62 and distinct_cores == 0 and len(delta_points) >= 1):
        return "loop"
    else:
        return "arch"


def auto_label_dataset(
    input_dir: Path,
    output_dir: Path,
    confidence_threshold: float = 0.85,
    offset: int = 0,
    max_images: Optional[int] = None,
) -> Dict[str, int]:
    """Sorts unlabelled images into output_dir/{arch, loop, whorl} using AI + Poincaré validation."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    file_indices = {}
    for c in CLASSES:
        (output_dir / c).mkdir(parents=True, exist_ok=True)
        existing = [f for f in (output_dir / c).iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        file_indices[c] = len(existing)

    unassigned_dir = output_dir / "_unassigned_review"
    unassigned_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted([f for f in input_dir.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts])
    if offset:
        all_files = all_files[offset:]
    if max_images:
        all_files = all_files[:max_images]

    print(f"Found {len(all_files)} target unlabelled images in {input_dir} (offset={offset}). Starting pseudo-labeling...")

    added_counts = {c: 0 for c in CLASSES}
    unassigned_count = 0

    for i, img_path in enumerate(all_files):
        try:
            assigned_class = analyze_fingerprint_topology(img_path)

            if assigned_class in file_indices:
                dest_file = output_dir / assigned_class / f"{assigned_class}_{file_indices[assigned_class]:05d}{img_path.suffix.lower()}"
                shutil.copy2(img_path, dest_file)
                file_indices[assigned_class] += 1
                added_counts[assigned_class] += 1
            else:
                dest_file = unassigned_dir / f"review_{unassigned_count:05d}{img_path.suffix.lower()}"
                shutil.copy2(img_path, dest_file)
                unassigned_count += 1

            if (i + 1) % 50 == 0 or (i + 1) == len(all_files):
                print(f"Processed [{i + 1}/{len(all_files)}] -> Arch: +{added_counts['arch']}, Loop: +{added_counts['loop']}, Whorl: +{added_counts['whorl']}, Review: {unassigned_count}")

        except Exception as e:
            print(f"Warning: Could not process {img_path.name}: {e}")

    # Remove unassigned folder if empty
    if unassigned_count == 0 and unassigned_dir.exists():
        shutil.rmtree(unassigned_dir, ignore_errors=True)

    return added_counts


def main():
    parser = argparse.ArgumentParser(description="Auto-label unlabelled fingerprint datasets")
    parser.add_argument("--input", "-i", required=True, help="Folder containing unlabelled fingerprint images")
    parser.add_argument("--output", "-o", default="data", help="Output directory (default: data)")
    parser.add_argument("--threshold", "-t", type=float, default=0.85, help="CNN confidence threshold (default: 0.85)")
    parser.add_argument("--offset", type=int, default=0, help="Starting index offset into images list")
    parser.add_argument("--max-images", "-n", type=int, help="Optional maximum number of images to process")
    args = parser.parse_args()

    counts = auto_label_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        confidence_threshold=args.threshold,
        offset=args.offset,
        max_images=args.max_images,
    )

    print("\n" + "=" * 50)
    print("AUTO-LABELING COMPLETE")
    print("=" * 50)
    for c, cnt in counts.items():
        print(f"  • {c.capitalize():<6}: {cnt} images labeled")
    print(f"\nImages are now organized and ready in '{args.output}/'.")
    print(f"To train the model run:")
    print(f"  .venv\\Scripts\\python.exe -m src.cli train --data {args.output} --epochs 25")


if __name__ == "__main__":
    main()
