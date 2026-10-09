#!/usr/bin/env python3
"""Validate labeled fingerprint sources and create deterministic group-safe splits.

Input CSV columns: image,label,group_id. Image paths are relative to --source-root
or absolute. Labels must come from source metadata or a documented human review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError


CLASSES = ("arch", "tented_arch", "left_loop", "right_loop", "whorl")
TF_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
LABEL_ALIASES = {
    "arch": "arch", "plain arch": "arch", "pa": "arch", "a": "arch",
    "tented arch": "tented_arch", "tented_arch": "tented_arch", "ta": "tented_arch", "t": "tented_arch",
    "left loop": "left_loop", "left_loop": "left_loop", "left slant loop": "left_loop", "ls": "left_loop", "l": "left_loop",
    "right loop": "right_loop", "right_loop": "right_loop", "right slant loop": "right_loop", "rs": "right_loop", "r": "right_loop",
    "whorl": "whorl", "whirl": "whorl", "plain whorl": "whorl", "w": "whorl", "pw": "whorl",
}


def normalize_label(raw: str) -> str:
    key = re.sub(r"\s+", " ", raw.strip().lower().replace("-", " "))
    key = key.replace("_", " ")
    # Keep documented source aliases explicit; don't guess from filenames.
    normalized = LABEL_ALIASES.get(key)
    if normalized is None:
        raise ValueError(f"unsupported or ambiguous source label {raw!r}")
    return normalized


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_source(csv_path: Path, source_root: Path):
    required = {"image", "label", "group_id"}
    rows = []
    errors = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"input CSV is missing required columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, start=2):
            try:
                rel = (row.get("image") or "").strip()
                raw_label = (row.get("label") or "").strip()
                group = (row.get("group_id") or "").strip()
                if not rel or not raw_label or not group:
                    raise ValueError("image, label, and group_id must be non-empty")
                path = Path(rel)
                if not path.is_absolute():
                    path = source_root / path
                path = path.resolve(strict=True)
                if not path.is_file():
                    raise ValueError("image path is not a file")
                if path.suffix.lower() not in TF_IMAGE_EXTENSIONS:
                    raise ValueError(f"unsupported TensorFlow image extension {path.suffix!r}; use PNG, JPEG, BMP, or GIF")
                label = normalize_label(raw_label)
                with Image.open(path) as im:
                    im.verify()
                digest = sha256_file(path)
                rows.append({
                    "image_path": str(path), "source_image": rel, "raw_label": raw_label,
                    "label": label, "label_id": CLASSES.index(label), "group_id": group,
                    "source": (row.get("source") or "unspecified").strip() or "unspecified",
                    "sha256": digest,
                })
            except (ValueError, OSError, UnidentifiedImageError) as exc:
                errors.append(f"line {line}: {exc}")
    if errors:
        raise ValueError("source validation failed:\n  " + "\n  ".join(errors[:50]) + ("\n  …" if len(errors) > 50 else ""))
    if not rows:
        raise ValueError("source CSV contains no images")

    paths = [r["image_path"] for r in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("the same image path appears more than once")
    groups = defaultdict(set)
    hashes = defaultdict(set)
    for r in rows:
        groups[r["group_id"]].add(r["label"])
        hashes[r["sha256"]].add(r["group_id"])
    mixed = [g for g, labels in groups.items() if len(labels) != 1]
    if mixed:
        raise ValueError(f"group_id must have one class label; conflicting groups: {', '.join(mixed[:10])}")
    duplicate_hashes = [h for h, gs in hashes.items() if len(gs) > 1]
    if duplicate_hashes:
        raise ValueError(f"identical image content occurs in multiple groups ({len(duplicate_hashes)} hash(es)); fix group IDs or remove duplicate copies")

    image_counts = Counter(r["label"] for r in rows)
    group_counts = Counter(next(iter(labels)) for labels in groups.values())
    missing_classes = [c for c in CLASSES if image_counts[c] == 0]
    if missing_classes:
        raise ValueError(f"source is missing required classes: {', '.join(missing_classes)}")
    insufficient = [f"{c} ({group_counts[c]} groups)" for c in CLASSES if group_counts[c] < 3]
    if insufficient:
        raise ValueError("need at least three independent groups per class for train/validation/test: " + ", ".join(insufficient))
    return rows, groups


def split_groups(groups, seed: int, ratios):
    result = {"train": set(), "validation": set(), "test": set()}
    by_class = defaultdict(list)
    for group, labels in groups.items():
        by_class[next(iter(labels))].append(group)
    rng = random.Random(seed)
    for label in CLASSES:
        items = sorted(by_class[label])
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * ratios[0]))
        n_val = max(1, int(n * ratios[1]))
        if n_train + n_val >= n:
            n_train, n_val = n - 2, 1
        result["train"].update(items[:n_train])
        result["validation"].update(items[n_train:n_train + n_val])
        result["test"].update(items[n_train + n_val:])
    return result


def write_csv(path: Path, rows):
    fields = ["image_path", "label", "label_id", "group_id", "source", "source_image", "raw_label", "sha256"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-csv", required=True, type=Path, help="CSV containing image,label,group_id and optional source")
    p.add_argument("--source-root", type=Path, default=Path.cwd(), help="base directory for relative image paths")
    p.add_argument("--output", required=True, type=Path, help="directory for manifests and provenance report")
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15), metavar=("TRAIN", "VALIDATION", "TEST"))
    args = p.parse_args()
    if any(x <= 0 for x in args.ratios) or abs(sum(args.ratios) - 1.0) > 1e-8:
        p.error("split ratios must be positive and sum to 1")
    try:
        rows, groups = read_source(args.source_csv.resolve(), args.source_root.resolve())
        split = split_groups(groups, args.seed, args.ratios)
        args.output.mkdir(parents=True, exist_ok=True)
        for name, ids in split.items():
            subset = [r for r in rows if r["group_id"] in ids]
            write_csv(args.output / f"{name}.csv", subset)
        overlap = (split["train"] & split["validation"]) | (split["train"] & split["test"]) | (split["validation"] & split["test"])
        if overlap:
            raise AssertionError("internal error: group leakage across splits")
        summary = {
            "seed": args.seed,
            "ratios": dict(zip(("train", "validation", "test"), args.ratios)),
            "class_order": list(CLASSES),
            "input_csv": str(args.source_csv.resolve()),
            "source_root": str(args.source_root.resolve()),
            "images": len(rows), "unique_groups": len(groups),
            "class_image_counts": dict(Counter(r["label"] for r in rows)),
            "class_group_counts": {c: sum(1 for labels in groups.values() if c in labels) for c in CLASSES},
            "splits": {name: {"images": sum(r["group_id"] in ids for r in rows), "groups": len(ids), "class_images": dict(Counter(r["label"] for r in rows if r["group_id"] in ids))} for name, ids in split.items()},
            "augmentation": "Applied by the training model only; no augmented files are written into validation or test.",
        }
        (args.output / "dataset_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
