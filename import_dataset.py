"""Utility to unpack, organize, and import downloaded fingerprint datasets into the project.

Supports:
1. Datasets organized in folders (e.g. Arch, Loop, Whorl, Left Loop, Plain Whorl, etc.)
2. NIST Special Database 4 (SD4) with accompanying .txt metadata (Class: A, L, R, W, T)
3. CSV manifests mapping image paths to labels
4. Direct ZIP extraction from Downloads
"""

import argparse
import csv
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from src.constants import CLASSES, LABEL_ALIASES

PROJECT_DATA_DIR = Path("data")


def parse_nist_txt_label(txt_path: Path) -> Optional[str]:
    """Extracts pattern class (A, T, L, R, W) from NIST SD4 text metadata."""
    try:
        content = txt_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"Class:\s*([A-Za-z]+)", content, re.IGNORECASE)
        if match:
            raw = match.group(1).strip().lower()
            return LABEL_ALIASES.get(raw)
    except Exception:
        pass
    return None


def import_from_directory_or_zip(
    source_path: Path,
    target_data_dir: Path = PROJECT_DATA_DIR,
    max_per_class: Optional[int] = None,
) -> Dict[str, int]:
    """Scans and copies images into target_data_dir/{arch, loop, whorl}."""
    source_path = Path(source_path)
    target_data_dir = Path(target_data_dir)

    for c in CLASSES:
        (target_data_dir / c).mkdir(parents=True, exist_ok=True)

    extracted_temp: Optional[Path] = None

    # Handle ZIP file
    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        print(f"Unzipping {source_path.name}...")
        extracted_temp = target_data_dir / "_temp_extract"
        extracted_temp.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_path, "r") as zf:
            zf.extractall(extracted_temp)
        search_root = extracted_temp
    elif source_path.is_dir():
        search_root = source_path
    else:
        raise ValueError(f"Source path not found: {source_path}")

    counts = {c: 0 for c in CLASSES}
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    # Pass 1: Check for folder-based classification (e.g. */arch/*, */loop/*)
    for img_file in search_root.rglob("*"):
        if img_file.is_file() and img_file.suffix.lower() in valid_exts:
            # Check parent folder names for class keywords
            parent_parts = [p.lower() for p in img_file.parts]
            matched_class = None
            for part in reversed(parent_parts[:-1]):
                cleaned = part.replace("-", " ").replace("_", " ").strip()
                if cleaned in LABEL_ALIASES:
                    matched_class = LABEL_ALIASES[cleaned]
                    break
                # Substring check
                if "whorl" in cleaned or "whirl" in cleaned:
                    matched_class = "whorl"
                    break
                elif "loop" in cleaned:
                    matched_class = "loop"
                    break
                elif "arch" in cleaned:
                    matched_class = "arch"
                    break

            # Pass 2: Check for NIST .txt counterpart (e.g. f0001_01.txt)
            if not matched_class:
                txt_candidate = img_file.with_suffix(".txt")
                if txt_candidate.exists():
                    matched_class = parse_nist_txt_label(txt_candidate)

            if matched_class and matched_class in counts:
                if max_per_class is None or counts[matched_class] < max_per_class:
                    dest_name = f"{matched_class}_{counts[matched_class]:05d}{img_file.suffix.lower()}"
                    dest_file = target_data_dir / matched_class / dest_name
                    shutil.copy2(img_file, dest_file)
                    counts[matched_class] += 1

    # Cleanup temp extraction
    if extracted_temp and extracted_temp.exists():
        shutil.rmtree(extracted_temp, ignore_errors=True)

    return counts


def main():
    parser = argparse.ArgumentParser(description="Import and organize fingerprint dataset for training")
    parser.add_argument("source", help="Path to downloaded ZIP file or extracted folder")
    parser.add_argument("--target", default="data", help="Target project data folder (default: data)")
    parser.add_argument("--max-per-class", type=int, help="Optional limit of images per class")
    args = parser.parse_args()

    print(f"Importing dataset from {args.source} into {args.target}...")
    counts = import_from_directory_or_zip(Path(args.source), Path(args.target), args.max_per_class)
    print("\nImport Summary:")
    for c, cnt in counts.items():
        print(f"  • {c.capitalize()}: {cnt} images")
    print(f"\nImages are organized in: {Path(args.target).resolve()}")
    print("You can now train the model with:")
    print(f"  .venv\\Scripts\\python.exe -m src.cli train --data {args.target} --epochs 25")


if __name__ == "__main__":
    main()
