"""Constants and configurations for the Fingerprint Pattern Classification system.

Matches the 3 core pattern categories (Arch, Loop, Whorl) and preprocessing
standards specified in the Parul University research project.
"""

from typing import Dict, List, Tuple

# 3 Core Biometric Pattern Categories as defined in the project slides
CLASSES: Tuple[str, ...] = ("arch", "loop", "whorl")
NUM_CLASSES: int = len(CLASSES)

CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {i: c for i, c in enumerate(CLASSES)}

# Display metadata and anatomical ridge characteristics from Project Slides 1, 6
CLASS_METADATA: Dict[str, Dict[str, str]] = {
    "arch": {
        "title": "Arch Pattern",
        "description": "Ridges enter from one side, rise smoothly in the center, and exit the opposite side without delta or recurving loop.",
        "key_features": "No delta, no core, simplest ridge flow.",
        "prevalence": "~5% of population",
        "color": "#10B981",  # Emerald Green
    },
    "loop": {
        "title": "Loop Pattern",
        "description": "Ridges enter and exit from the same side, curving back on themselves around a central core point with exactly one delta.",
        "key_features": "One core, one delta, recurving ridge flow (left or right slant).",
        "prevalence": "~60-65% of population",
        "color": "#3B82F6",  # Blue
    },
    "whorl": {
        "title": "Whorl Pattern",
        "description": "Ridges form circular, spiral, or whorled configurations around a central core, typically accompanied by two deltas.",
        "key_features": "Two deltas, central circular/spiral whorl, complex ridge topology.",
        "prevalence": "~30-35% of population",
        "color": "#8B5CF6",  # Purple
    },
}

# Aliases to accommodate external datasets (NIST SD4, FVC, Sokoto, etc.)
# Maps 5-class NIST conventions (plain arch, tented arch, left loop, right loop, whorl) into the 3 core categories
LABEL_ALIASES: Dict[str, str] = {
    # Arch aliases
    "arch": "arch",
    "plain arch": "arch",
    "plain_arch": "arch",
    "pa": "arch",
    "a": "arch",
    "tented arch": "arch",
    "tented_arch": "arch",
    "ta": "arch",
    "t": "arch",
    # Loop aliases
    "loop": "loop",
    "left loop": "loop",
    "left_loop": "loop",
    "left slant loop": "loop",
    "ls": "loop",
    "l": "loop",
    "right loop": "loop",
    "right_loop": "loop",
    "right slant loop": "loop",
    "rs": "loop",
    "r": "loop",
    "radial loop": "loop",
    "ulnar loop": "loop",
    # Whorl aliases
    "whorl": "whorl",
    "whirl": "whorl",
    "plain whorl": "whorl",
    "plain_whorl": "whorl",
    "pw": "whorl",
    "w": "whorl",
    "double loop": "whorl",
    "double_loop": "whorl",
    "accidental whorl": "whorl",
    "accidental_whorl": "whorl",
}

# Image Preprocessing Standards (Slide 7)
DEFAULT_IMAGE_SIZE: int = 128
DEFAULT_BATCH_SIZE: int = 32
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Borderline Near-Miss Margin threshold (Slide 11, 14)
# When top1_prob - top2_prob < BORDERLINE_MARGIN_THRESHOLD, flag as near-miss / borderline
BORDERLINE_MARGIN_THRESHOLD: float = 0.20
