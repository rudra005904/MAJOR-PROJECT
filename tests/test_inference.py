"""Unit tests for the FingerprintClassifier inference engine."""

import numpy as np
import pytest

from src.constants import CLASSES
from src.dataset import generate_synthetic_fingerprint
from src.inference import FingerprintClassifier
from src.model import FingerprintCNN


def test_classifier_predict_synthetic_print():
    model = FingerprintCNN()
    classifier = FingerprintClassifier(model=model)

    img = generate_synthetic_fingerprint("arch", size=64, seed=42)
    result = classifier.predict(img, include_visualizations=True)

    assert result["predicted_class"] in CLASSES
    assert "confidence" in result
    assert "probabilities" in result
    assert len(result["probabilities"]) == 3
    assert "borderline_analysis" in result
    assert "preprocessing_stages" in result

    stages = result["preprocessing_stages"]
    for step_name in ("original", "denoised", "enhanced", "segmented", "mask", "orientation"):
        assert step_name in stages
        assert stages[step_name].startswith("data:image/png;base64,")


def test_classifier_predict_without_visualizations():
    model = FingerprintCNN()
    classifier = FingerprintClassifier(model=model)
    img = np.ones((64, 64), dtype=np.uint8) * 150
    result = classifier.predict(img, include_visualizations=False)

    assert result["predicted_class"] in CLASSES
    assert "preprocessing_stages" not in result
