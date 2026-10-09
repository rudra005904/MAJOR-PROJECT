"""Inference Engine for Fingerprint Pattern Classification.

Provides end-to-end classification for single images and batch scans,
including full preprocessing visualization stages, near-miss borderline diagnosis,
and biometric pattern interpretation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from src.constants import (
    BORDERLINE_MARGIN_THRESHOLD,
    CLASS_METADATA,
    CLASSES,
    IDX_TO_CLASS,
    NUM_CLASSES,
)
from src.model import FingerprintCNN
from src.preprocessing import PreprocessingPipeline


class FingerprintClassifier:
    """Production-ready inference service for automated fingerprint pattern recognition."""

    def __init__(
        self,
        model: Optional[FingerprintCNN] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = PreprocessingPipeline()

        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint_path is not None and Path(checkpoint_path).exists():
            self.model = FingerprintCNN.load_checkpoint(checkpoint_path, device=self.device)
        else:
            # Initialize with fresh architecture
            self.model = FingerprintCNN().to(self.device)

        self.model.eval()

    def predict(
        self,
        image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
        include_visualizations: bool = True,
        borderline_threshold: float = BORDERLINE_MARGIN_THRESHOLD,
    ) -> Dict[str, Any]:
        """Classifies a fingerprint image and returns a detailed anatomical report.

        Args:
            image_input: Image filepath, bytes, numpy array, or PIL Image.
            include_visualizations: If True, returns base64 images of all preprocessing steps.
            borderline_threshold: Margin between top 2 classes to flag as borderline pattern.

        Returns:
            Dictionary with predicted_class, confidence, probabilities, borderline_alert,
            anatomical_details, and step visualizations.
        """
        # Run preprocessing
        if include_visualizations:
            tensor_np, intermediate_info = self.pipeline.process(image_input, return_intermediates=True)
        else:
            tensor_np = self.pipeline.process(image_input, return_intermediates=False)
            intermediate_info = None

        # Convert to torch tensor: shape (1, 1, H, W)
        tensor_img = torch.from_numpy(tensor_np).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_img)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Class ranking
        sorted_indices = np.argsort(probs)[::-1]
        top1_idx = int(sorted_indices[0])
        top2_idx = int(sorted_indices[1])

        top1_class = IDX_TO_CLASS[top1_idx]
        top2_class = IDX_TO_CLASS[top2_idx]
        top1_prob = float(probs[top1_idx])
        top2_prob = float(probs[top2_idx])
        margin = top1_prob - top2_prob

        is_borderline = margin < borderline_threshold

        # Probability distribution dict
        prob_dict = {IDX_TO_CLASS[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)}

        # Anatomical metadata for predicted pattern
        meta = CLASS_METADATA.get(top1_class, {})

        result: Dict[str, Any] = {
            "predicted_class": top1_class,
            "predicted_class_title": meta.get("title", top1_class.capitalize()),
            "confidence": round(top1_prob, 4),
            "confidence_percentage": f"{top1_prob * 100:.2f}%",
            "probabilities": prob_dict,
            "borderline_analysis": {
                "is_borderline": is_borderline,
                "margin": round(margin, 4),
                "runner_up_class": top2_class,
                "runner_up_prob": round(top2_prob, 4),
                "threshold": borderline_threshold,
                "note": (
                    f"Warning: Low margin ({margin:.2%}) between {top1_class} and {top2_class}. "
                    "Review core and delta ridge flow carefully."
                    if is_borderline
                    else "High confidence separation from secondary pattern."
                ),
            },
            "anatomical_description": meta.get("description", ""),
            "key_features": meta.get("key_features", ""),
            "population_prevalence": meta.get("prevalence", ""),
            "color_theme": meta.get("color", "#3B82F6"),
        }

        if include_visualizations and intermediate_info is not None:
            result["preprocessing_stages"] = intermediate_info["steps"]
            result["preprocessing_metrics"] = intermediate_info["meta"]

        return result
