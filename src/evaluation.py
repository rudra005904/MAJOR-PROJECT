"""Evaluation metrics, Confusion Matrix generation, and Borderline Near-Miss analysis.

Fulfills Phase 4 and Slides 11, 12, 14 of the project:
1. Overall Accuracy.
2. Macro & Weighted Precision, Recall, and F1-Score.
3. Per-Class Metrics breakdown (Arch, Loop, Whorl).
4. Confusion Matrix (raw counts and normalized percentages).
5. Borderline / Near-Miss pattern identification (low margin between top-2 predictions).
6. Confusion matrix plot visualization (base64 encoded PNG).
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.constants import BORDERLINE_MARGIN_THRESHOLD, CLASSES, NUM_CLASSES


def plot_confusion_matrix_b64(matrix: Any, classes: Optional[List[str]] = None) -> str:
    """Generates a publication-grade confusion matrix visualization as a base64 PNG data URL."""
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    if classes is None:
        classes = list(CLASSES)

    num_classes = len(classes)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels([c.capitalize() for c in classes], fontsize=11, fontweight="bold")
    ax.set_yticklabels([c.capitalize() for c in classes], fontsize=11, fontweight="bold")

    ax.set_xlabel("Predicted Pattern Class", fontsize=12, labelpad=10, fontweight="bold")
    ax.set_ylabel("True Pattern Class", fontsize=12, labelpad=10, fontweight="bold")
    ax.set_title("Biometric Pattern Confusion Matrix", fontsize=13, pad=12, fontweight="bold")

    # Overlay counts and percentages
    total_samples = np.sum(matrix)
    thresh = matrix.max() / 2.0 if matrix.size > 0 and matrix.max() > 0 else 1.0
    for i in range(num_classes):
        for j in range(num_classes):
            count = int(matrix[i, j])
            pct = (count / total_samples * 100) if total_samples > 0 else 0
            color = "white" if count > thresh else "black"
            ax.text(
                j,
                i,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
                fontweight="semibold",
            )

    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


class EvaluationReport:
    """Rigorous evaluation report matching Slide 12 and 14."""

    def __init__(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_probs: Optional[np.ndarray] = None,
        sample_ids: Optional[List[str]] = None,
    ):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_probs = y_probs
        self.sample_ids = sample_ids or [f"sample_{i}" for i in range(len(y_true))]

        self.num_samples = len(y_true)
        self.classes = list(CLASSES)

        # Core global metrics (Slide 12)
        self.accuracy = float(accuracy_score(self.y_true, self.y_pred))
        self.precision_macro = float(precision_score(self.y_true, self.y_pred, average="macro", zero_division=0))
        self.recall_macro = float(recall_score(self.y_true, self.y_pred, average="macro", zero_division=0))
        self.f1_macro = float(f1_score(self.y_true, self.y_pred, average="macro", zero_division=0))
        self.f1_weighted = float(f1_score(self.y_true, self.y_pred, average="weighted", zero_division=0))

        # Per-class metrics (Slide 12 item 6)
        p_class = precision_score(self.y_true, self.y_pred, average=None, labels=range(NUM_CLASSES), zero_division=0)
        r_class = recall_score(self.y_true, self.y_pred, average=None, labels=range(NUM_CLASSES), zero_division=0)
        f_class = f1_score(self.y_true, self.y_pred, average=None, labels=range(NUM_CLASSES), zero_division=0)

        self.per_class: Dict[str, Dict[str, float]] = {}
        for idx, cname in enumerate(self.classes):
            support = int(np.sum(self.y_true == idx))
            self.per_class[cname] = {
                "precision": float(p_class[idx]),
                "recall": float(r_class[idx]),
                "f1_score": float(f_class[idx]),
                "support": support,
            }

        # Confusion Matrix (Slide 12 item 5)
        self.cm = confusion_matrix(self.y_true, self.y_pred, labels=range(NUM_CLASSES))
        self.cm_list = self.cm.tolist()

        # Near-Miss Borderline Analysis (Slide 11 & 14)
        self.near_misses: List[Dict[str, Any]] = []
        if self.y_probs is not None:
            self._analyze_near_misses()

    def _analyze_near_misses(self, margin_threshold: float = BORDERLINE_MARGIN_THRESHOLD) -> None:
        """Finds instances where model confidence was close between top 2 classes."""
        for i, probs in enumerate(self.y_probs):
            sorted_indices = np.argsort(probs)[::-1]
            top1_idx, top2_idx = sorted_indices[0], sorted_indices[1]
            top1_prob, top2_prob = float(probs[top1_idx]), float(probs[top2_idx])
            margin = top1_prob - top2_prob

            is_borderline = margin < margin_threshold
            is_error = int(self.y_true[i]) != int(self.y_pred[i])

            if is_borderline or is_error:
                self.near_misses.append({
                    "sample_id": self.sample_ids[i],
                    "true_class": self.classes[self.y_true[i]],
                    "predicted_class": self.classes[self.y_pred[i]],
                    "top1_class": self.classes[top1_idx],
                    "top1_prob": round(top1_prob, 4),
                    "top2_class": self.classes[top2_idx],
                    "top2_prob": round(top2_prob, 4),
                    "margin": round(margin, 4),
                    "is_borderline": is_borderline,
                    "is_misclassified": is_error,
                })

    def render_confusion_matrix_plot(self) -> str:
        """Generates a publication-grade confusion matrix visualization as a base64 PNG data URL."""
        return plot_confusion_matrix_b64(self.cm, self.classes)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes complete report to JSON-serializable dictionary."""
        return {
            "summary": {
                "total_samples": self.num_samples,
                "accuracy": round(self.accuracy, 4),
                "accuracy_percentage": f"{self.accuracy * 100:.2f}%",
                "precision_macro": round(self.precision_macro, 4),
                "recall_macro": round(self.recall_macro, 4),
                "f1_macro": round(self.f1_macro, 4),
                "f1_weighted": round(self.f1_weighted, 4),
            },
            "per_class": self.per_class,
            "confusion_matrix": {
                "labels": self.classes,
                "matrix": self.cm_list,
            },
            "near_misses_count": len(self.near_misses),
            "near_misses": self.near_misses[:20],  # top 20
        }

    def print_text_summary(self) -> str:
        """Constructs formatted human-readable report."""
        lines = [
            "=" * 64,
            "AUTOMATED FINGERPRINT PATTERN CLASSIFICATION EVALUATION REPORT",
            "=" * 64,
            f"Total Evaluated Samples: {self.num_samples}",
            f"Overall Accuracy:        {self.accuracy * 100:.2f}%",
            f"Macro F1-Score:          {self.f1_macro * 100:.2f}%",
            f"Weighted F1-Score:       {self.f1_weighted * 100:.2f}%",
            "-" * 64,
            "PER-CLASS PERFORMANCE BREAKDOWN (Slide 12):",
            f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}",
            "-" * 64,
        ]
        for cname, metrics in self.per_class.items():
            lines.append(
                f"{cname.capitalize():<12} "
                f"{metrics['precision'] * 100:>10.2f}% "
                f"{metrics['recall'] * 100:>10.2f}% "
                f"{metrics['f1_score'] * 100:>10.2f}% "
                f"{metrics['support']:>8}"
            )
        lines.append("-" * 64)
        lines.append("CONFUSION MATRIX (True \\ Pred):")
        header = f"{'':<10}" + " ".join(f"{c.capitalize():>10}" for c in self.classes)
        lines.append(header)
        for i, cname in enumerate(self.classes):
            row = f"{cname.capitalize():<10}" + " ".join(f"{self.cm[i, j]:>10}" for j in range(NUM_CLASSES))
            lines.append(row)
        lines.append("-" * 64)
        lines.append(f"Borderline / Near-Miss Predictions: {len(self.near_misses)}")
        lines.append("=" * 64)
        return "\n".join(lines)
