"""Unit tests for the evaluation metrics and confusion matrix suite (Slides 11, 12, 14)."""

import numpy as np
import pytest

from src.constants import NUM_CLASSES
from src.evaluation import EvaluationReport


def test_evaluation_report_metrics_perfect():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 2, 2]
    y_probs = np.array([
        [0.9, 0.05, 0.05],
        [0.85, 0.1, 0.05],
        [0.05, 0.9, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.05, 0.9],
        [0.02, 0.08, 0.9],
    ])

    report = EvaluationReport(y_true=y_true, y_pred=y_pred, y_probs=y_probs)
    assert report.accuracy == 1.0
    assert report.precision_macro == 1.0
    assert report.recall_macro == 1.0
    assert report.f1_macro == 1.0
    assert report.cm[0, 0] == 2
    assert report.cm[1, 1] == 2
    assert report.cm[2, 2] == 2
    assert len(report.near_misses) == 0


def test_evaluation_report_borderline_detection():
    y_true = [1]
    y_pred = [1]
    # Margin between top 1 (0.52) and top 2 (0.44) is 0.08 < 0.20 threshold
    y_probs = np.array([[0.04, 0.52, 0.44]])

    report = EvaluationReport(y_true=y_true, y_pred=y_pred, y_probs=y_probs)
    assert len(report.near_misses) == 1
    nm = report.near_misses[0]
    assert nm["is_borderline"] is True
    assert nm["top1_class"] == "loop"
    assert nm["top2_class"] == "whorl"
    assert pytest.approx(nm["margin"], abs=1e-4) == 0.08


def test_evaluation_report_confusion_matrix_plot():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 2, 2]
    report = EvaluationReport(y_true=y_true, y_pred=y_pred)
    plot_data_url = report.render_confusion_matrix_plot()
    assert plot_data_url.startswith("data:image/png;base64,")


def test_evaluation_report_serialization():
    y_true = [0, 1, 2]
    y_pred = [0, 1, 2]
    report = EvaluationReport(y_true=y_true, y_pred=y_pred)

    d = report.to_dict()
    assert "summary" in d
    assert "per_class" in d
    assert "confusion_matrix" in d
    assert d["summary"]["accuracy"] == 1.0

    summary_text = report.print_text_summary()
    assert "AUTOMATED FINGERPRINT PATTERN CLASSIFICATION" in summary_text
    assert "Arch" in summary_text
    assert "Loop" in summary_text
    assert "Whorl" in summary_text
