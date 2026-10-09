"""Complete End-to-End Pipeline Integration Test.

Tests the full 5-phase workflow from the project presentation:
- Phase 1: Data Collection & Cleaning
- Phase 2: Preprocessing Pipeline (Grayscale, resize, noise reduction, CLAHE, segmentation, normalization)
- Phase 3: CNN Model Design & Training
- Phase 4: Evaluation & Confusion Matrix Reporting
- Phase 5: Inference & Demonstration
"""

import tempfile
from pathlib import Path

from src.constants import CLASSES
from src.dataset import create_sample_dataset, load_samples_from_directory, stratified_split
from src.inference import FingerprintClassifier
from src.model import FingerprintCNN
from src.preprocessing import PreprocessingPipeline
from src.trainer import FingerprintTrainer


def test_complete_end_to_end_workflow():
    with tempfile.TemporaryDirectory() as tmpdir:
        root_dir = Path(tmpdir)
        data_dir = root_dir / "data"
        model_path = root_dir / "fingerprint_cnn.pth"

        # Phase 1: Data Collection
        counts = create_sample_dataset(data_dir, samples_per_class=6, seed=99)
        assert len(counts) == 3
        samples = load_samples_from_directory(data_dir)
        assert len(samples) == 18

        # Stratified Split (Slide 5, 7)
        train_s, val_s, test_s = stratified_split(samples, ratios=(0.60, 0.20, 0.20), seed=99)
        assert len(train_s) > 0 and len(val_s) > 0 and len(test_s) > 0

        # Phase 2: Preprocessing Pipeline (Slide 7)
        pipeline = PreprocessingPipeline(target_size=64)
        sample_img = samples[0].image_source
        tensor, info = pipeline.process(sample_img, return_intermediates=True)
        assert tensor.shape == (1, 64, 64)
        assert len(info["steps"]) == 6

        # Phase 3: Model Design & Training (Slide 8)
        model = FingerprintCNN()
        trainer = FingerprintTrainer(model=model, learning_rate=0.005)
        fit_summary = trainer.fit(
            train_samples=train_s,
            val_samples=val_s,
            epochs=2,
            batch_size=4,
            save_best_path=model_path,
            verbose=False,
        )
        assert fit_summary["epochs_run"] == 2
        assert model_path.exists()

        # Phase 4: Evaluation & Confusion Matrix (Slide 11, 12)
        report = trainer.evaluate_test_set(test_s, batch_size=4)
        assert report.num_samples == len(test_s)
        assert 0.0 <= report.accuracy <= 1.0
        assert report.cm.shape == (3, 3)

        report_dict = report.to_dict()
        assert "per_class" in report_dict
        for c in CLASSES:
            assert c in report_dict["per_class"]

        # Phase 5: Demonstration Inference (Slide 13)
        classifier = FingerprintClassifier(checkpoint_path=model_path)
        pred = classifier.predict(sample_img, include_visualizations=True)
        assert pred["predicted_class"] in CLASSES
        assert 0.0 <= pred["confidence"] <= 1.0
        assert "preprocessing_stages" in pred
