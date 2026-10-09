"""Command-line interface (CLI) for Fingerprint Pattern Classification.

Provides subcommands:
- generate-dataset: Create synthetic biometric samples for Arch, Loop, and Whorl
- preprocess: Run and inspect the 6-stage image cleanup & enhancement pipeline
- train: Train CNN model with stratified splits, class weights, and early stopping
- evaluate: Run rigorous benchmark evaluation and export confusion matrix report
- predict: Classify a single image with probability distribution and borderline diagnosis
- serve: Run the interactive web dashboard & REST API
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import uvicorn

from src.constants import DEFAULT_BATCH_SIZE, DEFAULT_IMAGE_SIZE
from src.dataset import (
    create_sample_dataset,
    load_samples_from_directory,
    stratified_split,
)
from src.inference import FingerprintClassifier
from src.model import FingerprintCNN
from src.preprocessing import PreprocessingPipeline, load_to_grayscale
from src.trainer import FingerprintTrainer


def cmd_generate_dataset(args):
    """Generates synthetic dataset."""
    output_dir = Path(args.output)
    print(f"Generating synthetic biometric dataset into {output_dir}...")
    counts = create_sample_dataset(output_dir, samples_per_class=args.samples_per_class, seed=args.seed)
    print("Generation complete! Created samples:")
    for c, cnt in counts.items():
        print(f"  - {c.capitalize()}: {cnt} images")
    return 0


def cmd_preprocess(args):
    """Executes preprocessing pipeline on an image."""
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PreprocessingPipeline(target_size=args.size)
    tensor, info = pipeline.process(input_path, return_intermediates=True)

    print(f"Preprocessed {input_path} -> Tensor shape: {tensor.shape}")
    print(f"Preprocessing metrics: {json.dumps(info['meta'], indent=2)}")

    # Save output images
    cv2.imwrite(str(output_dir / "normalized_segmented.png"), (tensor[0] * 255).astype("uint8"))
    print(f"Saved processed output to {output_dir}")
    return 0


def cmd_train(args):
    """Trains the CNN model."""
    data_dir = Path(args.data)
    output_model = Path(args.output)

    samples = load_samples_from_directory(data_dir)
    if not samples:
        print(f"ERROR: No fingerprint samples found in {data_dir}. Run generate-dataset first.", file=sys.stderr)
        return 1

    print(f"Loaded {len(samples)} total samples from {data_dir}.")
    train_s, val_s, test_s = stratified_split(samples, ratios=(0.70, 0.15, 0.15), seed=args.seed)
    print(f"Stratified split -> Train: {len(train_s)} | Val: {len(val_s)} | Test: {len(test_s)}")

    model = FingerprintCNN()
    trainer = FingerprintTrainer(model=model, learning_rate=args.lr)

    print(f"Starting CNN training for {args.epochs} epochs with patience {args.patience}...")
    summary = trainer.fit(
        train_samples=train_s,
        val_samples=val_s,
        epochs=args.epochs,
        batch_size=args.batch,
        patience=args.patience,
        save_best_path=output_model,
        verbose=True,
    )

    print("\nTraining Finished!")
    print(f"Best Val Loss: {summary['best_val_loss']} | Best Val Acc: {summary['best_val_acc'] * 100:.2f}%")
    print(f"Saved model to: {output_model}")

    print("\nEvaluating on Held-Out Test Split (Slide 11 & 12)...")
    report = trainer.evaluate_test_set(test_s, batch_size=args.batch)
    print(report.print_text_summary())

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"Saved evaluation report to {report_path}")

    return 0


def cmd_evaluate(args):
    """Evaluates a trained model checkpoint."""
    model_path = Path(args.model)
    data_dir = Path(args.data)

    samples = load_samples_from_directory(data_dir)
    if not samples:
        print(f"ERROR: No samples found in {data_dir}", file=sys.stderr)
        return 1

    model = FingerprintCNN.load_checkpoint(model_path)
    trainer = FingerprintTrainer(model=model)
    report = trainer.evaluate_test_set(samples)

    print(report.print_text_summary())
    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"Saved JSON report to {args.output_json}")

    return 0


def cmd_predict(args):
    """Classifies a fingerprint image."""
    img_path = Path(args.image)
    model_path = Path(args.model) if args.model else None

    classifier = FingerprintClassifier(checkpoint_path=model_path)
    result = classifier.predict(img_path, include_visualizations=False)

    print("\n--- CLASSIFICATION RESULT ---")
    print(f"Predicted Pattern: {result['predicted_class_title'].upper()}")
    print(f"Confidence:        {result['confidence_percentage']}")
    print(f"Key Features:      {result['key_features']}")
    print(f"Prevalence:        {result['population_prevalence']}")
    print("Class Probabilities:")
    for c, p in result["probabilities"].items():
        print(f"  - {c.capitalize():<6}: {p * 100:.2f}%")

    if result["borderline_analysis"]["is_borderline"]:
        print("\n[BORDERLINE ALERT]: Low margin separation detected between classes!")
        print(f"  Runner-up: {result['borderline_analysis']['runner_up_class']} ({result['borderline_analysis']['runner_up_prob'] * 100:.2f}%)")
    return 0


def cmd_serve(args):
    """Runs the web application."""
    print(f"Starting Fingerprint Classification server on http://{args.host}:{args.port}")
    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Automated Fingerprint Pattern Classification System (Parul University)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # generate-dataset
    p_gen = subparsers.add_parser("generate-dataset", help="Generate synthetic fingerprint dataset")
    p_gen.add_argument("--output", "-o", default="sample_data", help="Output directory")
    p_gen.add_argument("--samples-per-class", "-n", type=int, default=30, help="Samples per class")
    p_gen.add_argument("--seed", type=int, default=42, help="Random seed")
    p_gen.set_defaults(func=cmd_generate_dataset)

    # preprocess
    p_pre = subparsers.add_parser("preprocess", help="Run preprocessing on an image")
    p_pre.add_argument("--input", "-i", required=True, help="Input image path")
    p_pre.add_argument("--output", "-o", default="output/preprocessed", help="Output directory")
    p_pre.add_argument("--size", type=int, default=DEFAULT_IMAGE_SIZE, help="Target resolution")
    p_pre.set_defaults(func=cmd_preprocess)

    # train
    p_tr = subparsers.add_parser("train", help="Train CNN model")
    p_tr.add_argument("--data", "-d", default="sample_data", help="Dataset directory")
    p_tr.add_argument("--output", "-o", default="models/best_fingerprint_cnn.pth", help="Checkpoint output path")
    p_tr.add_argument("--epochs", "-e", type=int, default=15, help="Number of epochs")
    p_tr.add_argument("--batch", "-b", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    p_tr.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    p_tr.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    p_tr.add_argument("--seed", type=int, default=42, help="Random seed")
    p_tr.add_argument("--report-json", help="Path to save evaluation JSON report")
    p_tr.set_defaults(func=cmd_train)

    # evaluate
    p_ev = subparsers.add_parser("evaluate", help="Evaluate model on dataset")
    p_ev.add_argument("--model", "-m", default="models/best_fingerprint_cnn.pth", help="Model checkpoint")
    p_ev.add_argument("--data", "-d", default="sample_data", help="Dataset directory")
    p_ev.add_argument("--output-json", help="Path to save JSON metrics report")
    p_ev.set_defaults(func=cmd_evaluate)

    # predict
    p_pr = subparsers.add_parser("predict", help="Predict class of a fingerprint image")
    p_pr.add_argument("--image", "-i", required=True, help="Image path")
    p_pr.add_argument("--model", "-m", help="Optional model checkpoint")
    p_pr.set_defaults(func=cmd_predict)

    # serve
    p_srv = subparsers.add_parser("serve", help="Run web application")
    p_srv.add_argument("--host", default="127.0.0.1", help="Host address")
    p_srv.add_argument("--port", type=int, default=8000, help="Port")
    p_srv.add_argument("--reload", action="store_true", help="Auto reload")
    p_srv.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
