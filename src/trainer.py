"""Training engine for the Fingerprint CNN model.

Fulfills Phase 3 & 4 (Slides 8, 11, 14):
1. Training on train split with data augmentation.
2. Validation evaluation after each epoch.
3. Class-weighted CrossEntropyLoss to address class imbalance.
4. Early stopping with best checkpoint retention.
5. Rigorous test split scoring returning an EvaluationReport.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.constants import DEFAULT_BATCH_SIZE, NUM_CLASSES
from src.dataset import (
    BiometricAugmentation,
    FingerprintDataset,
    FingerprintSample,
    compute_class_weights,
)
from src.evaluation import EvaluationReport
from src.model import FingerprintCNN
from src.preprocessing import PreprocessingPipeline


class FingerprintTrainer:
    """Orchestrates model optimization, validation, early stopping, and evaluation."""

    def __init__(
        self,
        model: Optional[FingerprintCNN] = None,
        device: Optional[str] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (model or FingerprintCNN()).to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def train_epoch(self, dataloader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Runs a single training epoch with backpropagation."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total_samples += batch_size

        epoch_loss = total_loss / max(1, total_samples)
        epoch_acc = correct / max(1, total_samples)
        return epoch_loss, epoch_acc

    def evaluate_loss_acc(self, dataloader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Calculates loss and accuracy on validation/test set without gradients."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                loss = criterion(logits, labels)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total_samples += batch_size

        return total_loss / max(1, total_samples), correct / max(1, total_samples)

    def fit(
        self,
        train_samples: List[FingerprintSample],
        val_samples: List[FingerprintSample],
        epochs: int = 15,
        batch_size: int = DEFAULT_BATCH_SIZE,
        patience: int = 5,
        save_best_path: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Full training loop with validation checks and early stopping."""
        pipeline = PreprocessingPipeline()
        aug = BiometricAugmentation()

        # Class weights to handle imbalance (Slide 7 & 14)
        class_weights = compute_class_weights(train_samples).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        train_ds = FingerprintDataset(train_samples, pipeline=pipeline, augmentation=aug)
        val_ds = FingerprintDataset(val_samples, pipeline=pipeline, augmentation=None)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_score = -float("inf")
        epochs_no_improve = 0
        best_state: Optional[Dict[str, Any]] = None

        start_time = time.time()
        for epoch in range(1, epochs + 1):
            t_loss, t_acc = self.train_epoch(train_loader, criterion)
            v_loss, v_acc = self.evaluate_loss_acc(val_loader, criterion)

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(round(t_loss, 4))
            self.history["train_acc"].append(round(t_acc, 4))
            self.history["val_loss"].append(round(v_loss, 4))
            self.history["val_acc"].append(round(v_acc, 4))

            if verbose:
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} | "
                    f"Train Loss: {t_loss:.4f} Acc: {t_acc * 100:.1f}% | "
                    f"Val Loss: {v_loss:.4f} Acc: {v_acc * 100:.1f}%"
                )

            # Model selection criterion: prioritize validation accuracy, then validation loss
            val_score = v_acc - 0.1 * v_loss
            if val_score > best_score:
                best_score = val_score
                best_val_loss = v_loss
                best_val_acc = v_acc
                epochs_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                if save_best_path:
                    self.model.save_checkpoint(
                        save_best_path,
                        metadata={
                            "epoch": epoch,
                            "val_loss": round(v_loss, 4),
                            "val_acc": round(v_acc, 4),
                        },
                    )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                    break

        # Restore weights of best epoch
        if best_state is not None:
            self.model.load_state_dict(best_state)

        duration = time.time() - start_time
        return {
            "epochs_run": len(self.history["epoch"]),
            "best_val_loss": round(best_val_loss, 4),
            "best_val_acc": round(best_val_acc, 4),
            "training_time_sec": round(duration, 2),
            "history": self.history,
        }

    def evaluate_test_set(
        self,
        test_samples: List[FingerprintSample],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> EvaluationReport:
        """Runs the model over the held-out test split and generates the EvaluationReport."""
        self.model.eval()
        pipeline = PreprocessingPipeline()
        test_ds = FingerprintDataset(test_samples, pipeline=pipeline, augmentation=None)
        loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                logits = self.model(images)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
                all_probs.extend(probs.tolist())

        sample_ids = [Path(s.image_source).stem if isinstance(s.image_source, (str, Path)) else f"sample_{i}" for i, s in enumerate(test_samples)]

        report = EvaluationReport(
            y_true=all_labels,
            y_pred=all_preds,
            y_probs=np.array(all_probs),
            sample_ids=sample_ids,
        )
        return report
