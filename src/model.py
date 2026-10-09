"""CNN Architecture for Fingerprint Pattern Classification.

Implements Phase 3 and Slide 8 of the project:
- Convolutional Feature Extractor:
  * Stacked Conv2d + BatchNorm + ReLU blocks learning local ridge orientation and curvature.
  * Pooling layers progressively reducing spatial size while preserving key ridge topology.
  * Deeper layers capturing core and delta-level pattern structures.
- Classification Head:
  * Adaptive pooling / flattening.
  * Dense fully-connected layers with Dropout (0.35) to prevent overfitting.
  * Linear projection with Softmax output over the 3 pattern classes (Arch, Loop, Whorl).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import CLASSES, NUM_CLASSES


class ConvBlock(nn.Module):
    """Convolution + Batch Normalization + ReLU block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class FingerprintCNN(nn.Module):
    """Deep Convolutional Neural Network for 3-class Biometric Pattern Recognition."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = 1,
        dropout_rate: float = 0.35,
    ):
        super().__init__()
        self.num_classes = num_classes

        # -------------------------------------------------------------
        # 1. Feature Extractor (Slide 8)
        # -------------------------------------------------------------
        # Stage 1: Local ridge orientation and curvature (1 -> 32 channels)
        self.stage1_block1 = ConvBlock(in_channels, 32, kernel_size=3)
        self.stage1_block2 = ConvBlock(32, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128x128 -> 64x64

        # Stage 2: Intermediate ridge-flow & recurvature (32 -> 64 channels)
        self.stage2_block1 = ConvBlock(32, 64, kernel_size=3)
        self.stage2_block2 = ConvBlock(64, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 -> 32x32

        # Stage 3: Core and Delta structural features (64 -> 128 channels)
        self.stage3_block1 = ConvBlock(64, 128, kernel_size=3)
        self.stage3_block2 = ConvBlock(128, 128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16

        # Stage 4: Global topology & high-level fingerprint geometry (128 -> 128 channels)
        self.stage4_block = ConvBlock(128, 128, kernel_size=3)
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))  # spatial 4x4

        # -------------------------------------------------------------
        # 2. Classification Head (Slide 8)
        # -------------------------------------------------------------
        self.flatten = nn.Flatten()
        feature_dim = 128 * 4 * 4  # 2048
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc_relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts convolutional feature maps through all stages."""
        x = self.stage1_block1(x)
        x = self.stage1_block2(x)
        x = self.pool1(x)

        x = self.stage2_block1(x)
        x = self.stage2_block2(x)
        x = self.pool2(x)

        x = self.stage3_block1(x)
        x = self.stage3_block2(x)
        x = self.pool3(x)

        x = self.stage4_block(x)
        x = self.global_pool(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass emitting raw logits."""
        feats = self.extract_features(x)
        flat = self.flatten(feats)
        dense = self.fc_relu(self.fc1(flat))
        dropped = self.dropout(dense)
        logits = self.fc2(dropped)
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Emits normalized softmax probability distribution over classes."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs

    def count_parameters(self) -> Dict[str, int]:
        """Calculates trainable and total network parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}

    def save_checkpoint(self, path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Serializes model weights and architecture configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "num_classes": self.num_classes,
            "classes": list(CLASSES),
            "metadata": metadata or {},
        }
        torch.save(checkpoint, str(path))

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path], device: str = "cpu") -> "FingerprintCNN":
        """Instantiates and loads model state from a saved checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        checkpoint = torch.load(str(path), map_location=device)
        model = cls(num_classes=checkpoint.get("num_classes", NUM_CLASSES))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
