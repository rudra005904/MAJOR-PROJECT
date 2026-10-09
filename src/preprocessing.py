"""Preprocessing pipeline for fingerprint images.

Fulfills Phase 2 and Slides 3, 4, 7 of the project requirements:
1. Grayscale conversion and fixed input resizing.
2. Noise reduction (Gaussian / bilateral filtering).
3. Ridge contrast enhancement via CLAHE.
4. Fingerprint region segmentation from background (Otsu threshold + morphology).
5. Pixel intensity normalization.
6. Ridge orientation and gradient feature map extraction.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.constants import DEFAULT_IMAGE_SIZE


def load_to_grayscale(
    image_input: Union[str, Path, bytes, np.ndarray, Image.Image]
) -> np.ndarray:
    """Loads an image from diverse source formats and converts it to a 2D uint8 grayscale array."""
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not decode image at path: {path}")
        return img

    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to decode image from provided byte buffer.")
        return img

    elif isinstance(image_input, Image.Image):
        # Convert PIL image to grayscale numpy array
        return np.array(image_input.convert("L"), dtype=np.uint8)

    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return image_input.astype(np.uint8)
        elif image_input.ndim == 3:
            if image_input.shape[2] == 1:
                return image_input.squeeze(axis=2).astype(np.uint8)
            elif image_input.shape[2] in (3, 4):
                # Convert BGR/RGB to grayscale
                return cv2.cvtColor(image_input[:, :, :3].astype(np.uint8), cv2.COLOR_BGR2GRAY)
        raise ValueError(f"Unsupported numpy image array shape: {image_input.shape}")

    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def resize_image(image: np.ndarray, target_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Resizes image to (target_size, target_size) using area/cubic interpolation."""
    if image.shape[0] == target_size and image.shape[1] == target_size:
        return image.copy()
    interpolation = cv2.INTER_AREA if (image.shape[0] > target_size) else cv2.INTER_CUBIC
    return cv2.resize(image, (target_size, target_size), interpolation=interpolation)


def reduce_noise(image: np.ndarray, method: str = "gaussian", ksize: int = 3) -> np.ndarray:
    """Applies noise filtering to attenuate acquisition sensor noise and speckles."""
    if method == "bilateral":
        # Edge-preserving bilateral filter
        return cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)
    elif method == "median":
        return cv2.medianBlur(image, ksize if ksize % 2 != 0 else ksize + 1)
    else:  # default gaussian
        k = ksize if ksize % 2 != 0 else ksize + 1
        return cv2.GaussianBlur(image, (k, k), sigmaX=0.8)


def enhance_ridge_contrast(
    image: np.ndarray, clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Enhances ridge-valley contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def segment_fingerprint(
    image: np.ndarray,
    threshold_method: str = "otsu",
    morph_kernel_size: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segments the fingerprint foreground region from noisy borders or blank background.

    Returns:
        (segmented_image, binary_mask) where background pixels in segmented_image are normalized.
    """
    # Fingerprints have dark ridges on light background (or vice-versa).
    # In standard biometric scans, background is light (high intensity), ridges are dark.
    # We apply Otsu's thresholding to distinguish the active fingerprint pattern.
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, inv_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing then opening to connect ridge clusters and remove isolated dust
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    closed = cv2.morphologyEx(inv_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find the largest connected component (the actual fingerprint pad)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    mask = np.zeros_like(image)
    if num_labels > 1:
        # Ignore background label 0, find largest component
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask[labels == largest_label] = 255
        # Fill holes in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    else:
        # Fallback if no distinct component found
        mask = np.ones_like(image) * 255

    # Composite: keep enhanced image inside mask, set background to 255 (neutral light)
    segmented = image.copy()
    segmented[mask == 0] = 255

    return segmented, mask


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    """Normalizes pixel values from uint8 [0, 255] to float32 [0.0, 1.0]."""
    return (image.astype(np.float32) / 255.0).clip(0.0, 1.0)


def extract_ridge_orientation_map(image: np.ndarray) -> np.ndarray:
    """Computes ridge orientation vector field and converts it to a visually interpretable RGB color map.

    Colors correspond to ridge angles (HSV cyclic map), highlighting core whorls and delta intersections.
    """
    # Calculate gradients along x and y
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Local ridge orientation: theta = 0.5 * atan2(2*gx*gy, gx^2 - gy^2) + pi/2
    gxx = gx * gx
    gyy = gy * gy
    gxy = gx * gy

    # Smooth the gradient tensors
    gxx_smooth = cv2.GaussianBlur(gxx, (5, 5), 1.0)
    gyy_smooth = cv2.GaussianBlur(gyy, (5, 5), 1.0)
    gxy_smooth = cv2.GaussianBlur(gxy, (5, 5), 1.0)

    theta = 0.5 * np.arctan2(2 * gxy_smooth, gxx_smooth - gyy_smooth) + np.pi / 2
    # Map theta [0, pi] to Hue [0, 180]
    hue = (np.degrees(theta) % 180).astype(np.uint8)

    # Coherence / gradient strength as saturation and value
    mag = np.sqrt(gxx_smooth + gyy_smooth)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    sat = np.ones_like(hue, dtype=np.uint8) * 200
    val = np.clip(mag_norm * 1.5, 40, 255).astype(np.uint8)

    hsv = cv2.merge([hue, sat, val])
    rgb_orientation = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb_orientation


def encode_image_base64(img_array: np.ndarray) -> str:
    """Converts a numpy array (uint8 2D grayscale or 3D RGB) to base64 PNG data URL."""
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)

    if img_array.ndim == 2:
        pil_img = Image.fromarray(img_array, mode="L")
    elif img_array.ndim == 3:
        pil_img = Image.fromarray(img_array, mode="RGB")
    else:
        raise ValueError(f"Cannot encode image with shape {img_array.shape}")

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


class PreprocessingPipeline:
    """Comprehensive, reproducible preprocessing pipeline implementing Phase 2."""

    def __init__(
        self,
        target_size: int = DEFAULT_IMAGE_SIZE,
        noise_filter: str = "gaussian",
        clahe_clip: float = 2.5,
        segment: bool = True,
    ):
        self.target_size = target_size
        self.noise_filter = noise_filter
        self.clahe_clip = clahe_clip
        self.segment = segment

    def process(
        self,
        image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
        return_intermediates: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """Executes the full preprocessing pipeline.

        Returns:
            - Normalized tensor array [1, target_size, target_size] in range [0.0, 1.0]
            - If return_intermediates=True, also returns a dictionary of base64 preview images and step metrics.
        """
        # Step 1: Ingestion & Grayscale
        raw_gray = load_to_grayscale(image_input)

        # Step 2: Resolution Standardization
        resized = resize_image(raw_gray, self.target_size)

        # Step 3: Noise Filtering
        denoised = reduce_noise(resized, method=self.noise_filter)

        # Step 4: Ridge Contrast Enhancement via CLAHE
        enhanced = enhance_ridge_contrast(denoised, clip_limit=self.clahe_clip)

        # Step 5: Region Segmentation
        if self.segment:
            segmented, mask = segment_fingerprint(enhanced)
        else:
            segmented = enhanced
            mask = np.ones_like(enhanced) * 255

        # Step 6: Normalization to [0.0, 1.0]
        normalized = normalize_intensity(segmented)
        # Format as (1, H, W) for PyTorch tensor compatibility
        tensor_array = np.expand_dims(normalized, axis=0)

        if not return_intermediates:
            return tensor_array

        # Compute Orientation Gradient Map for explainability
        orientation_map = extract_ridge_orientation_map(enhanced)

        intermediates = {
            "steps": {
                "original": encode_image_base64(resized),
                "denoised": encode_image_base64(denoised),
                "enhanced": encode_image_base64(enhanced),
                "segmented": encode_image_base64(segmented),
                "mask": encode_image_base64(mask),
                "orientation": encode_image_base64(orientation_map),
            },
            "meta": {
                "target_size": (self.target_size, self.target_size),
                "mean_intensity": float(np.mean(normalized)),
                "std_intensity": float(np.std(normalized)),
                "foreground_ratio": float(np.mean(mask > 0)),
            },
        }

        return tensor_array, intermediates
