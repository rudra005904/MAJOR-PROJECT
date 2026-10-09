"""Unit tests for the preprocessing pipeline (Slides 3, 4, 7)."""

import numpy as np
import pytest
from PIL import Image

from src.constants import DEFAULT_IMAGE_SIZE
from src.preprocessing import (
    PreprocessingPipeline,
    enhance_ridge_contrast,
    extract_ridge_orientation_map,
    load_to_grayscale,
    normalize_intensity,
    reduce_noise,
    resize_image,
    segment_fingerprint,
)


def test_load_to_grayscale_numpy_2d():
    arr = np.zeros((100, 100), dtype=np.uint8)
    gray = load_to_grayscale(arr)
    assert gray.shape == (100, 100)
    assert gray.dtype == np.uint8


def test_load_to_grayscale_numpy_rgb():
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, :, 0] = 255  # Red
    gray = load_to_grayscale(arr)
    assert gray.shape == (100, 100)
    assert gray.dtype == np.uint8


def test_load_to_grayscale_pil_image():
    pil_img = Image.new("RGB", (80, 80), color="blue")
    gray = load_to_grayscale(pil_img)
    assert gray.shape == (80, 80)
    assert gray.dtype == np.uint8


def test_load_to_grayscale_invalid_type():
    with pytest.raises(TypeError):
        load_to_grayscale(12345)


def test_load_to_grayscale_nonexistent_path():
    with pytest.raises(FileNotFoundError):
        load_to_grayscale("nonexistent_path_xyz_123.png")


def test_resize_image():
    img = np.ones((64, 64), dtype=np.uint8) * 120
    resized = resize_image(img, target_size=128)
    assert resized.shape == (128, 128)

    same_size = resize_image(img, target_size=64)
    assert same_size.shape == (64, 64)


def test_reduce_noise():
    img = (np.random.rand(64, 64) * 255).astype(np.uint8)
    denoised_gauss = reduce_noise(img, method="gaussian")
    assert denoised_gauss.shape == (64, 64)

    denoised_bilateral = reduce_noise(img, method="bilateral")
    assert denoised_bilateral.shape == (64, 64)

    denoised_median = reduce_noise(img, method="median")
    assert denoised_median.shape == (64, 64)


def test_enhance_ridge_contrast():
    img = (np.random.rand(64, 64) * 100 + 50).astype(np.uint8)
    enhanced = enhance_ridge_contrast(img)
    assert enhanced.shape == (64, 64)
    assert enhanced.dtype == np.uint8


def test_segment_fingerprint():
    # Create synthetic circle (foreground) on white background
    img = np.ones((100, 100), dtype=np.uint8) * 240
    img[30:70, 30:70] = 40  # dark patch
    segmented, mask = segment_fingerprint(img)
    assert segmented.shape == (100, 100)
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


def test_normalize_intensity():
    img = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    norm = normalize_intensity(img)
    assert norm.shape == (2, 2)
    assert norm.dtype == np.float32
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0


def test_extract_ridge_orientation_map():
    img = np.zeros((64, 64), dtype=np.uint8)
    img[::4, :] = 255  # horizontal lines
    orient = extract_ridge_orientation_map(img)
    assert orient.shape == (64, 64, 3)
    assert orient.dtype == np.uint8


def test_preprocessing_pipeline_full():
    pipeline = PreprocessingPipeline(target_size=128)
    img = (np.random.rand(90, 110) * 255).astype(np.uint8)

    # Without intermediates
    tensor = pipeline.process(img, return_intermediates=False)
    assert tensor.shape == (1, 128, 128)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() <= tensor.max() <= 1.0

    # With intermediates
    tensor, info = pipeline.process(img, return_intermediates=True)
    assert tensor.shape == (1, 128, 128)
    assert "steps" in info
    assert "original" in info["steps"]
    assert "denoised" in info["steps"]
    assert "enhanced" in info["steps"]
    assert "segmented" in info["steps"]
    assert "mask" in info["steps"]
    assert "orientation" in info["steps"]
    assert info["steps"]["original"].startswith("data:image/png;base64,")
