"""
UNIT 1 concepts: digital image fundamentals, image representation,
sampling/quantization, grayscale vs color images.

This module handles loading, validation and the first two pipeline stages:
resize/normalization and grayscale conversion.
"""
import cv2
import numpy as np

from app.config import PROCESSING_SIZE


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode raw uploaded bytes into a BGR uint8 array (OpenCV convention)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. File may be corrupted or in an unsupported format.")
    return img


def resize_and_normalize(img: np.ndarray, size=PROCESSING_SIZE) -> np.ndarray:
    """
    Resample the image to a fixed working resolution (demonstrates the
    sampling concept from Unit 1: changing the spatial resolution/grid the
    image is represented on) and normalize pixel intensities to the full
    0-255 uint8 range (quantization stays at 8 bits/channel, the standard
    representation used throughout the rest of the pipeline).
    """
    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    normalized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert the 3-channel color image to a single-channel grayscale image
    using the luminosity-weighted conversion (ITU-R BT.601: 0.299R + 0.587G
    + 0.114B, as implemented by cv2.COLOR_BGR2GRAY). Most defect-detection
    features here (edges, texture, thresholds) are computed on intensity
    alone, since surface defects are primarily structural, not chromatic.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def color_statistics(img_bgr: np.ndarray) -> dict:
    """Per-channel mean/std, used both for the pipeline explanation and as
    optional color-based features (Unit 3: color image processing)."""
    means = img_bgr.reshape(-1, 3).mean(axis=0)
    stds = img_bgr.reshape(-1, 3).std(axis=0)
    return {
        "mean_b": float(means[0]), "mean_g": float(means[1]), "mean_r": float(means[2]),
        "std_b": float(stds[0]), "std_g": float(stds[1]), "std_r": float(stds[2]),
    }
