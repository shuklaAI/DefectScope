"""
UNIT 2 concepts: histogram processing, histogram equalization/CLAHE,
spatial filtering, low-pass filtering, sharpening.
UNIT 3 concepts: noise models, noise reduction, MMSE/Wiener-style
restoration.
"""
import cv2
import numpy as np
from scipy.signal import wiener


def compute_histogram(gray: np.ndarray, bins: int = 256) -> dict:
    hist, edges = np.histogram(gray, bins=bins, range=(0, 256))
    return {"counts": hist.astype(float).tolist(), "bin_edges": edges.astype(float).tolist()}


def denoise(gray: np.ndarray, method: str = "median") -> np.ndarray:
    """
    Removes sensor/acquisition noise before structural analysis. Real
    industrial camera feeds commonly show salt-and-pepper noise (impulse
    noise from sensor defects/dust) and Gaussian noise (electronic/thermal
    noise) - median filtering suppresses the former without blurring edges
    as much as a mean filter would, while Gaussian blur suppresses the
    latter. Wiener filtering is offered as an MMSE-style restoration
    alternative (Unit 3) that adapts to local variance.
    """
    if method == "median":
        return cv2.medianBlur(gray, 5)
    if method == "gaussian":
        return cv2.GaussianBlur(gray, (5, 5), sigmaX=1.0)
    if method == "wiener":
        restored = wiener(gray.astype(np.float64), (5, 5))
        restored = np.nan_to_num(restored, nan=0.0)
        return np.clip(restored, 0, 255).astype(np.uint8)
    raise ValueError(f"Unknown denoise method: {method}")


def enhance_contrast(gray: np.ndarray, method: str = "clahe") -> np.ndarray:
    """
    Improves local contrast so faint defects (hairline scratches, shallow
    dents) become visible before edge/threshold operations. Plain global
    histogram equalization can over-amplify noise/background texture on
    industrial surfaces, so CLAHE (Contrast Limited Adaptive Histogram
    Equalization) is used by default: it equalizes in small tiles with a
    clip limit that caps noise amplification.
    """
    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    if method == "global_eq":
        return cv2.equalizeHist(gray)
    raise ValueError(f"Unknown enhancement method: {method}")


def sharpen(gray: np.ndarray) -> np.ndarray:
    """High-pass style sharpening via an unsharp mask (Unit 2)."""
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
