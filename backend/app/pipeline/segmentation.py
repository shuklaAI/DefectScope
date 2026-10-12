"""
UNIT 5 concepts: image segmentation, thresholding (global/adaptive/Otsu),
region-based segmentation.
"""
import cv2
import numpy as np


def otsu_threshold(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Otsu's method picks a single global threshold that minimizes
    intra-class intensity variance between foreground and background,
    computed directly from the image histogram - well suited to industrial
    surfaces with a fairly uniform background and a distinct defect
    intensity band.
    """
    thresh_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, float(thresh_val)


def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive (local) mean thresholding: the threshold is recomputed per
    neighborhood, which handles uneven illumination across a surface far
    better than a single global threshold.
    """
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize=25, C=5
    )


def segment(gray_enhanced: np.ndarray, edges: np.ndarray, method: str = "otsu_edges") -> tuple[np.ndarray, float]:
    """
    Combines edge information with intensity thresholding: a defect region
    is treated as a candidate only where it is BOTH an intensity outlier
    (Otsu on the enhanced grayscale image) AND sits on a strong, locally
    dense edge cluster (dilated Canny edges). Requiring agreement between
    the two cues (rather than either one alone) makes the mask far more
    robust to background sensor noise, which after CLAHE enhancement can
    otherwise register as scattered false-positive edges/intensity
    outliers on their own.
    """
    otsu_mask, thresh_val = otsu_threshold(gray_enhanced)

    # A clean/background-dominated Otsu split can come out with the
    # "object" class being the majority of the image; invert so the mask
    # always represents the minority (anomalous) region.
    if np.mean(otsu_mask) > 127:
        otsu_mask = cv2.bitwise_not(otsu_mask)

    edges_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    combined = cv2.bitwise_and(otsu_mask, edges_dilated)
    return combined, thresh_val
