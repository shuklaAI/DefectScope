"""
UNIT 5 concepts: point/line/edge detection, Hough transform.
"""
import cv2
import numpy as np


def sobel_edges(gray: np.ndarray) -> np.ndarray:
    """First-derivative (gradient) edge operator. Sensitive to gradient
    magnitude in x and y independently - useful for showing directional
    edge response in the viva."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = np.clip((mag / (mag.max() + 1e-6)) * 255, 0, 255)
    return mag.astype(np.uint8)


def canny_edges(gray: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """
    Canny is used as the primary edge detector: it applies Gaussian
    smoothing, computes intensity gradients, performs non-maximum
    suppression to thin edges, then hysteresis thresholding (with `low`
    and `high`) to link strong edges through weak ones - producing thin,
    well-connected boundaries that are far more useful for contour
    extraction than the raw Sobel gradient magnitude.
    """
    return cv2.Canny(gray, low, high)


def detect_lines_hough(edges: np.ndarray):
    """
    Probabilistic Hough transform, used to demonstrate line/boundary
    detection (Unit 5) - useful for elongated defects like scratches which
    are approximately linear. Returns line segment endpoints.
    """
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=25, maxLineGap=8)
    if lines is None:
        return []
    return [tuple(l[0]) for l in lines]
