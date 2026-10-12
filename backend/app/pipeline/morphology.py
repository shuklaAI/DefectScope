"""
UNIT 4 concepts: morphological image processing (erosion, dilation,
opening, closing), hole filling, connected components,
thinning/skeletonization.
"""
import cv2
import numpy as np


def clean_mask(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Opening (erosion -> dilation) removes small isolated noise specks left
    over from thresholding without significantly shrinking real defect
    regions. Closing (dilation -> erosion) afterward fills small internal
    gaps so a single defect isn't fragmented into several components.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fills enclosed background holes inside foreground blobs (e.g. the
    center of a ring-shaped contamination mark) via flood fill from the
    border, so area measurements reflect the full defect region."""
    h, w = binary.shape
    flood = binary.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return binary | flood_inv


def connected_components(binary: np.ndarray, min_area: int = 25, max_area_fraction: float = 0.4):
    """
    Labels each 8-connected blob in the mask (Unit 4) and filters out
    components below `min_area` pixels, which discards residual noise
    specks that survived morphological cleaning. Returns the label map and
    a list of stats dicts per surviving component.

    Components larger than `max_area_fraction` of the total frame are also
    discarded: a genuine isolated surface defect should occupy a small
    fraction of the inspected surface, so a blob spanning most of the
    frame is far more likely to be a global segmentation failure (e.g.
    illumination artifacts or noise amplified by contrast enhancement)
    than an actual defect, and is treated as inconclusive rather than a
    false positive.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    total_px = binary.shape[0] * binary.shape[1]
    max_area = total_px * max_area_fraction
    components = []
    for i in range(1, num_labels):  # skip background label 0
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        components.append({
            "label": i,
            "x": int(stats[i, cv2.CC_STAT_LEFT]),
            "y": int(stats[i, cv2.CC_STAT_TOP]),
            "w": int(stats[i, cv2.CC_STAT_WIDTH]),
            "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
            "area": area,
            "centroid": (float(centroids[i][0]), float(centroids[i][1])),
        })
    return labels, components


def skeletonize(binary: np.ndarray) -> np.ndarray:
    """
    Thinning/skeletonization reduces a blob to a 1px-wide medial skeleton -
    useful for characterizing scratch/crack shapes (long and thin vs
    round). Implemented as a morphological skeleton: repeatedly erode the
    mask and accumulate the difference between each eroded image and its
    opening (OpenCV erode/dilate, no scipy involved).
    """
    img = (binary > 0).astype(np.uint8)
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = img.copy()
    while True:
        eroded = cv2.erode(temp, element)
        opened = cv2.dilate(eroded, element)
        subset = cv2.subtract(temp, opened)
        skel = cv2.bitwise_or(skel, subset)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break
    return skel * 255
