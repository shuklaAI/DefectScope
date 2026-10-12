"""
UNIT 6 concepts: feature extraction ahead of classification.

This module is the single source of truth for feature extraction and is
imported by BOTH the live inference pipeline (app/pipeline/pipeline_runner.py)
and the offline training script (backend/ml/train_classifier.py). Using the
exact same function in both places is what guarantees the trained model's
inputs match what it sees at inference time - a common and serious bug in
ML projects is training on one feature computation and predicting with a
subtly different one.

Two feature groups are computed and merged into one flat, fixed-order
vector (FEATURE_NAMES):

1. GLOBAL TEXTURE / STATISTICAL features - computed over the whole enhanced
   grayscale image. They are well suited to a dataset like NEU-CLS where the
   entire image is already a defect crop (there is no separate "background"
   region to segment away) and the defect signature is a texture pattern
   covering much of the frame. (For the shipped model, these 11 features
   account for most of the RandomForest's impurity-based feature importance;
   that is a descriptive property of that model, not an ablation study.)
2. REGIONAL / GEOMETRIC features - computed from the segmented,
   morphologically-cleaned connected components (candidate defect regions
   produced by DIP segmentation; NEU-CLS has no ground-truth masks, so these
   are not verified defect boundaries). The rule-based Demo Mode classifier
   relies on these shape descriptors.

Both groups are always computed and always present in the vector so a
single model/feature contract works for both Demo Mode and Real ML Mode.
"""
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

# Fixed, ordered feature names. Both training (backend/ml/train_classifier.py)
# and inference (classifier.py) must build/read vectors in exactly this order.
FEATURE_NAMES = [
    # --- global statistical ---
    "mean_intensity", "std_intensity", "min_intensity", "max_intensity",
    # --- global texture ---
    "entropy", "local_variance_mean", "edge_density_global",
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
    # --- regional / geometric (largest component) ---
    "largest_area", "largest_perimeter", "largest_aspect_ratio",
    "largest_extent", "largest_solidity", "largest_circularity", "largest_eccentricity",
    # --- aggregate across all surviving components ---
    "num_components", "total_defect_area", "avg_component_area",
    "total_defect_percent", "contour_count",
]


def _glcm_features(gray: np.ndarray) -> dict:
    """
    Gray-Level Co-occurrence Matrix texture descriptors (Unit 6 / classical
    texture analysis): contrast, homogeneity, energy and correlation,
    computed at a single pixel offset/angle and averaged, which is a
    standard lightweight GLCM configuration for defect-texture
    discrimination without excessive computation cost.
    """
    quantized = (gray / 32).astype(np.uint8)  # 8 gray levels keeps the GLCM small & fast
    glcm = graycomatrix(quantized, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                         levels=8, symmetric=True, normed=True)
    return {
        "glcm_contrast": float(np.mean(graycoprops(glcm, "contrast"))),
        "glcm_homogeneity": float(np.mean(graycoprops(glcm, "homogeneity"))),
        "glcm_energy": float(np.mean(graycoprops(glcm, "energy"))),
        "glcm_correlation": float(np.mean(np.nan_to_num(graycoprops(glcm, "correlation")))),
    }


def _local_variance_mean(gray: np.ndarray, ksize: int = 9) -> float:
    """Mean of a local-variance map (box-filter of squared deviation from a
    local mean), a simple, fast texture-roughness indicator."""
    gray_f = gray.astype(np.float64)
    mean = cv2.blur(gray_f, (ksize, ksize))
    sq_mean = cv2.blur(gray_f ** 2, (ksize, ksize))
    variance = np.clip(sq_mean - mean ** 2, 0, None)
    return float(np.mean(variance))


def extract_global_features(gray: np.ndarray) -> dict:
    """Statistical + texture features computed over the entire (enhanced)
    grayscale frame - see the module docstring for why they suit a
    whole-image texture-classification dataset like NEU-CLS."""
    edges = cv2.Canny(gray, 50, 150)
    feats = {
        "mean_intensity": round(float(np.mean(gray)), 3),
        "std_intensity": round(float(np.std(gray)), 3),
        "min_intensity": float(np.min(gray)),
        "max_intensity": float(np.max(gray)),
        "entropy": round(float(shannon_entropy(gray)), 4),
        "local_variance_mean": round(_local_variance_mean(gray), 3),
        "edge_density_global": round(float(np.mean(edges > 0)), 4),
    }
    feats.update({k: round(v, 4) for k, v in _glcm_features(gray).items()})
    return feats


def extract_regional_features(mask: np.ndarray, components: list) -> dict:
    """
    Geometric/shape features from the largest surviving connected component,
    plus aggregate statistics across ALL surviving components (not just the
    largest) - so multiple simultaneous defect regions contribute to the
    feature vector rather than being silently discarded.
    """
    total_px = mask.shape[0] * mask.shape[1]

    if not components:
        return {
            "largest_area": 0.0, "largest_perimeter": 0.0, "largest_aspect_ratio": 0.0,
            "largest_extent": 0.0, "largest_solidity": 0.0, "largest_circularity": 0.0,
            "largest_eccentricity": 0.0, "num_components": 0, "total_defect_area": 0.0,
            "avg_component_area": 0.0, "total_defect_percent": 0.0, "contour_count": 0,
        }

    largest = max(components, key=lambda c: c["area"])
    x, y, w, h = largest["x"], largest["y"], largest["w"], largest["h"]
    region_mask = (mask[y:y + h, x:x + w] > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    perimeter, aspect_ratio, extent, solidity, circularity, eccentricity = (0.0,) * 6
    area = float(largest["area"])
    aspect_ratio = float(w) / float(h) if h > 0 else 0.0
    extent = area / float(w * h) if w * h > 0 else 0.0

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(cnt, True))
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(min(area / hull_area, 1.0)) if hull_area > 0 else 0.0
        circularity = float(min(4 * np.pi * area / (perimeter ** 2), 1.0)) if perimeter > 0 else 0.0
        if len(cnt) >= 5:
            (_, _), (ma, mi), _ = cv2.fitEllipse(cnt)
            major, minor = max(ma, mi), min(ma, mi)
            eccentricity = float(np.sqrt(1 - (minor / major) ** 2)) if major > 0 else 0.0

    all_mask_u8 = (mask > 0).astype(np.uint8) * 255
    all_contours, _ = cv2.findContours(all_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = float(sum(c["area"] for c in components))
    avg_area = total_area / len(components) if components else 0.0

    return {
        "largest_area": area,
        "largest_perimeter": round(perimeter, 2),
        "largest_aspect_ratio": round(aspect_ratio, 3),
        "largest_extent": round(extent, 3),
        "largest_solidity": round(solidity, 3),
        "largest_circularity": round(circularity, 3),
        "largest_eccentricity": round(eccentricity, 3),
        "num_components": len(components),
        "total_defect_area": total_area,
        "avg_component_area": round(avg_area, 2),
        "total_defect_percent": round((total_area / total_px) * 100, 3),
        "contour_count": len(all_contours),
    }


def extract_features(gray: np.ndarray, mask: np.ndarray, components: list) -> dict:
    """Canonical, order-stable feature dict used by both training and
    inference. Always returns every key in FEATURE_NAMES, with NaN/Inf
    values defensively replaced by 0.0 (e.g. a fully uniform input image
    can otherwise drive a divide-by-zero inside a texture statistic)."""
    feats = {}
    feats.update(extract_global_features(gray))
    feats.update(extract_regional_features(mask, components))
    result = {}
    for name in FEATURE_NAMES:
        value = feats.get(name, 0.0)
        if value is None or not np.isfinite(value):
            value = 0.0
        result[name] = value
    return result


def feature_vector(features: dict) -> np.ndarray:
    """Flattens a feature dict into a fixed-order numpy row, in FEATURE_NAMES
    order, defensively sanitizing any NaN/Inf to 0.0."""
    row = []
    for name in FEATURE_NAMES:
        value = features.get(name, 0.0)
        if value is None or not np.isfinite(value):
            value = 0.0
        row.append(value)
    return np.array([row], dtype=np.float64)


# --- Backward-compatible name used by the older demo rule-based classifier ---
def extract_region_features(gray: np.ndarray, mask: np.ndarray, components: list) -> dict:
    """Deprecated alias kept for the rule-based demo classifier, which only
    needs the regional subset. New code should use extract_features()."""
    return extract_regional_features(mask, components)
