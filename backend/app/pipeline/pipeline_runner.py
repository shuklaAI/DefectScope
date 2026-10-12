"""
Orchestrates the full DIP pipeline end-to-end, saving each intermediate
stage image to disk and returning a structured trace consumed by the API
layer (app/schemas.py: StageResult) and rendered stage-by-stage on the
frontend's Pipeline page.

Stages 1-9 (preprocessing through connected components) are IDENTICAL
regardless of mode - this is deliberate: the same DIP pipeline that trains
the real model also runs at inference time (see backend/ml/train_classifier.py,
which calls the same app.pipeline.* functions in the same order). Only the
final candidate-region/classification step differs between:

- mode="demo": defect-vs-no-defect + rule-based classifier, run against
  synthetic images (backend/scripts/generate_demo_dataset.py).
- mode="real": whole-image + regional feature vector fed to the trained
  RandomForestClassifier on the real NEU-CLS dataset. Raises
  classifier.ModelNotAvailableError if no trained model exists - the API
  layer converts this into a clear error response, never a fake result.
"""
import time
import uuid

import cv2
import numpy as np

from app.config import RESULTS_DIR, PROCESSING_SIZE
from app.pipeline import preprocessing, filtering, edge_detection, segmentation, morphology, features, classifier


def _save(img: np.ndarray, run_id: str, stage_key: str) -> str:
    """Persists a stage image and returns its relative URL path (served
    statically by FastAPI under /inspection-assets/...)."""
    filename = f"{run_id}_{stage_key}.png"
    path = RESULTS_DIR / filename
    cv2.imwrite(str(path), img)
    return f"/inspection-assets/{filename}"


def run_pipeline(image_bytes: bytes, mode: str = "demo") -> dict:
    """
    mode: "demo" (synthetic data, rule-based classifier, defect/no-defect +
    candidate-region detection) or "real" (trained RandomForest on the real NEU-CLS
    dataset, multi-class prediction). See module docstring.
    """
    if mode not in ("demo", "real"):
        raise ValueError(f"Unknown mode '{mode}'. Expected 'demo' or 'real'.")

    start = time.perf_counter()
    run_id = uuid.uuid4().hex[:12]
    stages = []

    # STEP 1: original image
    original = preprocessing.load_image_from_bytes(image_bytes)
    original_url = _save(original, run_id, "01_original")
    stages.append({
        "step": 1, "key": "original", "title": "Original Image",
        "technique": "Raw upload (digital image representation)",
        "explanation": "The uploaded image as a discrete 2D grid of pixels, each an 8-bit "
                        "quantized intensity per color channel (Unit 1: image representation, "
                        "sampling & quantization).",
        "parameters": {"width": original.shape[1], "height": original.shape[0], "channels": original.shape[2]},
        "image_url": original_url,
    })

    # STEP 2: resize/normalize
    resized = preprocessing.resize_and_normalize(original, PROCESSING_SIZE)
    resized_url = _save(resized, run_id, "02_resized")
    stages.append({
        "step": 2, "key": "resized", "title": "Resize & Normalization",
        "technique": "Area resampling + min-max normalization",
        "explanation": "Resampled to a fixed working resolution so downstream kernel sizes and "
                        "thresholds behave consistently, then intensities stretched to the full "
                        "0-255 range. This is the exact same resizing used when the model was "
                        "trained (see ml/train_classifier.py), so inference and training see "
                        "consistently scaled inputs.",
        "parameters": {"target_size": f"{PROCESSING_SIZE[0]}x{PROCESSING_SIZE[1]}"},
        "image_url": resized_url,
    })

    # STEP 3: grayscale
    gray = preprocessing.to_grayscale(resized)
    gray_url = _save(gray, run_id, "03_grayscale")
    color_stats = preprocessing.color_statistics(resized)
    stages.append({
        "step": 3, "key": "grayscale", "title": "Grayscale Conversion",
        "technique": "Luminosity-weighted conversion (ITU-R BT.601)",
        "explanation": "Converted to single-channel intensity, since surface defects are "
                        "primarily structural/textural rather than color-based.",
        "parameters": {k: round(v, 2) for k, v in color_stats.items()},
        "image_url": gray_url,
    })

    # STEP 4: histogram of raw grayscale (for display/comparison)
    hist_before = filtering.compute_histogram(gray)

    # STEP 5: noise reduction
    denoised = filtering.denoise(gray, method="median")
    denoised_url = _save(denoised, run_id, "04_denoised")
    stages.append({
        "step": 4, "key": "denoised", "title": "Noise Reduction",
        "technique": "Median filtering (5x5 kernel)",
        "explanation": "Suppresses impulse (salt-and-pepper) sensor noise while better preserving "
                        "edges than an equivalent mean/Gaussian filter would.",
        "parameters": {"kernel_size": 5, "method": "median"},
        "image_url": denoised_url,
    })

    # STEP 6: contrast enhancement + histogram after
    enhanced = filtering.enhance_contrast(denoised, method="clahe")
    enhanced_url = _save(enhanced, run_id, "05_enhanced")
    hist_after = filtering.compute_histogram(enhanced)
    stages.append({
        "step": 5, "key": "enhanced", "title": "Contrast Enhancement",
        "technique": "CLAHE (Contrast Limited Adaptive Histogram Equalization)",
        "explanation": "Boosts local contrast in tiles (8x8) with a clip limit to avoid "
                        "over-amplifying noise, making faint defects/texture patterns easier to "
                        "separate before edge/threshold operations and before feature extraction.",
        "parameters": {"clip_limit": 3.0, "tile_grid_size": "8x8"},
        "image_url": enhanced_url,
        "histogram": {"before": hist_before["counts"], "after": hist_after["counts"]},
    })

    # STEP 7: edge detection
    edges = edge_detection.canny_edges(enhanced)
    edges_url = _save(edges, run_id, "06_edges")
    stages.append({
        "step": 6, "key": "edges", "title": "Edge Detection",
        "technique": "Canny edge detector",
        "explanation": "Applies Gaussian smoothing, gradient computation, non-maximum "
                        "suppression, then hysteresis thresholding to produce thin, connected "
                        "edge boundaries, used both for candidate-region segmentation and as a global "
                        "edge-density feature.",
        "parameters": {"low_threshold": 50, "high_threshold": 150},
        "image_url": edges_url,
    })

    # STEP 8: thresholding / segmentation
    seg_mask, otsu_val = segmentation.segment(enhanced, edges)
    seg_url = _save(seg_mask, run_id, "07_segmented")
    stages.append({
        "step": 7, "key": "segmented", "title": "Thresholding / Segmentation",
        "technique": "Otsu global thresholding combined with dilated Canny edges",
        "explanation": "Otsu automatically selects the intensity threshold that best separates "
                        "foreground from background from the image histogram; requiring agreement "
                        "with edge information avoids flagging noise-only regions.",
        "parameters": {"otsu_threshold_value": round(otsu_val, 1)},
        "image_url": seg_url,
    })

    # STEP 9: morphological processing
    cleaned = morphology.clean_mask(seg_mask, kernel_size=3)
    filled = morphology.fill_holes(cleaned)
    morph_url = _save(filled, run_id, "08_morphology")
    stages.append({
        "step": 8, "key": "morphology", "title": "Morphological Processing",
        "technique": "Opening -> Closing -> Hole filling (elliptical 3x3 kernel)",
        "explanation": "Opening (erosion then dilation) removes small noise specks left by "
                        "thresholding; closing (dilation then erosion) reconnects fragments of "
                        "the same defect region; hole filling closes small enclosed gaps. A 3x3 "
                        "kernel is used (rather than larger) so thin scratch-like defects are not "
                        "eroded away.",
        "parameters": {"kernel_size": 3, "kernel_shape": "ellipse"},
        "image_url": morph_url,
    })

    # STEP 10: connected components (multiple components are ALL kept, not just the largest)
    labels, components = morphology.connected_components(filled, min_area=25, max_area_fraction=0.4)
    label_vis = np.zeros((*filled.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    colors = rng.integers(50, 255, size=(max(len(components), 1) + 1, 3))
    for idx, comp in enumerate(components, start=1):
        label_vis[labels == comp["label"]] = colors[idx]
    cc_url = _save(label_vis, run_id, "09_components")
    stages.append({
        "step": 9, "key": "components", "title": "Connected Component Analysis",
        "technique": "8-connectivity labeling (cv2.connectedComponentsWithStats)",
        "explanation": "Groups foreground pixels into distinct blobs, discards components below "
                        "the minimum area (residual noise) or above the maximum area fraction "
                        "(likely a global segmentation failure), and keeps every remaining "
                        "component as a candidate defect region - not only the largest one.",
        "parameters": {"connectivity": 8, "min_area_px": 25, "max_area_fraction": 0.4,
                        "components_found": len(components)},
        "image_url": cc_url,
    })

    # STEP 10: candidate defect region detection (ALL surviving components boxed, not just one)
    #
    # IMPORTANT: NEU-CLS provides image-level classification labels, not
    # pixel-level or bounding-box ground truth. The boxes drawn here come
    # entirely from the classical DIP segmentation above (Otsu + edge
    # agreement + morphology + connected components) and have NOT been
    # verified against any ground-truth defect boundary. They are shown as
    # "candidate regions" - useful for demonstrating the DIP pipeline and
    # for the heuristic demo classifier's shape features - not as a claim
    # of verified defect boundaries.
    localized = resized.copy()
    for comp in components:
        x, y, w, h = comp["x"], comp["y"], comp["w"], comp["h"]
        cv2.rectangle(localized, (x, y), (x + w, y + h), (0, 0, 255), 2)
    loc_url = _save(localized, run_id, "10_localized")
    total_area = sum(c["area"] for c in components)
    image_area = PROCESSING_SIZE[0] * PROCESSING_SIZE[1]
    defect_area_percent = round((total_area / image_area) * 100, 2)
    largest_area = max((c["area"] for c in components), default=0)
    stages.append({
        "step": 10, "key": "localized", "title": "Candidate Defect Regions (DIP Segmentation)",
        "technique": "Bounding boxes for every surviving connected component",
        "explanation": "Candidate regions are generated by the Digital Image Processing pipeline "
                        "(Otsu thresholding + edge agreement + morphology + connected components). "
                        "The NEU-CLS dataset does not provide ground-truth segmentation masks, so "
                        "these regions should not be interpreted as verified defect boundaries - "
                        "they demonstrate the DIP segmentation pipeline and feed the demo "
                        "classifier's shape features, nothing more.",
        "parameters": {
            "regions_found": len(components),
            "total_candidate_area_percent": defect_area_percent,
            "largest_region_area_px": int(largest_area),
        },
        "image_url": loc_url,
    })

    # STEP 11: feature extraction (global texture + regional geometry - identical to training)
    feat = features.extract_features(enhanced, filled, components)

    # STEP 12: classification (mode-dependent; may raise ModelNotAvailableError for mode="real")
    cls_result = classifier.classify(feat, image_area, mode=mode)

    # Final highlighted result image
    highlighted = resized.copy()
    if components:
        largest = max(components, key=lambda c: c["area"])
        x, y, w, h = largest["x"], largest["y"], largest["w"], largest["h"]
        cv2.rectangle(highlighted, (x, y), (x + w, y + h), (0, 0, 255), 3)
        label_text = cls_result["predicted_class"] or ""
        if label_text:
            cv2.putText(highlighted, label_text, (x, max(y - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    highlighted_url = _save(highlighted, run_id, "11_highlighted")

    if cls_result["classifier_used"] == "random_forest_trained":
        technique = f"scikit-learn RandomForestClassifier v{cls_result.get('model_version', '1')} (trained on NEU-CLS)"
        explanation = (
            "The 23-value feature vector (global texture + regional geometry) was scaled with the "
            "training-set StandardScaler and passed to the trained RandomForest. Reported value is "
            "the model's real predict_proba() output for its top class, labeled 'model probability' "
            "- this is not a calibrated confidence score."
        )
    else:
        technique = "Rule-based demo classifier (thresholds on geometric features)"
        explanation = ("No trained model was requested/available, so a transparent rule-based "
                        "classifier maps region shape/edge features to a likely defect category. "
                        "This is explicitly a demo heuristic, not a validated ML result.")

    stages.append({
        "step": 11, "key": "classification", "title": "Classification",
        "technique": technique,
        "explanation": explanation,
        "parameters": feat,
        "image_url": highlighted_url,
    })

    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "run_id": run_id,
        "mode": mode,
        "original_image_url": original_url,
        "highlighted_image_url": highlighted_url,
        "stages": stages,
        "features": feat,
        "num_regions": len(components),
        "defect_area_percent": defect_area_percent,
        "candidate_regions_disclaimer": (
            "Candidate regions are generated by the Digital Image Processing pipeline. The "
            "NEU-CLS dataset does not provide ground-truth segmentation masks, so these regions "
            "should not be interpreted as verified defect boundaries."
        ),
        "predicted_class": cls_result["predicted_class"],
        "classifier_used": cls_result["classifier_used"],
        "model_mode": cls_result["model_mode"],
        "model_probability": cls_result.get("model_probability"),
        "heuristic_score": cls_result.get("heuristic_score"),
        "per_class_probability": cls_result.get("per_class_probability"),
        "model_version": cls_result.get("model_version"),
        "result": ("DEFECT_DETECTED" if (mode == "demo" and cls_result["is_defect"]) else
                   "NO_DEFECT" if mode == "demo" else "CLASSIFIED"),
        "processing_time_ms": round(elapsed_ms, 2),
    }
