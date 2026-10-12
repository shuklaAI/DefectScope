"""
UNIT 6 concepts: classification.

Two classifier paths, kept strictly separate and always labeled honestly in
every API response and UI screen:

1. DEMO MODE - rule_based_demo
   A small, fully transparent decision procedure over geometric features
   (aspect ratio, circularity, edge density, relative area), used with the
   synthetic demo images (backend/scripts/generate_demo_dataset.py). It
   reports a "heuristic_score" - explicitly NOT a probability, NOT a
   calibrated confidence, just a fixed number reflecting how strongly the
   rule fired. It answers "is there a candidate defect region, and roughly what
   shape is it" on images that may or may not contain a defect.

2. REAL ML MODE - random_forest_trained
   A scikit-learn RandomForestClassifier trained by backend/ml/train_classifier.py
   on the real NEU-CLS steel surface defect dataset (see backend/data/DATASET_INFO.md).
   NEU-CLS has no "normal / no-defect" class - every image is already a crop
   of one of six known defect textures - so this path answers a different
   question: "which of the six known defect categories does this image most
   resemble?", reported as "model_probability" using the classifier's real
   predict_proba() output. This is NOT calibrated confidence (no calibration
   step, e.g. Platt scaling, has been performed) and the API/UI must not
   call it that.

If Real ML Mode is requested but no trained model exists at
app.config.TRAINED_MODEL_PATH, this module raises ModelNotAvailableError
rather than silently falling back to a fake result - the API layer turns
this into a clear 400/503 error telling the user to run the training script.
"""
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from app.config import TRAINED_MODEL_PATH, SCALER_PATH, METADATA_PATH
from app.pipeline.features import FEATURE_NAMES, feature_vector

DEMO_DEFECT_LABELS = ["Scratch", "Crack", "Dent", "Contamination", "Hole", "Surface Irregularity"]


class ModelNotAvailableError(Exception):
    """Raised when Real ML Mode is requested but no trained model artifact exists."""
    pass


class FeatureContractError(Exception):
    """
    Raised when the feature names/order recorded in a saved model's
    metadata.json no longer match app.pipeline.features.FEATURE_NAMES -
    e.g. because the feature-extraction code was changed after the model
    was trained. Predicting anyway would silently feed the model a
    misaligned vector, so this is treated as a hard, actionable error
    rather than a place to fall back to a guess.
    """
    pass


_model_cache: Optional[dict] = None


def is_trained_model_available() -> bool:
    return TRAINED_MODEL_PATH.exists() and SCALER_PATH.exists() and METADATA_PATH.exists()


def load_model_bundle() -> dict:
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not is_trained_model_available():
        raise ModelNotAvailableError(
            "No trained model found. Run `python ml/train_classifier.py` from the backend "
            "directory first (see README.md / DATASET_INFO.md)."
        )
    model = joblib.load(TRAINED_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metadata = json.loads(Path(METADATA_PATH).read_text())

    saved_feature_names = metadata.get("feature_names")
    if saved_feature_names != FEATURE_NAMES:
        raise FeatureContractError(
            "The saved model's feature contract does not match the current feature-extraction "
            "code (app/pipeline/features.py::FEATURE_NAMES). This model was trained against a "
            "different feature set and must be retrained with `python ml/train_classifier.py` "
            "before Real ML Mode can be used safely.\n"
            f"Model expects ({len(saved_feature_names or [])}): {saved_feature_names}\n"
            f"Code currently produces ({len(FEATURE_NAMES)}): {FEATURE_NAMES}"
        )

    _model_cache = {"model": model, "scaler": scaler, "metadata": metadata}
    return _model_cache


def get_model_metadata() -> Optional[dict]:
    if not METADATA_PATH.exists():
        return None
    return json.loads(Path(METADATA_PATH).read_text())


def get_evaluation_results() -> Optional[dict]:
    from app.config import EVALUATION_PATH
    if not EVALUATION_PATH.exists():
        return None
    return json.loads(Path(EVALUATION_PATH).read_text())


# ---------------------------------------------------------------------------
# DEMO MODE: transparent rule-based classifier
# ---------------------------------------------------------------------------

def rule_based_classify(features: dict, image_area_px: int) -> tuple[bool, Optional[str], Optional[float]]:
    """Returns (is_defect, defect_type, heuristic_score). heuristic_score is
    explicitly a fixed heuristic weight, not a probability."""
    area = features.get("largest_area", 0.0)
    if area <= 0:
        return False, None, None

    area_percent = (area / image_area_px) * 100
    aspect_ratio = features.get("largest_aspect_ratio", 1.0)
    circularity = features.get("largest_circularity", 0.0)
    edge_density = features.get("edge_density_global", 0.0)

    if area_percent < 0.05:
        return False, None, None

    elongated = aspect_ratio > 3.0 or aspect_ratio < 0.33
    very_elongated = aspect_ratio > 6.0 or aspect_ratio < 0.166
    round_ish = circularity > 0.65

    if very_elongated and edge_density > 0.15:
        defect_type, score = "Crack", 0.75
    elif elongated:
        defect_type, score = "Scratch", 0.7
    elif round_ish and area_percent > 1.5:
        defect_type, score = "Dent", 0.65
    elif round_ish:
        defect_type, score = "Hole", 0.6
    elif edge_density > 0.25:
        defect_type, score = "Contamination", 0.6
    else:
        defect_type, score = "Surface Irregularity", 0.55

    return True, defect_type, score


def classify_demo(features: dict, image_area_px: int) -> dict:
    is_defect, defect_type, score = rule_based_classify(features, image_area_px)
    return {
        "predicted_class": defect_type,
        "is_defect": is_defect,
        "model_probability": None,
        "heuristic_score": score,
        "classifier_used": "rule_based_demo",
        "model_mode": "DEMO / SYNTHETIC DATA",
    }


# ---------------------------------------------------------------------------
# REAL ML MODE: trained RandomForestClassifier on NEU-CLS
# ---------------------------------------------------------------------------

def classify_real(features: dict) -> dict:
    bundle = load_model_bundle()
    clf = bundle["model"]
    scaler = bundle["scaler"]
    metadata = bundle["metadata"]
    label_map = {int(k): v for k, v in metadata["label_map"].items()}

    x_raw = feature_vector(features)  # fixed FEATURE_NAMES order
    x_scaled = scaler.transform(x_raw)

    pred = clf.predict(x_scaled)[0]
    proba = clf.predict_proba(x_scaled)[0]
    model_probability = float(np.max(proba))
    predicted_class = label_map[int(pred)]

    per_class_proba = {label_map[i]: round(float(p), 4) for i, p in enumerate(proba)}

    return {
        "predicted_class": predicted_class,
        "is_defect": True,  # every NEU-CLS image is a known defect crop by dataset design
        "model_probability": round(model_probability, 4),
        "heuristic_score": None,
        "classifier_used": "random_forest_trained",
        "model_mode": "TRAINED MODEL",
        "per_class_probability": per_class_proba,
        "model_version": metadata.get("model_version"),
    }


def classify(features: dict, image_area_px: int, mode: str = "demo") -> dict:
    if mode == "real":
        return classify_real(features)  # raises ModelNotAvailableError if no model - not caught here
    return classify_demo(features, image_area_px)
