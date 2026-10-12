"""
Genuine ML training pipeline for Real ML Mode, run against the real NEU-CLS
steel surface defect dataset (backend/data/raw/NEU-CLS/, see
backend/data/DATASET_INFO.md for provenance and class list).

Workflow (matches the assignment's required pipeline exactly):

    Dataset -> Preprocessing -> DIP feature extraction -> Train/test split
    -> Feature scaling -> Random Forest -> Evaluation -> Saved model
    -> Inference through FastAPI (app/pipeline/classifier.py loads what
       this script saves)

Every preprocessing/feature-extraction call here uses the SAME functions
(app.pipeline.preprocessing/filtering/edge_detection/segmentation/morphology/
features) as the live inference pipeline (app/pipeline/pipeline_runner.py),
in the same order, so there is no train/inference skew.

Usage (from the backend/ directory):
    python ml/train_classifier.py
    python ml/train_classifier.py --data-dir data/raw/NEU-CLS --test-size 0.2

Outputs (all real, nothing invented):
    models/defect_classifier.joblib   - trained RandomForestClassifier
    models/scaler.joblib              - StandardScaler fit on the TRAIN split only
    models/metadata.json              - model/dataset/version metadata
    models/evaluation.json            - accuracy/precision/recall/F1/confusion matrix
                                         computed on the held-out TEST split only
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (  # noqa: E402
    RAW_DATA_DIR, TRAINED_MODEL_PATH, SCALER_PATH, METADATA_PATH, EVALUATION_PATH,
    PROCESSING_SIZE, RANDOM_SEED,
)
from app.pipeline import preprocessing, filtering, edge_detection, segmentation, morphology, features as feat_mod  # noqa: E402

MODEL_VERSION = "1.0.0"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def compute_features_for_image(path: Path) -> dict:
    """
    Runs the exact same DIP sequence as app/pipeline/pipeline_runner.py's
    stages 1-11 (minus saving intermediate PNGs, which would only slow
    training down - the pixel operations themselves are identical) and
    returns the canonical feature dict.
    """
    data = path.read_bytes()
    img = preprocessing.load_image_from_bytes(data)
    resized = preprocessing.resize_and_normalize(img, PROCESSING_SIZE)
    gray = preprocessing.to_grayscale(resized)
    denoised = filtering.denoise(gray, method="median")
    enhanced = filtering.enhance_contrast(denoised, method="clahe")
    edges = edge_detection.canny_edges(enhanced)
    seg_mask, _ = segmentation.segment(enhanced, edges)
    cleaned = morphology.clean_mask(seg_mask, kernel_size=3)
    filled = morphology.fill_holes(cleaned)
    _, components = morphology.connected_components(filled, min_area=25, max_area_fraction=0.4)
    return feat_mod.extract_features(enhanced, filled, components)


def load_dataset(data_dir: Path):
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if len(class_dirs) < 2:
        raise SystemExit(f"Need at least 2 class folders under {data_dir}, found {len(class_dirs)}.")

    label_map = {i: d.name for i, d in enumerate(class_dirs)}
    X, y, paths = [], [], []
    class_counts = {}

    print(f"Loading dataset from {data_dir} ...")
    for idx, d in enumerate(class_dirs):
        images = sorted([p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        class_counts[d.name] = len(images)
        print(f"  [{idx}] {d.name}: {len(images)} images")
        for img_path in images:
            try:
                feats = compute_features_for_image(img_path)
            except Exception as e:
                print(f"    WARNING: skipped {img_path.name} ({e})")
                continue
            X.append([feats.get(name, 0.0) for name in feat_mod.FEATURE_NAMES])
            y.append(idx)
            paths.append(str(img_path))

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64), label_map, class_counts


def main(data_dir: str, test_size: float):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise SystemExit(
            f"Dataset directory not found: {data_path}\n"
            f"See backend/data/DATASET_INFO.md for how to obtain/place the NEU-CLS dataset."
        )

    t0 = time.perf_counter()
    X, y, label_map, class_counts = load_dataset(data_path)

    if len(X) < 20:
        raise SystemExit(f"Too few usable samples ({len(X)}) to train meaningfully.")

    print(f"\nTotal usable samples: {len(X)}  |  Feature dimensionality: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train/test split: {len(X_train)} train / {len(X_test)} test (stratified, seed={RANDOM_SEED})")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit on TRAIN ONLY
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=RANDOM_SEED, n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)

    # --- Real evaluation on the held-out test set. Nothing here is invented. ---
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    report_dict = classification_report(
        y_test, y_pred, target_names=[label_map[i] for i in sorted(label_map)],
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n--- HELD-OUT TEST SET EVALUATION (real, not invented) ---")
    print(classification_report(y_test, y_pred, target_names=[label_map[i] for i in sorted(label_map)],
                                 zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    for row, name in zip(cm, [label_map[i] for i in sorted(label_map)]):
        print(f"  {name:>16}: {row}")

    # --- Save artifacts ---
    TRAINED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, TRAINED_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metadata = {
        "model_type": "RandomForestClassifier",
        "model_version": MODEL_VERSION,
        "feature_names": feat_mod.FEATURE_NAMES,
        "label_map": {str(k): v for k, v in label_map.items()},
        "class_names": [label_map[i] for i in sorted(label_map)],
        "dataset_name": "NEU-CLS (NEU Surface Defect Database, classification variant)",
        "dataset_class_counts": class_counts,
        "total_samples": int(len(X)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "random_seed": RANDOM_SEED,
        "test_size": test_size,
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_random_forest_params": {"n_estimators": 300, "max_depth": None},
        "training_duration_seconds": round(time.perf_counter() - t0, 2),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    evaluation = {
        "evaluated_on": "held-out test split (never seen during training)",
        "test_set_size": int(len(X_test)),
        "train_set_size": int(len(X_train)),
        "accuracy": round(float(accuracy), 4),
        "macro_precision": round(float(precision_macro), 4),
        "macro_recall": round(float(recall_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "per_class_report": report_dict,
        "confusion_matrix": cm,
        "class_order": [label_map[i] for i in sorted(label_map)],
        "class_distribution": class_counts,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    EVALUATION_PATH.write_text(json.dumps(evaluation, indent=2))

    print(f"\nSaved model      -> {TRAINED_MODEL_PATH}")
    print(f"Saved scaler     -> {SCALER_PATH}")
    print(f"Saved metadata   -> {METADATA_PATH}")
    print(f"Saved evaluation -> {EVALUATION_PATH}")
    print("\nThe API will now use this model automatically for mode='real' /api/inspect calls.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Real ML Mode classifier on NEU-CLS")
    parser.add_argument("--data-dir", default=str(RAW_DATA_DIR))
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    main(args.data_dir, args.test_size)
