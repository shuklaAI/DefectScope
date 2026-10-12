"""
Central configuration for the backend.

All paths are resolved relative to this file so the app has no hardcoded
absolute paths and can be moved / deployed anywhere. Values that a real
deployment would want to change (DB url, CORS origins, storage roots) are
read from environment variables with sensible local-dev defaults.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/

STORAGE_DIR = Path(os.getenv("STORAGE_DIR", BASE_DIR / "storage"))
UPLOAD_DIR = STORAGE_DIR / "uploads"
RESULTS_DIR = STORAGE_DIR / "results"

# Model artifacts live in backend/models/ (not storage/), matching the
# project's ml/train_classifier.py output location.
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))

# Dataset locations (backend/data/raw/NEU-CLS/<class>/*.jpg, see data/DATASET_INFO.md)
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "NEU-CLS"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

for d in (UPLOAD_DIR, RESULTS_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# SQLite for local development. Swapping to Postgres/Supabase later only
# requires changing this connection string (SQLAlchemy handles the rest).
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{STORAGE_DIR / 'inspections.db'}")

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", 15))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Trained model artifacts, produced by ml/train_classifier.py against the
# real NEU-CLS dataset (backend/data/raw/NEU-CLS/). If these files do not
# exist, Real ML Mode is unavailable and the API returns a clear error
# rather than silently using Demo Mode instead.
TRAINED_MODEL_PATH = MODEL_DIR / "defect_classifier.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
EVALUATION_PATH = MODEL_DIR / "evaluation.json"

# Standard working resolution the pipeline resizes inputs to. Keeping this
# fixed makes downstream thresholds/kernel sizes behave consistently
# regardless of the original upload resolution.
PROCESSING_SIZE = (512, 512)

# Fixed random seed used everywhere randomness matters (train/test split,
# RandomForest) so training is reproducible.
RANDOM_SEED = 42
