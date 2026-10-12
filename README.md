# AI-Based Industrial Surface Defect Detection and Classification Using Digital Image Processing

A B.Tech Digital Image Processing project. A web application runs steel-surface images through an
inspectable DIP pipeline (OpenCV / NumPy / scikit-image / SciPy), extracts a 23-value feature vector,
and classifies it with a scikit-learn Random Forest trained and evaluated on the public **NEU-CLS**
steel surface defect dataset.

Every metric in this repository was computed by an actual experiment and can be reproduced with the
commands below. Where something is only a demonstration, it is labelled as such.

---

## Overview

| | **Demo Mode** | **Real ML Mode** |
|---|---|---|
| Label in the UI | `DEMO / SYNTHETIC DATA` | `TRAINED MODEL` |
| Data | Any image; best with the synthetic images from `scripts/generate_demo_dataset.py` | NEU-CLS-style steel surface images |
| Classifier | Rule-based heuristic on shape features | Trained `RandomForestClassifier` |
| Score shown | `heuristic_score`: a fixed weight, **not** a probability | `model_probability`: the Random Forest's `predict_proba()` output, **not** a calibrated confidence |
| Answers | "Is there a candidate defect region, and what shape is it?" | "Which of the six known NEU-CLS categories does this image most resemble?" |

NEU-CLS has no "normal / no-defect" class, so Real ML Mode always returns one of six defect classes; it
does not say whether a defect is present. The two modes are never mixed: the Dashboard reports their
statistics separately and model-evaluation metrics never come from inspection history.

## Features

- 11-stage DIP pipeline; every intermediate image, technique and parameter is viewable per inspection
- Real ML Mode (trained Random Forest) and Demo Mode (rule-based), clearly separated
- Model Evaluation page: accuracy, macro and per-class precision/recall/F1, confusion matrix, split sizes
- Inspection history with timestamp, filename, prediction, mode, model version, candidate-region count and area, processing time
- Downloadable result image and text inspection report
- In-app Methodology page and viva question bank written to match what the code does
- 36 automated backend tests (unit, API, and real-image end-to-end)

**## Screenshots**

<img width="1853" height="957" alt="image" src="https://github.com/user-attachments/assets/3c499114-3543-489c-ae7e-789402d2b038" />
<img width="1853" height="957" alt="image" src="https://github.com/user-attachments/assets/98fbea25-be6f-42d8-bc86-901746851e0b" />
<img width="1853" height="957" alt="image" src="https://github.com/user-attachments/assets/464a9d79-d9e6-4f80-8f69-1f40546dd6cc" />
<img width="1862" height="628" alt="image" src="https://github.com/user-attachments/assets/9fe2b576-7eca-4957-8441-901715845010" />
<img width="1862" height="780" alt="image" src="https://github.com/user-attachments/assets/a51b746d-59bf-48dc-a3b4-fc0ce3ed43fa" />
<img width="1864" height="946" alt="image" src="https://github.com/user-attachments/assets/c8ddfb3f-13ba-451b-82a8-947ed0d587fb" />

## Architecture

```
React + Vite + Tailwind  --REST/JSON, multipart upload-->  FastAPI  -->  SQLite
                                                              |
                                                              v
                       DIP pipeline (app/pipeline/*)  -- one code path for training AND inference
                                                              |
                          23-feature vector --> StandardScaler --> RandomForest  (models/*.joblib)
```

In development Vite proxies `/api`, `/inspection-assets` and `/uploads` to the backend on port 8000
(`frontend/vite.config.js`). The model is trained offline by `ml/train_classifier.py` and only *loaded*
by the API. On load, the API checks that the model's saved feature names equal the current
`FEATURE_NAMES`; on a mismatch it refuses to predict instead of silently misaligning features.

## Technology Stack

React 18, Vite 5, Tailwind CSS 3, Recharts | FastAPI, SQLAlchemy, SQLite | OpenCV, NumPy,
scikit-image, SciPy | scikit-learn, joblib | pytest

## DIP Pipeline

| # | Stage | Technique (parameters) | Module |
|---|---|---|---|
| 1 | Original image | decode + validate | `preprocessing.py` |
| 2 | Resize & normalise | 512x512, `INTER_AREA`, min-max to 0-255 | `preprocessing.py` |
| 3 | Grayscale | BT.601 luminosity weighting | `preprocessing.py` |
| 4 | Noise reduction | median filter, 5x5 | `filtering.py` |
| 5 | Contrast enhancement | CLAHE, clip limit 3.0, 8x8 tiles | `filtering.py` |
| 6 | Edge detection | Canny, thresholds 50 / 150 | `edge_detection.py` |
| 7 | Segmentation | Otsu **AND** dilated (5x5) Canny edges | `segmentation.py` |
| 8 | Morphology | 3x3 elliptical opening, closing, hole filling | `morphology.py` |
| 9 | Connected components | 8-connectivity; drop regions < 25 px or > 40% of frame | `morphology.py` |
| 10 | Candidate defect regions | bounding box for every surviving region | `pipeline_runner.py` |
| 11 | Feature extraction + classification | 23 features -> classifier | `features.py`, `classifier.py` |

**Candidate regions are generated by the Digital Image Processing pipeline. The NEU-CLS dataset does not
provide ground-truth segmentation masks, so these regions should not be interpreted as verified defect
boundaries.** NEU-CLS provides image-level defect categories rather than pixel-level segmentation
annotations, so no segmentation accuracy is claimed. Sobel, Wiener filtering, Hough lines and
skeletonisation are implemented but are not part of the main pipeline.

## Dataset

**NEU-CLS** (Northeastern University Surface Defect Database, classification variant): real grayscale
200x200 images of hot-rolled steel strip. Provenance, licence note and limitations are in
[`backend/data/DATASET_INFO.md`](backend/data/DATASET_INFO.md).

| Class | Images |
|---|---|
| crazing | 300 |
| inclusion | 300 |
| patches | 299 |
| pitted_surface | 300 |
| rolled-in_scale | 300 |
| scratches | 300 |
| **Total** | **1,799** |

The original release has 1,800 images. A full MD5 scan found exactly one exact duplicate
(`patches_101.jpg` = `patches_105.jpg`); `patches_105.jpg` was removed and the model retrained.

### Dataset structure

```
backend/data/
  raw/NEU-CLS/<class_name>/*.jpg     # bundled, 6 class folders
  processed/                         # reserved; currently unused
backend/models/                      # defect_classifier.joblib, scaler.joblib, metadata.json, evaluation.json
```

## Feature Extraction

One function (`app/pipeline/features.py::extract_features`) produces the vector in the fixed order of
`FEATURE_NAMES`, for both training and inference. Non-finite values are replaced by 0, and an empty
segmentation yields zeroed regional features.

- **Global (11):** mean, std, min, max intensity; Shannon entropy; mean local variance; global Canny
  edge density; GLCM contrast, homogeneity, energy, correlation
- **Regional / aggregate (12):** largest region's area, perimeter, aspect ratio, extent, solidity,
  circularity, eccentricity; number of regions, total area, average area, total candidate-area %, contour count

For the shipped model (v1.0.0), the 11 global features carry about 88% of the forest's impurity-based
feature importance and the 12 regional ones about 12%. This describes that model only; no ablation
study was run.

## Machine Learning

`RandomForestClassifier` (300 trees, unlimited depth, `random_state=42`) on the 23 features, preceded
by a `StandardScaler` fitted on the training split only. Tree models do not need scaling, so the scaler
is kept for pipeline uniformity, not accuracy.

### Training procedure

1. Load every image from `data/raw/NEU-CLS/<class>/`
2. Run the same DIP functions, in the same order, as live inference
3. Build `X` (1,799 x 23) and `y` (6 classes)
4. Stratified 80/20 split, seed 42 (1,439 train / 360 test)
5. Fit scaler on train; fit Random Forest on scaled train
6. Evaluate once on the untouched test split
7. Save model, scaler, `metadata.json` (feature names, class names, dataset, seed, version, date) and `evaluation.json`

## Evaluation

**Random Forest achieved 91.39% accuracy on a stratified held-out test split of the NEU-CLS dataset**
(360 test images, 1,439 training images, 6 classes, random seed 42). Macro precision 91.40%,
macro recall 91.39%, macro F1 91.38%. These are computed on the held-out test split only, never on
training data or inspection history.

| Class | Precision | Recall | F1 | Test support |
|---|---|---|---|---|
| crazing | 96.61% | 95.00% | 95.80% | 60 |
| inclusion | 85.25% | 86.67% | 85.95% | 60 |
| patches | 95.00% | 95.00% | 95.00% | 60 |
| pitted_surface | 81.36% | 80.00% | 80.67% | 60 |
| rolled-in_scale | 93.55% | 96.67% | 95.08% | 60 |
| scratches | 96.61% | 95.00% | 95.80% | 60 |

Confusion matrix (rows = true, columns = predicted, order as above):

```
crazing          [57,  0,  0,  0,  3,  0]
inclusion        [ 0, 52,  0,  8,  0,  0]
patches          [ 0,  0, 57,  2,  1,  0]
pitted_surface   [ 1,  6,  3, 48,  0,  2]
rolled-in_scale  [ 1,  0,  0,  1, 58,  0]
scratches        [ 0,  3,  0,  0,  0, 57]
```

`pitted_surface` is the weakest class and is mostly confused with `inclusion` and `patches`. This is a
single split of 360 images (binomial standard error of the accuracy about 1.5 percentage points); no
cross-validation was run. Numbers in this section are copied from `backend/models/evaluation.json`; the
Dashboard's **Model Evaluation** tab renders that file live. Retraining from a clean virtual environment
with the pinned requirements reproduced these metrics exactly.

## Installation

Tested with **Python 3.12.3**, **Node 22.22.2**, npm 10.9.7 (other versions not verified).

### Backend setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used. The saved model was pickled with scikit-learn 1.8.0;
with a different version, retrain (below).

### Frontend setup

```bash
cd frontend
npm install
```

### Dataset setup

The dataset is already in `backend/data/raw/NEU-CLS/`. If you replace or re-download it, keep one
folder per class under that path. To check for exact duplicates (Linux):

```bash
cd backend/data/raw/NEU-CLS
find . -name '*.jpg' | xargs md5sum | sort | uniq -w32 -d     # prints nothing if there are none
```

### Model training

A trained model is already included. To retrain (about a minute), from `backend/` with the venv active:

```bash
python ml/train_classifier.py
```

### Running the application

#### Local development

Terminal 1 (from `backend/`, virtual environment active):

```bash
uvicorn app.main:app --reload --port 8000
```

Terminal 2 (from `frontend/`):

```bash
npm run dev
```

Open http://127.0.0.1:5173. Vite proxies `/api`, `/inspection-assets` and `/uploads` to
`http://127.0.0.1:8000`. The React `/results` page is not proxied; result images are served by the
backend under `/inspection-assets/`. The backend also keeps `/results/` as a legacy static-image
alias so image paths already stored in inspection history continue to work.

Interactive API docs: http://127.0.0.1:8000/docs.

#### Production deployment

Build the frontend with the backend's public origin set at build time:

```bash
cd frontend
VITE_API_BASE_URL=https://api.example.org npm run build
```

Replace `https://api.example.org` with the origin that serves this project's FastAPI backend. The
frontend uses this value for API calls, result images, upload images and image downloads. If the
frontend and backend are served from one origin, leave `VITE_API_BASE_URL` unset or empty and
configure the web server to forward `/api/`, `/inspection-assets/` and `/uploads/` to FastAPI. Serve
the built frontend as a single-page application: unknown frontend paths, including `/results` and
`/history/<id>`, must return `index.html`. Do not forward the frontend `/results` page to FastAPI;
the backend's `/results/` path is only a legacy image alias.

For a separate frontend and API origin, set the backend's `CORS_ORIGINS` to the frontend's exact
origin (for example, `https://app.example.org`) before starting FastAPI. The production build uses
the API URL embedded when it is built, so set `VITE_API_BASE_URL` again if the API origin changes.
Actual public-host deployment and TLS/proxy configuration depend on the deployment environment;
they are not included or validated by this repository.

The development proxy remains configured in `frontend/vite.config.js` and does not apply to the
production bundle.

Optional sample images for Demo Mode (written to `backend/storage/demo_dataset/`):

```bash
python scripts/generate_demo_dataset.py --count 40
```

Tests (from `backend/`): `pytest tests/ -v`. They use a temporary database and storage directory and do not
touch your real inspection history.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | liveness + whether a trained model is available |
| `POST` | `/api/inspect` | multipart `file` + form `mode` (`demo` or `real`); runs the pipeline, stores and returns the result |
| `GET` | `/api/inspections?limit=&mode=` | history list |
| `GET` | `/api/inspections/{id}` | full stored inspection with all 11 stages |
| `GET` | `/api/dashboard` | inspection-activity statistics, split by mode (never model accuracy) |
| `GET` | `/api/model/info` | model metadata and held-out evaluation results |
| `POST` | `/api/process/{grayscale,filter,segment,morphology}` | single-operation endpoints for exploration |

Errors: `400` invalid/unsupported/empty/corrupted file or invalid mode; `404` unknown inspection;
`503` Real ML Mode requested but no trained model exists (a fake result is never returned);
`500` feature-contract mismatch or processing failure. The UI shows a clear message when the backend
cannot be reached.

## Limitations

- NEU-CLS consists of laboratory/benchmark images, not live production-line images.
- NEU-CLS provides classification labels, not segmentation ground truth, so DIP candidate regions are not verified defect masks and segmentation quality is not measured.
- Performance may change with different cameras, lighting, materials or manufacturing environments; the model is trained on one dataset and should not be taken as applicable to all industrial defects.
- The model always outputs one of six classes (no "normal" class, no reject option): a blank image is still assigned a class.
- `model_probability` is an uncalibrated Random Forest output.
- Single train/test split, no cross-validation or confidence interval.
- Real production deployment would require additional validation on the target imaging setup.

## Future Scope

Cross-validation and evaluation on further datasets; a "normal" class or anomaly-detection stage;
a CNN baseline for comparison; probability calibration; annotated masks to quantify segmentation
accuracy; testing on images from a real inspection setup.

## Known Issues

- A local browser regression pass covered all seven pages, direct navigation and refresh for Results
  and History, a real NEU-CLS inspection, pipeline images, history retrieval, dashboard data, and
  result/report download payloads in development and production preview. A public deployment,
  other browsers, and physical download behavior still require environment-specific/manual checks.
- Demo Mode segmentation can produce ring-like or fragmented masks on solid synthetic blobs, so its
  rule-based labels can be wrong (a solid disc was labelled "Surface Irregularity"). Treat them as illustrative.
- Real-mode candidate-region counts vary widely across NEU images (0 to over 100 in a 10-image
  check); the segmentation thresholds were not tuned or validated against ground truth.
- The first inspection after server start is slower (model loading, roughly 1 s versus about 0.15 s).
- Production deployment requires either a same-origin reverse proxy for `/api/`, `/inspection-assets/` and `/uploads/`, or a separately reachable API with `VITE_API_BASE_URL` and matching backend `CORS_ORIGINS`. No hosting-specific proxy configuration is included.
- `npm audit` reports 12 advisories (5 moderate, 7 high); two moderate advisories affect production dependencies through React Router. Most fixes offered by npm require major-version upgrades, so dependencies were not broadly upgraded during this maintenance pass.
- `vite build` prints a chunk-size advisory (main bundle about 635 kB); pytest prints one Starlette/httpx deprecation warning. Neither affects behaviour.
- The installation versions above were tested previously. This maintenance pass also ran on Python 3.14.4 and Node 24.18.0 (npm 12.0.2); a public hosting environment was not tested.
