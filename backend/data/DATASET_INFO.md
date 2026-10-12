# Dataset: NEU-CLS (NEU Surface Defect Database — Classification variant)

## Source

Collected and released by the Surface Inspection Laboratory, Northeastern University (NEU),
China. Original page: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html

The copy bundled in `data/raw/NEU-CLS/` in this repository was obtained from a public GitHub
mirror (Mirlenko/Surface_Defects_Classification) that re-hosts the same 1,800 images as plain
JPEG files, which made it retrievable in this project's sandboxed build environment. Every image
was visually spot-checked against the known NEU-CLS defect appearance before being included.

## Why this dataset

- **Real, not synthetic**: genuine grayscale photographs of hot-rolled steel strip surfaces from
  an actual production line, not a rendered or procedurally generated image.
- **Labeled, balanced, manageable**: 1,799 images total after removing one exact duplicate
  (originally 1,800), approximately 300 per class, 200×200 grayscale — small enough to preprocess
  and train classical features on a CPU (about a minute on the development machine), while still
  being a genuine multi-class problem.
- **Well suited to classical DIP + handcrafted features + ML**: the six defect types are
  primarily textural/geometric (fine cracks, blotches, linear scratches, pitting, scale patches),
  which is exactly what edge density, GLCM texture statistics, and connected-component shape
  descriptors are designed to capture — this dataset does not require a deep CNN to get a
  meaningful, explainable signal.
- **Widely used academically**, so results are easy to benchmark and discuss credibly in a viva.

## Dataset deduplication

An exact MD5 duplicate was found between `patches/patches_101.jpg` and `patches/patches_105.jpg`
(identical file hash — a true copy, not merely a visually similar image). A full MD5 scan across
all 1,800 original images confirmed this was the **only** exact duplicate in the dataset.
`patches_105.jpg` was removed; `patches_101.jpg` was kept. The model was retrained from scratch
on the corrected 1,799-image dataset — see `models/evaluation.json` for the resulting metrics.

## Composition (after deduplication)

| Property | Value |
|---|---|
| Total images | 1,799 (1,800 originally; 1 exact duplicate removed) |
| Classes | 6 |
| Images per class | 300 each, except `patches` (299 after deduplication) |
| Image format | Grayscale JPEG, 200×200 px |

**Class names** (folder names used in `data/raw/NEU-CLS/`):
`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`

## Important limitation of this dataset (stated explicitly)

**NEU-CLS has no "normal / no-defect" class.** Every image is a crop that already contains one
of the six defect types. This means a classifier trained on it answers *"which of these six known
defect textures does this image most resemble?"*, not *"is there a defect present at all?"*. This
project is explicit about that distinction:

**NEU-CLS provides image-level defect categories rather than pixel-level segmentation
annotations.** There are no ground-truth segmentation masks or bounding boxes. Any "candidate
defect region" shown by this application's DIP pipeline (Otsu thresholding + edge agreement +
morphology + connected components) is produced entirely by classical image processing and has
not been validated against ground truth — it is labeled a "candidate region" everywhere in the
app, never a verified defect boundary.

- **Demo Mode** (synthetic data, `scripts/generate_demo_dataset.py`) demonstrates the
  defect-vs-no-defect + candidate-region-detection workflow, using the rule-based classifier.
- **Real ML Mode** (this dataset, `ml/train_classifier.py`) demonstrates genuine multi-class
  texture classification among the six real defect categories, using a trained RandomForest.

The two are never mixed in the UI, the database, or the dashboard.

## Dataset structure

```
backend/data/
├── raw/
│   └── NEU-CLS/
│       ├── crazing/            (300 .jpg)
│       ├── inclusion/          (300 .jpg)
│       ├── patches/            (299 .jpg — 1 exact duplicate removed, see below)
│       ├── pitted_surface/     (300 .jpg)
│       ├── rolled-in_scale/    (300 .jpg)
│       └── scratches/          (300 .jpg)
└── processed/                  (reserved; the training script does not need to cache
                                  intermediate files for a dataset this size, so this
                                  directory is currently empty — kept for future use,
                                  e.g. caching extracted feature CSVs for faster re-runs)
```

## Expected preprocessing

Identical to the live inference pipeline (`app/pipeline/`), so training and inference are
never inconsistent: resize/normalize → grayscale (already grayscale here, so this is a no-op
type check) → median denoise → CLAHE contrast enhancement → Canny edge detection → Otsu +
edge-agreement thresholding → morphological opening/closing/hole-filling → connected-component
labeling → combined global texture features + regional shape/aggregate features.

## Obtaining the dataset yourself (if you remove it from this repo)

If you clean the dataset out of the repository (e.g. to keep a git history smaller), it can be
re-obtained from either:
- The original NEU faculty page linked above, or
- Kaggle: search "NEU Metal Surface Defects Data" (community mirror), or
- Any of several public GitHub mirrors that host it as plain image files (search "NEU-CLS
  crazing inclusion patches pitted_surface rolled-in_scale scratches github").

Whichever source is used, place images into `data/raw/NEU-CLS/<class_name>/*.jpg` using the six
class folder names above before running `ml/train_classifier.py`.

## License / academic use note

This dataset is widely redistributed and used in academic computer-vision research under the
original NEU release terms. It is included here strictly for non-commercial academic/educational
project use. If publishing results, cite the original NEU Surface Defect Database.
