import React from "react";

const SECTIONS = [
  {
    title: "1. Problem Statement",
    body: "Manual visual inspection of manufactured surfaces is slow and varies between inspectors. This project investigates whether a classical Digital Image Processing pipeline, combined with a conventional machine-learning classifier, can categorise steel-surface defect images, and it keeps every processing step visible so the method can be examined and explained.",
  },
  {
    title: "2. Objectives",
    body: "(1) Implement a documented, inspectable sequence of DIP operations. (2) Extract a fixed, reproducible feature vector from that processing. (3) Train and evaluate a Random Forest on a real public dataset using a held-out test split. (4) Report only metrics that were actually computed, and keep the rule-based Demo Mode strictly separate from the trained model.",
  },
  {
    title: "3. Dataset",
    body: "NEU-CLS (NEU Surface Defect Database, classification variant): 1,799 real grayscale 200x200 steel-strip surface photographs across 6 classes (crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches): 300 images each except patches at 299, after one exact duplicate (identical MD5 hash) was removed from the original 1,800. NEU-CLS provides image-level defect categories rather than pixel-level segmentation annotations, and it contains no 'normal / no-defect' class. Real ML Mode therefore answers 'which of the six known categories does this image most resemble', not 'is a defect present'. Provenance and licence notes: backend/data/DATASET_INFO.md.",
  },
  {
    title: "4. Image Preprocessing",
    body: "Each image is decoded, resized to a fixed 512x512 working resolution (OpenCV INTER_AREA) and min-max normalised to the full 0-255 range, so kernel sizes and thresholds behave the same for every input. The 200x200 NEU images are therefore resampled to 512x512, identically during training and inference.",
  },
  {
    title: "5. Grayscale Conversion",
    body: "The image is converted to one channel with OpenCV's BT.601 luminosity weighting (0.299R + 0.587G + 0.114B). The NEU-CLS images are already grayscale, so for them this step changes little; it matters for colour uploads in Demo Mode.",
  },
  {
    title: "6. Noise Reduction",
    body: "A 5x5 median filter suppresses impulse (salt-and-pepper) noise while preserving edges better than a mean filter. Gaussian and Wiener filtering are implemented in filtering.py and exposed via the standalone /api/process/filter endpoint, but the main pipeline uses the median filter.",
  },
  {
    title: "7. Contrast Enhancement",
    body: "CLAHE (Contrast Limited Adaptive Histogram Equalisation; clip limit 3.0, 8x8 tiles) boosts local contrast without the global noise amplification of plain histogram equalisation. The Pipeline page plots the intensity histogram before and after this step.",
  },
  {
    title: "8. Edge Detection",
    body: "Canny edge detection (hysteresis thresholds 50 and 150) produces thin, connected edges. They are used both as one cue for candidate-region segmentation and, as a global edge-density measure, as a classifier feature. A Sobel gradient operator and a Hough line detector are implemented but not used in the main pipeline.",
  },
  {
    title: "9. Segmentation",
    body: "A pixel is kept as a candidate only if it is both an Otsu-threshold foreground pixel (on the enhanced image) and lies within a dilated Canny edge. Requiring both cues reduces false regions from background noise. NEU-CLS provides image-level defect categories rather than pixel-level segmentation annotations, so no segmentation accuracy can be reported: the resulting regions are candidate regions for demonstrating DIP and for feature extraction, not verified defect boundaries.",
  },
  {
    title: "10. Morphological Processing",
    body: "A 3x3 elliptical opening removes small noise specks, a closing reconnects nearby fragments, and flood-fill hole filling closes enclosed gaps. The small kernel is deliberate so that thin, scratch-like structures are not eroded away.",
  },
  {
    title: "11. Connected Components",
    body: "8-connectivity labelling groups mask pixels into regions. Regions smaller than 25 px (noise) or larger than 40% of the frame (likely a global segmentation failure) are discarded. Every remaining region contributes to the aggregate features, not only the largest. If none survive, the regional features are set to zero and the model still receives a valid vector (no NaN or infinity values are ever passed on).",
  },
  {
    title: "12. Feature Extraction",
    body: "A fixed-order vector of 23 features, computed by one function (app/pipeline/features.py) used identically for training and inference. Global (11): mean/std/min/max intensity, Shannon entropy, mean local variance, global Canny edge density, and four GLCM statistics (contrast, homogeneity, energy, correlation). Regional and aggregate (12): area, perimeter, aspect ratio, extent, solidity, circularity and eccentricity of the largest region, plus component count, total and average region area, total candidate-area percentage and contour count. The saved model records its feature names, and the API refuses to use a model whose feature list differs from the current code.",
  },
  {
    title: "13. Random Forest Classification",
    body: "A scikit-learn RandomForestClassifier (300 trees, unlimited depth, random seed 42) is trained on a stratified 80/20 split of the 23-feature vectors. A StandardScaler is fit on the training split only and applied identically at inference; tree models do not require scaling, so it is kept for pipeline uniformity rather than because it changes the splits. The reported 'model probability' is the Random Forest's predict_proba() output for its top class (the average of the trees' class-probability estimates). No calibration has been performed, so it must not be read as a calibrated confidence. Demo Mode instead uses a rule-based heuristic and reports a fixed 'heuristic score', which is not a probability.",
  },
  {
    title: "14. Model Evaluation",
    body: "Accuracy, macro and per-class precision / recall / F1, and the confusion matrix are computed once on the held-out test split and written to backend/models/evaluation.json. The Dashboard's Model Evaluation tab renders that file; the figures are never recomputed from, or mixed with, inspection history. This is a single train/test split on one benchmark dataset; no cross-validation, calibration analysis, or ablation study has been performed.",
  },
  {
    title: "15. Limitations",
    body: "The dataset consists of laboratory/benchmark images, not live production-line images. NEU-CLS provides classification labels rather than segmentation ground truth, so the candidate regions produced by the DIP pipeline are not verified defect masks. Performance may change with different cameras, lighting conditions, materials or manufacturing environments, and the model is trained on one specific dataset, so it should not be read as applicable to all industrial defects. It always outputs one of the six known classes, because the dataset has no 'normal' class and the model has no reject option. The rule-based Demo Mode thresholds are heuristics tested only on synthetic images. Real production deployment would require additional validation on the target imaging setup.",
  },
  {
    title: "16. Future Scope",
    body: "Evaluate on additional public datasets and with cross-validation; add a 'normal' class or an anomaly-detection stage so defect presence can be assessed; compare against a CNN baseline on the same split; calibrate the predicted probabilities; collect or annotate segmentation masks so the DIP segmentation can be quantitatively evaluated; and test on images from a real inspection setup before any practical use.",
  },
];

const VIVA_QA = [
  ["What is Digital Image Processing?", "The use of computer algorithms to analyze, enhance, or extract information from digital images, represented as discrete grids of quantized pixel values."],
  ["Why is image preprocessing required?", "Raw images vary in resolution, illumination and noise; preprocessing standardizes them (resize, grayscale, denoise, contrast) so later stages and the feature vector behave predictably."],
  ["Why grayscale conversion?", "The defect signal in this dataset is structural/textural, not chromatic (the source images are already grayscale); working on one intensity channel simplifies edge/threshold/texture computations."],
  ["What is histogram equalization / CLAHE?", "A technique that redistributes pixel intensities to use more of the available range; CLAHE applies this locally in tiles with a clip limit, improving contrast without over-amplifying noise the way global equalization can."],
  ["Why median filtering?", "To suppress impulse (salt-and-pepper) sensor noise before edge/threshold/texture operations, which are sensitive to spurious intensity spikes, while preserving edges better than a mean filter."],
  ["What is edge detection, and why Canny?", "Edge detection finds pixels where intensity changes sharply. Canny adds Gaussian smoothing, non-maximum suppression and hysteresis thresholding on top of a raw gradient (Sobel), producing thin, well-connected edges."],
  ["What is thresholding, and why Otsu?", "Converting a grayscale image to a binary mask by classifying pixels as foreground/background relative to a threshold; Otsu picks that threshold automatically by minimizing intra-class variance in the image's histogram."],
  ["Why morphological opening and closing?", "Opening (erosion then dilation) removes small noise specks without shrinking real regions; closing (dilation then erosion) reconnects small gaps within the same region. Together they clean a noisy segmentation mask."],
  ["What are connected components, and how are multiple defects handled?", "Groups of 8-connected foreground pixels, each treated as one candidate region. Every surviving component (not only the largest) contributes to the aggregate feature count and area. Multiple regions are retained."],
  ["Which features are extracted, and why these?", "23 features: global statistical/texture (intensity stats, entropy, local variance, edge density, 4 GLCM descriptors) capture whole-image texture patterns; regional geometric features (area, perimeter, aspect ratio, solidity, circularity, eccentricity) plus aggregate counts capture shape. Together they suit both a whole-image texture dataset (NEU-CLS) and a candidate-region demo scenario."],
  ["Why RandomForest, and how is scaling handled?", "RandomForest handles a moderate-size tabular feature vector well, needs little hyperparameter tuning, and exposes interpretable per-feature importances. Features are passed through a StandardScaler fit ONLY on the training split (to avoid leakage); trees are insensitive to monotonic scaling, so this is for pipeline uniformity, not accuracy."],
  ["How was the model evaluated, and on what data?", "Accuracy, macro precision/recall/F1, per-class metrics and a confusion matrix were computed once on a stratified 20% held-out test split that the model never saw during training (see the Dashboard's Model Evaluation tab for the actual numbers)."],
  ["Why does Real ML Mode not say 'no defect found'?", "NEU-CLS has no 'normal' class. Each image belongs to one of six known defect types. The model classifies among those categories; it does not detect whether a defect is present."],
  ["What is the difference between 'model probability' and 'confidence'?", "'Model probability' is the RandomForest predict_proba() output for its top class. It represents the model's vote distribution. It is not calibrated confidence; no calibration step such as Platt scaling has been performed."],
  ["What are the limitations of this system?", "See the Limitations section: Real ML Mode has only been evaluated on NEU-CLS and no claim is made beyond it; Demo Mode's rule thresholds are heuristic and untested against real defects; classical features will generally underperform a deep CNN; segmentation quality varies with real-world lighting."],
  ["How would you extend this to detect 'no defect' at all?", "Add a real dataset that includes non-defective surface images under the same imaging conditions, and retrain as a 7-class (6 defects + normal) or binary-then-multiclass cascade."],
  ["How could this be deployed in a real production line (future scope)?", "A future version could process frames from a line-scan or area camera and route flagged parts for review. The pipeline would first need validation under the target line's imaging conditions."],
  ["Why keep Demo Mode at all?", "It demonstrates the DIP segmentation and candidate-region-detection workflow (bounding boxes on an otherwise uniform surface) that NEU-CLS's whole-image-is-a-defect design doesn't exercise, using synthetic images so the UI is testable without depending on any external dataset."],
  ["How is reproducibility ensured?", "A fixed random seed (42) is used for the train/test split and the RandomForest; the scaler is persisted alongside the model; and training metadata (dataset name, counts, date, seed, version) is saved to models/metadata.json."],
  ["What would you benchmark this against?", "Published NEU-CLS results using deep CNNs (which typically report higher accuracy than classical handcrafted features), to honestly contextualize where a classical DIP+ML pipeline sits relative to deep learning on the same dataset."],
  ["Does the system provide verified defect localization?", "No. The bounding boxes shown are candidate regions produced by the classical DIP segmentation (Otsu + edge agreement + morphology + connected components). NEU-CLS provides only image-level class labels, with no ground-truth segmentation masks or bounding boxes, so these regions are explicitly labeled as candidates throughout the app, not verified defect boundaries."],
];

export default function About() {
  return (
    <div className="max-w-4xl mx-auto px-5 py-14">
      <h1 className="text-3xl font-bold mb-2">Methodology</h1>
      <p className="text-slate-400 mb-10">
        Full project documentation, structured for both the written report and the live viva.
        Every explanation below corresponds to what the running code actually does.
      </p>

      <div className="space-y-8 mb-16">
        {SECTIONS.map((s) => (
          <div key={s.title}>
            <h2 className="text-lg font-semibold text-accent-400 mb-1.5">{s.title}</h2>
            <p className="text-slate-300 text-sm leading-relaxed">{s.body}</p>
          </div>
        ))}
      </div>

      <h2 className="text-2xl font-bold mb-2">Viva Question Bank</h2>
      <p className="text-slate-400 mb-6 text-sm">
        Answers describe what the running system does. Cross-check against the
        Processing Pipeline page and the Dashboard's Model Evaluation tab for the exact values
        on any given run.
      </p>
      <div className="space-y-3">
        {VIVA_QA.map(([q, a], i) => (
          <details key={i} className="card p-4 group">
            <summary className="cursor-pointer font-medium text-slate-200 list-none flex items-center justify-between">
              {q}
              <span className="text-slate-500 group-open:rotate-180 transition-transform">⌄</span>
            </summary>
            <p className="text-sm text-slate-400 mt-3 leading-relaxed">{a}</p>
          </details>
        ))}
      </div>
    </div>
  );
}
