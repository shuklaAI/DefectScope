import React from "react";
import { Link } from "react-router-dom";

const PIPELINE_STEPS = [
  "Original Image", "Resize & Normalize", "Grayscale", "Noise Reduction",
  "Contrast Enhancement", "Edge Detection", "Segmentation", "Morphology",
  "Connected Components", "Candidate Regions", "Classification",
];

const CAPABILITIES = [
  { title: "Image processing", desc: "Resize and normalize, grayscale conversion, median filtering, CLAHE, Canny edges, Otsu thresholding, morphology and connected components." },
  { title: "Stage outputs", desc: "Review each intermediate image alongside the operation and parameters used at that stage." },
  { title: "Classifier", desc: "The Random Forest was trained on 1,799 NEU-CLS images across six defect categories. Demo mode uses a separate rule-based classifier." },
  { title: "Model score", desc: "Real mode shows the model's predict_proba output. The probability is not calibrated." },
  { title: "Evaluation", desc: "The Dashboard reports metrics from the 360-image held-out test split, separate from inspection history." },
  { title: "Inspection records", desc: "Each run is saved with its prediction, processing time, candidate regions and stage outputs." },
];

const STACK = ["React", "Vite", "Tailwind CSS", "FastAPI", "OpenCV", "NumPy", "scikit-image", "SciPy", "scikit-learn", "SQLite"];

export default function Home() {
  return (
    <div>
      <section className="border-b border-ink-600">
        <div className="max-w-7xl mx-auto px-5 py-20 grid lg:grid-cols-2 gap-12 items-center">
          <div>
            <div className="label-chip border-ink-600 text-slate-400 mb-5">
              <span className="w-1.5 h-1.5 rounded-full bg-ok-500" /> DIGITAL IMAGE PROCESSING · FINAL YEAR PROJECT
            </div>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight leading-[1.1]">
              Steel surface defect inspection, with each processing step visible
            </h1>
            <p className="mt-5 text-slate-400 text-lg leading-relaxed max-w-xl">
              Upload a surface image to resize it to 512 × 512, inspect 11 image-processing stages
              and extract a 23-value feature vector. Real ML mode classifies NEU-CLS images into
              six defect categories. Demo mode applies a separate rule-based classifier intended
              for synthetic samples.
            </p>
            <div className="mt-8 flex gap-3">
              <Link to="/inspection" className="px-5 py-3 rounded-md bg-accent-500 hover:bg-accent-600 text-white font-semibold transition-colors">
                Start Inspection
              </Link>
              <Link to="/about" className="px-5 py-3 rounded-md border border-ink-600 hover:border-ink-500 font-semibold transition-colors">
                View Methodology
              </Link>
            </div>
          </div>
          <div className="card p-5">
            <div className="text-xs uppercase tracking-wider text-slate-500 mb-4">Processing Pipeline</div>
            <div className="space-y-2">
              {PIPELINE_STEPS.map((s, i) => (
                <div key={s} className="flex items-center gap-3">
                  <div className="w-6 h-6 rounded-sm bg-ink-700 border border-ink-600 flex items-center justify-center text-[10px] font-mono text-accent-400 shrink-0">
                    {i + 1}
                  </div>
                  <div className="h-px flex-1 bg-ink-600" />
                  <div className="text-sm font-mono text-slate-300 w-48 text-right">{s}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="max-w-7xl mx-auto px-5 py-16">
        <h2 className="text-2xl font-semibold mb-2">How it works</h2>
        <p className="text-slate-400 max-w-2xl mb-8">
          The image is segmented using Otsu thresholding and dilated Canny edges. Morphology
          cleans the mask before connected components are measured. Regions are candidates, not
          verified defect boundaries.
        </p>
        <div className="grid md:grid-cols-3 gap-4">
          {[
            { n: "01", t: "Preprocess", d: "Resize, normalize, convert to grayscale, denoise, and enhance contrast (CLAHE) so subtle defects become measurable." },
            { n: "02", t: "Segment", d: "Canny edges combined with Otsu thresholding isolate anomalous regions; morphology and connected-component analysis produce candidate defect regions (not verified ground-truth boundaries, since NEU-CLS provides no segmentation masks)." },
            { n: "03", t: "Classify", d: "Demo mode applies a rule-based heuristic. Real ML mode uses the trained Random Forest to return one of six NEU-CLS classes and its uncalibrated predict_proba score." },
          ].map((c) => (
            <div key={c.n} className="card p-5">
              <div className="text-3xl font-mono text-slate-500 font-bold">{c.n}</div>
              <div className="font-semibold mt-2">{c.t}</div>
              <div className="text-sm text-slate-400 mt-1.5 leading-relaxed">{c.d}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="max-w-7xl mx-auto px-5 py-16 border-t border-ink-600">
        <h2 className="text-2xl font-semibold mb-8">Key capabilities</h2>
        <div className="grid md:grid-cols-3 gap-4">
          {CAPABILITIES.map((c) => (
            <div key={c.title} className="card p-5">
              <div className="font-semibold text-accent-400">{c.title}</div>
              <div className="text-sm text-slate-400 mt-1.5 leading-relaxed">{c.desc}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="max-w-7xl mx-auto px-5 py-16 border-t border-ink-600">
        <h2 className="text-2xl font-semibold mb-6">Technology stack</h2>
        <div className="flex flex-wrap gap-2">
          {STACK.map((t) => (
            <span key={t} className="label-chip border-ink-600 text-slate-300 font-mono">{t}</span>
          ))}
        </div>
      </section>
    </div>
  );
}
