import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import UploadDropzone from "../components/UploadDropzone.jsx";
import { useInspection } from "../context/InspectionContext.jsx";
import { inspectImage, getModelInfo } from "../api/client.js";

export default function Inspection() {
  const {
    pendingFile, setPendingFile, pendingPreviewUrl, setPendingPreviewUrl,
    mode, setMode, isProcessing, setIsProcessing,
    setCurrentResult, error, setError, reset,
  } = useInspection();
  const navigate = useNavigate();
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    getModelInfo().then(setModelInfo).catch(() => setModelInfo({ model_available: false }));
  }, []);

  const handleFile = (file, url) => {
    setPendingFile(file);
    setPendingPreviewUrl(url);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!pendingFile) return;
    setIsProcessing(true);
    setError(null);
    try {
      const result = await inspectImage(pendingFile, mode);
      setCurrentResult(result);
      navigate("/pipeline");
    } catch (e) {
      setError(e.message || "Inspection failed");
    } finally {
      setIsProcessing(false);
    }
  };

  const realDisabled = modelInfo && !modelInfo.model_available;

  return (
    <div className="max-w-4xl mx-auto px-5 py-14">
      <div className="mb-8">
        <div className="label-chip border-ink-600 text-slate-400 mb-3">STEP 1</div>
        <h1 className="text-3xl font-bold">Upload for Inspection</h1>
        <p className="text-slate-400 mt-2">
          Choose a mode, then upload an image. The system runs the same DIP pipeline in both
          modes; only the final classification step differs.
        </p>
      </div>

      <div className="card p-6">
        <label className="text-xs uppercase tracking-wider text-slate-500">Mode</label>
        <div className="grid sm:grid-cols-2 gap-3 mt-2">
          <button
            type="button"
            onClick={() => setMode("demo")}
            className={`text-left p-4 rounded-sm border-2 transition-colors ${
              mode === "demo" ? "border-accent-500 bg-accent-500/5" : "border-ink-600 hover:border-ink-500"
            }`}
          >
            <div className="label-chip border-ink-600 text-slate-400 mb-2">DEMO / SYNTHETIC DATA</div>
            <div className="font-semibold">Demo Mode</div>
            <div className="text-xs text-slate-400 mt-1 leading-relaxed">
              Rule-based classifier. Works on any image, but is most useful with a synthetic surface
              (see scripts/generate_demo_dataset.py). It demonstrates DIP segmentation and candidate
              defect-region detection on a surface with an isolated defect.
            </div>
          </button>

          <button
            type="button"
            disabled={realDisabled}
            onClick={() => !realDisabled && setMode("real")}
            className={`text-left p-4 rounded-sm border-2 transition-colors ${
              realDisabled ? "opacity-40 cursor-not-allowed border-ink-600" :
              mode === "real" ? "border-ok-500 bg-ok-500/5" : "border-ink-600 hover:border-ink-500"
            }`}
          >
            <div className="label-chip border-ok-500/40 text-ok-500 bg-ok-500/10 mb-2">TRAINED MODEL</div>
            <div className="font-semibold">Real ML Mode</div>
            <div className="text-xs text-slate-400 mt-1 leading-relaxed">
              Trained RandomForest on the real NEU-CLS steel surface defect dataset (6 classes:
              crazing, inclusion, patches, pitted surface, rolled-in scale, scratches). Predicts
              which known defect category the image most resembles.
              {realDisabled && " (No trained model found. Run ml/train_classifier.py.)"}
            </div>
          </button>
        </div>

        {mode === "real" && (
          <div className="mt-3 px-3 py-2 rounded-md bg-accent-500/10 border border-accent-500/30 text-xs text-slate-300">
            This model was trained and validated only on the NEU-CLS steel surface dataset
            (grayscale, ~200×200px crops). Results on unrelated images are not meaningful.
            For a genuine test, upload one of the sample images from
            <code className="mx-1 font-mono">backend/data/raw/NEU-CLS/&lt;class&gt;/</code>.
          </div>
        )}

        <div className="mt-6">
          <UploadDropzone file={pendingFile} previewUrl={pendingPreviewUrl} onFile={handleFile} />
        </div>

        {error && (
          <div className="mt-5 px-4 py-3 rounded-md bg-warn-500/10 border border-warn-500/30 text-warn-500 text-sm">
            {error}
          </div>
        )}

        <div className="mt-6 flex gap-3">
          <button
            onClick={handleAnalyze}
            disabled={!pendingFile || isProcessing}
            className="px-5 py-2.5 rounded-md bg-accent-500 hover:bg-accent-600 disabled:bg-ink-700 disabled:text-slate-500 text-white font-semibold transition-colors flex items-center gap-2"
          >
            {isProcessing && (
              <span className="w-3.5 h-3.5 rounded-full border-2 border-white/40 border-t-white animate-spin" />
            )}
            {isProcessing ? "Processing…" : "Analyze Image"}
          </button>
          <button
            onClick={reset}
            disabled={isProcessing}
            className="px-5 py-2.5 rounded-md border border-ink-600 hover:border-ink-500 font-semibold transition-colors"
          >
            Reset
          </button>
        </div>
      </div>
    </div>
  );
}
