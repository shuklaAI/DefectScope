import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useInspection } from "../context/InspectionContext.jsx";
import StageCard from "../components/StageCard.jsx";
import HistogramChart from "../components/HistogramChart.jsx";

export default function Pipeline() {
  const { currentResult } = useInspection();
  const [activeStep, setActiveStep] = useState(0);
  const navigate = useNavigate();

  if (!currentResult) {
    return (
      <div className="max-w-3xl mx-auto px-5 py-20 text-center">
        <h1 className="text-2xl font-semibold">No inspection loaded</h1>
        <p className="text-slate-400 mt-2">Run an inspection first to view its processing pipeline.</p>
        <Link to="/inspection" className="inline-block mt-6 px-5 py-2.5 rounded-md bg-accent-500 hover:bg-accent-600 text-white font-semibold">
          Start Inspection
        </Link>
      </div>
    );
  }

  const stages = currentResult.stages;
  const stage = stages[activeStep];

  return (
    <div className="max-w-6xl mx-auto px-5 py-12">
      <div className="flex items-center justify-between mb-8 flex-wrap gap-4">
        <div>
          <div className="label-chip border-ink-600 text-slate-400 mb-2">STEP 2</div>
          <h1 className="text-3xl font-bold">Processing Pipeline</h1>
          <p className="text-slate-400 mt-1">Click through each stage to see the technique, output image and parameters.</p>
        </div>
        <button
          onClick={() => navigate("/results")}
          className="px-5 py-2.5 rounded-md bg-accent-500 hover:bg-accent-600 text-white font-semibold"
        >
          View Final Result →
        </button>
      </div>

      <div className="flex gap-2 overflow-x-auto pb-3 mb-6">
        {stages.map((s, i) => (
          <button
            key={s.key}
            onClick={() => setActiveStep(i)}
            className={`whitespace-nowrap px-3.5 py-2 rounded-md text-sm font-medium border transition-colors ${
              i === activeStep
                ? "border-accent-500 bg-accent-500/10 text-accent-400"
                : "border-ink-600 text-slate-400 hover:border-ink-500"
            }`}
          >
            {s.step}. {s.title}
          </button>
        ))}
      </div>

      <StageCard stage={stage} />

      {stage.histogram && (
        <div className="mt-6">
          <HistogramChart before={stage.histogram.before} after={stage.histogram.after} />
        </div>
      )}

      <div className="flex justify-between mt-8">
        <button
          onClick={() => setActiveStep((s) => Math.max(0, s - 1))}
          disabled={activeStep === 0}
          className="px-4 py-2 rounded-md border border-ink-600 disabled:opacity-30 text-sm font-medium"
        >
          ← Previous Stage
        </button>
        <button
          onClick={() => setActiveStep((s) => Math.min(stages.length - 1, s + 1))}
          disabled={activeStep === stages.length - 1}
          className="px-4 py-2 rounded-md border border-ink-600 disabled:opacity-30 text-sm font-medium"
        >
          Next Stage →
        </button>
      </div>
    </div>
  );
}
