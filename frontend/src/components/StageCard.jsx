import React from "react";
import { assetUrl } from "../api/client.js";

export default function StageCard({ stage, active }) {
  if (!stage) return null;
  return (
    <div className="grid md:grid-cols-2 gap-6">
      <div className="card p-3 flex items-center justify-center min-h-[320px]">
        {stage.image_url ? (
          <img
            src={assetUrl(stage.image_url)}
            alt={stage.title}
            className="max-h-[420px] max-w-full rounded-md border border-ink-600"
          />
        ) : (
          <div className="text-slate-500 text-sm">No image for this stage</div>
        )}
      </div>
      <div className="space-y-4">
        <div>
          <div className="label-chip border-accent-500/40 text-accent-400 bg-accent-500/10">
            STEP {stage.step} / 11
          </div>
          <h3 className="text-2xl font-semibold mt-3">{stage.title}</h3>
          <div className="text-sm font-mono text-slate-400 mt-1">{stage.technique}</div>
        </div>
        <p className="text-slate-300 leading-relaxed text-sm">{stage.explanation}</p>

        {stage.parameters && Object.keys(stage.parameters).length > 0 && (
          <div>
            <div className="text-xs uppercase tracking-wider text-slate-500 mb-2">Parameters</div>
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(stage.parameters).map(([k, v]) => (
                <div key={k} className="card px-3 py-2">
                  <div className="text-[10px] font-mono text-slate-500 uppercase">{k.replace(/_/g, " ")}</div>
                  <div className="text-sm font-mono text-slate-200">
                    {typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(3)) : String(v)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
