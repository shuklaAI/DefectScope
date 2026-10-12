import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { assetUrl, listInspections } from "../api/client.js";

export default function History() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  useEffect(() => {
    listInspections(100)
      .then(setItems)
      .catch((e) => setErr(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="max-w-6xl mx-auto px-5 py-12">
      <h1 className="text-3xl font-bold mb-2">Inspection History</h1>
      <p className="text-slate-400 mb-8">All previous inspections, most recent first.</p>

      {loading && <div className="text-slate-400">Loading…</div>}
      {err && <div className="text-warn-500">{err}</div>}
      {!loading && !err && items.length === 0 && (
        <div className="card p-10 text-center text-slate-400">
          No inspections yet. <Link to="/inspection" className="text-accent-400 font-medium">Run your first inspection →</Link>
        </div>
      )}

      <div className="space-y-2">
        {items.map((r) => {
          const isPositive = r.result === "DEFECT_DETECTED" || r.result === "CLASSIFIED";
          return (
            <Link
              key={r.id}
              to={`/history/${r.id}`}
              className="card p-4 flex items-center gap-4 hover:border-ink-500 transition-colors"
            >
              <img src={assetUrl(r.original_image_url)} alt="" className="w-14 h-14 object-cover rounded-md border border-ink-600" />
              <div className="flex-1 min-w-0">
                <div className="font-medium truncate">{r.original_filename}</div>
                <div className="text-xs text-slate-500 font-mono mt-0.5 flex items-center gap-2 flex-wrap">
                  <span>{new Date(r.timestamp).toLocaleString()}</span>
                  <span className={`label-chip !px-1.5 !py-0 border-ink-600 ${r.mode === "real" ? "text-ok-500" : "text-slate-400"}`}>
                    {r.model_mode}
                  </span>
                  {r.model_version && <span>model v{r.model_version}</span>}
                </div>
                <div className="text-[11px] text-slate-500 font-mono mt-0.5">
                  {r.num_regions} candidate region{r.num_regions === 1 ? "" : "s"}
                  {r.defect_area_percent != null && ` · candidate area ${r.defect_area_percent}%`}
                </div>
              </div>
              <div className="text-right">
                <div className={`text-sm font-semibold ${isPositive ? "text-warn-500" : "text-ok-500"}`}>
                  {r.predicted_class || (isPositive ? "Defect" : "No Defect")}
                </div>
                <div className="text-xs text-slate-500 font-mono">
                  {r.mode === "real" && r.model_probability != null
                    ? `model prob. ${(r.model_probability * 100).toFixed(1)}% · `
                    : ""}
                  {r.processing_time_ms} ms
                </div>
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
