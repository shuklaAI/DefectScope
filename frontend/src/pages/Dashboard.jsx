import React, { useEffect, useState } from "react";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend, BarChart, Bar, XAxis, YAxis, CartesianGrid } from "recharts";
import { Link } from "react-router-dom";
import { getDashboard, getModelInfo } from "../api/client.js";

const COLORS = ["#8a4f24", "#a54232", "#39714d", "#a67c36", "#6e675e", "#8a6a59"];

export default function Dashboard() {
  const [tab, setTab] = useState("activity");
  const [stats, setStats] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([getDashboard(), getModelInfo()])
      .then(([s, m]) => { setStats(s); setModelInfo(m); })
      .catch((e) => setErr(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="max-w-6xl mx-auto px-5 py-20 text-slate-400">Loading dashboard…</div>;
  if (err) return <div className="max-w-6xl mx-auto px-5 py-20 text-warn-500">{err}</div>;

  return (
    <div className="max-w-6xl mx-auto px-5 py-12">
      <h1 className="text-3xl font-bold mb-2">Dashboard</h1>
      <p className="text-slate-400 mb-6">
        Inspection Activity (from stored inspections) and Model Evaluation (from the held-out
        test split at training time) are kept strictly separate and are never blended.
      </p>

      <div className="flex gap-2 mb-8 border-b border-ink-600">
        {[
          { id: "activity", label: "Inspection Activity" },
          { id: "evaluation", label: "Model Evaluation" },
        ].map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`px-4 py-2.5 text-sm font-semibold border-b-2 transition-colors ${
              tab === t.id ? "border-accent-500 text-accent-400" : "border-transparent text-slate-500 hover:text-slate-300"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "activity" && <ActivityTab stats={stats} />}
      {tab === "evaluation" && <EvaluationTab modelInfo={modelInfo} />}
    </div>
  );
}

function ActivityTab({ stats }) {
  if (!stats.has_data) {
    return (
      <div className="max-w-xl mx-auto text-center py-16">
        <h2 className="text-xl font-semibold">No inspection data yet</h2>
        <p className="text-slate-400 mt-2 text-sm">
          Statistics are calculated from completed inspections. Run an inspection to populate
          this view.
        </p>
        <Link to="/inspection" className="inline-block mt-6 px-5 py-2.5 rounded-md bg-accent-500 hover:bg-accent-600 text-white font-semibold">
          Run an Inspection
        </Link>
      </div>
    );
  }

  return (
    <div>
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <Stat label="Total Inspections" value={stats.total_inspections} />
        <Stat label="Demo Mode Runs" value={stats.demo_inspections} />
        <Stat label="Real ML Mode Runs" value={stats.real_inspections} accent="text-ok-500" />
        <Stat label="Avg. Processing Time" value={`${stats.avg_processing_time_ms} ms`} />
      </div>

      <div className="grid lg:grid-cols-2 gap-6 mb-6">
        <DistributionCard
          title="Real ML Mode: predicted classes"
          data={stats.real_class_distribution}
          emptyText="No Real ML Mode inspections yet."
        />
        <DistributionCard
          title="Demo Mode: heuristic labels"
          data={stats.demo_class_distribution}
          emptyText="No Demo Mode inspections yet."
          footnote={`Demo results: ${stats.demo_defect_detected} with a candidate defect, ${stats.demo_no_defect} without.`}
        />
      </div>

      <div className="card p-5">
        <div className="text-xs uppercase tracking-wider text-slate-500 mb-4">Recent Inspections</div>
        <div className="space-y-2">
          {stats.recent_inspections.map((r) => (
            <Link key={r.id} to={`/history/${r.id}`} className="flex items-center justify-between text-sm px-3 py-2 rounded-md hover:bg-ink-700 transition-colors">
              <span className="truncate flex-1">{r.original_filename}</span>
              <span className="text-[10px] font-mono text-slate-500 mr-2">{r.mode === "real" ? "REAL" : "DEMO"}</span>
              <span className={r.result === "NO_DEFECT" ? "text-ok-500" : "text-warn-500"}>
                {r.predicted_class || (r.result === "NO_DEFECT" ? "OK" : "N/A")}
              </span>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}

function DistributionCard({ title, data, emptyText, footnote }) {
  const pieData = Object.entries(data).map(([name, value]) => ({ name, value }));
  return (
    <div className="card p-5">
      <div className="text-xs uppercase tracking-wider text-slate-500 mb-4">{title}</div>
      {pieData.length === 0 ? (
        <div className="text-slate-500 text-sm py-10 text-center">{emptyText}</div>
      ) : (
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie data={pieData} dataKey="value" nameKey="name" outerRadius={80} label>
              {pieData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #d1d1ca", fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
          </PieChart>
        </ResponsiveContainer>
      )}
      {footnote && <div className="text-xs text-slate-500 mt-2">{footnote}</div>}
    </div>
  );
}

function EvaluationTab({ modelInfo }) {
  if (!modelInfo || !modelInfo.model_available) {
    return (
      <div className="max-w-xl mx-auto text-center py-16">
        <h2 className="text-xl font-semibold">No trained model found</h2>
        <p className="text-slate-400 mt-2 text-sm">
          {modelInfo?.message || "Run ml/train_classifier.py against backend/data/raw/NEU-CLS/ to train and evaluate a model."}
        </p>
      </div>
    );
  }

  const { metadata, evaluation } = modelInfo;
  const perClassRows = evaluation.class_order.map((cls) => ({
    name: cls,
    precision: evaluation.per_class_report[cls]?.precision ?? 0,
    recall: evaluation.per_class_report[cls]?.recall ?? 0,
    f1: evaluation.per_class_report[cls]?.["f1-score"] ?? 0,
  }));

  return (
    <div>
      <div className="mb-6 px-4 py-3 rounded-md bg-accent-500/10 border border-accent-500/30 text-xs text-slate-300 space-y-1.5">
        <div className="font-semibold text-slate-200 uppercase tracking-wider text-[10px]">
          Evaluation on held-out test set
        </div>
        <div>
          {metadata.model_type} achieved {(evaluation.accuracy * 100).toFixed(2)}% accuracy on a
          stratified held-out test split of the NEU-CLS dataset ({evaluation.test_set_size} test
          images; {evaluation.train_set_size} training images; {evaluation.class_order.length} classes;
          random seed {metadata.random_seed}).
        </div>
        <div className="text-slate-400">
          Methodology: one stratified train/test split; StandardScaler fit on the training split only;
          metrics computed once on the test split, which the model never saw during training. They are
          not computed from training data or from inspection history. This is a single split on one
          benchmark dataset, not a validation on production-line images.
        </div>
      </div>

      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <Stat label="Accuracy" value={`${(evaluation.accuracy * 100).toFixed(2)}%`} accent="text-ok-500" />
        <Stat label="Macro Precision" value={`${(evaluation.macro_precision * 100).toFixed(2)}%`} />
        <Stat label="Macro Recall" value={`${(evaluation.macro_recall * 100).toFixed(2)}%`} />
        <Stat label="Macro F1" value={`${(evaluation.macro_f1 * 100).toFixed(2)}%`} />
      </div>

      <div className="grid lg:grid-cols-2 gap-6 mb-8">
        <div className="card p-5">
          <div className="text-xs uppercase tracking-wider text-slate-500 mb-4">Per-Class Metrics (test set)</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={perClassRows}>
              <CartesianGrid stroke="#d1d1ca" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 9, fill: "#727970" }} angle={-20} textAnchor="end" height={60} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: "#727970" }} width={30} />
              <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #d1d1ca", fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="precision" fill="#8a4f24" />
              <Bar dataKey="recall" fill="#39714d" />
              <Bar dataKey="f1" fill="#a67c36" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card p-5">
          <div className="text-xs uppercase tracking-wider text-slate-500 mb-1">NEU-CLS test split confusion matrix</div>
          <p className="text-xs text-slate-500 mb-4">Counts from {evaluation.test_set_size} held-out images. Rows are actual classes; columns are predicted classes.</p>
          <div className="overflow-x-auto">
            <table className="text-xs font-mono w-full">
              <thead>
                <tr>
                  <th className="text-left p-1"></th>
                  {evaluation.class_order.map((c) => (
                    <th key={c} className="p-1 text-slate-500 font-normal">{c.slice(0, 4)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {evaluation.confusion_matrix.map((row, i) => (
                  <tr key={i}>
                    <td className="p-1 text-slate-500 whitespace-nowrap">{evaluation.class_order[i].slice(0, 10)}</td>
                    {row.map((val, j) => (
                      <td key={j} className={`p-1 text-center rounded ${i === j ? "bg-ok-500/20 text-ok-500" : val > 0 ? "bg-warn-500/10 text-warn-500" : "text-slate-600"}`}>
                        {val}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="card p-5 mb-8">
        <div className="text-xs uppercase tracking-wider text-slate-500 mb-3">Per-Class Precision / Recall / F1 (held-out test set)</div>
        <div className="overflow-x-auto">
          <table className="text-sm font-mono w-full">
            <thead>
              <tr className="text-slate-500 text-xs">
                <th className="text-left p-1.5 font-normal">Class</th>
                <th className="text-right p-1.5 font-normal">Precision</th>
                <th className="text-right p-1.5 font-normal">Recall</th>
                <th className="text-right p-1.5 font-normal">F1</th>
                <th className="text-right p-1.5 font-normal">Test support</th>
              </tr>
            </thead>
            <tbody>
              {evaluation.class_order.map((cls) => {
                const r = evaluation.per_class_report[cls] || {};
                const pct = (v) => (v != null ? `${(v * 100).toFixed(2)}%` : "N/A");
                return (
                  <tr key={cls} className="border-t border-ink-600">
                    <td className="p-1.5">{cls}</td>
                    <td className="p-1.5 text-right">{pct(r.precision)}</td>
                    <td className="p-1.5 text-right">{pct(r.recall)}</td>
                    <td className="p-1.5 text-right">{pct(r["f1-score"])}</td>
                    <td className="p-1.5 text-right">{r.support ?? "N/A"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card p-5">
        <div className="text-xs uppercase tracking-wider text-slate-500 mb-3">Dataset & Training</div>
        <div className="grid sm:grid-cols-3 gap-4 text-sm">
          <div><span className="text-slate-500">Dataset:</span> {metadata.dataset_name}</div>
          <div><span className="text-slate-500">Total samples:</span> {metadata.total_samples} ({evaluation.class_order.length} classes)</div>
          <div><span className="text-slate-500">Train / Test:</span> {metadata.train_samples} / {metadata.test_samples}</div>
          <div><span className="text-slate-500">Model:</span> {metadata.model_type} v{metadata.model_version}</div>
          <div><span className="text-slate-500">Random seed:</span> {metadata.random_seed}</div>
          <div><span className="text-slate-500">Trained:</span> {new Date(metadata.training_date_utc).toLocaleString()}</div>
        </div>
        <div className="mt-4">
          <div className="text-xs text-slate-500 mb-2">Class distribution (full dataset)</div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(metadata.dataset_class_counts).map(([cls, count]) => (
              <span key={cls} className="label-chip border-ink-600 text-slate-300 font-mono">{cls}: {count}</span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value, accent }) {
  return (
    <div className="card px-4 py-3">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`text-2xl font-bold font-mono mt-0.5 ${accent || ""}`}>{value}</div>
    </div>
  );
}
