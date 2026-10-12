import React from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from "recharts";

export default function HistogramChart({ before, after }) {
  if (!before || !after) return null;
  const data = before.map((v, i) => ({ bin: i, before: v, after: after[i] }));
  return (
    <div className="card p-4">
      <div className="text-xs uppercase tracking-wider text-slate-500 mb-3">
        Intensity histogram: before and after enhancement
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data}>
          <CartesianGrid stroke="#d1d1ca" vertical={false} />
          <XAxis dataKey="bin" tick={{ fontSize: 10, fill: "#727970" }} tickCount={6} />
          <YAxis tick={{ fontSize: 10, fill: "#727970" }} width={40} />
          <Tooltip
            contentStyle={{ background: "#ffffff", border: "1px solid #d1d1ca", fontSize: 12 }}
            labelStyle={{ color: "#596159" }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line type="monotone" dataKey="before" stroke="#727970" dot={false} strokeWidth={1.5} name="Before" />
          <Line type="monotone" dataKey="after" stroke="#8a4f24" dot={false} strokeWidth={1.5} name="After (CLAHE)" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
