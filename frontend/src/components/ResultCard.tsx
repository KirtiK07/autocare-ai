import { Gauge } from "lucide-react";
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Card, CardBody, CardHeader, EmptyState, RiskBadge, Skeleton } from "@/components/ui/primitives";
import type { Prediction } from "@/lib/types";
import { pct, RISK_STYLES } from "@/lib/utils";

/** Semicircle gauge: arc length = fault probability; colour = risk band. */
export function RiskGauge({ probability, band }: { probability: number; band: Prediction["risk_band"] }) {
  const r = 80;
  const circumference = Math.PI * r;
  const filled = Math.max(0.005, Math.min(1, probability)) * circumference;
  return (
    <svg viewBox="0 0 200 116" className="mx-auto w-full max-w-[260px]" role="img" aria-label={`Fault probability ${pct(probability)}`}>
      <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#e2e8f0" strokeWidth="16" strokeLinecap="round" />
      <path
        d="M20 100 A80 80 0 0 1 180 100"
        fill="none"
        stroke={RISK_STYLES[band].hex}
        strokeWidth="16"
        strokeLinecap="round"
        strokeDasharray={`${filled} ${circumference}`}
        style={{ transition: "stroke-dasharray 600ms ease" }}
      />
      <text x="100" y="88" textAnchor="middle" className="fill-slate-900" style={{ fontSize: 30, fontWeight: 600 }}>
        {pct(probability)}
      </text>
      <text x="100" y="108" textAnchor="middle" className="fill-slate-500" style={{ fontSize: 11 }}>
        fault probability
      </text>
    </svg>
  );
}

export function DriversChart({ prediction }: { prediction: Prediction }) {
  const data = prediction.top_features.map((f) => ({
    name: f.label,
    deviation: f.deviation,
    value: f.value,
    direction: f.direction,
  }));
  // Symmetric axis around 0 so "lower than usual" and "higher than usual" read at a glance.
  const bound = Math.ceil(Math.max(1, ...data.map((d) => Math.abs(d.deviation))));
  return (
    <div className="h-40">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
          <XAxis
            type="number"
            domain={[-bound, bound]}
            allowDecimals={false}
            tick={{ fontSize: 11, fill: "#64748b" }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 12, fill: "#334155" }} tickLine={false} axisLine={false} />
          <ReferenceLine x={0} stroke="#94a3b8" />
          <Tooltip
            cursor={{ fill: "#f1f5f9" }}
            formatter={(v: number, _n, item) => [`${v > 0 ? "+" : ""}${v.toFixed(2)} σ (reading ${item.payload.value})`, "vs healthy mean"]}
          />
          <Bar dataKey="deviation" radius={[3, 3, 3, 3]} barSize={18}>
            {data.map((d) => (
              <Cell key={d.name} fill={d.direction === "high" ? "#1f63d8" : "#64748b"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ResultCard({ prediction, loading }: { prediction: Prediction | null; loading?: boolean }) {
  return (
    <Card className="h-full">
      <CardHeader
        title="Fault risk"
        description="Model estimate from this snapshot, not an inspection."
        action={prediction && !loading ? <RiskBadge band={prediction.risk_band} /> : null}
      />
      <CardBody>
        {loading ? (
          <div className="space-y-4" aria-busy="true" aria-label="Loading prediction">
            <Skeleton className="mx-auto h-32 w-56 rounded-full" />
            <Skeleton className="h-4 w-1/3" />
            <Skeleton className="h-28 w-full" />
          </div>
        ) : prediction ? (
          <div className="space-y-4" data-testid="result">
            <RiskGauge probability={prediction.probability} band={prediction.risk_band} />
            <p className="text-center text-sm text-slate-600">
              Classified as <strong className="font-semibold text-slate-900">{prediction.label}</strong>
            </p>
            <div>
              <h3 className="text-sm font-medium text-slate-800">Readings that drove this result</h3>
              <p className="mb-2 text-xs text-slate-500">Distance from the healthy-engine average, in standard deviations (σ).</p>
              <DriversChart prediction={prediction} />
              <ul className="mt-2 space-y-1 text-sm text-slate-600">
                {prediction.top_features.map((f) => (
                  <li key={f.feature}>
                    <span className="font-medium text-slate-800">{f.label}</span> is {f.direction === "high" ? "higher" : "lower"} than
                    usual ({f.value}, {f.deviation > 0 ? "+" : ""}
                    {f.deviation.toFixed(1)} σ)
                  </li>
                ))}
              </ul>
            </div>
            <p className="text-xs text-slate-400">Model {prediction.model_version}</p>
          </div>
        ) : (
          <EmptyState icon={<Gauge className="size-5" />} title="No prediction yet">
            Enter readings or load a sample, then press <em>Predict fault risk</em>.
          </EmptyState>
        )}
      </CardBody>
    </Card>
  );
}
