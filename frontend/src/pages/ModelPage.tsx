import { useQuery } from "@tanstack/react-query";
import { CartesianGrid, Bar, BarChart, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, ErrorBar } from "recharts";

import { PageHeader } from "@/components/Layout";
import { Alert, Badge, Card, CardBody, CardHeader, Skeleton, Table, Td, Th } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import type { ModelMetrics, ModelResult } from "@/lib/types";
import { cn } from "@/lib/utils";

const FEATURE_LABELS: Record<string, string> = {
  engine_rpm: "Engine RPM",
  lub_oil_pressure: "Oil pressure",
  fuel_pressure: "Fuel pressure",
  coolant_pressure: "Coolant pressure",
  lub_oil_temp: "Oil temperature",
  coolant_temp: "Coolant temperature",
  oil_fuel_ratio: "Oil / fuel pressure",
  coolant_oil_ratio: "Coolant / oil pressure",
  temp_delta: "Oil − coolant temp",
  rpm_band_low: "RPM band: low",
  rpm_band_mid: "RPM band: mid",
  rpm_band_high: "RPM band: high",
};

const f4 = (n: number) => (Number.isFinite(n) ? n.toFixed(4) : "—");

function ComparisonTable({ m }: { m: ModelMetrics }) {
  return (
    <Table>
      <thead>
        <tr>
          <Th>Model</Th>
          <Th>Imbalance</Th>
          <Th className="text-right">Threshold</Th>
          <Th className="text-right">ROC-AUC</Th>
          <Th className="text-right">Balanced acc.</Th>
          <Th className="text-right">MCC</Th>
          <Th className="text-right">Accuracy</Th>
          <Th className="text-right">Precision</Th>
          <Th className="text-right">Recall</Th>
          <Th className="text-right">F1</Th>
        </tr>
      </thead>
      <tbody>
        {m.models.map((r) => {
          const served = r.key === m.served_model;
          return (
            <tr key={r.key} className={cn(served && "bg-accent-50/60")}>
              <Td className="font-medium text-slate-900">
                {r.name} {served && <Badge className="ml-1 bg-accent-100 text-accent-700 ring-accent-100">served</Badge>}
              </Td>
              <Td className="text-slate-500">{r.strategy}</Td>
              {[
                r.test.threshold,
                r.test.roc_auc,
                r.test.balanced_accuracy,
                r.test.mcc,
                r.test.accuracy,
                r.test.precision,
                r.test.recall,
                r.test.f1,
              ].map((v, i) => (
                <Td key={i} className="text-right">
                  {i === 0 ? v.toFixed(2) : f4(v)}
                </Td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </Table>
  );
}

function ConfusionMatrix({ r }: { r: ModelResult }) {
  const [[tn, fp], [fn, tp]] = r.test.confusion_matrix;
  const total = tn + fp + fn + tp;
  const cell = (n: number, label: string, good: boolean) => (
    <div className={cn("rounded-lg p-3 text-center", good ? "bg-accent-50" : "bg-slate-100")}>
      <div className="text-xl font-semibold text-slate-900 tabular-nums">{n.toLocaleString()}</div>
      <div className="text-xs text-slate-500">
        {label} · {((n / total) * 100).toFixed(1)}%
      </div>
    </div>
  );
  return (
    <div>
      <div className="mb-1 grid grid-cols-[5rem_1fr_1fr] gap-2 text-center text-xs font-medium text-slate-500">
        <span />
        <span>Predicted healthy</span>
        <span>Predicted faulty</span>
      </div>
      <div className="grid grid-cols-[5rem_1fr_1fr] items-center gap-2">
        <span className="text-xs font-medium text-slate-500">Actually healthy</span>
        {cell(tn, "true negative", true)}
        {cell(fp, "false alarm", false)}
        <span className="text-xs font-medium text-slate-500">Actually faulty</span>
        {cell(fn, "missed fault", false)}
        {cell(tp, "caught fault", true)}
      </div>
    </div>
  );
}

export function ModelPage() {
  const q = useQuery({ queryKey: ["model-metrics"], queryFn: api.modelMetrics, staleTime: Infinity });
  if (q.isLoading)
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-48 w-full" />
        <Skeleton className="h-72 w-full" />
      </div>
    );
  if (q.isError || !q.data) return <Alert>{(q.error as Error | null)?.message ?? "Metrics unavailable"}</Alert>;
  const m = q.data;
  const served = m.models.find((r) => r.key === m.served_model)!;
  const importance = m.feature_importance
    .filter((f) => f.importance > 0.0005)
    .map((f) => ({ name: FEATURE_LABELS[f.feature] ?? f.feature, value: f.importance, err: f.std }));
  const synthetic = m.dataset.source === "synthetic";

  return (
    <>
      <PageHeader
        title="Model"
        description={`Served: ${served.name} · version ${m.model_version} · test set of ${m.split_sizes.test.toLocaleString()} readings`}
      />
      {synthetic && (
        <div className="mb-4">
          <Alert tone="info">
            These metrics come from a <strong>synthetic</strong> stand-in dataset, used for development. Retrain on the real Kaggle data
            before quoting them.
          </Alert>
        </div>
      )}
      <div className="space-y-6">
        <Card>
          <CardHeader
            title="Model comparison"
            description={`Held-out test set. Each threshold is tuned on validation for recall ≥ ${m.min_recall} (missing a fault costs more than a false alarm); the served model has the best validation ROC-AUC. F1 is shown but not used to choose: with faults as the majority class, always predicting "faulty" already scores F1 0.77, so balanced accuracy and MCC are the fairer comparison.`}
          />
          <ComparisonTable m={m} />
        </Card>

        <div className="grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader title="Confusion matrix" description={`${served.name} at threshold ${served.test.threshold.toFixed(2)}`} />
            <CardBody>
              <ConfusionMatrix r={served} />
            </CardBody>
          </Card>
          <Card>
            <CardHeader title="ROC curve" description={`${served.name} · AUC ${served.test.roc_auc.toFixed(3)}`} />
            <CardBody className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={m.roc_curve} margin={{ left: 0, right: 12, top: 8, bottom: 16 }}>
                  <CartesianGrid stroke="#f1f5f9" />
                  <XAxis
                    dataKey="fpr"
                    type="number"
                    domain={[0, 1]}
                    tickFormatter={(v: number) => v.toFixed(1)}
                    tick={{ fontSize: 11, fill: "#64748b" }}
                    label={{ value: "False positive rate", position: "insideBottom", offset: -8, fontSize: 11, fill: "#64748b" }}
                  />
                  <YAxis
                    type="number"
                    domain={[0, 1]}
                    tickFormatter={(v: number) => v.toFixed(1)}
                    tick={{ fontSize: 11, fill: "#64748b" }}
                    label={{ value: "True positive rate", angle: -90, position: "insideLeft", offset: 16, fontSize: 11, fill: "#64748b" }}
                  />
                  <Tooltip formatter={(v: number) => v.toFixed(3)} labelFormatter={(v: number) => `FPR ${Number(v).toFixed(3)}`} />
                  <Line
                    data={[
                      { fpr: 0, tpr: 0 },
                      { fpr: 1, tpr: 1 },
                    ]}
                    dataKey="tpr"
                    stroke="#cbd5e1"
                    strokeDasharray="4 4"
                    dot={false}
                    isAnimationActive={false}
                    name="Chance"
                  />
                  <Line dataKey="tpr" stroke="#1f63d8" strokeWidth={2} dot={false} name="TPR" />
                </LineChart>
              </ResponsiveContainer>
            </CardBody>
          </Card>
        </div>

        <Card>
          <CardHeader
            title="Feature importance"
            description="Permutation importance: drop in ROC-AUC when a feature is shuffled (mean ± std over 5 repeats)."
          />
          <CardBody style={{ height: Math.max(180, importance.length * 34 + 40) }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={importance} layout="vertical" margin={{ left: 8, right: 24 }}>
                <XAxis type="number" tick={{ fontSize: 11, fill: "#64748b" }} tickFormatter={(v: number) => v.toFixed(3)} />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={160}
                  tick={{ fontSize: 12, fill: "#334155" }}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip formatter={(v: number) => v.toFixed(4)} cursor={{ fill: "#f1f5f9" }} />
                <Bar dataKey="value" fill="#1f63d8" radius={[0, 3, 3, 0]} barSize={16} name="AUC drop">
                  <ErrorBar dataKey="err" width={4} stroke="#334155" direction="x" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="How the model works" />
          <CardBody className="space-y-2 text-sm leading-relaxed text-slate-600">
            <p>
              Each reading is cleaned (median imputation, clipping at the 1st/99th percentile), extended with engineered features (pressure
              ratios, the oil–coolant temperature gap, RPM bands) and standardised with a scaler fitted on the training split only. Training
              and the API load the same saved preprocessor, so they never drift.
            </p>
            <p>
              Baselines (Dummy, Logistic Regression, Random Forest, Gradient Boosting) are tuned with ROC-AUC scoring and 5-fold stratified
              cross-validation, comparing class weighting against SMOTE inside the pipeline. The PyTorch network (12 → 64 → 32 → 1,
              BatchNorm, dropout 0.2) trains with Adam and early stopping on validation loss
              {m.training_history.length ? ` (stopped after ${m.training_history.length} epochs)` : ""}.
            </p>
            <p>
              The "readings that drove this result" on the Diagnose page come from occlusion: each sensor is reset to a typical healthy
              value and the change in fault probability is measured. This is faithful to the model, but it reflects the model's
              associations, not mechanical cause and effect.
            </p>
          </CardBody>
        </Card>
      </div>
    </>
  );
}
