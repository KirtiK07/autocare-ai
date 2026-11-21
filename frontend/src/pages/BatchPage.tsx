import { useMutation } from "@tanstack/react-query";
import { Download, FileUp, Loader2, Play } from "lucide-react";
import { useRef, useState, type DragEvent } from "react";

import { PageHeader } from "@/components/Layout";
import { Alert, Button, Card, CardBody, CardHeader, EmptyState, RiskBadge, Table, Td, Th } from "@/components/ui/primitives";
import { api, ApiError } from "@/lib/api";
import { SENSOR_KEYS, type BatchResponse } from "@/lib/types";
import { cn, pct, toCsv } from "@/lib/utils";

const TEMPLATE = `${SENSOR_KEYS.join(",")}\n700,2.5,11.0,3.1,84.0,81.0\n900,3.4,6.2,2.2,77.0,78.0\n`;

function parsePreview(text: string): string[][] {
  return text
    .split(/\r?\n/)
    .filter((l) => l.trim())
    .slice(0, 11)
    .map((l) => l.split(","));
}

function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  URL.revokeObjectURL(url);
}

export function BatchPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string[][]>([]);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const run = useMutation<BatchResponse, ApiError, File>({ mutationFn: api.predictBatch });

  const pick = async (f: File | undefined) => {
    if (!f) return;
    setFile(f);
    run.reset();
    setPreview(parsePreview(await f.text()));
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    void pick(e.dataTransfer.files[0]);
  };

  const exportResults = () => {
    if (!run.data) return;
    download(
      "autocare_predictions.csv",
      toCsv(
        run.data.results.map((r) => ({
          row: r.row,
          ...r.inputs,
          probability: r.prediction?.probability ?? "",
          label: r.prediction?.label ?? "",
          risk_band: r.prediction?.risk_band ?? "",
          top_driver: r.prediction?.top_features[0]?.label ?? "",
          error: r.error ?? "",
        })),
      ),
    );
  };

  return (
    <>
      <PageHeader
        title="Batch prediction"
        description="Upload a CSV of readings (up to 1,000 rows). Results are not saved to history."
        action={
          <Button variant="outline" size="sm" onClick={() => download("autocare_template.csv", TEMPLATE)}>
            <Download className="size-3.5" /> CSV template
          </Button>
        }
      />
      <div className="space-y-6">
        <Card>
          <CardBody>
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
              className={cn(
                "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-6 py-10 text-center transition-colors",
                dragging ? "border-accent-500 bg-accent-50" : "border-slate-300 hover:border-slate-400",
              )}
            >
              <FileUp className="size-6 text-slate-400" />
              <p className="text-sm font-medium text-slate-700">{file ? file.name : "Drop a CSV here, or click to choose"}</p>
              <p className="text-xs text-slate-500">Columns: {SENSOR_KEYS.join(", ")} (Kaggle headers also work)</p>
              <input
                ref={inputRef}
                type="file"
                accept=".csv,text/csv"
                className="hidden"
                onChange={(e) => void pick(e.target.files?.[0])}
              />
            </div>
          </CardBody>
        </Card>

        {preview.length > 0 && (
          <Card>
            <CardHeader
              title="Preview"
              description="First 10 rows"
              action={
                <Button onClick={() => file && run.mutate(file)} disabled={run.isPending}>
                  {run.isPending ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-4" />}
                  Run predictions
                </Button>
              }
            />
            <Table>
              <thead>
                <tr>
                  {preview[0].map((h, i) => (
                    <Th key={i}>{h}</Th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.slice(1).map((r, i) => (
                  <tr key={i}>
                    {r.map((c, j) => (
                      <Td key={j}>{c}</Td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </Table>
          </Card>
        )}

        {run.isError && <Alert>{run.error.message}</Alert>}

        {run.data && (
          <Card>
            <CardHeader
              title="Results"
              description={`${run.data.succeeded} scored · ${run.data.failed} failed`}
              action={
                <Button variant="outline" size="sm" onClick={exportResults}>
                  <Download className="size-3.5" /> Download CSV
                </Button>
              }
            />
            {run.data.results.length === 0 ? (
              <EmptyState icon={<FileUp className="size-5" />} title="The file had no data rows" />
            ) : (
              <Table>
                <thead>
                  <tr>
                    <Th>Row</Th>
                    <Th>Risk</Th>
                    <Th>Probability</Th>
                    <Th>Top driver</Th>
                    <Th>Note</Th>
                  </tr>
                </thead>
                <tbody>
                  {run.data.results.map((r) => (
                    <tr key={r.row}>
                      <Td>{r.row}</Td>
                      <Td>{r.prediction ? <RiskBadge band={r.prediction.risk_band} /> : "—"}</Td>
                      <Td>{r.prediction ? pct(r.prediction.probability, 1) : "—"}</Td>
                      <Td>{r.prediction?.top_features[0]?.label ?? "—"}</Td>
                      <Td className="text-red-600">{r.error ?? ""}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            )}
          </Card>
        )}
      </div>
    </>
  );
}
