import type {
  BatchResponse,
  DiagnoseResponse,
  FieldError,
  Health,
  ModelInfo,
  ModelMetrics,
  PredictionDetail,
  PredictionPage,
  PredictionRecord,
  RiskBand,
  SensorReading,
} from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";

/** Error carrying the API's `{error, detail}` body. `fields` is set for 422s. */
export class ApiError extends Error {
  status: number;
  code: string;
  fields: FieldError[];

  constructor(status: number, code: string, message: string, fields: FieldError[] = []) {
    super(message);
    this.status = status;
    this.code = code;
    this.fields = fields;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new ApiError(0, "network_error", "Can't reach the server. Is the backend running?");
  }
  if (res.ok) return (await res.json()) as T;
  let body: { error?: string; detail?: unknown } = {};
  try {
    body = await res.json();
  } catch {
    /* non-JSON error */
  }
  const code = body.error ?? "http_error";
  if (Array.isArray(body.detail)) {
    const fields = body.detail as FieldError[];
    const message = fields.map((f) => `${f.field}: ${f.message}`).join("; ") || "Invalid input";
    throw new ApiError(res.status, code, message, fields);
  }
  const message = typeof body.detail === "string" ? body.detail : `Request failed (${res.status})`;
  throw new ApiError(res.status, code, message);
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>("/health"),
  modelInfo: () => request<ModelInfo>("/model/info"),
  modelMetrics: () => request<ModelMetrics>("/model/metrics"),
  predict: (reading: SensorReading) => request<PredictionRecord>("/predict", json(reading)),
  predictBatch: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<BatchResponse>("/predict/batch", { method: "POST", body: form });
  },
  predictions: (page = 1, pageSize = 20, riskBand?: RiskBand) => {
    const q = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
    if (riskBand) q.set("risk_band", riskBand);
    return request<PredictionPage>(`/predictions?${q}`);
  },
  prediction: (id: number) => request<PredictionDetail>(`/predictions/${id}`),
  diagnose: (question: string, predictionId?: number | null) =>
    request<DiagnoseResponse>("/diagnose", json({ question, prediction_id: predictionId ?? null })),
};
