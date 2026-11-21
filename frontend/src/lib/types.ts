export type RiskBand = "low" | "medium" | "high";

export const SENSOR_KEYS = ["engine_rpm", "lub_oil_pressure", "fuel_pressure", "coolant_pressure", "lub_oil_temp", "coolant_temp"] as const;
export type SensorKey = (typeof SENSOR_KEYS)[number];
export type SensorReading = Record<SensorKey, number>;

export interface FeatureContribution {
  feature: SensorKey;
  label: string;
  value: number;
  deviation: number;
  direction: "high" | "low";
}

export interface Prediction {
  probability: number;
  label: "faulty" | "healthy";
  risk_band: RiskBand;
  top_features: FeatureContribution[];
  model_version: string;
}

export interface PredictionRecord extends Prediction {
  id: number;
  created_at: string;
  inputs: SensorReading;
}

export interface Citation {
  source: string;
  page: number | null;
}

export interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  supported: boolean | null;
  created_at: string;
}

export interface PredictionDetail extends PredictionRecord {
  messages: ChatMessage[];
}

export interface PredictionPage {
  items: PredictionRecord[];
  total: number;
  page: number;
  page_size: number;
}

export interface BatchRowResult {
  row: number;
  inputs: Record<string, unknown>;
  prediction: Prediction | null;
  error: string | null;
}

export interface BatchResponse {
  rows: number;
  succeeded: number;
  failed: number;
  results: BatchRowResult[];
}

export interface Snippet {
  id: string;
  source: string;
  page: number | null;
  text: string;
  score: number;
}

export interface SentenceCheck {
  text: string;
  supported: boolean;
  score: number | null;
}

export interface DiagnoseResponse {
  answer: string;
  citations: Citation[];
  snippets: Snippet[];
  sentences: SentenceCheck[];
  supported: boolean;
  covered: boolean;
  mode: string;
  message_id: number | null;
}

export interface Health {
  status: "ok" | "degraded";
  model_loaded: boolean;
  model_version: string | null;
  vector_store: { ready: boolean; chunks: number; error?: string };
  llm_provider: string;
  guardrail: string;
  database: boolean;
}

export interface SensorRange {
  min: number;
  max: number;
  healthy_p05: number;
  healthy_p95: number;
}

export interface ModelInfo {
  model_version: string;
  served_model: string;
  threshold: number;
  risk_bands: { medium: number; high: number };
  samples: { healthy: SensorReading; faulty: SensorReading };
  sensor_ranges: Record<SensorKey, SensorRange>;
  sensor_labels: Record<SensorKey, string>;
  dataset: { source: string; name?: string; url?: string; rows: number; fault_share: number };
}

export interface ModelMetricsRow {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  threshold: number;
  confusion_matrix: number[][];
}

export interface ModelResult {
  key: string;
  name: string;
  strategy: string;
  params: Record<string, unknown>;
  cv_f1?: number;
  val: ModelMetricsRow;
  test: ModelMetricsRow;
}

export interface ModelMetrics {
  model_version: string;
  served_model: string;
  imbalanced: boolean;
  min_recall: number;
  split_sizes: { train: number; val: number; test: number };
  models: ModelResult[];
  roc_curve: { fpr: number; tpr: number }[];
  feature_importance: { feature: string; importance: number; std: number }[];
  training_history: { epoch: number; train_loss: number; val_loss: number }[];
  dataset: ModelInfo["dataset"];
}

export interface FieldError {
  field: string;
  message: string;
}
