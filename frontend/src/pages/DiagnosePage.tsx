import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";

import { ChatPanel } from "@/components/ChatPanel";
import { PageHeader } from "@/components/Layout";
import { ResultCard } from "@/components/ResultCard";
import { SensorForm } from "@/components/SensorForm";
import { Alert } from "@/components/ui/primitives";
import { api, ApiError } from "@/lib/api";
import type { PredictionRecord, SensorReading } from "@/lib/types";
import { formatDate } from "@/lib/utils";

export function DiagnosePage() {
  const [params, setParams] = useSearchParams();
  const predictionId = params.get("prediction") ? Number(params.get("prediction")) : null;
  const qc = useQueryClient();

  const info = useQuery({ queryKey: ["model-info"], queryFn: api.modelInfo, staleTime: Infinity, retry: 1 });
  const detail = useQuery({
    queryKey: ["prediction", predictionId],
    queryFn: () => api.prediction(predictionId!),
    enabled: predictionId !== null,
  });

  const predict = useMutation<PredictionRecord, ApiError, SensorReading>({
    mutationFn: api.predict,
    onSuccess: (rec) => {
      qc.setQueryData(["prediction", rec.id], { ...rec, messages: [] });
      qc.invalidateQueries({ queryKey: ["predictions"] });
      setParams({ prediction: String(rec.id) });
    },
  });

  const current = detail.data ?? null;

  return (
    <>
      <PageHeader
        title="Diagnose"
        description={
          current
            ? `Prediction #${current.id} · ${formatDate(current.created_at)}`
            : "Enter engine sensor readings to get a fault-risk score, then ask what it means."
        }
      />
      {info.isError && (
        <div className="mb-4">
          <Alert>
            {(info.error as ApiError).status === 503
              ? "The model isn't trained yet. Run `python -m ml.train` in backend/ and restart the API."
              : (info.error as Error).message}
          </Alert>
        </div>
      )}
      {predict.isError && predict.error.fields.length === 0 && (
        <div className="mb-4">
          <Alert>{predict.error.message}</Alert>
        </div>
      )}
      <div className="grid gap-6 lg:grid-cols-2">
        <SensorForm
          onSubmit={(r) => predict.mutate(r)}
          loading={predict.isPending}
          serverErrors={predict.error?.fields}
          samples={info.data?.samples}
          ranges={info.data?.sensor_ranges}
          initial={current?.inputs ?? null}
        />
        <ResultCard prediction={current} loading={predict.isPending || (detail.isLoading && predictionId !== null)} />
      </div>
      <div className="mt-6">
        <ChatPanel predictionId={current?.id ?? null} history={detail.data?.messages} />
      </div>
    </>
  );
}
