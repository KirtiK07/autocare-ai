import { FlaskConical, Loader2, Sparkles } from "lucide-react";
import { useEffect, useState, type FormEvent } from "react";

import { Button, Card, CardBody, CardHeader, Input } from "@/components/ui/primitives";
import type { FieldError, SensorKey, SensorRange, SensorReading } from "@/lib/types";
import { SENSORS, validateReading } from "@/lib/utils";

const GROUPS = ["Engine", "Oil", "Fuel", "Coolant"] as const;

interface Props {
  onSubmit: (reading: SensorReading) => void;
  loading?: boolean;
  serverErrors?: FieldError[];
  samples?: { healthy: SensorReading; faulty: SensorReading };
  ranges?: Partial<Record<SensorKey, SensorRange>>;
  initial?: SensorReading | null;
}

const toStrings = (r: SensorReading) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, String(v)])) as Record<string, string>;

export function SensorForm({ onSubmit, loading, serverErrors = [], samples, ranges, initial }: Props) {
  const [values, setValues] = useState<Record<string, string>>(() => (initial ? toStrings(initial) : {}));
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (initial) setValues(toStrings(initial));
  }, [initial]);

  useEffect(() => {
    if (serverErrors.length) setErrors(Object.fromEntries(serverErrors.map((e) => [e.field, e.message])));
  }, [serverErrors]);

  const load = (r: SensorReading) => {
    setValues(toStrings(r));
    setErrors({});
  };

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const errs = validateReading(values);
    setErrors(errs);
    if (Object.keys(errs).length) return;
    onSubmit(Object.fromEntries(SENSORS.map((s) => [s.key, Number(values[s.key])])) as SensorReading);
  };

  return (
    <Card>
      <CardHeader
        title="Sensor readings"
        description="One snapshot from the engine's sensors."
        action={
          samples && (
            <div className="flex flex-wrap justify-end gap-2">
              <Button type="button" variant="outline" size="sm" onClick={() => load(samples.healthy)}>
                <Sparkles className="size-3.5" /> Healthy sample
              </Button>
              <Button type="button" variant="outline" size="sm" onClick={() => load(samples.faulty)}>
                <FlaskConical className="size-3.5" /> Faulty sample
              </Button>
            </div>
          )
        }
      />
      <CardBody>
        <form onSubmit={submit} noValidate className="space-y-5">
          {GROUPS.map((group) => (
            <fieldset key={group}>
              <legend className="mb-2 text-xs font-semibold tracking-wide text-slate-500 uppercase">{group}</legend>
              <div className="grid gap-3 sm:grid-cols-2">
                {SENSORS.filter((s) => s.group === group).map((s) => {
                  const range = ranges?.[s.key];
                  const err = errors[s.key];
                  return (
                    <label key={s.key} className="block">
                      <span className="mb-1 flex items-baseline justify-between text-sm font-medium text-slate-700">
                        {s.label}
                        <span className="text-xs font-normal text-slate-400">{s.unit}</span>
                      </span>
                      <Input
                        name={s.key}
                        type="number"
                        inputMode="decimal"
                        step={s.step}
                        value={values[s.key] ?? ""}
                        invalid={!!err}
                        aria-describedby={`${s.key}-help`}
                        onChange={(e) => setValues((v) => ({ ...v, [s.key]: e.target.value }))}
                      />
                      <span id={`${s.key}-help`} className={err ? "mt-1 block text-xs text-red-600" : "mt-1 block text-xs text-slate-400"}>
                        {err ??
                          (range ? `Typical healthy: ${range.healthy_p05}–${range.healthy_p95} ${s.unit}` : `${s.min}–${s.max} ${s.unit}`)}
                      </span>
                    </label>
                  );
                })}
              </div>
            </fieldset>
          ))}
          <Button type="submit" className="w-full" disabled={loading}>
            {loading && <Loader2 className="size-4 animate-spin" />}
            {loading ? "Predicting…" : "Predict fault risk"}
          </Button>
        </form>
      </CardBody>
    </Card>
  );
}
