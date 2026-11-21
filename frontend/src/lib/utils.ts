import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

import type { RiskBand, SensorKey } from "./types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const pct = (p: number, digits = 0) => `${(p * 100).toFixed(digits)}%`;

export const RISK_STYLES: Record<RiskBand, { text: string; bg: string; ring: string; hex: string; label: string }> = {
  low: { text: "text-green-700", bg: "bg-green-50", ring: "ring-green-600/20", hex: "#16a34a", label: "Low" },
  medium: { text: "text-amber-700", bg: "bg-amber-50", ring: "ring-amber-600/20", hex: "#d97706", label: "Medium" },
  high: { text: "text-red-700", bg: "bg-red-50", ring: "ring-red-600/20", hex: "#dc2626", label: "High" },
};

export interface SensorMeta {
  key: SensorKey;
  label: string;
  unit: string;
  group: "Engine" | "Oil" | "Fuel" | "Coolant";
  min: number;
  max: number;
  step: number;
}

/** Bounds mirror the API's SensorReading model so most errors are caught client-side. */
export const SENSORS: SensorMeta[] = [
  { key: "engine_rpm", label: "Engine RPM", unit: "rpm", group: "Engine", min: 0, max: 8000, step: 1 },
  { key: "lub_oil_pressure", label: "Oil pressure", unit: "bar", group: "Oil", min: 0, max: 30, step: 0.01 },
  { key: "lub_oil_temp", label: "Oil temperature", unit: "°C", group: "Oil", min: -40, max: 200, step: 0.1 },
  { key: "fuel_pressure", label: "Fuel pressure", unit: "bar", group: "Fuel", min: 0, max: 50, step: 0.01 },
  { key: "coolant_pressure", label: "Coolant pressure", unit: "bar", group: "Coolant", min: 0, max: 30, step: 0.01 },
  { key: "coolant_temp", label: "Coolant temperature", unit: "°C", group: "Coolant", min: -40, max: 250, step: 0.1 },
];

export function validateReading(values: Record<string, string>): Record<string, string> {
  const errors: Record<string, string> = {};
  for (const s of SENSORS) {
    const raw = values[s.key];
    if (raw === undefined || raw.trim() === "") {
      errors[s.key] = "Required";
      continue;
    }
    const n = Number(raw);
    if (!Number.isFinite(n)) errors[s.key] = "Enter a number";
    else if (n < s.min || n > s.max) errors[s.key] = `Must be between ${s.min} and ${s.max} ${s.unit}`;
  }
  return errors;
}

export function formatDate(iso: string) {
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function toCsv(rows: Record<string, unknown>[]): string {
  if (!rows.length) return "";
  const headers = Object.keys(rows[0]);
  const esc = (v: unknown) => {
    const s = v === null || v === undefined ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [headers.join(","), ...rows.map((r) => headers.map((h) => esc(r[h])).join(","))].join("\n");
}
