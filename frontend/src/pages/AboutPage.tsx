import { useQuery } from "@tanstack/react-query";
import { ExternalLink, TriangleAlert } from "lucide-react";

import { PageHeader } from "@/components/Layout";
import { Card, CardBody, CardHeader } from "@/components/ui/primitives";
import { api } from "@/lib/api";

const REPO_URL = (import.meta.env.VITE_REPO_URL as string | undefined) ?? "https://github.com/";

const STACK: [string, string][] = [
  ["ML", "Pandas, NumPy, scikit-learn, imbalanced-learn, PyTorch (CPU)"],
  ["RAG", "LangChain, sentence-transformers, Chroma, NLI cross-encoder guardrail, Groq or Gemini"],
  ["Backend", "FastAPI, Pydantic, SQLAlchemy, Alembic, PostgreSQL"],
  ["Frontend", "React, Vite, TypeScript, Tailwind CSS, Recharts, TanStack Query"],
  ["Ops", "Docker, Docker Compose, GitHub Actions CI/CD, Hugging Face Spaces"],
];

/** Architecture: UI → Caddy → FastAPI (services) → Postgres / Chroma / LLM. Offline jobs feed artifacts. */
function ArchitectureDiagram() {
  const box = "fill-white stroke-slate-300";
  const text = { fontSize: 12, fill: "#0f172a" } as const;
  const sub = { fontSize: 10, fill: "#64748b" } as const;
  return (
    <svg viewBox="0 0 760 300" className="w-full" role="img" aria-label="Architecture diagram">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M0 0 L10 5 L0 10 z" fill="#94a3b8" />
        </marker>
      </defs>
      {/* UI */}
      <rect x="10" y="110" width="130" height="70" rx="10" className={box} />
      <text x="75" y="140" textAnchor="middle" style={text}>
        React UI
      </text>
      <text x="75" y="158" textAnchor="middle" style={sub}>
        Vite · Tailwind
      </text>
      {/* Caddy */}
      <rect x="180" y="110" width="120" height="70" rx="10" className={box} />
      <text x="240" y="140" textAnchor="middle" style={text}>
        Caddy
      </text>
      <text x="240" y="158" textAnchor="middle" style={sub}>
        HTTPS · /api proxy
      </text>
      {/* FastAPI */}
      <rect x="340" y="60" width="200" height="170" rx="12" className="fill-accent-50 stroke-accent-500" />
      <text x="440" y="84" textAnchor="middle" style={{ ...text, fontWeight: 600 }}>
        FastAPI backend
      </text>
      {["FaultPredictor", "DiagnosticAgent", "ManualRetriever", "LLMClient · Guardrail"].map((t, i) => (
        <g key={t}>
          <rect x="360" y={98 + i * 31} width="160" height="24" rx="6" className={box} />
          <text x="440" y={114 + i * 31} textAnchor="middle" style={{ fontSize: 11, fill: "#0f172a" }}>
            {t}
          </text>
        </g>
      ))}
      {/* Right side */}
      {[
        { y: 20, t: "PostgreSQL", s: "predictions · chat" },
        { y: 115, t: "Chroma", s: "manual chunks" },
        { y: 210, t: "Groq / Gemini", s: "LLM API" },
      ].map(({ y, t, s }) => (
        <g key={t}>
          <rect x="590" y={y} width="150" height="60" rx="10" className={box} />
          <text x="665" y={y + 27} textAnchor="middle" style={text}>
            {t}
          </text>
          <text x="665" y={y + 44} textAnchor="middle" style={sub}>
            {s}
          </text>
          <line x1="540" y1="145" x2="588" y2={y + 30} stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arrow)" />
        </g>
      ))}
      <line x1="140" y1="145" x2="178" y2="145" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arrow)" />
      <line x1="300" y1="145" x2="338" y2="145" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arrow)" />
      {/* Offline */}
      <rect x="340" y="252" width="200" height="40" rx="8" className="fill-slate-50 stroke-slate-300" strokeDasharray="4 3" />
      <text x="440" y="270" textAnchor="middle" style={{ fontSize: 11, fill: "#334155" }}>
        Offline: ml.train · rag.ingest
      </text>
      <text x="440" y="285" textAnchor="middle" style={sub}>
        artifacts/ and vectorstore/ loaded at startup
      </text>
    </svg>
  );
}

export function AboutPage() {
  const info = useQuery({ queryKey: ["model-info"], queryFn: api.modelInfo, staleTime: Infinity, retry: 1 });
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  const ds = info.data?.dataset;
  return (
    <>
      <PageHeader
        title="About AutoCare AI"
        description="Predicts engine faults from sensor readings and explains them from vehicle manuals."
      />
      <div className="space-y-6">
        <div className="flex gap-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          <TriangleAlert className="mt-0.5 size-4 shrink-0" />
          <p>
            This is a portfolio project, not a substitute for a mechanic. The risk score is a statistical estimate from snapshot readings;
            have a qualified mechanic inspect any vehicle you're concerned about.
          </p>
        </div>
        <Card>
          <CardHeader title="Architecture" description="The UI only talks to the backend, which keeps API keys server-side." />
          <CardBody>
            <ArchitectureDiagram />
          </CardBody>
        </Card>
        <div className="grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader title="Tech stack" />
            <CardBody>
              <dl className="space-y-2 text-sm">
                {STACK.map(([k, v]) => (
                  <div key={k} className="grid grid-cols-[5.5rem_1fr] gap-2">
                    <dt className="font-medium text-slate-800">{k}</dt>
                    <dd className="text-slate-600">{v}</dd>
                  </div>
                ))}
              </dl>
            </CardBody>
          </Card>
          <Card>
            <CardHeader title="Data and sources" />
            <CardBody className="space-y-3 text-sm text-slate-600">
              <p>
                <span className="font-medium text-slate-800">Sensor data: </span>
                {ds ? (
                  <>
                    {ds.url ? (
                      <a className="text-accent-600 hover:underline" href={ds.url} target="_blank" rel="noreferrer">
                        {ds.name ?? ds.source}
                      </a>
                    ) : (
                      (ds.name ?? ds.source)
                    )}{" "}
                    · {ds.rows.toLocaleString()} rows · {(ds.fault_share * 100).toFixed(1)}% labelled faulty
                  </>
                ) : (
                  "—"
                )}
              </p>
              <p>
                <span className="font-medium text-slate-800">Manuals indexed: </span>
                {health.data ? `${health.data.vector_store.chunks} chunks` : "—"} (sources listed in knowledge/SOURCES.md)
              </p>
              <p>
                <span className="font-medium text-slate-800">Assistant: </span>
                {health.data ? `${health.data.llm_provider} · guardrail ${health.data.guardrail}` : "—"}
              </p>
              <a
                className="inline-flex items-center gap-1 font-medium text-accent-600 hover:underline"
                href={REPO_URL}
                target="_blank"
                rel="noreferrer"
              >
                Source code on GitHub <ExternalLink className="size-3.5" />
              </a>
            </CardBody>
          </Card>
        </div>
      </div>
    </>
  );
}
