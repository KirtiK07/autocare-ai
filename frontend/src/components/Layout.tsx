import { useQuery } from "@tanstack/react-query";
import { Activity, BarChart3, FileSpreadsheet, Gauge, History, Info, Menu, X } from "lucide-react";
import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";

import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/", label: "Diagnose", icon: Gauge, end: true },
  { to: "/batch", label: "Batch", icon: FileSpreadsheet },
  { to: "/history", label: "History", icon: History },
  { to: "/model", label: "Model", icon: BarChart3 },
  { to: "/about", label: "About", icon: Info },
];

function StatusDot() {
  const { data, isError } = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 30_000 });
  const ok = data?.status === "ok";
  const label = isError ? "API offline" : !data ? "Connecting…" : ok ? "All systems ready" : "Degraded";
  return (
    <div
      className="flex items-center gap-2 text-xs text-slate-400"
      title={data ? `Model ${data.model_version ?? "—"} · LLM ${data.llm_provider} · guardrail ${data.guardrail}` : undefined}
    >
      <span className={cn("size-2 rounded-full", isError ? "bg-red-500" : ok ? "bg-green-500" : "bg-amber-500")} />
      {label}
    </div>
  );
}

function NavItems({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <>
      {NAV.map(({ to, label, icon: Icon, end }) => (
        <NavLink
          key={to}
          to={to}
          end={end}
          onClick={onNavigate}
          className={({ isActive }) =>
            cn(
              "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
              isActive ? "bg-white/10 text-white" : "text-slate-400 hover:bg-white/5 hover:text-white",
            )
          }
        >
          <Icon className="size-4" />
          {label}
        </NavLink>
      ))}
    </>
  );
}

export function Layout() {
  const [open, setOpen] = useState(false);
  return (
    <div className="flex min-h-full flex-col lg:flex-row">
      {/* Sidebar (desktop) */}
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col bg-slate-900 px-3 py-5 lg:flex">
        <Brand />
        <nav className="mt-8 flex flex-col gap-1">
          <NavItems />
        </nav>
        <div className="mt-auto px-3">
          <StatusDot />
        </div>
      </aside>

      {/* Top bar (tablet / mobile) */}
      <header className="sticky top-0 z-40 bg-slate-900 px-4 py-3 lg:hidden">
        <div className="flex items-center justify-between">
          <Brand />
          <button className="rounded p-1 text-slate-300" onClick={() => setOpen((o) => !o)} aria-label="Toggle navigation">
            {open ? <X className="size-5" /> : <Menu className="size-5" />}
          </button>
        </div>
        {open && (
          <nav className="mt-3 flex flex-col gap-1 pb-2">
            <NavItems onNavigate={() => setOpen(false)} />
            <div className="px-3 pt-2">
              <StatusDot />
            </div>
          </nav>
        )}
      </header>

      <main className="min-w-0 flex-1">
        <div className="mx-auto max-w-7xl px-4 py-6 md:px-8 lg:py-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}

function Brand() {
  return (
    <div className="flex items-center gap-2 px-3">
      <div className="flex size-8 items-center justify-center rounded-lg bg-accent-600">
        <Activity className="size-4 text-white" />
      </div>
      <div>
        <p className="text-sm leading-tight font-semibold text-white">AutoCare AI</p>
        <p className="text-[11px] leading-tight text-slate-400">Engine diagnostics</p>
      </div>
    </div>
  );
}

export function PageHeader({ title, description, action }: { title: string; description?: string; action?: React.ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-slate-900">{title}</h1>
        {description && <p className="mt-1 text-sm text-slate-500">{description}</p>}
      </div>
      {action}
    </div>
  );
}
