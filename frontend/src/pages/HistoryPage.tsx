import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { History } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { PageHeader } from "@/components/Layout";
import { Alert, Button, Card, EmptyState, RiskBadge, Skeleton, Table, Td, Th } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import type { RiskBand } from "@/lib/types";
import { cn, formatDate, pct } from "@/lib/utils";

const FILTERS: (RiskBand | undefined)[] = [undefined, "high", "medium", "low"];
const PAGE_SIZE = 15;

export function HistoryPage() {
  const [band, setBand] = useState<RiskBand | undefined>();
  const [page, setPage] = useState(1);
  const navigate = useNavigate();
  const q = useQuery({
    queryKey: ["predictions", page, band],
    queryFn: () => api.predictions(page, PAGE_SIZE, band),
    placeholderData: keepPreviousData,
  });
  const pages = q.data ? Math.max(1, Math.ceil(q.data.total / PAGE_SIZE)) : 1;

  return (
    <>
      <PageHeader title="History" description="Past predictions, newest first. Open one to see its readings and chat." />
      <div className="mb-4 flex flex-wrap gap-2" role="group" aria-label="Filter by risk">
        {FILTERS.map((f) => (
          <button
            key={f ?? "all"}
            onClick={() => {
              setBand(f);
              setPage(1);
            }}
            className={cn(
              "rounded-full px-3 py-1 text-sm font-medium ring-1 ring-inset",
              band === f ? "bg-slate-900 text-white ring-slate-900" : "bg-white text-slate-600 ring-slate-300 hover:bg-slate-50",
            )}
          >
            {f ? `${f[0].toUpperCase()}${f.slice(1)}` : "All"}
          </button>
        ))}
      </div>
      {q.isError && <Alert>{(q.error as Error).message}</Alert>}
      <Card>
        {q.isLoading ? (
          <div className="space-y-2 p-5">
            {Array.from({ length: 5 }, (_, i) => (
              <Skeleton key={i} className="h-8 w-full" />
            ))}
          </div>
        ) : q.data && q.data.items.length > 0 ? (
          <>
            <Table>
              <thead>
                <tr>
                  <Th>#</Th>
                  <Th>Time</Th>
                  <Th>Risk</Th>
                  <Th>Probability</Th>
                  <Th>Top driver</Th>
                </tr>
              </thead>
              <tbody>
                {q.data.items.map((p) => (
                  <tr
                    key={p.id}
                    tabIndex={0}
                    onClick={() => navigate(`/?prediction=${p.id}`)}
                    onKeyDown={(e) => e.key === "Enter" && navigate(`/?prediction=${p.id}`)}
                    className="cursor-pointer hover:bg-slate-50 focus:bg-slate-50 focus:outline-none"
                  >
                    <Td className="text-slate-500">{p.id}</Td>
                    <Td>{formatDate(p.created_at)}</Td>
                    <Td>
                      <RiskBadge band={p.risk_band} />
                    </Td>
                    <Td>{pct(p.probability, 1)}</Td>
                    <Td>
                      {p.top_features[0]?.label} <span className="text-slate-400">({p.top_features[0]?.direction})</span>
                    </Td>
                  </tr>
                ))}
              </tbody>
            </Table>
            <div className="flex items-center justify-between px-5 py-3 text-sm text-slate-500">
              <span>
                {q.data.total} prediction{q.data.total === 1 ? "" : "s"}
              </span>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
                  Previous
                </Button>
                <span>
                  {page} / {pages}
                </span>
                <Button variant="outline" size="sm" disabled={page >= pages} onClick={() => setPage((p) => p + 1)}>
                  Next
                </Button>
              </div>
            </div>
          </>
        ) : (
          <EmptyState icon={<History className="size-5" />} title="No predictions yet">
            <Link to="/" className="font-medium text-accent-600 hover:underline">
              Try a sample reading
            </Link>{" "}
            on the Diagnose page.
          </EmptyState>
        )}
      </Card>
    </>
  );
}
