import { useMutation } from "@tanstack/react-query";
import { BookOpen, MessageSquare, SendHorizontal, ShieldAlert } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";

import { Alert, Button, Card, CardBody, CardHeader, EmptyState, Sheet } from "@/components/ui/primitives";
import { api, ApiError } from "@/lib/api";
import type { ChatMessage, Citation, DiagnoseResponse, SentenceCheck, Snippet } from "@/lib/types";
import { cn } from "@/lib/utils";

export const DEFAULT_QUESTION = "Why is the risk high and what should I check first?";

interface Turn {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  snippets?: Snippet[];
  sentences?: SentenceCheck[];
  covered?: boolean;
  mode?: string;
}

const CITE = /\s*\[([^\]]+)\]/g;
const citeLabel = (c: Citation) => (c.page ? `${c.source}, page ${c.page}` : c.source);

/** Renders an answer with unsupported sentences underlined, citations as chips. */
function AnswerText({ turn, onCite }: { turn: Turn; onCite: (label: string) => void }) {
  const parts = turn.sentences?.length ? turn.sentences : [{ text: turn.content, supported: true, score: null }];
  return (
    <div className="space-y-1 leading-relaxed">
      <p>
        {parts.map((s, i) => {
          const labels = [...s.text.matchAll(CITE)].map((m) => m[1]);
          const body = s.text.replace(CITE, "").trim();
          return (
            <span key={i}>
              <span
                className={cn(!s.supported && "underline decoration-amber-500 decoration-wavy underline-offset-4")}
                title={s.supported ? undefined : "Not supported by the retrieved manual text"}
              >
                {body}
              </span>
              {labels.map((l) => (
                <button
                  key={l}
                  type="button"
                  onClick={() => onCite(l)}
                  className="mx-0.5 inline-flex max-w-[16rem] translate-y-[-1px] items-center gap-1 truncate rounded bg-accent-50 px-1.5 py-0.5 align-middle text-[11px] font-medium text-accent-700 ring-1 ring-accent-100 hover:bg-accent-100"
                  title={`Show source: ${l}`}
                >
                  <BookOpen className="size-3 shrink-0" />
                  <span className="truncate">{l}</span>
                </button>
              ))}{" "}
            </span>
          );
        })}
      </p>
      {turn.sentences?.some((s) => !s.supported) && (
        <p className="flex items-center gap-1 text-xs text-amber-700">
          <ShieldAlert className="size-3.5" /> Underlined sentences aren't backed by the manuals.
        </p>
      )}
      {turn.mode === "extractive" && (
        <p className="text-xs text-slate-400">Offline mode: quoting the manuals directly (no LLM key configured).</p>
      )}
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 px-1 py-2" aria-label="Assistant is typing">
      {[0, 1, 2].map((i) => (
        <span key={i} className="typing-dot size-2 rounded-full bg-slate-400" />
      ))}
    </div>
  );
}

export function ChatPanel({ predictionId, history }: { predictionId: number | null; history?: ChatMessage[] }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const [drawer, setDrawer] = useState<{ label: string; snippets: Snippet[] } | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setTurns((history ?? []).map((m) => ({ role: m.role, content: m.content, citations: m.citations })));
    setQuestion(DEFAULT_QUESTION);
  }, [predictionId, history]);

  useEffect(() => {
    if (turns.length > 0) endRef.current?.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
  }, [turns.length]);

  const ask = useMutation<DiagnoseResponse, ApiError, string>({
    mutationFn: (q) => api.diagnose(q, predictionId),
    onSuccess: (r) =>
      setTurns((t) => [
        ...t,
        {
          role: "assistant",
          content: r.answer,
          citations: r.citations,
          snippets: r.snippets,
          sentences: r.covered ? r.sentences : [],
          covered: r.covered,
          mode: r.mode,
        },
      ]),
  });

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const q = question.trim();
    if (q.length < 3 || ask.isPending) return;
    setTurns((t) => [...t, { role: "user", content: q }]);
    setQuestion("");
    ask.mutate(q);
  };

  const openSource = (turn: Turn, label: string) => {
    const pool = turn.snippets ?? [];
    const lower = label.toLowerCase();
    const matching = pool.filter((s) => citeLabel(s).toLowerCase() === lower);
    setDrawer({ label, snippets: matching.length ? matching : pool });
  };

  return (
    <Card>
      <CardHeader
        title="Ask the assistant"
        description={
          predictionId
            ? "Answers come only from the indexed manuals, with sources."
            : "Run a prediction first to ground answers in your readings, or ask a general question."
        }
      />
      <CardBody className="space-y-4">
        <div className="max-h-[28rem] space-y-4 overflow-y-auto" aria-live="polite">
          {turns.length === 0 && !ask.isPending && (
            <EmptyState icon={<MessageSquare className="size-5" />} title="No questions yet">
              Try the suggested question below.
            </EmptyState>
          )}
          {turns.map((t, i) =>
            t.role === "user" ? (
              <div key={i} className="flex justify-end">
                <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-accent-600 px-4 py-2 text-sm text-white">{t.content}</div>
              </div>
            ) : (
              <div
                key={i}
                className="max-w-[95%] rounded-2xl rounded-bl-sm bg-slate-100 px-4 py-3 text-sm text-slate-800"
                data-testid="assistant-message"
              >
                <AnswerText turn={t} onCite={(l) => openSource(t, l)} />
              </div>
            ),
          )}
          {ask.isPending && <TypingIndicator />}
          <div ref={endRef} />
        </div>
        {ask.isError && (
          <Alert>
            {ask.error.message}
            {ask.error.status === 503 && " Retry in a few seconds."}
          </Alert>
        )}
        <form onSubmit={submit} className="flex gap-2">
          <input
            aria-label="Your question"
            className="h-10 flex-1 rounded-md border border-slate-300 bg-white px-3 text-sm focus-visible:ring-2 focus-visible:ring-accent-500 focus-visible:outline-none"
            value={question}
            maxLength={1000}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask about the readings, a warning light, what to check…"
          />
          <Button type="submit" disabled={ask.isPending || question.trim().length < 3} aria-label="Send question">
            <SendHorizontal className="size-4" />
          </Button>
        </form>
      </CardBody>

      <Sheet open={!!drawer} onClose={() => setDrawer(null)} title="Source">
        {drawer && (
          <div className="space-y-4">
            <p className="text-sm font-medium text-slate-800">{drawer.label}</p>
            {drawer.snippets.length === 0 && <p className="text-sm text-slate-500">The passage text isn't stored for earlier messages.</p>}
            {drawer.snippets.map((s) => (
              <figure key={s.id} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <figcaption className="mb-1 flex justify-between text-xs text-slate-500">
                  <span>{citeLabel(s)}</span>
                  <span>relevance {s.score.toFixed(2)}</span>
                </figcaption>
                <blockquote className="text-sm leading-relaxed text-slate-700">{s.text}</blockquote>
              </figure>
            ))}
          </div>
        )}
      </Sheet>
    </Card>
  );
}
