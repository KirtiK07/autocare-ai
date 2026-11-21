"""RAG evaluation over eval/rag_questions.jsonl.

    python -m rag.evaluate            # retrieval hit rate + guardrail support share
    python -m rag.evaluate --no-llm   # retrieval only (fast, no LLM calls)

Each line: {"question": "...", "expected_keywords": ["thermostat", ...]}  or
           {"question": "...", "should_refuse": true}   (off-topic; the assistant must decline)
A retrieval hit means a top-k chunk contains one of the expected keywords.
Results are written to artifacts/rag_eval.json for the README.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.core.config import BACKEND_DIR, get_settings
from app.core.container import build_container

EVAL_FILE = BACKEND_DIR / "eval" / "rag_questions.jsonl"


def load_questions(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(use_llm: bool) -> dict[str, Any]:
    settings = get_settings()
    c = build_container(settings)
    questions = load_questions(EVAL_FILE)
    rows: list[dict[str, Any]] = []
    for q in questions:
        refuse = bool(q.get("should_refuse"))
        chunks = c.retriever.search(q["question"], settings.retrieval_k)
        keywords = [k.lower() for k in q.get("expected_keywords", [])]
        hit = (not refuse) and any(k in ch.text.lower() for ch in chunks for k in keywords)
        row: dict[str, Any] = {
            "question": q["question"],
            "should_refuse": refuse,
            "hit": hit,
            "retrieved": [ch.citation for ch in chunks],
        }
        if use_llm:
            ans = c.agent.answer(q["question"])
            claims = [v for v in ans.sentences if v.score is not None]
            row |= {
                "covered": ans.covered,
                "fully_supported": ans.covered and ans.supported,
                "claim_support": (sum(v.supported for v in claims) / len(claims)) if claims else None,
                "answer": ans.answer,
            }
        rows.append(row)

    answerable = [r for r in rows if not r["should_refuse"]]
    off_topic = [r for r in rows if r["should_refuse"]]
    n = len(answerable)
    summary: dict[str, Any] = {
        "answerable_questions": n,
        "off_topic_questions": len(off_topic),
        "k": settings.retrieval_k,
        "retrieval_hit_rate_at_k": round(sum(r["hit"] for r in answerable) / n, 4),
        "llm": c.llm.name if use_llm else None,
        "guardrail": c.guardrail.name if use_llm else None,
    }
    if use_llm:
        covered = [r for r in answerable if r["covered"]]
        scored = [r["claim_support"] for r in covered if r["claim_support"] is not None]
        summary |= {
            "answered_share": round(len(covered) / n, 4),
            "fully_supported_share": round(sum(r["fully_supported"] for r in answerable) / n, 4),
            "mean_claim_support_when_answered": round(sum(scored) / len(scored), 4) if scored else None,
            "off_topic_refusal_rate": (
                round(sum(not r["covered"] for r in off_topic) / len(off_topic), 4) if off_topic else None
            ),
        }
    out = settings.artifacts_dir / "rag_eval.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and grounding")
    parser.add_argument("--no-llm", action="store_true", help="Only measure retrieval hit rate")
    args = parser.parse_args()
    print(json.dumps(run(use_llm=not args.no_llm), indent=2))


if __name__ == "__main__":
    main()
