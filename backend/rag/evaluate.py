"""RAG evaluation over eval/rag_questions.jsonl.

    python -m rag.evaluate            # retrieval hit rate + guardrail support share
    python -m rag.evaluate --no-llm   # retrieval only (fast, no LLM calls)

Each line: {"question": "...", "expected_source": "<source title>", "expected_page": 12?}
A hit means a chunk from the expected source (and page, if given) is in the top k.
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
        chunks = c.retriever.search(q["question"], settings.retrieval_k)
        hit = any(
            ch.source == q["expected_source"] and ("expected_page" not in q or ch.page == q["expected_page"])
            for ch in chunks
        )
        row: dict[str, Any] = {
            "question": q["question"],
            "hit": hit,
            "retrieved": [ch.citation for ch in chunks],
        }
        if use_llm:
            ans = c.agent.answer(q["question"])
            row |= {
                "covered": ans.covered,
                "fully_supported": ans.covered and ans.supported,
                "answer": ans.answer,
            }
        rows.append(row)

    n = len(rows)
    summary: dict[str, Any] = {
        "questions": n,
        "k": settings.retrieval_k,
        "hit_rate_at_k": round(sum(r["hit"] for r in rows) / n, 4),
        "llm": c.llm.name if use_llm else None,
        "guardrail": c.guardrail.name if use_llm else None,
    }
    if use_llm:
        summary["fully_supported_share"] = round(sum(r["fully_supported"] for r in rows) / n, 4)
        summary["covered_share"] = round(sum(r["covered"] for r in rows) / n, 4)
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
