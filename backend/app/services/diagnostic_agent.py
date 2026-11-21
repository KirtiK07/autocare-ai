"""DiagnosticAgent: retriever + LLM + guardrail -> a grounded, cited answer."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas import Prediction
from app.services.guardrail import Guardrail, SentenceVerdict
from app.services.llm import ChatTurn, LLMClient, normalise_citations
from app.services.manual_retriever import Chunk, Retriever

NOT_COVERED = "This isn't covered in the manuals I have. Please have a qualified mechanic inspect the engine."

SYSTEM_PROMPT = """You are AutoCare AI, a vehicle diagnostics assistant.
Rules:
1. Answer ONLY from the CONTEXT passages. Do not use outside knowledge.
2. After every factual sentence, cite its passage in square brackets exactly as written in the passage header, e.g. [Owner's Manual, page 12]. Never cite anything that is not a passage header.
3. If the context does not answer the question, reply exactly: "Not covered in the manuals."
4. For anything safety-related (overheating, oil pressure loss, brakes, fuel leaks), end by recommending a qualified mechanic.
5. Stay close to the passages' wording. Do not add causes, numbers or explanations that no passage states, even if you believe them.
6. Be concise: at most 6 sentences, plain language, the most likely cause and the first check first.
7. The fault-risk score comes from a statistical model, not an inspection; never present it as certain."""

_CITE = re.compile(r"\[([^\]]+?)\]")


@dataclass
class DiagnosticAnswer:
    answer: str
    citations: list[dict[str, object]]
    snippets: list[Chunk]
    sentences: list[SentenceVerdict]
    supported: bool
    covered: bool
    mode: str
    query: str = ""
    raw_answer: str = ""
    extra: dict[str, object] = field(default_factory=dict)


def prediction_context(prediction: Prediction | None) -> str:
    if prediction is None:
        return ""
    drivers = (
        ", ".join(
            f"{'high' if f.direction == 'high' else 'low'} {f.label.lower()}"
            for f in prediction.top_features
            if f.impact > 0
        )
        or "no single reading stands out"
    )
    return f"{prediction.risk_band} fault risk ({prediction.probability:.0%}); top drivers: {drivers}"


def prediction_premise(prediction: Prediction) -> str:
    drivers = " ".join(
        f"The {f.label.lower()} reading is {'higher' if f.direction == 'high' else 'lower'} than usual."
        for f in prediction.top_features
        if f.impact > 0
    )
    return (
        f"The model estimates a {prediction.risk_band} fault risk of {prediction.probability:.0%}. "
        f"The readings that drove the result: {drivers}"
    )


class DiagnosticAgent:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        guardrail: Guardrail,
        k: int = 5,
        min_supported_share: float = 0.5,
    ):
        self.retriever = retriever
        self.llm = llm
        self.guardrail = guardrail
        self.k = k
        self.min_supported_share = min_supported_share

    def build_query(self, question: str, prediction: Prediction | None) -> str:
        ctx = prediction_context(prediction)
        return f"{question} ({ctx})" if ctx else question

    def retrieve(self, question: str, prediction: Prediction | None) -> list[Chunk]:
        """The question alone decides whether the manuals are relevant at all;
        the prediction context (top drivers) then helps rank what is retrieved.
        Without this gate, an off-topic question plus "low oil pressure" context
        would always retrieve oil-pressure passages."""
        by_question = self.retriever.search(question, self.k)
        if not by_question or prediction is None:
            return by_question
        merged = {c.id: c for c in self.retriever.search(self.build_query(question, prediction), self.k)}
        for c in by_question:
            if c.id not in merged or c.score > merged[c.id].score:
                merged[c.id] = c
        return sorted(merged.values(), key=lambda c: c.score, reverse=True)[: self.k]

    @staticmethod
    def format_context(chunks: list[Chunk]) -> str:
        return "\n\n".join(f"[{c.citation}]\n{c.text}" for c in chunks)

    def _citations(self, answer: str, chunks: list[Chunk]) -> list[dict[str, object]]:
        by_label = {c.citation.lower(): c for c in chunks}
        seen: dict[str, dict[str, object]] = {}
        for raw in _CITE.findall(answer):
            for part in raw.split(";"):
                chunk = by_label.get(part.strip().lower())
                if chunk and chunk.citation not in seen:
                    seen[chunk.citation] = {"source": chunk.source, "page": chunk.page}
        return list(seen.values())

    def answer(
        self, question: str, prediction: Prediction | None = None, history: list[ChatTurn] | None = None
    ) -> DiagnosticAnswer:
        query = self.build_query(question, prediction)
        chunks = self.retrieve(question, prediction)
        mode = self.llm.name
        if not chunks:
            return DiagnosticAnswer(NOT_COVERED, [], [], [], True, False, mode, query, NOT_COVERED)

        ctx = prediction_context(prediction)
        user = (
            f"PREDICTION: {ctx}\n\n" if ctx else ""
        ) + f"QUESTION: {question}\n\nCONTEXT:\n{self.format_context(chunks)}"
        raw = normalise_citations(self.llm.complete(SYSTEM_PROMPT, user, history)).strip()
        if "not covered in the manuals" in raw.lower() and len(raw) < 80:
            return DiagnosticAnswer(NOT_COVERED, [], chunks, [], True, False, mode, query, raw)

        # The prediction itself is given data, so statements about the readings are checkable too.
        premises = [c.text for c in chunks] + ([prediction_premise(prediction)] if prediction else [])
        verdicts = self.guardrail.check(raw, premises)
        claims = [v for v in verdicts if v.score is not None]
        supported_claims = [v for v in claims if v.supported]
        share = len(supported_claims) / len(claims) if claims else 1.0
        if claims and not supported_claims:
            # Nothing in the answer is backed by the manuals: refuse rather than guess.
            return DiagnosticAnswer(
                NOT_COVERED, [], chunks, verdicts, False, False, mode, query, raw, {"supported_share": 0.0}
            )
        if share < self.min_supported_share:
            # Mostly unsupported: keep only the verified sentences (plus advice) and say so.
            kept = [v for v in verdicts if v.supported]
            removed = len(verdicts) - len(kept)
            text = " ".join(v.text for v in kept)
            text += f" (I removed {removed} sentence{'s' if removed != 1 else ''} the manuals don't support.)"
            return DiagnosticAnswer(
                answer=text,
                citations=self._citations(text, chunks),
                snippets=chunks,
                sentences=kept,
                supported=True,
                covered=True,
                mode=mode,
                query=query,
                raw_answer=raw,
                extra={"supported_share": share, "trimmed": removed},
            )
        return DiagnosticAnswer(
            answer=raw,
            citations=self._citations(raw, chunks),
            snippets=chunks,
            sentences=verdicts,
            supported=all(v.supported for v in verdicts),
            covered=True,
            mode=mode,
            query=query,
            raw_answer=raw,
            extra={"supported_share": share},
        )
