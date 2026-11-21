"""Hallucination guardrail: checks each answer sentence against retrieved chunks with NLI."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol

log = logging.getLogger(__name__)

_CITATION = re.compile(r"\[[^\]]+\]")
# Split after sentence punctuation, or after a trailing [citation], before a new sentence.
_SENTENCE_SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?]\]))\s+(?=[A-Z0-9\"'(])|(?<=\])\s+(?=[A-Z])")
# Sentences that are advice/disclaimers by design, not factual claims from the manual.
_WORD = re.compile(r"[a-z0-9]{3,}")
_EXEMPT = re.compile(r"qualified (mechanic|technician)|not covered in the manuals|consult|professional", re.I)


@dataclass
class SentenceVerdict:
    text: str
    supported: bool
    score: float | None


class Guardrail(Protocol):
    name: str

    def check(self, answer: str, contexts: list[str]) -> list[SentenceVerdict]: ...


def split_sentences(answer: str) -> list[str]:
    parts: list[str] = []
    for para in answer.split("\n"):
        para = para.strip().lstrip("-*•0123456789. ").strip()
        if para:
            parts.extend(s.strip() for s in _SENTENCE_SPLIT.split(para) if s.strip())
    return parts


class NLIGuardrail:
    """Cross-encoder NLI: a sentence is supported if some chunk entails it.

    Uses the max entailment probability over retrieved chunks. A near-verbatim
    quote of one chunk (>= 90% of its words) also counts as supported, because NLI
    models often label definitional sentences ("X is the Y that ...") as neutral
    even when copied word for word. It can miss claims that need several chunks
    combined, and it cannot judge whether the manual itself is correct.
    """

    name = "nli"
    QUOTE_COVERAGE = 0.9

    def __init__(self, model_name: str, threshold: float = 0.5):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device="cpu")
        self.threshold = threshold
        labels = {str(v).lower(): int(k) for k, v in (self.model.config.id2label or {}).items()}
        self.entail_idx = labels.get("entailment", 1)

    @staticmethod
    def _coverage(claim: str, context: str) -> float:
        words = _WORD.findall(claim.lower())
        if not words:
            return 0.0
        vocab = set(_WORD.findall(context.lower()))
        return sum(w in vocab for w in words) / len(words)

    def check(self, answer: str, contexts: list[str]) -> list[SentenceVerdict]:
        sentences = split_sentences(answer)
        if not sentences:
            return []
        verdicts: list[SentenceVerdict] = []
        for sentence in sentences:
            claim = _CITATION.sub("", sentence).strip()
            if not claim or _EXEMPT.search(claim) or not contexts:
                verdicts.append(SentenceVerdict(sentence, bool(_EXEMPT.search(claim)), None))
                continue
            scores = self.model.predict([(ctx, claim) for ctx in contexts], apply_softmax=True)
            entail = float(max(row[self.entail_idx] for row in scores))
            quoted = max(self._coverage(claim, ctx) for ctx in contexts) >= self.QUOTE_COVERAGE
            verdicts.append(SentenceVerdict(sentence, entail >= self.threshold or quoted, round(entail, 3)))
        return verdicts


class LexicalGuardrail:
    """Cheap fallback (no model download): share of content words found in context."""

    name = "lexical"
    _WORD = re.compile(r"[a-z0-9]{4,}")

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def check(self, answer: str, contexts: list[str]) -> list[SentenceVerdict]:
        vocab = set(self._WORD.findall(" ".join(contexts).lower()))
        out: list[SentenceVerdict] = []
        for sentence in split_sentences(answer):
            claim = _CITATION.sub("", sentence)
            if _EXEMPT.search(claim):
                out.append(SentenceVerdict(sentence, True, None))
                continue
            words = self._WORD.findall(claim.lower())
            score = sum(w in vocab for w in words) / len(words) if words else 1.0
            out.append(SentenceVerdict(sentence, score >= self.threshold, round(score, 3)))
        return out


class NoGuardrail:
    name = "disabled"

    def check(self, answer: str, contexts: list[str]) -> list[SentenceVerdict]:
        return [SentenceVerdict(s, True, None) for s in split_sentences(answer)]


def build_guardrail(enabled: bool, model_name: str, threshold: float) -> Guardrail:
    if not enabled:
        return NoGuardrail()
    try:
        return NLIGuardrail(model_name, threshold)
    except Exception as exc:  # noqa: BLE001 - model download can fail offline
        log.warning("NLI guardrail unavailable (%s); falling back to lexical overlap", exc)
        return LexicalGuardrail()
