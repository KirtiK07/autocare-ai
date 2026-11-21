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
# Recommendations ("Check the coolant level first.") are advice, not factual claims a
# passage can entail; they are shown but excluded from the support score.
_ADVICE = re.compile(
    r"^(please\s+)?(check|inspect|verify|ensure|make sure|stop|refill|top up|replace|have|look|let|turn|"
    r"switch|allow|avoid|do not|don't|never|always|consider|keep|start|begin|if|then|after|before|once)",
    re.I,
)
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


def premise_windows(contexts: list[str], max_len: int = 3) -> list[str]:
    """Sliding windows of 1..max_len consecutive sentences from each retrieved chunk.

    NLI models are trained on sentence-length premises; scoring a claim against a whole
    800-character chunk gives near-zero entailment even for close paraphrases.
    """
    out: list[str] = []
    for ctx in contexts:
        sents = [s for s in re.split(r"(?<=[.!?])\s+", " ".join(ctx.split())) if len(s) > 15]
        for k in range(1, max_len + 1):
            out += [" ".join(sents[i : i + k]) for i in range(len(sents) - k + 1)]
    return out


def _overlap(claim_words: set[str], text: str) -> float:
    words = set(_WORD.findall(text.lower()))
    return len(claim_words & words) / (len(claim_words) or 1)


class NLIGuardrail:
    """Cross-encoder NLI: a sentence is supported if some passage window entails it.

    Each claim is scored against the ``top_windows`` sentence windows (1-3 sentences)
    that share the most words with it; the max entailment probability decides. A
    near-verbatim quote (>= 90% of its words in one chunk) also counts as supported.
    It can miss claims that combine passages far apart, and it cannot judge whether
    the manual itself is correct.
    """

    name = "nli"
    QUOTE_COVERAGE = 0.9
    PARAPHRASE_OVERLAP = 0.6  # share of claim words found in one window
    CONTRADICTION_MAX = 0.5

    def __init__(self, model_name: str, threshold: float = 0.5, top_windows: int = 12):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device="cpu")
        self.threshold = threshold
        self.top_windows = top_windows
        labels = {str(v).lower(): int(k) for k, v in (self.model.config.id2label or {}).items()}
        self.entail_idx = labels.get("entailment", 1)
        self.contra_idx = labels.get("contradiction", 0)

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
        windows = premise_windows(contexts)
        pairs: list[tuple[str, str]] = []
        spans: dict[int, tuple[int, int]] = {}
        verdicts: list[SentenceVerdict | None] = []
        for i, sentence in enumerate(sentences):
            claim = _CITATION.sub("", sentence).strip()
            if not claim or _EXEMPT.search(claim) or _ADVICE.match(claim) or not windows:
                advice = bool(_EXEMPT.search(claim) or _ADVICE.match(claim))
                verdicts.append(SentenceVerdict(sentence, advice, None))
                continue
            claim_words = set(_WORD.findall(claim.lower()))
            best = sorted(windows, key=lambda w: _overlap(claim_words, w), reverse=True)[: self.top_windows]
            spans[i] = (len(pairs), len(pairs) + len(best))
            pairs += [(w, claim) for w in best]
            verdicts.append(None)
        # One batched forward pass for every claim keeps latency to ~1-2 s per answer.
        scores = self.model.predict(pairs, apply_softmax=True, batch_size=64) if pairs else []
        out: list[SentenceVerdict] = []
        for i, sentence in enumerate(sentences):
            v = verdicts[i]
            if v is not None:
                out.append(v)
                continue
            a, b = spans[i]
            claim = pairs[a][1]
            claim_words = set(_WORD.findall(claim.lower()))
            rows = scores[a:b]
            entail = float(max(row[self.entail_idx] for row in rows))
            contradiction = float(max(row[self.contra_idx] for row in rows))
            quoted = max(self._coverage(claim, ctx) for ctx in contexts) >= self.QUOTE_COVERAGE
            # Close paraphrase: most of the claim's words sit in one passage window and
            # no window contradicts it (catches claims that reverse what the manual says).
            paraphrase = (
                max(_overlap(claim_words, pairs[j][0]) for j in range(a, b)) >= self.PARAPHRASE_OVERLAP
                and contradiction < self.CONTRADICTION_MAX
            )
            supported = entail >= self.threshold or quoted or paraphrase
            out.append(SentenceVerdict(sentence, supported, round(entail, 3)))
        return out


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
