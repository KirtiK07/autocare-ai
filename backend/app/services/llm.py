"""LLM clients behind one abstract interface, so providers swap by config only."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx


class LLMError(RuntimeError):
    """Provider failure (HTTP error, bad response)."""


class LLMTimeoutError(LLMError):
    """Provider did not answer in time; the API maps this to 503."""


@dataclass
class ChatTurn:
    role: str  # "user" | "assistant"
    content: str


class LLMClient(ABC):
    """Generate a completion from a system prompt, prior turns and a user message."""

    name: str = "abstract"

    @abstractmethod
    def complete(self, system: str, user: str, history: list[ChatTurn] | None = None) -> str: ...


class _HTTPClient(LLMClient):
    def __init__(self, api_key: str, model: str, timeout: float):
        if not api_key:
            raise ValueError(f"{self.name} needs LLM_API_KEY")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _post(self, url: str, json: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
        try:
            resp = httpx.post(url, json=json, headers=headers, timeout=self.timeout)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"{self.name} timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.name} request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise LLMError(f"{self.name} returned {resp.status_code}: {resp.text[:300]}")
        return dict(resp.json())


class GroqClient(_HTTPClient):
    """Groq's OpenAI-compatible chat completions API."""

    name = "groq"
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "", timeout: float = 30.0):
        super().__init__(api_key, model or self.DEFAULT_MODEL, timeout)

    def complete(self, system: str, user: str, history: list[ChatTurn] | None = None) -> str:
        messages = [{"role": "system", "content": system}]
        messages += [{"role": t.role, "content": t.content} for t in history or []]
        messages.append({"role": "user", "content": user})
        data = self._post(
            self.URL,
            {"model": self.model, "messages": messages, "temperature": 0.1, "max_tokens": 700},
            {"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            return str(data["choices"][0]["message"]["content"]).strip()  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Unexpected Groq response") from exc


class GeminiClient(_HTTPClient):
    """Google Gemini generateContent API."""

    name = "gemini"
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: str, model: str = "", timeout: float = 30.0):
        super().__init__(api_key, model or self.DEFAULT_MODEL, timeout)

    def complete(self, system: str, user: str, history: list[ChatTurn] | None = None) -> str:
        contents = [
            {"role": "model" if t.role == "assistant" else "user", "parts": [{"text": t.content}]}
            for t in history or []
        ]
        contents.append({"role": "user", "parts": [{"text": user}]})
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        data = self._post(
            url,
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": contents,
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 700},
            },
            {"x-goog-api-key": self.api_key},
        )
        try:
            parts = data["candidates"][0]["content"]["parts"]  # type: ignore[index]
            return "".join(str(p.get("text", "")) for p in parts).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Unexpected Gemini response") from exc


class ExtractiveClient(LLMClient):
    """Offline fallback when no API key is set: quotes the best-matching context.

    It never invents text: every sentence comes from the retrieved manual passages,
    so the app stays usable (and honest) without a paid or free LLM key.
    """

    name = "extractive"
    _CTX = re.compile(r"\[(?P<cite>[^\]]+)\]\n(?P<body>.*?)(?=\n\n\[|\Z)", re.S)

    _WORD = re.compile(r"[a-z]{4,}")
    _STOP = frozenset(
        [
            "what",
            "should",
            "check",
            "first",
            "that",
            "this",
            "with",
            "from",
            "have",
            "does",
            "when",
            "there",
            "their",
            "about",
        ]
    )

    def complete(self, system: str, user: str, history: list[ChatTurn] | None = None) -> str:
        head, _, context = user.partition("CONTEXT:")
        blocks = self._CTX.findall(context)
        if not blocks:
            return "Not covered in the manuals."
        terms = {w for w in self._WORD.findall(head.lower()) if w not in self._STOP}
        scored: list[tuple[float, int, str, str]] = []
        for rank, (cite, body) in enumerate(blocks[:3]):
            text = " ".join(body.split())
            text = text.split(": ", 1)[1] if ": " in text[:60] else text  # drop a section heading
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                words = set(self._WORD.findall(sentence.lower()))
                overlap = len(words & terms) / (1 + len(terms))
                scored.append((overlap - 0.05 * rank, rank, sentence.strip(), cite))
        best = sorted(scored, key=lambda s: s[0], reverse=True)[:3]
        best.sort(key=lambda s: (s[1], scored.index(s)))
        lines = [f"{sentence} [{cite}]" for _, _, sentence, cite in best if sentence]
        return " ".join(lines) or "Not covered in the manuals."


def build_llm(provider: str, api_key: str, model: str, timeout: float) -> LLMClient:
    if provider == "none" or not api_key:
        return ExtractiveClient()
    if provider == "groq":
        return GroqClient(api_key, model, timeout)
    if provider == "gemini":
        return GeminiClient(api_key, model, timeout)
    raise ValueError(f"Unknown LLM provider: {provider}")
