"""ManualRetriever: similarity search over the manuals vector store."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    page: int | None
    text: str
    score: float

    @property
    def citation(self) -> str:
        return f"{self.source}, page {self.page}" if self.page is not None else self.source


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Chunk]: ...

    def status(self) -> dict[str, Any]: ...


def chunk_id(source: str, page: int | None, text: str) -> str:
    return hashlib.sha1(f"{source}|{page}|{text}".encode()).hexdigest()[:12]


class ManualRetriever:
    """Wraps a Chroma collection; drops chunks below a relevance floor."""

    def __init__(self, persist_dir: Path, collection: str, embedding_model: str, min_score: float = 0.2):
        from rag.store import make_embeddings, open_store

        self.persist_dir = persist_dir
        self.min_score = min_score
        self._store = open_store(persist_dir, collection, make_embeddings(embedding_model))

    def count(self) -> int:
        return int(self._store._collection.count())

    def status(self) -> dict[str, Any]:
        try:
            n = self.count()
        except Exception as exc:  # noqa: BLE001 - report, don't crash health checks
            return {"ready": False, "chunks": 0, "error": str(exc)}
        return {"ready": n > 0, "chunks": n}

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        if self.count() == 0:
            return []
        hits = self._store.similarity_search_with_relevance_scores(query, k=k)
        out: list[Chunk] = []
        for doc, score in hits:
            if score < self.min_score:
                continue
            source = str(doc.metadata.get("source_name") or doc.metadata.get("source", "manual"))
            page_raw = doc.metadata.get("page")
            page = int(page_raw) if isinstance(page_raw, int) and page_raw > 0 else None
            out.append(
                Chunk(chunk_id(source, page, doc.page_content), source, page, doc.page_content, float(score))
            )
        return out
