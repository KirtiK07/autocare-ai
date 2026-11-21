# Design decisions

Short log, newest first.

| Date | Decision | Why |
| --- | --- | --- |
| 2026-11-21 | **Threshold = highest validation precision with recall ≥ 0.85** instead of maximising F1/F2 | With faults as the majority class (~62%), F-beta tuning pushed every model to "always faulty" (threshold ≈ 0.1), matching the Dummy baseline. A recall floor states the real requirement (don't miss faults) and keeps false alarms down. |
| 2026-11-21 | **Serve whichever model wins on validation F1** (PyTorch on the current data) | Honest model selection. On tabular data, tree ensembles often match or beat small MLPs, so the comparison table shows all of them. The scikit-learn model is saved (compressed) only when it is the one served. |
| 2026-11-21 | **Relevance gate on the question alone** before adding prediction context to the query | Otherwise an off-topic question ("capital of France?") plus "low oil pressure" context still retrieved oil passages and produced a confident, irrelevant answer. |
| 2026-11-21 | **Guardrail = NLI entailment OR near-verbatim quote** | The NLI cross-encoders (xsmall, small and base all tested) label some sentences copied word for word as "neutral". A sentence that quotes ≥ 90% of its words from a chunk is grounded by construction. |
| 2026-11-21 | **Extractive fallback when no LLM key is set** | The app (and CI) stays fully usable without a paid or free key, and it never generates unsupported text. |
| 2026-11-21 | **Chroma over FAISS** | Persists to disk with metadata (source, page) out of the box and has a first-class LangChain integration. FAISS needs a separate docstore for metadata. At this corpus size, speed doesn't matter. |
| 2026-11-21 | **React + FastAPI over Streamlit** | A REST API plus a typed SPA is what real deployments look like, and it shows both halves. Streamlit would have been faster to build. |
| 2026-11-21 | **Groq / Gemini via plain HTTP** behind an abstract `LLMClient` | Two small classes instead of two SDKs. Switching providers is a config change. |
| 2026-11-21 | **SQLite locally, PostgreSQL in Docker/AWS** through the same SQLAlchemy models and Alembic migrations | Zero-setup development and tests, real database in production. `JSON().with_variant(JSONB)` gives JSONB on Postgres. |
| 2026-11-21 | **Python 3.12 in Docker** (plan said 3.11) | Matches the development machine, so pinned wheels are identical in both places. |
