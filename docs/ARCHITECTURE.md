# Architecture

Three containers: **Caddy + React** (frontend), **FastAPI** (backend: ML model and RAG agent) and **PostgreSQL** locally via Docker Compose. In production on Vercel, the React build is served from the CDN and FastAPI runs as a Python Function; one project routes `/api/*` to the function and everything else to the SPA. Training (`ml.train`) and ingestion (`rag.ingest`) are offline commands; the backend loads their outputs (`artifacts/`, `vectorstore/`) once at startup in FastAPI's lifespan.

```
Local (docker compose):  Browser ──HTTPS──▶ Caddy ──/api/*──▶ FastAPI
Production (Vercel):     Browser ──HTTPS──▶ Vercel ─┬─ /api/* ▶ FastAPI Python Function
                                                   └─ /*     ▶ React SPA on the CDN

Browser ──HTTPS──▶ Caddy ──/api/*──▶ FastAPI (uvicorn)
                    │                   ├─ FaultPredictor  ── artifacts/ (scaler.joblib, fault_net.pt, model_meta.json)
                    └─ static SPA       ├─ DiagnosticAgent ─┬─ ManualRetriever ── Chroma (vectorstore/)
                                        │                   ├─ LLMClient ── Groq | Gemini | Extractive
                                        │                   └─ Guardrail ── NLI cross-encoder
                                        └─ Repositories ── PostgreSQL (predictions, chat_messages)
```

The UI never calls the LLM or the database directly. API keys stay on the server.

## Backend layers

| Layer | Responsibility |
| --- | --- |
| `app/api/routes/` | HTTP only: parse, call a service, return a Pydantic response model |
| `app/services/` | Business logic as classes: `FaultPredictor`, `ManualRetriever`, `DiagnosticAgent`, `LLMClient` (+ `GroqClient`, `GeminiClient`, `ExtractiveClient`), guardrails |
| `app/db/` | SQLAlchemy models and `PredictionRepository` / `ChatRepository` |
| `app/core/` | Settings (pydantic-settings), the service `Container` built once at startup, error handlers, rate limiter |
| `ml/` | Shared preprocessing (`Preprocessor`), baselines, PyTorch network, evaluation, training entry point |
| `rag/` | Ingestion, vector-store construction, evaluation |

## Request flow: `POST /api/predict`

1. Pydantic validates `SensorReading`: every field is required and bounded, otherwise a 422 names the field.
2. `FaultPredictor.predict` runs the saved `Preprocessor` (impute → clip → engineer → scale), then the served model, which is PyTorch or a scikit-learn model, whichever `model_meta.json` names.
3. The probability is mapped to a risk band using the tuned threshold. The top 3 drivers come from occlusion: each sensor is reset to its healthy-class mean and the batch is re-scored. A driver's impact is how much the fault probability drops.
4. `PredictionRepository.add` stores the inputs, result and model version, and the API returns the record.

## Request flow: `POST /api/diagnose`

1. The per-IP rate limit is checked (protects the LLM key).
2. The prediction and the last 6 chat turns are loaded from the database.
3. `DiagnosticAgent.retrieve`: the **question alone** must retrieve chunks above the relevance floor. Otherwise the answer is "not covered". This stops an off-topic question plus "low oil pressure" context from always pulling oil passages. Results are then merged with a search on question + prediction context for ranking.
4. The LLM gets a fixed system prompt: answer only from the context, cite `[source, page]`, recommend a mechanic for safety issues.
5. The guardrail checks each factual sentence against 1–3 sentence windows of the retrieved passages, plus a sentence describing the prediction. A sentence is supported if the NLI cross-encoder (deberta-v3-small) finds it entailed, if it quotes a passage near-verbatim, or if it shares at least 60% of its words with one window and no window contradicts it. Advice sentences ("Check the oil level") are shown but not scored. If no claim is supported, the answer becomes "not covered". If fewer than half are supported, only the supported sentences are kept, with a note saying how many were removed.
6. Both turns are saved; the API returns the answer, citations, retrieved snippets (for the source drawer) and per-sentence verdicts (for underlining).

## Why each tool

See [DECISIONS.md](DECISIONS.md).
