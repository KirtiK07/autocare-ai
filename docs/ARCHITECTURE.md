# Architecture

Three containers: **Caddy + React** (frontend), **FastAPI** (backend: ML model and RAG agent) and **PostgreSQL**. Training (`ml.train`) and ingestion (`rag.ingest`) are offline commands; the backend loads their outputs (`artifacts/`, `vectorstore/`) once at startup in FastAPI's lifespan.

```
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
3. The probability is mapped to a risk band using the tuned threshold. The top 3 drivers are the sensors furthest from the healthy-class mean, in standard units.
4. `PredictionRepository.add` stores the inputs, result and model version, and the API returns the record.

## Request flow: `POST /api/diagnose`

1. The per-IP rate limit is checked (protects the LLM key).
2. The prediction and the last 6 chat turns are loaded from the database.
3. `DiagnosticAgent.retrieve`: the **question alone** must retrieve chunks above the relevance floor. Otherwise the answer is "not covered". This stops an off-topic question plus "low oil pressure" context from always pulling oil passages. Results are then merged with a search on question + prediction context for ranking.
4. The LLM gets a fixed system prompt: answer only from the context, cite `[source, page]`, recommend a mechanic for safety issues.
5. The guardrail checks each sentence. A sentence counts as supported if an NLI cross-encoder finds it entailed by a retrieved chunk, or if it quotes one near-verbatim. If fewer than half of the factual sentences are supported, the answer is replaced with the "not covered" message.
6. Both turns are saved; the API returns the answer, citations, retrieved snippets (for the source drawer) and per-sentence verdicts (for underlining).

## Why each tool

See [DECISIONS.md](DECISIONS.md).
