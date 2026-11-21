# AutoCare AI — Vehicle Fault Prediction & Diagnostic Assistant

[![CI](https://github.com/KirtiK07/autocare-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/KirtiK07/autocare-ai/actions/workflows/ci.yml)

**Predicts engine faults from sensor readings with a PyTorch model (benchmarked against scikit-learn), then explains them with a RAG assistant that answers only from vehicle manuals, cites the page, and flags or removes any sentence the manuals don't support.**

**Live demo:** _Vercel URL (after first deploy)_ · **Walkthrough (3 min):** _add link_

## What it does

1. **Enter or upload engine sensor readings** (RPM, oil/fuel/coolant pressure, oil/coolant temperature), or load a healthy/faulty sample in one click.
2. **Get a fault-risk score**: probability, a Low/Medium/High band, and the three readings that pushed it up, measured in probability points by occlusion.
3. **Ask the assistant** what it means and what to check. It retrieves passages from three public-domain US Army technical manuals plus a sensor glossary, answers with `[source, page]` citations, and an NLI guardrail underlines unsupported sentences or removes them. Off-topic questions are declined.

Also included: batch CSV scoring (up to 1,000 rows), prediction history with saved chats, and a Model page with the comparison table, confusion matrix, ROC curve and feature importance.

## Architecture

```
Production (Vercel, one project, two services)       Local (docker compose)
Browser ─▶ Vercel ─┬─ /api/* ─▶ FastAPI (Python fn)   Browser ─▶ Caddy ─┬─ static React app
                   └─ /*     ─▶ React app (CDN)                         └─ /api ─▶ FastAPI ─▶ PostgreSQL

FastAPI services: FaultPredictor (PyTorch / scikit-learn) · DiagnosticAgent ─┬─ ManualRetriever (Chroma)
                                                                            ├─ LLMClient (Groq · Gemini · extractive)
                                                                            └─ NLI guardrail
Offline jobs: python -m ml.train · python -m rag.ingest · python -m rag.evaluate
```

Routes are thin; business logic lives in OOP service classes behind an abstract `LLMClient`. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Tech stack

| Layer | Tools |
| --- | --- |
| ML | Python, Pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE), PyTorch |
| RAG / LLM | LangChain, sentence-transformers (MiniLM embeddings), Chroma vector DB, DeBERTa NLI guardrail, Groq (`gpt-oss-120b`) or Gemini |
| Backend | FastAPI, Pydantic, SQLAlchemy 2, Alembic, SQL (PostgreSQL / SQLite) |
| Frontend | React 19, TypeScript, Vite, Tailwind CSS, Recharts, TanStack Query |
| DevOps | Vercel (Services, Python Functions), Docker, Docker Compose, GitHub Actions CI |
| Quality | pytest, Vitest, Playwright, ruff, black, mypy, ESLint, Prettier, pre-commit |

## Results

**Fault model**: Kaggle *Automotive Vehicles Engine Health Dataset* (19,535 readings, 63% labelled faulty), held-out test set of 2,931 rows. Each threshold is tuned on validation for the highest precision with recall ≥ 0.85, and the served model has the best validation ROC-AUC.

| Model | Imbalance strategy | Threshold | ROC-AUC | Balanced acc. | MCC | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dummy (majority) | none | 0.50 | 0.5000 | 0.5000 | 0.0000 | 0.6305 | 0.6305 | 1.0000 | 0.7734 |
| Logistic Regression | smote | 0.38 | 0.6957 | 0.5975 | 0.2222 | 0.6602 | 0.6898 | 0.8377 | 0.7566 |
| Random Forest | smote | 0.38 | 0.6968 | 0.5996 | 0.2275 | 0.6626 | 0.6910 | 0.8409 | 0.7586 |
| Gradient Boosting | smote | 0.36 | 0.6962 | 0.5950 | 0.2213 | 0.6615 | 0.6872 | 0.8501 | 0.7600 |
| **PyTorch EngineFaultNet** (served) | pos_weight | 0.37 | **0.7008** | 0.5996 | 0.2253 | 0.6609 | 0.6915 | 0.8344 | 0.7563 |

The dataset is weakly separable: four very different models land within 0.005 ROC-AUC of each other. F1 isn't used to choose, because "always faulty" already scores 0.773. Balanced accuracy and MCC show the real gain over that baseline. Details are in the [model card](docs/MODEL_CARD.md).

**RAG assistant** (`python -m rag.evaluate`: 25 domain questions + 5 off-topic, Groq `gpt-oss-120b`, NLI guardrail):

| Metric | Result |
| --- | --- |
| Retrieval hit rate @5 (a top-5 passage contains the answer's key terms) | 100% (25/25) |
| Domain questions answered (not declined by the guardrail) | 96% (24/25) |
| Answers whose displayed sentences are all supported (after trimming) | 68% (17/25) |
| Mean share of supported claims per answered question | 88% |
| Off-topic questions declined | 100% (5/5) |

Corpus: 3 public-domain manuals (TM 9-8000, TM 9-2858, TM 10-570) + sensor glossary → 2,903 chunks. The retrieval metric is keyword-based, so treat it as an upper bound.

## Quick start

```bash
# backend
cd backend
python -m venv .venv && .venv/Scripts/activate       # macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
python scripts/download_data.py --csv path/to/engine_data.csv   # or plain (Kaggle API) / --synthetic
python -m ml.train          # trains 5 models, writes artifacts/ + results table
python -m rag.ingest        # indexes knowledge/ (PDFs + glossary) into Chroma
alembic upgrade head
uvicorn app.main:app --reload            # API docs: http://localhost:8000/api/docs

# frontend (new terminal)
cd frontend && npm install && npm run dev   # http://localhost:5173
```

Put `LLM_API_KEY=<groq key>` in a root `.env` (copy `.env.example`). Without a key, the assistant runs in extractive mode and quotes the manuals directly.

**Full stack in Docker:** `docker compose up --build` → http://localhost (Caddy + FastAPI + PostgreSQL).
**Deploy:** `vercel deploy --prod`, or push to `main` once the repo is connected to Vercel. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).
**Tests:** `cd backend && pytest` · `cd frontend && npm test && npm run test:e2e`

## Project structure

```
backend/
  app/            FastAPI: api/routes, services (FaultPredictor, DiagnosticAgent, ManualRetriever, LLMClient, guardrail),
                  db (SQLAlchemy models + repositories), core (settings, DI container, errors, rate limit)
  ml/             features.py (shared preprocessing), baselines.py, nets.py (PyTorch), evaluate.py, train.py
  rag/            ingest.py, store.py, evaluate.py
  artifacts/      trained model, scaler, metrics, results.md, rag_eval.json
  vectorstore/    prebuilt Chroma index (shipped with the Vercel function)
  knowledge/      pdfs/ (public-domain manuals), sensor_glossary.md, SOURCES.md
  eval/           rag_questions.jsonl     notebooks/ 01_eda.ipynb     alembic/     tests/
frontend/         React app, Vitest unit tests, Playwright smoke test, Caddyfile (compose)
docs/             ARCHITECTURE · MODEL_CARD · API · DEPLOYMENT · DECISIONS
```

## Limitations

- Snapshot readings from an unspecified engine type; no time series, so the model can't see trends, noises, leaks or wear.
- The manuals are military and older vehicle references. The principles carry over, but a specific vehicle's owner's manual takes precedence.
- The guardrail can't verify whether a manual is right, and NLI models occasionally entail false claims (1 of 5 in our hand-made test).
- This is a portfolio project, **not a substitute for a qualified mechanic**.

## Licence

MIT for the code. The manuals are US government works (public domain); the dataset keeps its Kaggle licence. See `backend/knowledge/SOURCES.md` and `backend/data/README.md`.
