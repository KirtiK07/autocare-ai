# AutoCare AI — Vehicle Fault Prediction & Diagnostic Assistant

[![CI](https://github.com/KirtiK07/autocare-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/KirtiK07/autocare-ai/actions/workflows/ci.yml)

**Predicts engine faults from sensor readings with a PyTorch model, then explains them in plain language with a RAG assistant that answers only from vehicle manuals and cites its sources.**

**Live demo:** _add after deploy_ · **Walkthrough (3 min):** _add link_

<!-- Replace with a 30-second GIF of the Diagnose page: docs/img/demo.gif -->

## What it does

1. **Enter or upload engine sensor readings** (RPM, oil/fuel/coolant pressure, oil/coolant temperature), or load a healthy/faulty sample in one click.
2. **Get a fault-risk score**: probability, a Low/Medium/High band, and the three readings that drove it.
3. **Ask the assistant** what it means and what to check. Answers come only from retrieved manual text, cite `[source, page]`, and an NLI guardrail flags any sentence the manuals don't support. If the manuals don't cover a question, the assistant says so.

Also included: batch CSV scoring (up to 1,000 rows), a prediction history with saved chats, and a Model page with the comparison table, confusion matrix, ROC curve and feature importance.

## Architecture

```
React UI ──▶ Caddy (HTTPS, /api proxy) ──▶ FastAPI ──▶ PostgreSQL   (predictions, chat)
                                             │   ├──▶ Chroma        (manual chunks)
                                             │   └──▶ Groq / Gemini (LLM, key server-side)
                                             └── loads artifacts/ (model) + vectorstore/ at startup
Offline: python -m ml.train   ·   python -m rag.ingest
```

The FastAPI routes are thin. Business logic lives in service classes: `FaultPredictor`, `ManualRetriever`, `DiagnosticAgent`, and an abstract `LLMClient` with `GroqClient` / `GeminiClient`. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Tech stack

| Layer | Tools |
| --- | --- |
| ML | Pandas, NumPy, scikit-learn, imbalanced-learn, PyTorch (CPU) |
| RAG | LangChain, sentence-transformers (all-MiniLM-L6-v2), Chroma, NLI cross-encoder guardrail, Groq Llama 3.3 or Gemini |
| Backend | FastAPI, Pydantic, SQLAlchemy 2, Alembic, PostgreSQL (SQLite for local dev/tests) |
| Frontend | React 19, Vite, TypeScript, Tailwind CSS v4, shadcn/ui-style components, Recharts, TanStack Query |
| Ops | Docker Compose, Caddy, GitHub Actions, GHCR, AWS EC2 |
| Quality | pytest, Vitest, Playwright, ruff, black, mypy, ESLint, Prettier, pre-commit |

## Results

> ⚠️ **The numbers below come from the synthetic stand-in dataset** (`scripts/download_data.py --synthetic`) that lets the pipeline run without Kaggle credentials. Retrain on the real dataset (`python scripts/download_data.py`, then `python -m ml.train`) and paste the new `backend/artifacts/results.md` here before quoting any number.

Test set (15% hold-out, 2,931 rows). Each model's threshold is tuned on validation for the highest precision with recall ≥ 0.85, because missing a real fault costs more than a false alarm. The served model is the one with the best validation F1.

| Model | Imbalance strategy | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dummy (majority) | none | 0.50 | 0.6240 | 0.6240 | 1.0000 | 0.7685 | 0.5000 |
| Logistic Regression | smote | 0.35 | 0.6991 | 0.7220 | 0.8420 | 0.7774 | 0.7612 |
| Random Forest | class_weight | 0.38 | 0.7049 | 0.7248 | 0.8496 | 0.7823 | 0.7706 |
| **PyTorch EngineFaultNet** (served) | pos_weight | 0.32 | 0.7199 | 0.7400 | 0.8496 | 0.7910 | 0.7906 |

**RAG** (`python -m rag.evaluate`, 25 questions in `backend/eval/rag_questions.jsonl`): retrieval hit rate @5 = 1.00, fully supported answers = 1.00. The current corpus is only the 8-section sensor glossary, so this is an easy baseline. Re-run it after adding the manual PDFs.

## Quick start

**Docker (whole stack):**

```bash
cp .env.example .env          # optionally add LLM_API_KEY (Groq or Gemini)
docker compose up --build     # → http://localhost
```

**Local development:**

```bash
# backend
cd backend
python -m venv .venv && .venv/Scripts/activate       # macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
python scripts/download_data.py      # or --synthetic for offline dev
python -m ml.train                   # trains, evaluates, writes artifacts/
python -m rag.ingest                 # builds the Chroma store from knowledge/
alembic upgrade head
uvicorn app.main:app --reload        # http://localhost:8000/api/docs

# frontend (new terminal)
cd frontend && npm install && npm run dev   # http://localhost:5173
```

No LLM key? The assistant runs in **extractive mode**: it quotes the most relevant manual sentences with citations instead of generating text.

**Tests:** `cd backend && pytest` · `cd frontend && npm test && npm run test:e2e`

## Project structure

```
backend/
  app/            FastAPI app: api/routes, services (OOP), db (SQLAlchemy + repositories), core (settings, DI, errors)
  ml/             features.py (shared preprocessing), baselines.py, nets.py (PyTorch), evaluate.py, train.py
  rag/            ingest.py, store.py, evaluate.py
  artifacts/      trained model, scaler, metrics (committed; small)
  knowledge/      sensor_glossary.md, SOURCES.md, pdfs/ (manuals; not committed)
  eval/           rag_questions.jsonl
  notebooks/      01_eda.ipynb
  alembic/        migrations
  tests/          unit + integration (fake LLM, no paid API calls)
frontend/         React + Vite app, Vitest unit tests, Playwright smoke test, Caddyfile
docs/             ARCHITECTURE, MODEL_CARD, API, DEPLOYMENT, DECISIONS
.github/workflows ci.yml (lint, types, tests, build) · deploy.yml (GHCR → EC2)
```

## Limitations

- The dataset holds single snapshot readings from an unspecified engine type and has no time series. The model can't see trends, noises, leaks or wear.
- The "readings that drove this result" are deviations from the healthy average, not causal attributions.
- The guardrail checks whether a sentence is entailed by retrieved text; it can't tell whether the manual is right, and it can miss claims that combine several passages.
- This is a portfolio project, **not a substitute for a qualified mechanic**.

## Licence

MIT for the code. Datasets and manuals keep their own licences (see `backend/data/README.md` and `backend/knowledge/SOURCES.md`).
