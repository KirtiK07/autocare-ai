# Deployment (Hugging Face Spaces, free)

The app ships as **one Docker container** on a Hugging Face Space (free CPU tier: 2 vCPU, 16 GB RAM, enough for PyTorch, the embedding model and the NLI guardrail). FastAPI serves the API under `/api` and the built React app on every other path.

```
push to main ──▶ CI (lint, types, tests, build) ──▶ Deploy workflow
                                                     └─ deploy/hf/push.py: stage backend/ + frontend/ + Dockerfile,
                                                        upload to the Space, set Space secrets
Hugging Face builds deploy/hf/Dockerfile: npm build → pip install → bake models → build Chroma index
```

## One-time setup (about 5 minutes)

1. Create a free account at huggingface.co.
2. Create a **write** token at https://huggingface.co/settings/tokens.
3. In the GitHub repo, open **Settings → Secrets and variables → Actions**:

   | Kind | Name | Value |
   | --- | --- | --- |
   | Secret | `HF_TOKEN` | The write token |
   | Secret | `LLM_API_KEY` | Your Groq key |
   | Variable | `HF_SPACE` | `<your-hf-username>/autocare-ai` |
   | Secret (optional) | `DATABASE_URL` | A free Postgres URL, e.g. from neon.tech, as `postgresql+psycopg://user:pass@host/db?sslmode=require` |

4. Run the **Deploy** workflow (Actions → Deploy → Run workflow), or push to `main`.

The workflow creates the Space if it doesn't exist and copies `LLM_API_KEY` (and `DATABASE_URL`) into the Space's secrets; they're never written into the uploaded files. The first build takes roughly 10–15 minutes, because it installs PyTorch and builds the vector index. After that, the app is live at `https://<hf-username>-autocare-ai.hf.space`.

## Database

Without `DATABASE_URL`, the Space uses SQLite in `/tmp`. That works, but the history resets whenever the Space restarts or sleeps (free Spaces sleep after about 48 hours without visits). Set `DATABASE_URL` to a free Neon Postgres database to keep the history. Alembic creates the tables on startup.

## Local alternatives

- `docker compose up --build`: the full three-container stack (Caddy + FastAPI + PostgreSQL) on http://localhost.
- Single container exactly as deployed: `docker build -f deploy/hf/Dockerfile -t autocare . && docker run -p 7860:7860 -e LLM_API_KEY=... autocare`. Run it from the repo root after `python deploy/hf/push.py`-style staging, or simply run `docker build` from a folder containing `backend/`, `frontend/` and that Dockerfile.

## Roll back

Each deploy is a commit in the Space repository. Revert the bad commit on GitHub `main` (CI and Deploy run again), or revert the commit in the Space's **Files → History** tab.
