# Deployment (Vercel, free Hobby plan)

One Vercel project with two **Services** (see `vercel.json`):

| Service | Root | What runs |
| --- | --- | --- |
| `web` | `frontend/` | Vite build of the React app, served from Vercel's CDN (SPA fallback to `index.html`) |
| `api` | `backend/` | FastAPI as a Python Vercel Function (`app/main.py`), Fluid Compute, 2 GB / 1 vCPU, 300 s max |

Top-level rewrites send `/api/*` to `api` and everything else to `web`, so the browser talks to one origin and needs no CORS.

## How the backend fits a serverless function

- **Dependencies:** `backend/pyproject.toml` lists the runtime packages; uv installs **CPU-only PyTorch** from the PyTorch index (the default CUDA wheel is several GB larger). The bundle uses Vercel's large-function support.
- **Read-only bundle, writable `/tmp`:** on a cold start (`VERCEL=1`), `app/core/runtime.py`:
  - copies the prebuilt Chroma store (`backend/vectorstore/`, committed, about 31 MB) to `/tmp/vectorstore`;
  - downloads the embedding (90 MB) and NLI (570 MB) models to `/tmp/hf`;
  - runs Alembic migrations.

  Expect the first request after the app has been idle to take roughly 30–60 s. After that, the instance is warm and requests take about a second, plus LLM time.
- **No ASGI lifespan needed:** the service container is built lazily on the first request (`ensure_container`), with a lock.
- **Database:** Neon Postgres (free) from the Vercel Marketplace (`vercel integration add neon`), which sets `DATABASE_URL` for all environments. The app converts the URL to the psycopg driver and runs migrations on a cold start. Without `DATABASE_URL`, SQLite lives in `/tmp`, so history resets on cold starts and isn't shared between instances.

## Deploy

```bash
npm i -g vercel
vercel login
vercel link --project autocare-ai
vercel env add LLM_API_KEY production --sensitive    # Groq key
vercel env add LLM_PROVIDER production               # groq
vercel deploy --prod
```

Connect the GitHub repo (Project → Settings → Git) to deploy every push to `main` automatically, with a preview deployment for every pull request.

## Rebuilding the vector store

The index ships prebuilt. After changing `backend/knowledge/`, run `python -m rag.ingest` locally and commit `backend/vectorstore/`.

## Local alternatives

- `docker compose up --build`: the full Caddy + FastAPI + PostgreSQL stack on http://localhost.
- Dev servers: `uvicorn app.main:app --reload` plus `npm run dev` (Vite proxies `/api`).

## Roll back

Vercel → Project → Deployments → pick an earlier deployment → **Promote to Production** (or `vercel rollback`).
