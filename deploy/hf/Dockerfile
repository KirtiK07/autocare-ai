# syntax=docker/dockerfile:1
# Single-container image for Hugging Face Spaces (Docker SDK, free CPU tier: 2 vCPU / 16 GB RAM).
# FastAPI serves both the API (/api) and the built React app. The deploy workflow copies this
# file to the Space root next to backend/ and frontend/.

# ---------- frontend build ----------
FROM node:20-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
ARG VITE_REPO_URL=""
ENV VITE_REPO_URL=$VITE_REPO_URL
RUN npm run build

# ---------- python deps + models + vector store ----------
FROM python:3.12-slim AS py
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HOME=/opt/hf
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY backend/requirements.txt .
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/nli-deberta-v3-small')"
WORKDIR /app
COPY backend/ ./
# Build the Chroma index at image build time so the Space starts in seconds.
RUN python -m rag.ingest

# ---------- runtime ----------
FROM python:3.12-slim
RUN useradd --create-home --uid 1000 user
ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=False \
    STATIC_DIR=/app/static \
    DATABASE_URL=sqlite:////tmp/autocare.db \
    ENVIRONMENT=production
COPY --from=py /opt/venv /opt/venv
COPY --from=py --chown=user /opt/hf /opt/hf
WORKDIR /app
COPY --from=py --chown=user /app /app
COPY --from=web --chown=user /web/dist /app/static
USER user
EXPOSE 7860
# DATABASE_URL can be overridden with a Space secret (e.g. a free Neon Postgres URL)
# so history survives restarts; otherwise SQLite in /tmp is used.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 7860 --proxy-headers --forwarded-allow-ips '*'"]
