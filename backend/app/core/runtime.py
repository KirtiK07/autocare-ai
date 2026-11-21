"""Serverless runtime preparation (Vercel Functions).

On Vercel the deployment bundle is read-only and only /tmp is writable, so:
* the Hugging Face model cache goes to /tmp/hf (models download on a cold start). /tmp holds
  about 500 MB, so the guardrail defaults to nli-deberta-v3-xsmall (280 MB) instead of -small (570 MB);
* the prebuilt Chroma store is copied from the bundle to /tmp/vectorstore;
* without DATABASE_URL, SQLite lives in /tmp (resets on cold starts; set a Postgres URL to keep history);
* Alembic migrations run once per cold start.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger(__name__)
TMP = Path("/tmp")


def on_vercel() -> bool:
    return bool(os.environ.get("VERCEL"))


def prepare_environment() -> None:
    """Set env defaults before settings and models load. No-op outside Vercel."""
    if not on_vercel():
        return
    os.environ.setdefault("HF_HOME", str(TMP / "hf"))
    os.environ.setdefault("GUARDRAIL_MODEL", "cross-encoder/nli-deberta-v3-xsmall")
    os.environ.setdefault("DATABASE_URL", f"sqlite:///{TMP / 'autocare.db'}")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def writable_vectorstore(bundled: Path) -> Path:
    """Copy the read-only bundled vector store to /tmp once per cold start."""
    if not on_vercel():
        return bundled
    target = TMP / "vectorstore"
    if not target.exists() and bundled.exists():
        shutil.copytree(bundled, target)
        log.info("Copied vector store to %s", target)
    return target


def run_migrations(database_url: str) -> None:
    from alembic.config import Config

    from alembic import command
    from app.core.config import BACKEND_DIR

    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(cfg, "head")
