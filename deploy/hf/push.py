"""Stage the repo for a Hugging Face Space and upload it (used by .github/workflows/deploy.yml).

    HF_TOKEN=... HF_SPACE=user/autocare-ai [LLM_API_KEY=...] [DATABASE_URL=...] python deploy/hf/push.py

Creates the Space (Docker SDK) if needed, copies backend/ + frontend/ with the Space
Dockerfile and card at the root, uploads it (large PDFs are handled by the Hub client),
and sets runtime secrets on the Space. Secrets never enter the uploaded files.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
IGNORE = shutil.ignore_patterns(
    ".venv", "node_modules", "dist", "__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "vectorstore", "data", "*.db", ".env", "playwright-report", "test-results",
    "notebooks", "tests", "e2e",
)


def main() -> None:
    token, space = os.environ.get("HF_TOKEN"), os.environ.get("HF_SPACE")
    if not token or not space:
        sys.exit("HF_TOKEN and HF_SPACE (e.g. your-hf-user/autocare-ai) are required")
    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        shutil.copytree(ROOT / "backend", stage / "backend", ignore=IGNORE)
        shutil.copytree(ROOT / "frontend", stage / "frontend", ignore=IGNORE)
        shutil.copy(ROOT / "deploy" / "hf" / "Dockerfile", stage / "Dockerfile")
        shutil.copy(ROOT / "deploy" / "hf" / "README.md", stage / "README.md")
        api.upload_folder(
            folder_path=str(stage),
            repo_id=space,
            repo_type="space",
            commit_message=f"Deploy {os.environ.get('GITHUB_SHA', 'local')[:7]}",
            delete_patterns=["backend/**", "frontend/**"],
        )

    for name in ("LLM_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "DATABASE_URL"):
        value = os.environ.get(name)
        if value:
            api.add_space_secret(space, name, value)
    print(f"Deployed to https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    main()
