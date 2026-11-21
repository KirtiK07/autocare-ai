# Contributing

## Branches and commits

- `main` is always deployable. Work on `feat/...`, `fix/...` or `docs/...` branches and merge through a pull request, even when working solo.
- Conventional commits: `feat: add batch prediction endpoint`, `fix: keep timezone on SQLite timestamps`, `docs: model card`.
- Tag releases (`v1.0.0`) with notes.

## Setup

```bash
pip install pre-commit && pre-commit install
cd backend && pip install -r requirements-dev.txt   # after the CPU torch install, see README
cd frontend && npm install
```

## Checks (the same ones CI runs)

```bash
# backend
cd backend
ruff check . && black --check . && mypy app ml rag && pytest

# frontend
cd frontend
npm run lint && npm run format:check && npm test && npm run build && npm run test:e2e
```

Tests never call a real LLM: they use `FakeLLM` and `FakeRetriever` from `tests/conftest.py`. A model sanity test fails if the served model doesn't beat the Dummy baseline F1.

## Windows note

If Windows Smart App Control blocks a compiled extension (`DLL load failed ... Application Control policy`), install that package from source as pure Python:

```bash
DISABLE_SQLALCHEMY_CEXT=1 pip install --force-reinstall --no-deps --no-binary sqlalchemy sqlalchemy==<version>
MYPY_USE_MYPYC=0 pip install --no-binary mypy mypy==1.15.0
```
