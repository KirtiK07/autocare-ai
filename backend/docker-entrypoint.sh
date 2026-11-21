#!/bin/sh
set -e

echo "Running database migrations..."
alembic upgrade head

# Build the vector store on first start (or after the volume is wiped).
if [ ! -f "${VECTORSTORE_DIR:-/app/vectorstore}/chroma.sqlite3" ]; then
  echo "Vector store missing; building it from knowledge/ ..."
  python -m rag.ingest
fi

exec "$@"
