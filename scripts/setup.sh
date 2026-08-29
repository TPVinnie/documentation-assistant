#!/usr/bin/env bash
# One-command setup: virtualenv, dependencies, .env, and the synthetic corpus.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

.venv/bin/python scripts/make_corpus.py

echo ""
echo "Setup complete. Next steps:"
echo "  source .venv/bin/activate"
echo "  python scripts/ingest.py"
echo "  uvicorn app.api.main:app --reload   # then open http://localhost:8000/ui"
