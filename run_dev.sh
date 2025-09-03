#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Activa el venv
if [ -f backend/.venv/bin/activate ]; then
  source backend/.venv/bin/activate
else
  python3 -m venv backend/.venv
  source backend/.venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r backend/requirements.txt
fi

# Lanza el backend (sirve UI en /app/)
exec uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
