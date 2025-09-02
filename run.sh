#!/usr/bin/env bash
set -e
cd "/home/apolox/Escritorio/sports-predictor" || exit 1
source backend/.venv/bin/activate

# abre la app en el navegador en 1s, en paralelo
( sleep 1; xdg-open "http://localhost:8000" ) >/dev/null 2>&1 &

# lanza el servidor (queda en esta terminal)
exec uvicorn backend.app:app --host 0.0.0.0 --port 8000
