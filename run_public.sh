#!/bin/bash
cd "$(dirname "$0")"

# Activar venv
source backend/.venv/bin/activate

# Exportar path
export PYTHONPATH=$PWD

# Levantar backend en segundo plano
uvicorn backend.app:app --host 0.0.0.0 --port 8000 &

# Guardar PID para detener después si quieres
SERVER_PID=$!

# Espera unos segundos para asegurar que arrancó
sleep 3

# Levantar ngrok
ngrok http 8000

# Cuando cierres ngrok, detener el backend
kill $SERVER_PID
