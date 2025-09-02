#!/bin/bash
cd ~/Escritorio/sports-predictor
source backend/.venv/bin/activate
export PYTHONPATH=$PWD
python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
