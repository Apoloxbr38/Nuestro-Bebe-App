@echo off
cd C:\Proyectos\sports-predictor
call .venv311\Scripts\activate.bat
start "Backend" cmd /k python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
timeout /t 2 >nul
start "Ngrok" cmd /k ngrok http 8000
start "" http://127.0.0.1:8000/app/
pause