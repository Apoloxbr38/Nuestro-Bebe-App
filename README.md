# Sports Predictor (v0.1 sexy)
Proyecto base minimalista para una app web de predicción de deportes (enfocado en fútbol por ahora).
- Backend: FastAPI + Poisson baseline.
- Frontend: HTML + JS vainilla.

## Requisitos
Python 3.10+

```bash
pip install -r backend/requirements.txt
```
## Ejecutar backend
```bash
uvicorn backend.app:app --reload
```

## Ejecutar frontend (simple)
Opción rápida:
```bash
cd frontend
python -m http.server 5173
```
Luego abre: http://localhost:5173

(El backend corre en http://127.0.0.1:8000; CORS habilitado)

## Flujo
1) Backend carga `backend/data/sample_matches.csv` y entrena un modelo Poisson sencillo.
2) Frontend pide `/teams` para llenar selects.
3) Al presionar **PREDICT**, llama `/predict?home=...&away=...` y muestra probabilidades y marcadores probables.

## Estructura
```
sports-predictor/
  backend/
    app.py
    requirements.txt
    models/
      baseline.py
    data/
      sample_matches.csv
  frontend/
    index.html
    assets/
      app.js
      style.css
```
