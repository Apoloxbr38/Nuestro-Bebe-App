# backend/app.py — cabecera ordenada
from fastapi import FastAPI, Query, Body
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import time
import joblib
import math
import numpy as np
import pandas as pd

# Puedes mantener estos imports absolutos si estás corriendo desde la raíz con:
#   uvicorn backend.app:app --reload
from .utils.data_status import compute_status, record_reload_marker
from backend.models.baseline import PoissonModel
from backend.data_loader import refresh_dataset, MERGED
from backend.train import MODEL_PATH
from backend.train_ou import MODEL_OU_PATH
from backend.train_btts import MODEL_BTTS_PATH
from backend.models.features import build_training_table

app = FastAPI(
    title="Sports Predictor API",
    version="0.8",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,   # opcional, útil si en el futuro usas cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ AHORA sí: define /data/status DESPUÉS de crear app
@app.get("/data/status")
def data_status():
    return compute_status()

# --- Cache global en memoria (para velocidad) ---
df_global: Optional[pd.DataFrame] = None
Xy_global: Optional[pd.DataFrame] = None
current_csv_path: Optional[Path] = None

def _set_data(df: pd.DataFrame, csv_path: Path | None = None, build_features: bool = True):
    """Carga dataset en memoria, re-entrena Poisson y opcionalmente construye features."""
    global df_global, Xy_global, current_csv_path, model
    df_global = df.copy()
    current_csv_path = csv_path
    model.fit(df_global)
    if build_features:
        try:
            Xy_global = build_training_table(df_global, window=10)
        except Exception as e:
            Xy_global = None
            print("[WARN] build_training_table falló:", e)

def _ensure_data_loaded():
    global df_global
    if df_global is None:
        df = pd.read_csv(DATA_PATH)
        _set_data(df, DATA_PATH, build_features=False)

def _ensure_features():
    """Construye Xy_global si aún no existe (perezoso)."""
    global Xy_global, df_global
    if Xy_global is None and df_global is not None:
        Xy_global = build_training_table(df_global, window=10)

# --- Helper para acceder siempre a Xy ---
def _get_Xy():
    _ensure_data_loaded()   # <- añade esto
    _ensure_features()
    return Xy_global if Xy_global is not None else build_training_table(df_global, window=10)

# Modelos y utilidades (asumimos que estos módulos ya existen en tu repo)
from backend.models.baseline import PoissonModel
from backend.data_loader import refresh_dataset, MERGED
from backend.train import MODEL_PATH
from backend.train_ou import MODEL_OU_PATH
from backend.train_btts import MODEL_BTTS_PATH
from backend.models.features import build_training_table

# Etiquetas para pequeñas explicaciones
FEATURE_LABELS = {
    "Elo_H": "Elo Local",
    "Elo_A": "Elo Visitante",
    "H_r_GF": "GF recientes (Local)",
    "H_r_GA": "GA recientes (Local)",
    "H_r_GD": "GD recientes (Local)",
    "H_r_W":  "Racha victorias (Local)",
    "H_r_D":  "Racha empates (Local)",
    "H_r_L":  "Racha derrotas (Local)",
    "H_r_HomeRate":"Fortaleza en casa",
    "A_r_GF": "GF recientes (Visita)",
    "A_r_GA": "GA recientes (Visita)",
    "A_r_GD": "GD recientes (Visita)",
    "A_r_W":  "Racha victorias (Visita)",
    "A_r_D":  "Racha empates (Visita)",
    "A_r_L":  "Racha derrotas (Visita)",
    "A_r_HomeRate":"Rend. visitante",
}

def top_drivers(row: pd.DataFrame, feature_list: list[str], k: int = 3):
    colz = [c for c in feature_list if c in row.columns]
    if not colz: return []
    vals = row[colz].iloc[0]
    med = np.median(vals.values)
    mad = np.median(np.abs(vals.values - med)) or 1.0
    score = (vals - med) / mad
    idx = np.argsort(-np.abs(score.values))[:k]
    out = []
    for i in idx:
        feat = colz[i]; s = score.values[i]
        pretty = FEATURE_LABELS.get(feat, feat)
        arrow = "↑" if s > 0 else "↓"
        out.append(f"{pretty} {arrow}")
    return out

app = FastAPI(
    title="Sports Predictor API",
    version="0.8",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import math

def _col_ok(df, c): 
    return (c in df.columns) and (df[c].notna().sum() > 0)

def _recent_mean_home(df, team, col, n=10):
    # Promedio reciente para el equipo cuando juega de local
    if not _col_ok(df, col): return None
    d = df.loc[df["HomeTeam"] == team, col].dropna().tail(n)
    return float(d.mean()) if len(d) else None

def _recent_mean_away(df, team, col, n=10):
    # Promedio reciente para el equipo cuando juega de visita
    if not _col_ok(df, col): return None
    d = df.loc[df["AwayTeam"] == team, col].dropna().tail(n)
    return float(d.mean()) if len(d) else None

def _sum_opt(a, b):
    if a is None and b is None: return None
    return (a or 0.0) + (b or 0.0)

def _poisson_cdf(k, lam):
    # P(X <= k) para Poisson(lam)
    if lam is None: return None
    k = int(k)
    s = 0.0
    for i in range(0, k+1):
        s += math.exp(-lam) * (lam**i) / math.factorial(i)
    return s

def _poisson_pmf(k, lam):
    if lam is None: return None
    return math.exp(-lam) * (lam**k) / math.factorial(k)

def _clip01(x):
    if x is None: return None
    return max(0.0, min(1.0, float(x)))

# --- Datos ---
LOCAL_SAMPLE = Path(__file__).parent / "data" / "sample_matches.csv"
DATA_PATH = MERGED if MERGED.exists() else LOCAL_SAMPLE

# --- Servir frontend estático ---
FRONT_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/app", StaticFiles(directory=str(FRONT_DIR), html=True), name="frontend")

# --- Esquema de respuesta ---
class PredictResponse(BaseModel):
    home: str
    away: str
    p_home: float
    p_draw: float
    p_away: float
    exp_goals_home: float
    exp_goals_away: float
    exp_goals_total: float
    ou_over25: float
    ou_under25: float
    btts_yes: float
    btts_no: float
    top_scorelines: list
    src_1x2: str
    src_ou25: str
    src_btts: str
    # --- NUEVO: métricas extra ---
    exp_corners_home: float | None = None
    exp_corners_away: float | None = None
    exp_corners_total: float | None = None
    corners_over95: float | None = None   # Prob total > 9.5
    corners_under95: float | None = None
    exp_yellows_home: float | None = None
    exp_yellows_away: float | None = None
    exp_yellows_total: float | None = None
    yellows_over45: float | None = None   # Prob total > 4.5
    yellows_under45: float | None = None
    exp_reds_home: float | None = None
    exp_reds_away: float | None = None
    exp_reds_total: float | None = None
    # Distribución rápida de goles totales (Poisson λ = xG total)
    p_goals_0: float | None = None
    p_goals_1: float | None = None
    p_goals_2: float | None = None
    p_goals_3plus: float | None = None

# --- Modelos en memoria ---
model = PoissonModel()
clf_bundle = None       # XGB 1X2
clf_ou_bundle = None    # XGB Over/Under 2.5
clf_btts_bundle = None  # XGB BTTS

@app.on_event("startup")
def _load_and_train():
    global clf_bundle, clf_ou_bundle, clf_btts_bundle
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception:
        df = pd.read_csv(LOCAL_SAMPLE)
    _set_data(df, DATA_PATH, build_features=False)

    # Carga modelos ML si existen
    clf_bundle     = joblib.load(MODEL_PATH)     if MODEL_PATH.exists()     else None
    clf_ou_bundle  = joblib.load(MODEL_OU_PATH)  if MODEL_OU_PATH.exists()  else None
    clf_btts_bundle= joblib.load(MODEL_BTTS_PATH)if MODEL_BTTS_PATH.exists()else None

    # ⚡ Precarga features (warmup automático)
    _ensure_data_loaded()
    _ensure_features()
    print("💜 Warmup completado: features precargadas.")

# --- Entrenamientos ---
@app.post("/train")
def train_endpoint():
    from backend.train import train as _train
    info = _train(str(DATA_PATH))
    global clf_bundle
    clf_bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    return {"status": "ok", **info}

@app.post("/train_ou")
def train_ou_endpoint():
    from backend.train_ou import train_ou as _train_ou
    info = _train_ou(str(DATA_PATH))
    global clf_ou_bundle
    clf_ou_bundle = joblib.load(MODEL_OU_PATH) if MODEL_OU_PATH.exists() else None
    return {"status": "ok", **info}

@app.post("/train_btts")
def train_btts_endpoint():
    from backend.train_btts import train_btts as _train_btts
    info = _train_btts(str(DATA_PATH))
    global clf_btts_bundle
    clf_btts_bundle = joblib.load(MODEL_BTTS_PATH) if MODEL_BTTS_PATH.exists() else None
    return {"status": "ok", **info}

# --- Recargas de datos ---
@app.post("/reload")
def reload_data(build_features: bool = False):
    path = refresh_dataset(leagues=("SP1",), start_years=(2023, 2024))
    df = pd.read_csv(path)
    _set_data(df, Path(path), build_features=build_features)
    record_reload_marker()  # ✅ ahora sí: marca recarga al completar
    return {"status": "ok", "rows": int(len(df)), "file": str(path), "built": build_features}

@app.post("/reload_multi")
def reload_multi(
    leagues: List[str] = Body(..., embed=True),
    start_years: Optional[List[int]] = Body(None, embed=True),
    last_n: Optional[int] = Body(4, embed=True),
    build_features: bool = Body(False, embed=True),
):
    kwargs = {}
    if start_years:
        kwargs["start_years"] = tuple(start_years); kwargs["last_n_years"] = None
    else:
        kwargs["start_years"] = tuple(); kwargs["last_n_years"] = int(last_n or 4)

    path = refresh_dataset(leagues=tuple(leagues), **kwargs)
    df = pd.read_csv(path)
    _set_data(df, Path(path), build_features=build_features)
    record_reload_marker()  # ✅ marca recarga exitosa aquí
    return {
        "status": "ok", "rows": int(len(df)), "file": str(path),
        "leagues": leagues, "years": (start_years if start_years else f"last_{last_n}"),
        "built": build_features
    }

# al final de tu lógica de recarga, si todo fue bien:
record_reload_marker()

# --- Warmup para precalentar cache ---
@app.post("/warmup")
def warmup():
    _ensure_data_loaded()
    _ensure_features()
    return {"status": "ok", "features_ready": Xy_global is not None}

# --- Helpers para columnas variables (League vs Div, etc.) ---
def _first_col(df, primary, alts):
    for c in (primary, *alts):
        if c in df.columns:
            return c
    return None

# --- Últimos partidos (global, normalizados) ---
@app.get("/recent")
def recent(limit: int = 10):
    _ensure_data_loaded()
    df = df_global
    if df is None or len(df) == 0:
        return {"matches": []}

    # Columnas posibles
    colL = _first_col(df, "League", ["Div"])
    colH = _first_col(df, "HomeTeam", ["Home", "Home_Team"])
    colA = _first_col(df, "AwayTeam", ["Away", "Away_Team"])
    colHG = _first_col(df, "FTHG", ["HG", "HomeGoals", "Home_Goals"])
    colAG = _first_col(df, "FTAG", ["AG", "AwayGoals", "Away_Goals"])
    colD  = _first_col(df, "Date", ["MatchDate", "DateStr", "Fecha"])

    # Si falta lo mínimo para mostrar, devolvemos vacío
    if not (colL and colH and colA):
        return {"matches": []}

    dfx = df[[c for c in [colD, colL, colH, colA, colHG, colAG] if c in df.columns]].copy()

    # Normaliza fecha
    if colD in dfx.columns:
        dfx["_d"] = pd.to_datetime(dfx[colD], errors="coerce", dayfirst=True)
    else:
        dfx["_d"] = pd.NaT

    # Ordena por fecha (más recientes primero) y toma los últimos N
    dfx = dfx.sort_values("_d", ascending=False).head(int(limit))

    # Construye salida con claves ESTÁNDAR
    rows = []
    for _, r in dfx.iterrows():
        rows.append({
            "Date": (r["_d"].strftime("%Y-%m-%d") if pd.notnull(r["_d"]) else str(r.get(colD, ""))),
            "League": str(r.get(colL, "")),
            "HomeTeam": str(r.get(colH, "")),
            "AwayTeam": str(r.get(colA, "")),
            "FTHG": (None if colHG not in dfx.columns or pd.isna(r.get(colHG)) else int(r.get(colHG))),
            "FTAG": (None if colAG not in dfx.columns or pd.isna(r.get(colAG)) else int(r.get(colAG))),
        })
    return {"matches": rows}

# --- Info (ligas y equipos filtrados) ---
@app.get("/leagues")
def leagues():
    _ensure_data_loaded()
    df = df_global
    if df is None:
        return {"leagues": []}
    colL = _first_col(df, "League", ["Div"])
    if not colL:
        return {"leagues": []}
    lgs = sorted([str(x) for x in df[colL].dropna().unique()])
    return {"leagues": lgs}

@app.get("/teams")
def teams(league: Optional[str] = Query(None, description="Código liga: ej E0, SP1, I1")):
    _ensure_data_loaded()
    df = df_global
    if df is None:
        return {"league": league, "teams": []}
    colL = _first_col(df, "League", ["Div"])
    if league and colL:
        df = df[df[colL] == league]
    home_col = _first_col(df, "HomeTeam", ["Home", "Home_Team"])
    away_col = _first_col(df, "AwayTeam", ["Away", "Away_Team"])
    if not home_col or not away_col:
        return {"league": league, "teams": []}
    homes = set(df[home_col].dropna().astype(str).unique())
    aways = set(df[away_col].dropna().astype(str).unique())
    all_teams = sorted(list(homes.union(aways)))
    return {"league": league, "teams": all_teams}

@app.get("/health")
def health():
    _ensure_data_loaded()
    df = df_global
    if df is None:
        return {"status": "empty", "teams": []}
    home_col = _first_col(df, "HomeTeam", ["Home", "Home_Team"])
    away_col = _first_col(df, "AwayTeam", ["Away", "Away_Team"])
    if not home_col or not away_col:
        return {"status": "ok", "teams": []}
    homes = set(df[home_col].dropna().astype(str).unique())
    aways = set(df[away_col].dropna().astype(str).unique())
    all_teams = sorted(list(homes.union(aways)))
    return {"status": "ok", "teams": all_teams}

# --- Predicción ---
class _PredictPayload(BaseModel):
    pass  # solo para recordar el response_model arriba

import time  # ponlo ARRIBA con los otros imports

@app.get("/predict", response_model=PredictResponse)
def predict(
    home: str = Query(..., description="Equipo local"),
    away: str = Query(..., description="Equipo visitante"),
):
    start_total = time.time()
    _ensure_data_loaded()

    # --- Poisson base ---
    t0 = time.time()
    res = model.predict(home, away)
    res["exp_goals_total"] = round(res["exp_goals_home"] + res["exp_goals_away"], 3)
    res["src_1x2"] = "poisson"
    res["src_ou25"] = "xg"
    res["src_btts"] = "poisson"
    print(f"[PERF] Poisson tomó {time.time() - t0:.3f} s")

    # --- 1X2 (ML) ---
    if clf_bundle:
        t1 = time.time()
        Xy = _get_Xy()

        def prof(Xy, team, is_home):
            cols = [c for c in Xy.columns if c.startswith("H_" if is_home else "A_")]
            mask = (Xy["HomeTeam"] == team) if is_home else (Xy["AwayTeam"] == team)
            return Xy.loc[mask, cols].tail(5).median() if len(Xy.loc[mask, cols]) else pd.Series({c: 0.0 for c in cols})

        hp, ap = prof(Xy, home, True), prof(Xy, away, False)
        Elo_H = Xy.loc[Xy["HomeTeam"] == home, "Elo_H"].tail(1)
        Elo_A = Xy.loc[Xy["AwayTeam"] == away, "Elo_A"].tail(1)
        if len(Elo_H) == 0: Elo_H = pd.Series([Xy["Elo_H"].median()])
        if len(Elo_A) == 0: Elo_A = pd.Series([Xy["Elo_A"].median()])

        row = pd.DataFrame([{
            "Elo_H": float(Elo_H.values[-1]), "Elo_A": float(Elo_A.values[-1]),
            **{k: float(v) for k, v in hp.items()},
            **{k: float(v) for k, v in ap.items()},
        }])
        feats = clf_bundle["features"]
        row = row.reindex(columns=feats).fillna(row.median(numeric_only=True))
        probs = clf_bundle["model"].predict_proba(row)[0]  # [p_away, p_draw, p_home]
        res["p_away"], res["p_draw"], res["p_home"] = map(lambda x: round(float(x), 4), probs)
        res["src_1x2"] = "ml"
        print(f"[PERF] ML 1X2 tomó {time.time() - t1:.3f} s")

    # --- Over/Under 2.5 (ML) ---
    if clf_ou_bundle:
        t2 = time.time()
        Xy = _get_Xy()

        def prof_ou(Xy, team, is_home):
            cols = [c for c in Xy.columns if c.startswith("H_" if is_home else "A_")]
            mask = (Xy["HomeTeam"] == team) if is_home else (Xy["AwayTeam"] == team)
            return Xy.loc[mask, cols].tail(5).median() if len(Xy.loc[mask, cols]) else pd.Series({c: 0.0 for c in cols})

        hp, ap = prof_ou(Xy, home, True), prof_ou(Xy, away, False)
        Elo_H = Xy.loc[Xy["HomeTeam"] == home, "Elo_H"].tail(1)
        Elo_A = Xy.loc[Xy["AwayTeam"] == away, "Elo_A"].tail(1)
        if len(Elo_H) == 0: Elo_H = pd.Series([Xy["Elo_H"].median()])
        if len(Elo_A) == 0: Elo_A = pd.Series([Xy["Elo_A"].median()])

        row_ou = pd.DataFrame([{
            "Elo_H": float(Elo_H.values[-1]), "Elo_A": float(Elo_A.values[-1]),
            **{k: float(v) for k, v in hp.items()},
            **{k: float(v) for k, v in ap.items()},
        }])
        feats_ou = clf_ou_bundle["features"]
        row_ou = row_ou.reindex(columns=feats_ou).fillna(row_ou.median(numeric_only=True))
        p_over = float(clf_ou_bundle["model"].predict_proba(row_ou)[0, 1])
        res["ou_over25"] = round(p_over, 4)
        res["ou_under25"] = round(1.0 - p_over, 4)
        res["src_ou25"] = "ml"
        print(f"[PERF] ML OU tomó {time.time() - t2:.3f} s")
    else:
        xt = res["exp_goals_total"]
        p_over = 1.0 / (1.0 + math.exp(-1.1 * (xt - 2.5)))
        res["ou_over25"] = round(p_over, 4)
        res["ou_under25"] = round(1.0 - p_over, 4)
        res["src_ou25"] = "xg"

    # --- BTTS (ML) ---
    if clf_btts_bundle:
        t3 = time.time()
        Xy = _get_Xy()

        def prof_bt(Xy, team, is_home):
            cols = [c for c in Xy.columns if c.startswith("H_" if is_home else "A_")]
            mask = (Xy["HomeTeam"] == team) if is_home else (Xy["AwayTeam"] == team)
            return Xy.loc[mask, cols].tail(5).median() if len(Xy.loc[mask, cols]) else pd.Series({c: 0.0 for c in cols})

        hp, ap = prof_bt(Xy, home, True), prof_bt(Xy, away, False)
        Elo_H = Xy.loc[Xy["HomeTeam"] == home, "Elo_H"].tail(1)
        Elo_A = Xy.loc[Xy["AwayTeam"] == away, "Elo_A"].tail(1)
        if len(Elo_H) == 0: Elo_H = pd.Series([Xy["Elo_H"].median()])
        if len(Elo_A) == 0: Elo_A = pd.Series([Xy["Elo_A"].median()])

        row_bt = pd.DataFrame([{
            "Elo_H": float(Elo_H.values[-1]), "Elo_A": float(Elo_A.values[-1]),
            **{k: float(v) for k, v in hp.items()},
            **{k: float(v) for k, v in ap.items()},
        }])
        feats_bt = clf_btts_bundle["features"]
        row_bt = row_bt.reindex(columns=feats_bt).fillna(row_bt.median(numeric_only=True))
        p_yes = float(clf_btts_bundle["model"].predict_proba(row_bt)[0, 1])
        res["btts_yes"] = round(p_yes, 4)
        res["btts_no"]  = round(1.0 - p_yes, 4)
        res["src_btts"] = "ml"
        print(f"[PERF] ML BTTS tomó {time.time() - t3:.3f} s")
    else:
        lam_h = res["exp_goals_home"]; lam_a = res["exp_goals_away"]
        p_yes = (1 - math.exp(-lam_h)) * (1 - math.exp(-lam_a))
        res["btts_yes"] = round(p_yes, 4)
        res["btts_no"]  = round(1.0 - p_yes, 4)
        res["src_btts"] = "poisson"

    # ====== CÓRNERS / TARJETAS / DISTR. GOLES ======
    df = df_global

    # Córners
    exp_ch = exp_ca = exp_ct = over95 = under95 = None
    if df is not None and _col_ok(df, "HC") and _col_ok(df, "AC"):
        exp_ch = _recent_mean_home(df, home, "HC", n=10)
        exp_ca = _recent_mean_away(df, away, "AC", n=10)
        exp_ct = _sum_opt(exp_ch, exp_ca)
        if exp_ct is not None:
            cdf9 = _poisson_cdf(9, exp_ct)   # línea 9.5
            if cdf9 is not None:
                over95  = _clip01(1.0 - cdf9)
                under95 = _clip01(cdf9)
    res["exp_corners_home"]  = round(exp_ch, 2) if exp_ch is not None else None
    res["exp_corners_away"]  = round(exp_ca, 2) if exp_ca is not None else None
    res["exp_corners_total"] = round(exp_ct, 2) if exp_ct is not None else None
    res["corners_over95"]    = round(over95, 4) if over95 is not None else None
    res["corners_under95"]   = round(under95, 4) if under95 is not None else None

    # Amarillas
    yh = ya = yt = y_over = y_under = None
    if df is not None and _col_ok(df, "HY") and _col_ok(df, "AY"):
        yh = _recent_mean_home(df, home, "HY", n=10)
        ya = _recent_mean_away(df, away, "AY", n=10)
        yt = _sum_opt(yh, ya)
        if yt is not None:
            cdf4 = _poisson_cdf(4, yt)  # línea 4.5
            if cdf4 is not None:
                y_over  = _clip01(1.0 - cdf4)
                y_under = _clip01(cdf4)
    res["exp_yellows_home"]  = round(yh, 2) if yh is not None else None
    res["exp_yellows_away"]  = round(ya, 2) if ya is not None else None
    res["exp_yellows_total"] = round(yt, 2) if yt is not None else None
    res["yellows_over45"]    = round(y_over, 4) if y_over is not None else None
    res["yellows_under45"]   = round(y_under, 4) if y_under is not None else None

    # Rojas
    rh = ra = rt = None
    if df is not None and _col_ok(df, "HR") and _col_ok(df, "AR"):
        rh = _recent_mean_home(df, home, "HR", n=12)
        ra = _recent_mean_away(df, away, "AR", n=12)
        rt = _sum_opt(rh, ra)
    res["exp_reds_home"]  = round(rh, 2) if rh is not None else None
    res["exp_reds_away"]  = round(ra, 2) if ra is not None else None
    res["exp_reds_total"] = round(rt, 2) if rt is not None else None

    # Distribución de goles
    lam = res.get("exp_goals_total", None)
    if lam is not None:
        p0 = _poisson_pmf(0, lam)
        p1 = _poisson_pmf(1, lam)
        p2 = _poisson_pmf(2, lam)
        if None not in (p0, p1, p2):
            res["p_goals_0"]     = round(p0, 4)
            res["p_goals_1"]     = round(p1, 4)
            res["p_goals_2"]     = round(p2, 4)
            res["p_goals_3plus"] = round(_clip01(1.0 - (p0+p1+p2)), 4)

    total_elapsed = time.time() - start_total
    print(f"[PERF] Predicción COMPLETA {home} vs {away} tomó {total_elapsed:.3f} s")

    return PredictResponse(**res)

# --- Explicaciones rápidas ---
@app.get("/explain")
def explain(
    home: str = Query(..., description="Equipo local"),
    away: str = Query(..., description="Equipo visitante"),
):
    _ensure_data_loaded()
    Xy = _get_Xy()

    def team_prof(Xy, team, is_home):
        cols = [c for c in Xy.columns if c.startswith("H_" if is_home else "A_")]
        mask = (Xy["HomeTeam"] == team) if is_home else (Xy["AwayTeam"] == team)
        return Xy.loc[mask, cols].tail(5).median() if len(Xy.loc[mask, cols]) else pd.Series({c:0.0 for c in cols})

    hp, ap = team_prof(Xy, home, True), team_prof(Xy, away, False)
    Elo_H = Xy.loc[Xy["HomeTeam"] == home, "Elo_H"].tail(1)
    Elo_A = Xy.loc[Xy["AwayTeam"] == away, "Elo_A"].tail(1)
    if len(Elo_H) == 0: Elo_H = pd.Series([Xy["Elo_H"].median()])
    if len(Elo_A) == 0: Elo_A = pd.Series([Xy["Elo_A"].median()])

    base_row = pd.DataFrame([{
        "Elo_H": float(Elo_H.values[-1]), "Elo_A": float(Elo_A.values[-1]),
        **{k: float(v) for k, v in hp.items()},
        **{k: float(v) for k, v in ap.items()},
    }])

    expl = {"home": home, "away": away, "reasons": {}}

    if clf_bundle:
        feats = clf_bundle["features"]
        row = base_row.reindex(columns=feats).fillna(base_row.median(numeric_only=True))
        expl["reasons"]["1x2"] = top_drivers(row, feats, 3)
    else:
        expl["reasons"]["1x2"] = ["Poisson por xG de ambos"]

    if clf_ou_bundle:
        feats = clf_ou_bundle["features"]
        row = base_row.reindex(columns=feats).fillna(base_row.median(numeric_only=True))
        expl["reasons"]["ou25"] = top_drivers(row, feats, 3)
    else:
        expl["reasons"]["ou25"] = ["xG total vs 2.5"]

    if clf_btts_bundle:
        feats = clf_btts_bundle["features"]
        row = base_row.reindex(columns=feats).fillna(base_row.median(numeric_only=True))
        expl["reasons"]["btts"] = top_drivers(row, feats, 3)
    else:
        expl["reasons"]["btts"] = ["Prob.(H>0)·Prob.(A>0) por Poisson"]

    return expl

# --- Rutas de comodidad ---
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/app/")

@app.get("/app/index.html", include_in_schema=False)
def app_index_file():
    return FileResponse(FRONT_DIR / "index.html")

@app.get("/status", response_class=HTMLResponse)
def status_page():
    csv = MERGED if MERGED.exists() else DATA_PATH
    df = pd.read_csv(csv)
    homes = set(df["HomeTeam"].dropna().astype(str).unique())
    aways = set(df["AwayTeam"].dropna().astype(str).unique())
    all_teams = sorted(list(homes.union(aways)))
    t0 = all_teams[0] if all_teams else "TeamA"
    t1 = all_teams[1] if len(all_teams) > 1 else "TeamB"
    html = f"""
    <html>
      <head><meta charset="utf-8"><title>Status</title></head>
      <body style="font-family:system-ui;background:#0f1115;color:#e9e9f1">
        <div style="max-width:900px;margin:40px auto">
          <h1>✅ API viva</h1>
          <p>Equipos cargados (todas las ligas): <b>{len(all_teams)}</b></p>
          <p>Prueba rápida: <code>/predict?home={t0}&amp;away={t1}</code></p>
          <p>Docs: <a href="/docs">/docs</a></p>
          <p>App: <a href="/app/">/app/</a></p>
        </div>
      </body>
    </html>
    """
    return html


# --- Estado del “bot” ---
@app.get("/bot_status")
def bot_status():
    return {
        "data_rows": (0 if df_global is None else int(len(df_global))),
        "features_ready": Xy_global is not None,
        "models": {
            "poisson": True,
            "ml_1x2": clf_bundle is not None,
            "ml_ou25": clf_ou_bundle is not None,
            "ml_btts": clf_btts_bundle is not None,
        },
        "current_csv": str(current_csv_path) if current_csv_path else None,
    }

# --- Importancias de features ---
@app.get("/importances")
def importances():
    if clf_bundle and "model" in clf_bundle:
        try:
            model = clf_bundle["model"]
            feats = clf_bundle["features"]
            if hasattr(model, "feature_importances_"):
                vals = model.feature_importances_
                top = sorted(zip(feats, vals), key=lambda x: -x[1])[:10]
                return {"top_features": [{"name": f, "importance": float(v)} for f, v in top]}
        except Exception as e:
            return {"error": str(e)}
    return {"top_features": []}

# --- Métricas básicas del dataset ---
@app.get("/metrics")
def metrics():
    _ensure_data_loaded()
    if df_global is None:
        return {"status": "empty"}

    n_matches = len(df_global)
    leagues = sorted(df_global[_first_col(df_global, "League", ["Div"])].unique()) if _first_col(df_global, "League", ["Div"]) else []
    teams = set(df_global[_first_col(df_global, "HomeTeam", ["Home", "Home_Team"])].dropna()) | set(df_global[_first_col(df_global, "AwayTeam", ["Away", "Away_Team"])].dropna())

    return {
        "matches": int(n_matches),
        "leagues": leagues,
        "n_teams": len(teams),
    }

# --- Mini panel de monitoreo ---
@app.get("/monitor", response_class=HTMLResponse)
def monitor():
    html = """
    <!doctype html>
    <html lang="es">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Monitor — Sexy Sports Predictor</title>
      <link rel="preconnect" href="https://cdn.jsdelivr.net">
      <style>
        :root{
          --bg:#0b0d12; --panel:#11151d; --muted:#9aa3b2; --text:#e6eaf2; --brand:#8b5cf6; --brand2:#06b6d4;
          --ring: 0 0 0 2px hsl(255 85% 65% / .35);
        }
        *{box-sizing:border-box}
        body{margin:0;background:radial-gradient(1200px 800px at 80% -10%,#13203a 0%,transparent 60%),
                         radial-gradient(1000px 600px at -10% 90%,#1e123a 0%,transparent 60%),var(--bg);
             color:var(--text);font:15px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial}
        .container{max-width:1100px;margin:0 auto;padding:22px}
        h1{margin:6px 0 18px;font-size:26px}
        .grid{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}
        .col-4{grid-column:span 4} .col-6{grid-column:span 6} .col-8{grid-column:span 8} .col-12{grid-column:span 12}
        @media (max-width:900px){.col-4,.col-6,.col-8{grid-column:span 12}}
        .card{background:color-mix(in oklab,var(--panel) 86%,black 14%);border:1px solid #1c2230;border-radius:18px;padding:18px;box-shadow:0 8px 30px #0008}
        .muted{color:var(--muted)}
        .pill{display:inline-flex;gap:8px;align-items:center;background:#0e1320;border:1px solid #20283a;padding:7px 10px;border-radius:999px;margin:4px 6px 0 0}
        .ok{color:#22c55e} .warn{color:#f59e0b}
        button{cursor:pointer;padding:10px 14px;border-radius:12px;border:1px solid #20283a;background:#0f1420;color:var(--text)}
        button.btn{background:linear-gradient(135deg,var(--brand),var(--brand2));color:white;border:none;font-weight:700}
        code{background:#0f1420;border:1px solid #20283a;border-radius:8px;padding:2px 6px}
        ul{margin:8px 0 0 18px}
        footer{margin:26px 0 8px;text-align:center;color:#7f8798;font-size:13px}
      </style>
    </head>
    <body>
      <div class="container">
        <h1>📊 Monitor — <span class="muted">Sexy Sports Predictor</span></h1>

        <div class="grid">
          <div class="col-8">
            <div class="card">
              <div style="display:flex;justify-content:space-between;align-items:center;gap:12px">
                <h3 style="margin:0">Estado del bot</h3>
                <div class="muted" id="csvPath">—</div>
              </div>
              <div id="botState" style="margin-top:8px"></div>
              <div style="margin-top:12px;display:flex;gap:10px;flex-wrap:wrap">
                <button class="btn" id="btnWarmup">⚡ Precalentar features</button>
                <button id="btnRefresh">🔄 Refrescar</button>
                <a href="/docs" target="_blank"><button>📚 API Docs</button></a>
                <a href="/app/" target="_blank"><button>🖥️ Abrir App</button></a>
              </div>
            </div>
          </div>

          <div class="col-4">
            <div class="card">
              <h3 style="margin:0 0 8px">Métricas</h3>
              <div class="muted" id="metricsSummary">Cargando…</div>
              <ul id="leaguesList"></ul>
            </div>
          </div>

          <div class="col-12">
            <div class="card">
              <h3 style="margin:0 0 8px">Top features (1X2)</h3>
              <canvas id="featChart" height="120"></canvas>
              <div class="muted" id="featNote" style="margin-top:8px"></div>
            </div>
          </div>
        </div>

        <footer>Hecho con 💜 por nosotros</footer>
      </div>

      <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
      <script>
        const $ = s => document.querySelector(s);
        const api = (p, opt) => fetch(p, opt).then(r => { if(!r.ok) throw new Error(r.status); return r.json(); });

        let chart;

        async function loadAll(){
          // bot_status
          const bs = await api('/bot_status');
          $('#csvPath').textContent = bs.current_csv || '—';
          const s = [];
          s.push(`<span class="pill">filas: <b>${bs.data_rows}</b></span>`);
          s.push(`<span class="pill">features: <b class="${bs.features_ready ? 'ok':'warn'}">${bs.features_ready ? 'listas':'no listas'}</b></span>`);
          s.push(`<span class="pill">Poisson: <b class="ok">${bs.models.poisson ? 'OK':'—'}</b></span>`);
          s.push(`<span class="pill">ML 1X2: <b class="${bs.models.ml_1x2 ? 'ok':'warn'}">${bs.models.ml_1x2 ? 'OK':'—'}</b></span>`);
          s.push(`<span class="pill">ML O/U: <b class="${bs.models.ml_ou25 ? 'ok':'warn'}">${bs.models.ml_ou25 ? 'OK':'—'}</b></span>`);
          s.push(`<span class="pill">ML BTTS: <b class="${bs.models.ml_btts ? 'ok':'warn'}">${bs.models.ml_btts ? 'OK':'—'}</b></span>`);
          $('#botState').innerHTML = s.join(' ');

          // metrics
          const m = await api('/metrics');
          if(m.status === 'empty'){
            $('#metricsSummary').textContent = 'Sin datos cargados.';
            $('#leaguesList').innerHTML = '';
          }else{
            $('#metricsSummary').textContent = `Partidos: ${m.matches} · Equipos: ${m.n_teams}`;
            const ul = $('#leaguesList'); ul.innerHTML = '';
            (m.leagues || []).slice(0,12).forEach(code=>{
              const li = document.createElement('li');
              li.textContent = code;
              ul.appendChild(li);
            });
          }

          // importances
          const imp = await api('/importances');
          const items = (imp.top_features || []);
          $('#featNote').textContent = items.length ? '' : 'Sin importancias disponibles (¿modelo ML 1X2 entrenado?).';

          const labels = items.map(x=>x.name);
          const data = items.map(x=>x.importance);

          const ctx = document.getElementById('featChart');
          const cfg = {
            type: 'bar',
            data: { labels, datasets:[{ label:'Importancia', data }] },
            options: {
              responsive:true,
              scales:{ y:{ beginAtZero:true } },
              plugins:{ legend:{ display:false } }
            }
          };
          if(chart){ chart.data = cfg.data; chart.update(); }
          else { chart = new Chart(ctx, cfg); }
        }

        $('#btnRefresh').addEventListener('click', ()=>loadAll());
        $('#btnWarmup').addEventListener('click', async ()=>{
          try{
            const r = await api('/warmup', {method:'POST'});
            alert('Warmup: ' + (r.features_ready ? 'features listas' : 'features no listas'));
            loadAll();
          }catch(e){ alert('No se pudo precalentar'); }
        });

        loadAll();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(html)
