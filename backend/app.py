# backend/app.py
from __future__ import annotations
from fastapi import FastAPI, Query, Body
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import math, time
import numpy as np
import pandas as pd
import joblib

# --- Rutas/Modelos internos (ya existen en tu repo) ---
from .live_router import live_router
from backend.utils.data_status import compute_status, record_reload_marker
from backend.models.baseline import PoissonModel
from backend.data_loader import refresh_dataset, MERGED
from backend.train import MODEL_PATH
from backend.train_ou import MODEL_OU_PATH
from backend.train_btts import MODEL_BTTS_PATH
from backend.models.features import build_training_table

# =========================
# FastAPI App + Middleware
# =========================
app = FastAPI(
    title="Sexy Sports Predictor ⚽✨",
    version="1.1.0",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir frontend
FRONT_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/app", StaticFiles(directory=str(FRONT_DIR), html=True), name="frontend")

# Router LIVE
app.include_router(live_router, prefix="/live")

# =========================
# Datos base (CSV actual)
# =========================
# Si existe MERGED (última descarga), úsalo; si no, CSV de ejemplo.
LOCAL_SAMPLE = Path(__file__).parent / "data" / "sample_matches.csv"
DATA_PATH = MERGED if MERGED.exists() else LOCAL_SAMPLE

# =========================
# Utilidades modelos/compat
# =========================
def _xgb_compat(model):
    try:
        defaults = {
            "n_estimators": None, "learning_rate": None, "max_depth": None,
            "min_child_weight": None, "subsample": None, "colsample_bytree": None,
            "colsample_bylevel": None, "colsample_bynode": None, "reg_alpha": None,
            "reg_lambda": None, "gamma": None, "max_delta_step": None, "objective": None,
            "booster": None, "n_jobs": None, "random_state": None, "verbosity": None,
            "tree_method": None, "predictor": None, "gpu_id": None,
            "use_label_encoder": False,
        }
        for k, v in defaults.items():
            if not hasattr(model, k):
                try: setattr(model, k, v)
                except Exception: pass
    except Exception:
        pass
    return model

# =========================
# Estado en memoria
# =========================
df_global: Optional[pd.DataFrame] = None
Xy_global: Optional[pd.DataFrame] = None
current_csv_path: Optional[Path] = None

model = PoissonModel()
clf_bundle = None       # XGB 1X2
clf_ou_bundle = None    # XGB OU
clf_btts_bundle = None  # XGB BTTS

def _set_data(df: pd.DataFrame, csv_path: Path | None = None, build_features: bool = True):
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
        try:
            df = pd.read_csv(DATA_PATH)
        except Exception:
            df = pd.read_csv(LOCAL_SAMPLE)
        _set_data(df, DATA_PATH, build_features=False)

def _ensure_features():
    global Xy_global, df_global
    if Xy_global is None and df_global is not None:
        try:
            Xy_global = build_training_table(df_global, window=10)
        except Exception as e:
            print("[WARN] features lazy falló:", e)

def _get_Xy():
    _ensure_data_loaded()
    _ensure_features()
    return Xy_global if Xy_global is not None else build_training_table(df_global, window=10)

def _team_profile(Xy: pd.DataFrame, team: str, is_home: bool, prefix: str) -> pd.Series:
    """Mediana de las últimas 5 filas para columnas que empiezan por H_ o A_. Si no hay, devuelve ceros."""
    cols = [c for c in Xy.columns if c.startswith(prefix)]
    if not cols:
        return pd.Series({})
    try:
        mask = (Xy["HomeTeam"] == team) if is_home else (Xy["AwayTeam"] == team)
        sub = Xy.loc[mask, cols]
        if sub.empty:
            return pd.Series({c: 0.0 for c in cols})
        return sub.tail(5).median(numeric_only=True).fillna(0.0)
    except Exception:
        return pd.Series({c: 0.0 for c in cols})

def _safe_last(Xy: pd.DataFrame, col: str, team: str, is_home: bool) -> float:
    """Último valor válido para una columna; si no hay, usa la mediana global; si tampoco, 0."""
    try:
        mask = (Xy["HomeTeam"] == team) if is_home else (Xy["AwayTeam"] == team)
        s = Xy.loc[mask, col].dropna()
        if len(s):
            return float(s.iloc[-1])
        s2 = Xy[col].dropna()
        if len(s2):
            return float(s2.median())
    except Exception:
        pass
    return 0.0

def _make_feature_row(Xy: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Fila base con Elo y perfiles H_/A_ lista para reindexar a features del modelo."""
    hp = _team_profile(Xy, home, True,  "H_")
    ap = _team_profile(Xy, away, False, "A_")
    return pd.DataFrame([{
        "Elo_H": _safe_last(Xy, "Elo_H", home, True),
        "Elo_A": _safe_last(Xy, "Elo_A", away, False),
        **{k: float(v) for k, v in hp.items()},
        **{k: float(v) for k, v in ap.items()},
    }])


# =========================
# Endpoints utilitarios
# =========================
@app.get("/data/status")
def data_status():
    return compute_status()

@app.on_event("startup")
def _startup_load():
    global clf_bundle, clf_ou_bundle, clf_btts_bundle
    _ensure_data_loaded()
    # Carga modelos si existen
    clf_bundle      = joblib.load(MODEL_PATH)      if MODEL_PATH.exists()      else None
    clf_ou_bundle   = joblib.load(MODEL_OU_PATH)   if MODEL_OU_PATH.exists()   else None
    clf_btts_bundle = joblib.load(MODEL_BTTS_PATH) if MODEL_BTTS_PATH.exists() else None
    # Compat XGB
    if clf_bundle and "model" in clf_bundle:       clf_bundle["model"]      = _xgb_compat(clf_bundle["model"])
    if clf_ou_bundle and "model" in clf_ou_bundle: clf_ou_bundle["model"]   = _xgb_compat(clf_ou_bundle["model"])
    if clf_btts_bundle and "model" in clf_btts_bundle: clf_btts_bundle["model"] = _xgb_compat(clf_btts_bundle["model"])
    # Precarga features
    _ensure_features()
    print("💜 Warmup completado.")

# =========================
# Últimos partidos (tabla)
# =========================
def _first_col(df, primary, alts):
    for c in (primary, *alts):
        if c in df.columns:
            return c
    return None

@app.get("/recent")
def recent(limit: int = 10):
    """
    Devuelve los últimos 'limit' partidos detectando columnas estándar.
    """
    _ensure_data_loaded()
    df = df_global
    if df is None or len(df) == 0:
        return {"matches": []}

    colL = _first_col(df, "League", ["Div"])
    colH = _first_col(df, "HomeTeam", ["Home", "Home_Team"])
    colA = _first_col(df, "AwayTeam", ["Away", "Away_Team"])
    colHG = _first_col(df, "FTHG", ["HG", "HomeGoals", "Home_Goals"])
    colAG = _first_col(df, "FTAG", ["AG", "AwayGoals", "Away_Goals"])
    colD  = _first_col(df, "Date",  ["MatchDate", "DateStr", "Fecha"])
    if not (colH and colA): return {"matches": []}

    keep = [c for c in [colD, colL, colH, colA, colHG, colAG] if c]
    dfx = df[keep].copy()

    # Normalización robusta de fechas
    if colD and colD in dfx.columns:
        _s = dfx[colD].astype(str)
        # Detecta si hay formato DD/MM/AAAA
        dayfirst = _s.str.contains(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", regex=True).any()
        dfx["_d"] = pd.to_datetime(_s, errors="coerce", dayfirst=bool(dayfirst))
    else:
        dfx["_d"] = pd.NaT

    dfx = dfx.sort_values("_d", ascending=False, na_position="last").head(int(limit))

    rows = []
    for _, r in dfx.iterrows():
        dd = r["_d"]
        date_out = (dd.strftime("%Y-%m-%d") if pd.notnull(dd) else (str(r.get(colD, "")) if colD else ""))
        rows.append({
            "Date": date_out,
            "League": ("" if not colL else str(r.get(colL, ""))),
            "HomeTeam": str(r.get(colH, "")),
            "AwayTeam": str(r.get(colA, "")),
            "FTHG": (None if not colHG or pd.isna(r.get(colHG)) else int(r.get(colHG))),
            "FTAG": (None if not colAG or pd.isna(r.get(colAG)) else int(r.get(colAG))),
        })
    return {"matches": rows}

# =========================
# Predicción + explicación
# =========================
class PredictResponse(BaseModel):
    home: str; away: str
    p_home: float; p_draw: float; p_away: float
    exp_goals_home: float; exp_goals_away: float; exp_goals_total: float
    ou_over25: float; ou_under25: float
    btts_yes: float; btts_no: float
    top_scorelines: list | None = None
    src_1x2: str; src_ou25: str; src_btts: str
    # extras
    exp_corners_home: float | None = None
    exp_corners_away: float | None = None
    exp_corners_total: float | None = None
    corners_over95: float | None = None
    corners_under95: float | None = None
    exp_yellows_home: float | None = None
    exp_yellows_away: float | None = None
    exp_yellows_total: float | None = None
    yellows_over45: float | None = None
    yellows_under45: float | None = None
    exp_reds_home: float | None = None
    exp_reds_away: float | None = None
    exp_reds_total: float | None = None
    p_goals_0: float | None = None
    p_goals_1: float | None = None
    p_goals_2: float | None = None
    p_goals_3plus: float | None = None

def _poisson_pmf(k, lam):
    if lam is None: return None
    return math.exp(-lam) * (lam**k) / math.factorial(k)

def _poisson_cdf(k, lam):
    if lam is None: return None
    s = 0.0
    for i in range(0, int(k)+1):
        s += math.exp(-lam) * (lam**i) / math.factorial(i)
    return s

def _clip01(x): return None if x is None else max(0.0, min(1.0, float(x)))

def _col_ok(df, c): 
    return (c in df.columns) and (df[c].notna().sum() > 0)

def _recent_mean_home(df, team, col, n=10):
    if not _col_ok(df, col): return None
    d = df.loc[df["HomeTeam"] == team, col].dropna().tail(n)
    return float(d.mean()) if len(d) else None

def _recent_mean_away(df, team, col, n=10):
    if not _col_ok(df, col): return None
    d = df.loc[df["AwayTeam"] == team, col].dropna().tail(n)
    return float(d.mean()) if len(d) else None

def _sum_opt(a, b):
    if a is None and b is None: return None
    return (a or 0.0) + (b or 0.0)

@app.get("/predict", response_model=PredictResponse)
def predict(home: str = Query(...), away: str = Query(...)):
    t0 = time.time()
    _ensure_data_loaded()
    res = model.predict(home, away)
    res["exp_goals_total"] = round(res["exp_goals_home"] + res["exp_goals_away"], 3)
    res["src_1x2"] = "poisson"; res["src_ou25"] = "xg"; res["src_btts"] = "poisson"

    # --- 1X2 (ML) ---
    if clf_bundle:
        try:
            t1 = time.time()
            Xy = _get_Xy()
            row = _make_feature_row(Xy, home, away)
            feats = list(clf_bundle["features"])
            row = row.reindex(columns=feats).fillna(0.0)
            m = _xgb_compat(clf_bundle["model"])
            probs = m.predict_proba(row)[0]  # [p_away, p_draw, p_home]
            res["p_away"], res["p_draw"], res["p_home"] = map(lambda x: round(float(x), 4), probs)
            res["src_1x2"] = "ml"
            print(f"[PERF] ML 1X2 tomó {time.time() - t1:.3f} s")
        except Exception as e:
            print("[WARN] ML 1X2 fallback por error:", e)
            # deja las de Poisson que ya estaban

    # --- Over/Under 2.5 (ML) ---
    if clf_ou_bundle:
        try:
            t2 = time.time()
            Xy = _get_Xy()
            row_ou = _make_feature_row(Xy, home, away)
            feats_ou = list(clf_ou_bundle["features"])
            row_ou = row_ou.reindex(columns=feats_ou).fillna(0.0)
            m_ou = _xgb_compat(clf_ou_bundle["model"])
            p_over = float(m_ou.predict_proba(row_ou)[0, 1])
            res["ou_over25"] = round(p_over, 4)
            res["ou_under25"] = round(1.0 - p_over, 4)
            res["src_ou25"] = "ml"
            print(f"[PERF] ML OU tomó {time.time() - t2:.3f} s")
        except Exception as e:
            print("[WARN] ML OU fallback por error:", e)
            xt = res["exp_goals_total"]
            p_over = 1.0 / (1.0 + math.exp(-1.1 * (xt - 2.5)))
            res["ou_over25"] = round(p_over, 4)
            res["ou_under25"] = round(1.0 - p_over, 4)
            res["src_ou25"] = "xg"

    # --- BTTS (ML) ---
    if clf_btts_bundle:
        try:
            t3 = time.time()
            Xy = _get_Xy()
            row_bt = _make_feature_row(Xy, home, away)
            feats_bt = list(clf_btts_bundle["features"])
            row_bt = row_bt.reindex(columns=feats_bt).fillna(0.0)
            m_bt = _xgb_compat(clf_btts_bundle["model"])
            p_yes = float(m_bt.predict_proba(row_bt)[0, 1])
            res["btts_yes"] = round(p_yes, 4)
            res["btts_no"]  = round(1.0 - p_yes, 4)
            res["src_btts"] = "ml"
            print(f"[PERF] ML BTTS tomó {time.time() - t3:.3f} s")
        except Exception as e:
            print("[WARN] ML BTTS fallback por error:", e)
            lam_h = res["exp_goals_home"]; lam_a = res["exp_goals_away"]
            p_yes = (1 - math.exp(-lam_h)) * (1 - math.exp(-lam_a))
            res["btts_yes"] = round(p_yes, 4)
            res["btts_no"]  = round(1.0 - p_yes, 4)
            res["src_btts"] = "poisson"

    # Córners / tarjetas / distribución de goles
    df = df_global
    exp_ch = exp_ca = exp_ct = over95 = under95 = None
    if df is not None and _col_ok(df, "HC") and _col_ok(df, "AC"):
        exp_ch = _recent_mean_home(df, home, "HC", n=10)
        exp_ca = _recent_mean_away(df, away, "AC", n=10)
        exp_ct = _sum_opt(exp_ch, exp_ca)
        if exp_ct is not None:
            cdf9 = _poisson_cdf(9, exp_ct); over95 = _clip01(1.0 - cdf9); under95 = _clip01(cdf9)
    res["exp_corners_home"]  = round(exp_ch, 2) if exp_ch is not None else None
    res["exp_corners_away"]  = round(exp_ca, 2) if exp_ca is not None else None
    res["exp_corners_total"] = round(exp_ct, 2) if exp_ct is not None else None
    res["corners_over95"]    = round(over95, 4) if over95 is not None else None
    res["corners_under95"]   = round(under95, 4) if under95 is not None else None

    yh = ya = yt = y_over = y_under = None
    if df is not None and _col_ok(df, "HY") and _col_ok(df, "AY"):
        yh = _recent_mean_home(df, home, "HY", n=10)
        ya = _recent_mean_away(df, away, "AY", n=10)
        yt = _sum_opt(yh, ya)
        if yt is not None:
            cdf4 = _poisson_cdf(4, yt); y_over = _clip01(1.0 - cdf4); y_under = _clip01(cdf4)
    res["exp_yellows_home"]  = round(yh, 2) if yh is not None else None
    res["exp_yellows_away"]  = round(ya, 2) if ya is not None else None
    res["exp_yellows_total"] = round(yt, 2) if yt is not None else None
    res["yellows_over45"]    = round(y_over, 4) if y_over is not None else None
    res["yellows_under45"]   = round(y_under, 4) if y_under is not None else None

    rh = ra = rt = None
    if df is not None and _col_ok(df, "HR") and _col_ok(df, "AR"):
        rh = _recent_mean_home(df, home, "HR", n=12)
        ra = _recent_mean_away(df, away, "AR", n=12)
        rt = _sum_opt(rh, ra)
    res["exp_reds_home"]  = round(rh, 2) if rh is not None else None
    res["exp_reds_away"]  = round(ra, 2) if ra is not None else None
    res["exp_reds_total"] = round(rt, 2) if rt is not None else None

    lam = res.get("exp_goals_total", None)
    if lam is not None:
        p0 = _poisson_pmf(0, lam); p1 = _poisson_pmf(1, lam); p2 = _poisson_pmf(2, lam)
        if None not in (p0, p1, p2):
            res["p_goals_0"]     = round(p0, 4)
            res["p_goals_1"]     = round(p1, 4)
            res["p_goals_2"]     = round(p2, 4)
            res["p_goals_3plus"] = round(_clip01(1.0 - (p0+p1+p2)), 4)

    print(f"[PERF] /predict {home} vs {away} en {time.time()-t0:.3f}s")
    return PredictResponse(**res)

@app.get("/explain")
def explain(home: str, away: str):
    _ensure_data_loaded()
    Xy = _get_Xy()
    def team_prof(Xy, team, is_home):
        cols = [c for c in Xy.columns if c.startswith("H_" if is_home else "A_")]
        mask = (Xy["HomeTeam"] == team) if is_home else (Xy["AwayTeam"] == team)
        return Xy.loc[mask, cols].tail(5).median() if len(Xy.loc[mask, cols]) else pd.Series({c:0.0 for c in cols})
    hp, ap = team_prof(Xy, home, True), team_prof(Xy, away, False)
    Elo_H = Xy.loc[Xy["HomeTeam"] == home, "Elo_H"].tail(1) or pd.Series([Xy["Elo_H"].median()])
    Elo_A = Xy.loc[Xy["AwayTeam"] == away, "Elo_A"].tail(1) or pd.Series([Xy["Elo_A"].median()])
    base_row = pd.DataFrame([{"Elo_H": float(Elo_H.values[-1]), "Elo_A": float(Elo_A.values[-1]), **hp.to_dict(), **ap.to_dict()}])

    def top_drivers(row, feats, k=3):
        colz = [c for c in feats if c in row.columns]
        if not colz: return []
        vals = row[colz].iloc[0]; med = np.median(vals.values)
        mad = np.median(np.abs(vals.values - med)) or 1.0
        score = (vals - med)/mad
        idx = np.argsort(-np.abs(score.values))[:k]
        return [colz[i] for i in idx]

    expl = {"home": home, "away": away, "reasons": {}}
    if clf_bundle:   expl["reasons"]["1x2"] = top_drivers(base_row.reindex(columns=clf_bundle["features"]).fillna(base_row.median(numeric_only=True)), clf_bundle["features"], 3)
    else:            expl["reasons"]["1x2"] = ["Poisson por xG"]
    if clf_ou_bundle: expl["reasons"]["ou25"] = top_drivers(base_row.reindex(columns=clf_ou_bundle["features"]).fillna(base_row.median(numeric_only=True)), clf_ou_bundle["features"], 3)
    else:             expl["reasons"]["ou25"] = ["xG total vs 2.5"]
    if clf_btts_bundle: expl["reasons"]["btts"] = top_drivers(base_row.reindex(columns=clf_btts_bundle["features"]).fillna(base_row.median(numeric_only=True)), clf_btts_bundle["features"], 3)
    else:               expl["reasons"]["btts"] = ["Prob(H>0)·Prob(A>0)"]
    return expl

# =========================
# Listas y salud
# =========================
@app.get("/leagues")
def leagues():
    _ensure_data_loaded()
    df = df_global
    if df is None: return {"leagues": []}
    colL = _first_col(df, "League", ["Div"])
    if not colL: return {"leagues": []}
    lgs = sorted([str(x) for x in df[colL].dropna().unique()])
    return {"leagues": lgs}

@app.get("/teams")
def teams(league: Optional[str] = Query(None)):
    _ensure_data_loaded()
    df = df_global
    if df is None: return {"league": league, "teams": []}
    colL = _first_col(df, "League", ["Div"])
    if league and colL: df = df[df[colL] == league]
    home_col = _first_col(df, "HomeTeam", ["Home", "Home_Team"])
    away_col = _first_col(df, "AwayTeam", ["Away", "Away_Team"])
    if not home_col or not away_col: return {"league": league, "teams": []}
    homes = set(df[home_col].dropna().astype(str).unique())
    aways = set(df[away_col].dropna().astype(str).unique())
    all_teams = sorted(list(homes.union(aways)))
    return {"league": league, "teams": all_teams}

@app.get("/health")
def health():
    _ensure_data_loaded()
    df = df_global
    if df is None: return {"status": "empty", "teams": []}
    home_col = _first_col(df, "HomeTeam", ["Home", "Home_Team"])
    away_col = _first_col(df, "AwayTeam", ["Away", "Away_Team"])
    if not home_col or not away_col: return {"status": "ok", "teams": []}
    homes = set(df[home_col].dropna().astype(str).unique())
    aways = set(df[away_col].dropna().astype(str).unique())
    all_teams = sorted(list(homes.union(aways)))
    return {"status": "ok", "teams": all_teams}

@app.get("/metrics")
def metrics():
    _ensure_data_loaded()
    if df_global is None: return {"status": "empty"}
    n_matches = len(df_global)
    colL = _first_col(df_global, "League", ["Div"])
    colH = _first_col(df_global, "HomeTeam", ["Home", "Home_Team"])
    colA = _first_col(df_global, "AwayTeam", ["Away", "Away_Team"])
    leagues = sorted(df_global[colL].unique()) if colL else []
    teams = set(df_global[colH].dropna()) | set(df_global[colA].dropna()) if (colH and colA) else set()
    return {"matches": int(n_matches), "leagues": leagues, "n_teams": len(teams)}

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

# =========================
# Recarga dataset (opcional)
# =========================
@app.post("/reload")
def reload_data(build_features: bool = False):
    path = refresh_dataset(leagues=("SP1",), start_years=(2023, 2024))
    df = pd.read_csv(path)
    _set_data(df, Path(path), build_features=build_features)
    record_reload_marker()
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
    record_reload_marker()
    return {"status": "ok", "rows": int(len(df)), "file": str(path),
            "leagues": leagues, "years": (start_years if start_years else f"last_{last_n}"),
            "built": build_features}

# =========================
# Rutas de conveniencia/UI
# =========================
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/app/")

@app.get("/app/index.html", include_in_schema=False)
def app_index_file():
    return FileResponse(FRONT_DIR / "index.html")

@app.get("/monitor", response_class=HTMLResponse)
def monitor():
    return HTMLResponse("<h1 style='font-family:system-ui'>Monitor ok</h1>")

# Fin app.py
