# backend/models/features.py
from __future__ import annotations
import pandas as pd
import numpy as np
from .elo import Elo

# Columnas opcionales que intentaremos usar si existen en el CSV
OPTIONAL_COLS = ["HF","AF","HY","AY","HR","AR","HC","AC"]  # Fouls, Yellow, Red, Corners

def tidy_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres básicos y conserva columnas opcionales si están presentes.
    Salida mínima: HomeTeam, AwayTeam, FTHG, FTAG, (Date opcional)
    Extra (si hay): HF, AF, HY, AY, HR, AR, HC, AC
    """
    cols = {c.lower(): c for c in df.columns}
    h  = cols.get("hometeam") or "HomeTeam"
    a  = cols.get("awayteam") or "AwayTeam"
    hg = cols.get("fthg") or "FTHG"
    ag = cols.get("ftag") or "FTAG"
    d  = cols.get("date")

    base = [h, a, hg, ag]
    if d:
        sel = [d] + base
        out = df[sel].copy()
        out.columns = ["Date","HomeTeam","AwayTeam","FTHG","FTAG"]
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.sort_values("Date")
    else:
        out = df[base].copy()
        out.columns = ["HomeTeam","AwayTeam","FTHG","FTAG"]

    # Adjunta extras si existen
    for c in OPTIONAL_COLS:
        if c in df.columns:
            out[c] = df[c].values

    return out.dropna(subset=["HomeTeam","AwayTeam","FTHG","FTAG"])

def _safe_col(g: pd.DataFrame, col: str) -> pd.Series:
    """Devuelve la serie si existe, si no una serie de NaN del mismo index."""
    if col in g.columns:
        return g[col]
    return pd.Series(np.nan, index=g.index)

def rolling_team_stats(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Construye vista long por equipo con GF/GA y stats opcionales,
    y calcula medias móviles (rolling) por equipo.
    """
    # Vista "long" con perspectiva de equipo
    home = pd.DataFrame({
        "Date": df["Date"] if "Date" in df.columns else pd.NaT,
        "Team": df["HomeTeam"],
        "GF": df["FTHG"],
        "GA": df["FTAG"],
        "is_home": 1,
        # Map de stats opcionales en perspectiva LOCAL
        "Fouls":  _safe_col(df, "HF"),
        "Yellow": _safe_col(df, "HY"),
        "Red":    _safe_col(df, "HR"),
        "Corners":_safe_col(df, "HC"),
    })
    away = pd.DataFrame({
        "Date": df["Date"] if "Date" in df.columns else pd.NaT,
        "Team": df["AwayTeam"],
        "GF": df["FTAG"],
        "GA": df["FTHG"],
        "is_home": 0,
        # Map de stats opcionales en perspectiva VISITANTE
        "Fouls":  _safe_col(df, "AF"),
        "Yellow": _safe_col(df, "AY"),
        "Red":    _safe_col(df, "AR"),
        "Corners":_safe_col(df, "AC"),
    })

    long = pd.concat([home, away], ignore_index=True)
    if "Date" in long.columns:
        long = long.sort_values("Date")

    # Resultados
    long["Win"]  = (long["GF"] > long["GA"]).astype(int)
    long["Draw"] = (long["GF"] == long["GA"]).astype(int)
    long["Loss"] = (long["GF"] < long["GA"]).astype(int)
    long["GD"]   = long["GF"] - long["GA"]

    # Para cada equipo, calcula rolling means de las métricas disponibles
    def _apply_roll(g: pd.DataFrame) -> pd.DataFrame:
        r = g.copy()
        roll = lambda s: s.rolling(window, min_periods=1).mean()
        r["r_GF"]        = roll(r["GF"])
        r["r_GA"]        = roll(r["GA"])
        r["r_GD"]        = roll(r["GD"])
        r["r_W"]         = roll(r["Win"])
        r["r_D"]         = roll(r["Draw"])
        r["r_L"]         = roll(r["Loss"])
        r["r_HomeRate"]  = roll(r["is_home"])
        # Extras (si la columna existe, ya viene con NaN si faltaba)
        r["r_Corners"]   = roll(r["Corners"])
        r["r_Yellow"]    = roll(r["Yellow"])
        r["r_Red"]       = roll(r["Red"])
        r["r_Fouls"]     = roll(r["Fouls"])
        return r

    feats = (
        long.groupby("Team", group_keys=False)
            .apply(_apply_roll)
            .reset_index(drop=True)
    )

    # Por si vinieran columnas duplicadas
    feats = feats.loc[:, ~feats.columns.duplicated()]
    return feats

def build_training_table(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Devuelve tabla con:
      - Date, HomeTeam, AwayTeam, FTHG, FTAG
      - Elo_H, Elo_A (previos al partido)
      - Features rolling para Local (H_*) y Visita (A_*)
      - y (clase 1X2): 0=away, 1=draw, 2=home
    Rellena NaN numéricos con la mediana.
    """
    df = tidy_matches(df)

    # Elo progresivo previo a cada partido
    elo = Elo()
    rows = []
    for _, r in df.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        hg, ag = int(r["FTHG"]), int(r["FTAG"])
        ra = elo.rating[h]
        rb = elo.rating[a]
        rows.append({
            "Date": r.get("Date", pd.NaT),
            "HomeTeam": h, "AwayTeam": a,
            "FTHG": hg, "FTAG": ag,
            "Elo_H": ra, "Elo_A": rb
        })
        elo.update_match(h, a, hg, ag)
    base = pd.DataFrame(rows)

    # Rolling features por equipo
    feats = rolling_team_stats(df, window=window)

    # Selección dinámica de columnas rolling (todas las r_*)
    ROLL_COLS = [c for c in feats.columns if c.startswith("r_")]

    def last_feats(team: str, date, home: bool) -> pd.Series:
        g = feats[feats["Team"] == team]
        if "Date" in feats.columns and pd.notna(date):
            g = g[g["Date"] < date]
        if len(g) == 0:
            return pd.Series([np.nan]*len(ROLL_COLS), index=ROLL_COLS)
        return g.iloc[-1][ROLL_COLS]

    # --- construir H y A sin duplicados de columnas y con índices limpios ---
    H_rows, A_rows = [], []
    for _, r in base.iterrows():
        H_rows.append(last_feats(r["HomeTeam"], r.get("Date", pd.NaT), home=True))
        A_rows.append(last_feats(r["AwayTeam"], r.get("Date", pd.NaT), home=False))

    H = pd.DataFrame(H_rows).add_prefix("H_")
    A = pd.DataFrame(A_rows).add_prefix("A_")

    # Deduplicar columnas si acaso
    def _dedup_cols(df_):
        if not df_.columns.is_unique:
            df_ = df_.loc[:, ~df_.columns.duplicated()]
        return df_

    H = _dedup_cols(H)
    A = _dedup_cols(A)
    base = base.loc[:, ~base.columns.duplicated()]

    # Alinear índices
    H = H.reset_index(drop=True)
    A = A.reset_index(drop=True)
    base = base.reset_index(drop=True)

    # Concatenación final
    Xy = pd.concat([base, H, A], axis=1)

    # Objetivo 1X2
    Xy["y"] = 0
    Xy.loc[Xy["FTHG"] == Xy["FTAG"], "y"] = 1
    Xy.loc[Xy["FTHG"] >  Xy["FTAG"], "y"] = 2

    # Rellenos prudentes (solo numéricos)
    num_cols = Xy.select_dtypes(include=[np.number]).columns
    Xy[num_cols] = Xy[num_cols].fillna(Xy[num_cols].median(numeric_only=True))
    Xy = Xy.dropna(subset=["HomeTeam","AwayTeam","FTHG","FTAG"])

    return Xy
