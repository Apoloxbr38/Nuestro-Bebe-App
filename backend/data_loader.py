# backend/data_loader.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import unicodedata
import re

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

MERGED = DATA_DIR / "merged.csv"

# Football-Data (temporadas europeas)
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

def _season_code(year: int) -> str:
    """2023 -> '2324'"""
    return f"{year % 100:02d}{(year + 1) % 100:02d}"

def _current_season_start_year(today: datetime | None = None) -> int:
    d = today or datetime.utcnow()
    # Temporada europea: Jul–Dic empieza este año; Ene–Jun pertenece al año anterior
    return d.year if d.month >= 7 else d.year - 1

def _last_n_start_years(n: int, today: datetime | None = None) -> list[int]:
    cur = _current_season_start_year(today)
    return [cur - i for i in range(n)]

def _clean_team(x: str) -> str:
    if not isinstance(x, str):
        return x
    x = x.strip()
    x = unicodedata.normalize("NFKC", x)
    x = re.sub(r"\s+", " ", x)
    return x

def _read_cached_or_download(league: str, year: int, max_age_hours: int = 24) -> pd.DataFrame | None:
    """
    Cachea en data/raw/{league}/{season}.csv y reutiliza si la caché es reciente.
    """
    season = _season_code(year)
    out_path = RAW_DIR / league / f"{season}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Usa caché si no está vieja
    if out_path.exists():
        try:
            mtime = datetime.fromtimestamp(out_path.stat().st_mtime)
            if datetime.now() - mtime < timedelta(hours=max_age_hours):
                return pd.read_csv(out_path)
        except Exception:
            # Si falló la lectura, borramos y seguimos
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass

    # Descarga directa
    url = BASE_URL.format(season=season, league=league)
    try:
        df = pd.read_csv(url)
    except Exception:
        return None

    # Persistir caché (best-effort)
    try:
        df.to_csv(out_path, index=False)
    except Exception:
        pass

    return df

def _normalize(df: pd.DataFrame, league: str, year: int) -> pd.DataFrame | None:
    """
    Normaliza columnas base y añade extras si existen.
    Base: HomeTeam, AwayTeam, FTHG, FTAG, (Date opcional)
    Extras: HF, AF, HY, AY, HR, AR, HC, AC
    """
    if df is None or df.empty:
        return None

    cols = df.columns
    req = ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    extras = ["HF", "AF", "HY", "AY", "HR", "AR", "HC", "AC"]
    avail = [c for c in extras if c in cols]

    # Selección base + extras
    out = df[req + avail].copy()

    # Fecha si existe en alguno de estos nombres
    date_cols = [c for c in ["Date", "DateUTC", "Date_GMT"] if c in cols]
    if date_cols:
        out["Date"] = df[date_cols[0]]

    # Metadatos
    out["League"] = league
    out["SeasonStart"] = year

    # Limpieza de nombres de equipos
    out["HomeTeam"] = out["HomeTeam"].map(_clean_team)
    out["AwayTeam"] = out["AwayTeam"].map(_clean_team)

    return out

def refresh_dataset(
    leagues: tuple[str, ...] = ("SP1",),             # LaLiga
    start_years: tuple[int, ...] | None = (2023, 2024),
    last_n_years: int | None = None,
    max_age_hours: int = 24,
) -> Path:
    """
    Descarga/combina datasets.
    - start_years=(2022,2023,2024,2025)  → años exactos de inicio de temporada
    - last_n_years=4                     → últimas 4 temporadas hasta hoy
    """
    if last_n_years and (not start_years or len(start_years) == 0):
        start_years = tuple(_last_n_start_years(last_n_years))
    elif start_years is None:
        start_years = tuple(_last_n_start_years(2))  # por defecto 2 temporadas

    frames: list[pd.DataFrame] = []
    for lg in leagues:
        for yr in start_years:
            raw = _read_cached_or_download(lg, yr, max_age_hours=max_age_hours)
            norm = _normalize(raw, lg, yr)
            if norm is None or norm.empty:
                continue
            frames.append(norm)

    # Si no hay nada, intenta devolver el último merged o crea vacío
    if not frames:
        if MERGED.exists():
            return MERGED
        empty = pd.DataFrame(columns=[
            "HomeTeam", "AwayTeam", "FTHG", "FTAG",
            "League", "SeasonStart", "Date",
            "HF","AF","HY","AY","HR","AR","HC","AC"
        ])
        empty.to_csv(MERGED, index=False)
        return MERGED

    full = pd.concat(frames, ignore_index=True)
    full = full.dropna(subset=["HomeTeam", "AwayTeam"])

    # Fecha en formato datetime (football-data usa dayfirst)
    if "Date" in full.columns:
        full["Date"] = pd.to_datetime(full["Date"], errors="coerce", dayfirst=True)
        full = full.sort_values(["Date", "League", "SeasonStart"], na_position="last")

    MERGED.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(MERGED, index=False)
    return MERGED
