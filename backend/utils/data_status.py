# backend/utils/data_status.py
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def _load_all_csvs():
    raw_dir = DATA_DIR / "raw"
    files = list(raw_dir.rglob("*.csv")) if raw_dir.exists() else []
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # normaliza la columna de fecha (prueba variantes comunes)
            date_col = None
            for col in ["Date", "date", "DATE", "MatchDate"]:
                if col in df.columns:
                    date_col = col
                    break
            if date_col is None:
                # si no hay fecha, salta ese archivo
                continue
            # dayfirst=True porque football-data suele venir dd/mm/yyyy
            df["date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True, utc=True)
            df["_source"] = f.name
            dfs.append(df[["date", "_source"]])
        except Exception:
            # si un archivo falla, seguimos con los demás
            continue
    if not dfs:
        return pd.DataFrame(columns=["date", "_source"])
    return pd.concat(dfs, ignore_index=True)

def record_reload_marker():
    (DATA_DIR / "last_reload.txt").write_text(datetime.now(timezone.utc).isoformat())

def compute_status():
    now = datetime.now(timezone.utc)
    merged = _load_all_csvs()
    total_files = len(list((DATA_DIR / "raw").rglob("*.csv"))) if (DATA_DIR / "raw").exists() else 0
    total_rows = int(merged.shape[0]) if not merged.empty else 0

    last_match_date = pd.to_datetime(merged["date"]).max() if not merged.empty else None
    last_match_date_iso = last_match_date.isoformat() if pd.notna(last_match_date) else None

    is_fresh = False
    freshness_reason = "No se encontraron fechas en los CSV."
    if last_match_date is not None and pd.notna(last_match_date):
        delta = now - last_match_date.to_pydatetime()
        # margen de 2 días por horarios/zonas/fixtures nocturnos
        is_fresh = delta <= timedelta(days=2)
        freshness_reason = f"Último partido hace {delta.days} día(s)."

    last_reload_file = DATA_DIR / "last_reload.txt"
    last_reload = last_reload_file.read_text().strip() if last_reload_file.exists() else None

    return {
        "total_raw_csvs": total_files,
        "total_rows_indexed": total_rows,
        "last_match_date_utc": last_match_date_iso,
        "is_fresh": bool(is_fresh),
        "reason": freshness_reason,
        "last_reload_at": last_reload,
    }
