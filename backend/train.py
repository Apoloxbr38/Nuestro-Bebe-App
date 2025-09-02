# backend/train.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

# Modelo: XGBoost si está disponible; si no, GradientBoosting como fallback
try:
    from xgboost import XGBClassifier
    def make_model():
        return XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=4,
            tree_method="hist",
        )
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier
    def make_model():
        return GradientBoostingClassifier()

from backend.models.features import build_training_table

DATA = Path(__file__).parent / "data" / "sample_matches.csv"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "model_1x2.pkl"

# Núcleo mínimo de features + extras (si existen en Xy)
CORE_FEATURES = [
    "Elo_H","Elo_A",
    "H_r_GF","H_r_GA","H_r_GD","H_r_W","H_r_D","H_r_L","H_r_HomeRate",
    "A_r_GF","A_r_GA","A_r_GD","A_r_W","A_r_D","A_r_L","A_r_HomeRate",
]
EXTRA_FEATURES = [
    "H_r_Corners","A_r_Corners",
    "H_r_Yellow","A_r_Yellow",
    "H_r_Red","A_r_Red",
    "H_r_Fouls","A_r_Fouls",
]

def _features_present(Xy: pd.DataFrame) -> list[str]:
    cols = []
    for c in CORE_FEATURES + EXTRA_FEATURES:
        if c in Xy.columns:
            cols.append(c)
    return cols

def train(csv_path: str | Path = DATA):
    df = pd.read_csv(csv_path)
    Xy = build_training_table(df, window=10)

    feats = _features_present(Xy)
    if len(feats) < 8:
        return {
            "rows": int(len(df)),
            "train_rows": 0,
            "val_rows": 0,
            "acc_argmax": None,
            "note": "Muy pocas columnas de features disponibles."
        }

    X = Xy[feats]
    y = Xy["y"]
    if len(X) < 60:
        return {
            "rows": int(len(df)),
            "train_rows": 0,
            "val_rows": 0,
            "acc_argmax": None,
            "note": "Muy pocos ejemplos tras preprocesado."
        }

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = make_model()
    model.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feats}, MODEL_PATH)

    # validación simple
    try:
        val_prob = model.predict_proba(X_val)
        acc = (np.argmax(val_prob, axis=1) == y_val.values).mean()
    except Exception:
        pred = model.predict(X_val)
        acc = (pred == y_val.values).mean()

    return {
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "acc_argmax": float(acc)
    }

if __name__ == "__main__":
    print(train())
