"""
src/model/drift_monitor.py
Monitor de drift / salud del modelo en producción.

Idea:
  El audit lookahead corre cortes históricos y muestra que el AUC degrada
  con el tiempo (0.609 en 2023 -> 0.446 en 2025-H2). Pero ese audit corre
  cada vez que se entrena el modelo y reporta sobre datos de entrenamiento.
  Este monitor mira al *real-time*: las señales que el modelo emitió en los
  últimos 30/60/90 días vs el retorno realizado, agrupado por ventana, con
  conteos y AUC/accuracy en cada bucket.

  Salida en `artifacts/drift_monitor.json` consumida por /api/drift_monitor.

Algoritmo:
  1. Carga features.csv (target ya computado: ret_14d_fwd).
  2. Carga forecast_snapshots.json (señales emitidas día por día, si existe).
  3. Para cada ventana de N días (30, 60, 90):
       - filtra registros en esa ventana con ret_14d_fwd no-NaN
       - computa accuracy (signo de la predicción vs signo del retorno)
       - computa AUC si hay clase mixta
       - reporta distribución de señales BUY/SELL/HOLD
  4. Salud: verde si accuracy >= 55%, amarillo 48-55%, rojo <48%.

Si no hay forecast_snapshots suficientes (caso pre-PMF), cae a usar las
predicciones del modelo *re-entrenado* sobre las features y las compara
con el target observado en cada ventana — equivalente a un "in-sample
rolling" que sirve para detectar que el régimen cambió.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FEATURES     = os.path.join(_PROJECT_ROOT, "data",      "features.csv")
_SNAPSHOTS    = os.path.join(_PROJECT_ROOT, "data",      "forecast_snapshots.json")
_MODEL_PATH   = os.path.join(_PROJECT_ROOT, "src", "model", "artifacts", "returns_model.joblib")
_OUT_PATH     = os.path.join(_PROJECT_ROOT, "artifacts", "drift_monitor.json")

WINDOWS = [30, 60, 90]


def _load_features() -> Optional[pd.DataFrame]:
    if not os.path.exists(_FEATURES):
        return None
    df = pd.read_csv(_FEATURES, parse_dates=["Date"])
    df = df.dropna(subset=["ret_14d_fwd"])
    return df.sort_values("Date").reset_index(drop=True)


def _load_signals_from_snapshots() -> Optional[pd.DataFrame]:
    if not os.path.exists(_SNAPSHOTS):
        return None
    try:
        with open(_SNAPSHOTS, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) < 5:
            return None
        rows = []
        for d in data:
            sig = d.get("signal")
            dt  = d.get("snapshot_date")
            if not sig or not dt:
                continue
            rows.append({"Date": pd.to_datetime(dt), "signal": sig})
        return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    except Exception:
        return None


def _predict_in_window(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Predice probabilidad usando el modelo guardado. Si no se puede, None."""
    try:
        import joblib
        saved = joblib.load(_MODEL_PATH)
    except Exception:
        return None
    model        = saved["model"]
    feature_cols = saved.get("feature_cols", [])
    if not feature_cols:
        return None
    cols_avail = [c for c in feature_cols if c in df.columns]
    if len(cols_avail) < len(feature_cols) * 0.7:
        return None
    X = df[cols_avail].fillna(0).astype(float)
    try:
        proba = model.predict_proba(X)[:, 1]
        return proba
    except Exception:
        return None


def _bucket_stats(df: pd.DataFrame, n_days: int, today: pd.Timestamp,
                  signals_df: Optional[pd.DataFrame]) -> dict:
    cutoff = today - pd.Timedelta(days=n_days)
    sub = df[df["Date"] >= cutoff].copy()
    if len(sub) < 5:
        return {
            "window_days": n_days, "n": len(sub),
            "accuracy": None, "auc": None,
            "signal_dist": {}, "status": "insufficient_data",
        }

    # Señales del modelo en esa ventana (predict_proba si existe; fallback a snapshots)
    proba = _predict_in_window(sub)
    if proba is None:
        # Fallback: use snapshots si están
        if signals_df is not None and not signals_df.empty:
            sub_s = signals_df[signals_df["Date"] >= cutoff].copy()
            if len(sub_s) < 5:
                return {
                    "window_days": n_days, "n": len(sub_s),
                    "accuracy": None, "auc": None,
                    "signal_dist": sub_s["signal"].value_counts().to_dict() if not sub_s.empty else {},
                    "status": "no_predictions",
                }
            # Accuracy proxy: BUY si ret>0, SELL si ret<0. HOLD se ignora.
            merged = pd.merge_asof(
                sub_s.sort_values("Date"),
                sub[["Date", "ret_14d_fwd"]].sort_values("Date"),
                on="Date", direction="forward",
            ).dropna(subset=["ret_14d_fwd"])
            buy_sell = merged[merged["signal"].isin(["BUY", "SELL"])]
            if len(buy_sell) < 3:
                return {
                    "window_days": n_days, "n": len(merged),
                    "accuracy": None, "auc": None,
                    "signal_dist": merged["signal"].value_counts().to_dict(),
                    "status": "no_active_signals",
                }
            correct = ((buy_sell["signal"] == "BUY") & (buy_sell["ret_14d_fwd"] > 0)) | \
                      ((buy_sell["signal"] == "SELL") & (buy_sell["ret_14d_fwd"] < 0))
            return {
                "window_days":  n_days,
                "n":            len(merged),
                "n_active":     int(len(buy_sell)),
                "accuracy":     round(float(correct.mean()) * 100, 1),
                "auc":          None,
                "signal_dist":  {k: int(v) for k, v in merged["signal"].value_counts().to_dict().items()},
                "status":       "ok_from_snapshots",
            }
        return {
            "window_days": n_days, "n": len(sub),
            "accuracy": None, "auc": None,
            "signal_dist": {}, "status": "no_model_or_snapshots",
        }

    y_true   = (sub["ret_14d_fwd"] > 0).astype(int).values
    pred_pos = (proba > 0.5).astype(int)
    accuracy = float((pred_pos == y_true).mean()) * 100

    auc = None
    if len(set(y_true)) > 1:
        try:
            from sklearn.metrics import roc_auc_score
            auc = round(float(roc_auc_score(y_true, proba)) * 100, 1)
        except Exception:
            auc = None

    sub["_signal"] = pd.cut(
        proba, bins=[-1, 0.42, 0.58, 2],
        labels=["SELL", "HOLD", "BUY"],
    )
    dist = {k: int(v) for k, v in sub["_signal"].value_counts().to_dict().items()}

    return {
        "window_days":  n_days,
        "n":            int(len(sub)),
        "accuracy":     round(accuracy, 1),
        "auc":          auc,
        "signal_dist":  dist,
        "status":       "ok",
    }


def _compute_health(buckets: list[dict]) -> str:
    """verde / amarillo / rojo segun la accuracy reciente."""
    accs = [b.get("accuracy") for b in buckets if b.get("accuracy") is not None]
    if not accs:
        return "gray"
    recent = accs[0]  # el bucket más corto (30d) está primero
    if recent >= 55: return "green"
    if recent >= 48: return "yellow"
    return "red"


def run() -> dict:
    df = _load_features()
    if df is None or df.empty:
        out = {"ok": False, "error": "features.csv no disponible",
               "generated_at": datetime.utcnow().isoformat()}
        _save(out); return out

    today      = pd.Timestamp(df["Date"].max())
    signals_df = _load_signals_from_snapshots()

    buckets = [_bucket_stats(df, n, today, signals_df) for n in WINDOWS]

    # Distribución total reciente vs histórica para detección de régimen
    proba_full = _predict_in_window(df)
    regime_shift = None
    if proba_full is not None:
        cutoff_recent = today - pd.Timedelta(days=90)
        recent_mask = df["Date"] >= cutoff_recent
        old_mask    = df["Date"] <  cutoff_recent
        if recent_mask.sum() > 10 and old_mask.sum() > 50:
            recent_mean = float(proba_full[recent_mask].mean())
            old_mean    = float(proba_full[old_mask].mean())
            recent_std  = float(proba_full[recent_mask].std() or 1e-9)
            old_std     = float(proba_full[old_mask].std()    or 1e-9)
            regime_shift = {
                "p_buy_recent_90d":  round(recent_mean, 3),
                "p_buy_historical":  round(old_mean,    3),
                "delta":             round(recent_mean - old_mean, 3),
                "z_delta":           round((recent_mean - old_mean) / max(old_std, 1e-9), 2),
            }

    out = {
        "ok":            True,
        "generated_at":  datetime.utcnow().isoformat(),
        "today":         today.date().isoformat(),
        "buckets":       buckets,
        "regime_shift":  regime_shift,
        "health":        _compute_health(buckets),
    }
    _save(out)
    return out


def _save(out: dict) -> None:
    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    res = run()
    print(json.dumps(res, indent=2, ensure_ascii=False))
