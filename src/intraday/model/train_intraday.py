"""
src/intraday/model/train_intraday.py
Entrena clasificador binario intradía (prob_up en horizonte H barras adelante).

Walk-forward CV con embargo proporcional al horizonte (default 2× horizon).

Inputs:
  - barras 5m enriquecidas (microstructure + regime + swing context)
Output:
  - artifacts/intraday/model_intraday.joblib
  - artifacts/intraday/intraday_train_metrics.json
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from src.intraday.data.tick_feed import fetch_intraday_bars
from src.intraday.data.session_calendar import annotate_sessions
from src.intraday.features.microstructure import build_intraday_features, feature_columns
from src.intraday.features.regime import add_regime_features
from src.intraday.features.context_swing import attach_swing_context


_OUT_DIR = os.path.join(_ROOT, "artifacts", "intraday")
_MODEL_PATH = os.path.join(_OUT_DIR, "model_intraday.joblib")
_METRICS_PATH = os.path.join(_OUT_DIR, "intraday_train_metrics.json")

INTERVAL    = "5m"
HORIZON_BARS = 12          # 60min sobre 5m
EMBARGO_BARS = HORIZON_BARS * 2
N_SPLITS     = 5

EXTRA_FEATURE_COLS = [
    "swing_bias_today", "swing_expected_ret",
    "swing_expected_vol", "swing_confidence", "swing_age_hours",
    "adx_14", "atr_zscore_60",
    "regime_trend", "regime_range", "regime_shock",
]


def build_dataset(interval: str = INTERVAL, horizon: int = HORIZON_BARS) -> pd.DataFrame:
    print(f"[train_intraday] fetching bars {interval} ...")
    bars = fetch_intraday_bars(interval=interval)
    if bars.empty:
        raise RuntimeError("no hay barras disponibles")

    print(f"[train_intraday] features microstructure ...")
    feat = build_intraday_features(bars, interval=interval)
    feat = add_regime_features(feat)
    feat = annotate_sessions(feat)
    feat = attach_swing_context(feat)

    # Target: signo del retorno futuro a H barras
    feat["future_close"] = feat["close"].shift(-horizon)
    feat["future_ret"]   = (feat["future_close"] - feat["close"]) / feat["close"]
    # Filtro de magnitud mínima para evitar etiquetas ruidosas (≥ 0.05% mov)
    feat["target"] = (feat["future_ret"] > 0.0005).astype(int)

    # Solo entrenar en RTH (mayor liquidez, target más limpio)
    feat = feat[feat["is_rth"] == 1].copy()
    feat = feat.dropna(subset=["future_ret"])
    print(f"[train_intraday] dataset: {len(feat)} barras RTH con target válido")
    return feat


def _feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols = [c for c in feature_columns() if c in df.columns]
    cols += [c for c in EXTRA_FEATURE_COLS if c in df.columns]
    # Excluir flags binarios redundantes y vwap raw (mantener vwap_dist)
    cols = [c for c in cols if c not in ("is_rth", "vwap_session")]
    X = df[cols].astype(float)
    return X, cols


def walk_forward_train(
    df: pd.DataFrame,
    n_splits: int = N_SPLITS,
    embargo: int = EMBARGO_BARS,
) -> tuple[object, dict, list[str]]:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise RuntimeError("xgboost no instalado: pip install xgboost")

    X, cols = _feature_matrix(df)
    y = df["target"].values

    n = len(df)
    fold_size = n // (n_splits + 1)
    fold_metrics = []

    print(f"[train_intraday] walk-forward {n_splits} folds, embargo={embargo} bars")
    for k in range(n_splits):
        train_end = (k + 1) * fold_size
        test_start = train_end + embargo
        test_end   = test_start + fold_size
        if test_end > n:
            break
        Xtr, ytr = X.iloc[:train_end], y[:train_end]
        Xte, yte = X.iloc[test_start:test_end], y[test_start:test_end]

        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
        # XGB no acepta NaN en algunas versiones — imputamos
        Xtr_f = Xtr.fillna(0); Xte_f = Xte.fillna(0)
        model.fit(Xtr_f, ytr)
        proba = model.predict_proba(Xte_f)[:, 1]
        pred  = (proba > 0.5).astype(int)
        try:
            auc = roc_auc_score(yte, proba) if len(set(yte)) > 1 else float("nan")
        except Exception:
            auc = float("nan")
        acc = accuracy_score(yte, pred)
        f1  = f1_score(yte, pred, zero_division=0)
        fold_metrics.append({"fold": k, "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
                             "auc": round(float(auc), 4), "acc": round(float(acc), 4),
                             "f1": round(float(f1), 4),
                             "pos_rate_train": round(float(ytr.mean()), 3),
                             "pos_rate_test":  round(float(yte.mean()), 3)})
        print(f"  fold {k}: AUC={auc:.3f} ACC={acc:.3f} F1={f1:.3f} "
              f"(n_tr={len(Xtr)} n_te={len(Xte)})")

    # Modelo final con todos los datos
    print("[train_intraday] entrenando modelo final con todo el dataset ...")
    final = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    final.fit(X.fillna(0), y)

    importances = sorted(
        zip(cols, final.feature_importances_), key=lambda x: x[1], reverse=True
    )[:15]
    metrics = {
        "interval":      INTERVAL,
        "horizon_bars":  HORIZON_BARS,
        "embargo_bars":  embargo,
        "n_total":       n,
        "n_features":    len(cols),
        "fold_metrics":  fold_metrics,
        "mean_auc":      round(float(np.nanmean([f["auc"] for f in fold_metrics])), 4),
        "mean_acc":      round(float(np.nanmean([f["acc"] for f in fold_metrics])), 4),
        "top_features":  [{"name": n, "importance": round(float(i), 4)} for n, i in importances],
    }
    return final, metrics, cols


def main():
    os.makedirs(_OUT_DIR, exist_ok=True)
    df = build_dataset()
    if len(df) < 500:
        print(f"[train_intraday] dataset chico ({len(df)} filas) — entreno pero metrics no serán confiables")

    model, metrics, cols = walk_forward_train(df)

    import joblib
    joblib.dump({"model": model, "feature_cols": cols,
                 "horizon_bars": HORIZON_BARS, "interval": INTERVAL},
                _MODEL_PATH)
    with open(_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n[train_intraday] OK")
    print(f"  modelo : {_MODEL_PATH}")
    print(f"  metrics: {_METRICS_PATH}")
    print(f"  AUC mean across folds: {metrics['mean_auc']}")
    print(f"  Top 5 features: {[f['name'] for f in metrics['top_features'][:5]]}")


if __name__ == "__main__":
    main()
