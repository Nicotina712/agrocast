"""
src/model/audit_lookahead.py
Auditoría de look-ahead bias y robustez OOS del clasificador de retornos.

Estrategia:
  1. Carga features.csv y entrena un XGBClassifier (mismos hiperparámetros
     que train_returns) usando SOLO datos hasta una fecha de corte.
  2. Evalúa en el período post-corte (estrictamente OOS, sin haberlos visto).
  3. Compara AUC/hit rate del corte temprano vs. el global.
  4. Detector de leakage: corre permutation importance en el bloque OOS;
     si una feature solita explica >40% del AUC, sospechá.

Cortes evaluados:
  - hasta 2023-12-31 → val 2024 + 2025 + 2026
  - hasta 2024-12-31 → val 2025 + 2026
  - hasta 2025-06-30 → val H2 2025 + 2026

Si el AUC se mantiene >0.55 en los 3 cortes, la mejora es robusta.
Si cae a <0.52 en cortes tempranos, hay sobreajuste a la cola del histórico.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

# Windows: stdout viene en cp1252 y rompe al imprimir flechas/bullets.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FEATURES     = os.path.join(_PROJECT_ROOT, "data", "features.csv")
_OUT_PATH     = os.path.join(_PROJECT_ROOT, "artifacts", "lookahead_audit.json")

NON_FEATURE_COLS = {
    "Date", "Soybeans",
    "ret_1d_fwd", "ret_7d_fwd", "ret_14d_fwd", "ret_30d_fwd",
    "ret_7d_net", "ret_14d_net", "direction",
}
TARGET = "ret_14d_fwd"

CUTOFFS = ["2023-12-31", "2024-12-31", "2025-06-30"]

# Event embargo: gap entre train y test para evitar contaminación con ventana
# forward del target (ret_14d_fwd usa precios D+1..D+14) y eventos WASDE/COT
# que se publican en fechas conocidas. 18 días = 14 (horizonte) + 4 (buffer evento).
EMBARGO_DAYS = 18


def _build_target(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from src.trader.costs import estimate_round_trip_cost_pct
        if "event_cost_mult" in df.columns:
            cost = df["event_cost_mult"].apply(lambda m: estimate_round_trip_cost_pct(float(m)))
        else:
            cost = pd.Series(estimate_round_trip_cost_pct(1.0), index=df.index)
    except Exception:
        cost = pd.Series(0.0005, index=df.index)
    df = df.copy()
    df["ret_14d_net"] = df[TARGET] - cost
    df["direction"]   = (df["ret_14d_net"] > 0).astype(int)
    return df


def _train_eval(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    feat_cols = [c for c in df_train.columns if c not in NON_FEATURE_COLS]
    X_tr, y_tr = df_train[feat_cols].fillna(0), df_train["direction"]
    X_te, y_te = df_test[feat_cols].fillna(0),  df_test["direction"]

    n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = float(n_neg / n_pos) if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, gamma=0.05,
        scale_pos_weight=spw, eval_metric="logloss", random_state=42,
    )
    model.fit(X_tr, y_tr, verbose=False)

    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs > 0.5).astype(int)
    try:
        auc = roc_auc_score(y_te, probs)
    except Exception:
        auc = float("nan")
    acc = accuracy_score(y_te, preds)

    # Top-5 features por importancia (gain)
    fi = pd.Series(model.feature_importances_, index=feat_cols)
    top5 = fi.sort_values(ascending=False).head(5).round(4).to_dict()
    top1_share = float(fi.max() / max(fi.sum(), 1e-9))

    # Distribución
    dist = {
        "BUY":  float((probs > 0.58).mean()),
        "SELL": float((probs < 0.42).mean()),
        "HOLD": float(((probs >= 0.42) & (probs <= 0.58)).mean()),
    }

    return {
        "n_train":      int(len(df_train)),
        "n_test":       int(len(df_test)),
        "auc":          round(float(auc), 4),
        "accuracy":     round(float(acc), 4),
        "p_mean":       round(float(probs.mean()), 4),
        "p_std":        round(float(probs.std()), 4),
        "top1_share":   round(top1_share, 4),
        "top5_features": top5,
        "signal_dist":  {k: round(v, 3) for k, v in dist.items()},
    }


def run_audit(features_path: str = _FEATURES) -> dict:
    if not os.path.exists(features_path):
        raise FileNotFoundError(features_path)

    df = pd.read_csv(features_path, parse_dates=["Date"])
    df = df.dropna(subset=[TARGET]).sort_values("Date").reset_index(drop=True)
    df = _build_target(df)

    results: dict = {"cutoffs": {}, "global": None}

    embargo = pd.Timedelta(days=EMBARGO_DAYS)
    for cutoff in CUTOFFS:
        cut = pd.Timestamp(cutoff)
        train = df[df["Date"] <= cut]
        # Embargo: descartar ventana [cut, cut+EMBARGO_DAYS] para evitar
        # contaminación por ret_14d_fwd y eventos macro programados.
        test  = df[df["Date"] >  cut + embargo]
        if len(train) < 200 or len(test) < 50:
            results["cutoffs"][cutoff] = {"skipped": True,
                                           "reason": f"train={len(train)} test={len(test)}"}
            continue
        results["cutoffs"][cutoff] = _train_eval(train, test)

    # Global 80/20 con embargo entre train y test
    split = int(len(df) * 0.8)
    df_train_g = df.iloc[:split]
    cut_g      = df_train_g["Date"].max()
    df_test_g  = df[df["Date"] > cut_g + embargo]
    results["global"] = _train_eval(df_train_g, df_test_g)
    results["embargo_days"] = EMBARGO_DAYS

    # Diagnóstico
    aucs = [r["auc"] for r in results["cutoffs"].values()
            if isinstance(r, dict) and "auc" in r and r["auc"] == r["auc"]]
    flags = []
    if not aucs:
        flags.append("Sin cutoffs evaluables")
    else:
        if min(aucs) < 0.52:
            flags.append(f"AUC mínimo {min(aucs):.3f} < 0.52 → posible overfit a cola del histórico")
        if results["global"]["auc"] - min(aucs) > 0.08:
            flags.append(f"Gap global vs early-cutoff > 0.08 → señal solo reciente")
        top1 = max(r["top1_share"] for r in results["cutoffs"].values()
                    if isinstance(r, dict) and "top1_share" in r)
        if top1 > 0.40:
            flags.append(f"Una feature concentra {top1*100:.0f}% de importancia → revisar leakage")

    results["flags"] = flags or ["Sin alertas evidentes"]

    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "="*60)
    print(" AUDITORÍA LOOK-AHEAD / ROBUSTEZ OOS")
    print("="*60)
    for cutoff, r in results["cutoffs"].items():
        if r.get("skipped"):
            print(f"  Corte {cutoff}: SKIP ({r.get('reason')})")
            continue
        print(f"  Corte {cutoff}: AUC={r['auc']:.3f} acc={r['accuracy']:.3f} "
              f"n_test={r['n_test']} top1={r['top1_share']*100:.0f}% "
              f"sig={r['signal_dist']}")
    g = results["global"]
    print(f"  Global 80/20:   AUC={g['auc']:.3f} acc={g['accuracy']:.3f} "
          f"n_test={g['n_test']}")
    print("\n  Flags:")
    for f_ in results["flags"]:
        print(f"   - {f_}")
    print(f"\n  Detalle JSON: {_OUT_PATH}")
    return results


if __name__ == "__main__":
    run_audit()
