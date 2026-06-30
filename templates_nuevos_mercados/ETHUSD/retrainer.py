"""
ETHUSD Ethereum Robot — Retrainer
Fetches historical bars, engineers features, trains a GradientBoosting model,
and validates with walk-forward cross-validation + embargo.

Run weekly (e.g., Sunday at 02:00 CT):
  python retrainer.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in [_HERE, _MVP_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import (
    SYMBOL, ARTIFACTS_DIR, MODEL_PATH,
    RETRAIN_TIMEFRAME, N_BARS_TRAIN, RETRAIN_HORIZON_BARS,
)
from mt5_bridge import initialize, fetch_mt5_bars, is_connected
from microstructure import build_intraday_features

N_SPLITS = 5
EMBARGO  = 24   # bars to skip between train/test to avoid leakage


def _make_labels(close: pd.Series, horizon: int = RETRAIN_HORIZON_BARS) -> pd.Series:
    """Binary label: 1 if close[t+horizon] > close[t], else 0."""
    future = close.shift(-horizon)
    return (future > close).astype(int)


def _walk_forward_cv(X: pd.DataFrame, y: pd.Series, n_splits: int = N_SPLITS, embargo: int = EMBARGO):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score

    n = len(X)
    fold_size = n // (n_splits + 1)
    results = []

    for i in range(n_splits):
        train_end   = (i + 1) * fold_size
        test_start  = train_end + embargo
        test_end    = test_start + fold_size

        if test_end > n:
            break

        X_tr = X.iloc[:train_end]
        y_tr = y.iloc[:train_end]
        X_te = X.iloc[test_start:test_end]
        y_te = y.iloc[test_start:test_end]

        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        clf.fit(X_tr, y_tr)

        preds = clf.predict(X_te)
        proba = clf.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_te, preds)
        auc = roc_auc_score(y_te, proba) if len(np.unique(y_te)) > 1 else 0.5

        results.append({"fold": i + 1, "accuracy": round(acc, 4), "auc": round(auc, 4), "n_train": len(X_tr), "n_test": len(X_te)})
        print(f"  Fold {i+1}: acc={acc:.4f} auc={auc:.4f} (train={len(X_tr)}, test={len(X_te)})")

    return results


def run_retrain():
    print(f"\n{'='*58}")
    print(f"ETHUSD Retrainer — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*58}")

    # ── 1. Connect & fetch ────────────────────────────────────────────────────
    if not is_connected():
        ok = initialize()
        if not ok:
            print("ERROR: MT5 connection failed")
            return

    print(f"Fetching {N_BARS_TRAIN} bars × {RETRAIN_TIMEFRAME} for {SYMBOL}...")
    bars = fetch_mt5_bars(RETRAIN_TIMEFRAME, N_BARS_TRAIN, SYMBOL)
    if bars is None or len(bars) < 500:
        print(f"ERROR: Not enough bars ({len(bars) if bars is not None else 0})")
        return

    print(f"Bars fetched: {len(bars)} | From: {bars.index[0]} → {bars.index[-1]}")

    # ── 2. Features ───────────────────────────────────────────────────────────
    feats = build_intraday_features(bars, RETRAIN_TIMEFRAME)
    y     = _make_labels(bars["close"], horizon=RETRAIN_HORIZON_BARS)

    # Drop NaN rows (feature warm-up period)
    valid = feats.notna().all(axis=1) & y.notna()
    X = feats[valid]
    y = y[valid]
    print(f"Valid samples after NaN drop: {len(X)} | Positive rate: {y.mean():.3f}")

    # ── 3. Walk-forward CV ────────────────────────────────────────────────────
    print(f"\nWalk-forward CV ({N_SPLITS} folds, embargo={EMBARGO} bars):")
    fold_results = _walk_forward_cv(X, y)

    avg_acc = np.mean([f["accuracy"] for f in fold_results])
    avg_auc = np.mean([f["auc"]      for f in fold_results])
    print(f"\nAvg accuracy: {avg_acc:.4f} | Avg AUC: {avg_auc:.4f}")

    # ── 4. Train final model ──────────────────────────────────────────────────
    from sklearn.ensemble import GradientBoostingClassifier
    print("\nTraining final model on all data...")
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    clf.fit(X, y)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")

    # ── 5. Feature importance ─────────────────────────────────────────────────
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    top10 = importances.nlargest(10)
    print("\nTop 10 features:")
    for feat, imp in top10.items():
        print(f"  {feat:<30s}: {imp:.4f}")

    # ── 6. Save metadata ──────────────────────────────────────────────────────
    import json
    meta = {
        "trained_at":   datetime.now().isoformat(),
        "symbol":       SYMBOL,
        "timeframe":    RETRAIN_TIMEFRAME,
        "n_bars":       len(bars),
        "n_samples":    len(X),
        "horizon_bars": RETRAIN_HORIZON_BARS,
        "avg_accuracy": round(avg_acc, 4),
        "avg_auc":      round(avg_auc, 4),
        "folds":        fold_results,
    }
    with open(MODEL_PATH.replace(".joblib", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nRetrain complete. AUC={avg_auc:.4f}")
    print("="*58)


if __name__ == "__main__":
    run_retrain()
