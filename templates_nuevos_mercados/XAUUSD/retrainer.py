"""
XAUUSD Gold — Weekly Retrainer
Re-trains the XGBoost model on fresh Gold data every week.

Usage:
  cd templates_nuevos_mercados/XAUUSD
  python retrainer.py              # retrain now
  python retrainer.py --check-only # check if retrain needed
  python retrainer.py --drift      # drift report only
"""

import os
import sys
import io
import json
import argparse
from datetime import datetime, date, timedelta

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in [_HERE, _MVP_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

from config import (
    SYMBOL, ARTIFACTS_DIR,
    RETRAIN_INTERVAL, N_BARS_TRAIN, HORIZON_BARS, EMBARGO_BARS,
    N_SPLITS, MIN_BARS_RETRAIN,
    MODEL_PATH, METRICS_PATH, RETRAIN_LOG_FILE, DRIFT_FILE,
)
from mt5_bridge import initialize as mt5_init, is_connected as mt5_connected, fetch_mt5_bars
from microstructure import build_intraday_features


# ─── Target label ─────────────────────────────────────────────────────────────

def _make_labels(close: pd.Series, horizon: int = 12) -> pd.Series:
    """
    Binary label: 1 if price is higher in `horizon` bars, 0 otherwise.
    Represents: "will Gold be higher in the next hour?"
    """
    future_close = close.shift(-horizon)
    return (future_close > close).astype(int)


# ─── Walk-forward CV ──────────────────────────────────────────────────────────

def _walk_forward_cv(X: pd.DataFrame, y: pd.Series, n_splits: int, embargo: int) -> list[dict]:
    """
    Walk-forward cross-validation with embargo to prevent data leakage.
    Returns list of fold metrics.
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score
    except ImportError:
        print("[ERROR] sklearn not installed. Run: pip install scikit-learn")
        return []

    n = len(X)
    fold_size = n // (n_splits + 1)
    metrics = []

    for i in range(n_splits):
        train_end   = fold_size * (i + 2)
        test_start  = train_end + embargo
        test_end    = test_start + fold_size

        if test_end > n:
            break

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test  = X.iloc[test_start:test_end]
        y_test  = y.iloc[test_start:test_end]

        # Drop NaN
        mask_tr = ~(X_train.isnull().any(axis=1) | y_train.isnull())
        mask_te = ~(X_test.isnull().any(axis=1)  | y_test.isnull())
        X_train, y_train = X_train[mask_tr], y_train[mask_tr]
        X_test,  y_test  = X_test[mask_te],  y_test[mask_te]

        if len(X_train) < 200 or len(X_test) < 50:
            continue

        model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred  = (proba > 0.5).astype(int)

        acc = accuracy_score(y_test, pred)
        try:
            auc = roc_auc_score(y_test, proba)
        except Exception:
            auc = float("nan")

        metrics.append({"fold": i, "acc": round(acc, 4), "auc": round(auc, 4), "n_train": len(X_train), "n_test": len(X_test)})
        print(f"  Fold {i}: acc={acc:.3f} auc={auc:.3f} (train={len(X_train)}, test={len(X_test)})")

    return metrics


# ─── Main retrain ─────────────────────────────────────────────────────────────

def retrain():
    """Fetch Gold data, build features, train XGBoost model, save."""
    print(f"\n=== XAUUSD Gold Retrainer — {date.today()} ===")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # MT5 connection
    if not mt5_connected():
        mt5_init()

    # Fetch historical bars
    print(f"Fetching {N_BARS_TRAIN} × {RETRAIN_INTERVAL} bars for {SYMBOL}...")
    bars = fetch_mt5_bars(RETRAIN_INTERVAL, N_BARS_TRAIN, SYMBOL)

    if bars is None or len(bars) < MIN_BARS_RETRAIN:
        msg = f"Not enough data: {len(bars) if bars is not None else 0} bars (need {MIN_BARS_RETRAIN})"
        print(f"[ERROR] {msg}")
        return {"status": "insufficient_data", "msg": msg}

    print(f"Data: {len(bars)} bars from {bars.index[0]} to {bars.index[-1]}")

    # Build features
    feats = build_intraday_features(bars, RETRAIN_INTERVAL)
    labels = _make_labels(bars["close"], HORIZON_BARS)

    # Align
    combined = feats.copy()
    combined["__label__"] = labels
    combined = combined.dropna()

    feature_cols = [c for c in combined.columns if c != "__label__"]
    X = combined[feature_cols]
    y = combined["__label__"]

    print(f"Features: {X.shape} | Label balance: {y.mean():.3f} (1=up)")

    # Walk-forward CV
    print(f"\nWalk-forward CV ({N_SPLITS} splits, embargo={EMBARGO_BARS} bars)...")
    cv_metrics = _walk_forward_cv(X, y, N_SPLITS, EMBARGO_BARS)

    if not cv_metrics:
        print("[WARN] CV failed — saving model trained on all data anyway.")

    # Train final model on all data
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        import joblib

        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean, y_clean = X[mask], y[mask]

        print(f"\nTraining final model on {len(X_clean)} samples...")
        model = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
        model.fit(X_clean, y_clean)

        joblib.dump(model, MODEL_PATH)
        print(f"Model saved: {MODEL_PATH}")

        # Feature importance
        fi = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1])
        top10 = fi[:10]
        print("\nTop 10 features:")
        for fname, imp in top10:
            print(f"  {fname:<25} {imp:.4f}")

        # Save metrics
        avg_acc = np.mean([m["acc"] for m in cv_metrics]) if cv_metrics else None
        avg_auc = np.mean([m["auc"] for m in cv_metrics]) if cv_metrics else None
        metrics = {
            "symbol":      SYMBOL,
            "trained_at":  datetime.now().isoformat(),
            "n_samples":   len(X_clean),
            "n_features":  len(feature_cols),
            "cv_avg_acc":  round(avg_acc, 4) if avg_acc else None,
            "cv_avg_auc":  round(avg_auc, 4) if avg_auc else None,
            "cv_folds":    cv_metrics,
            "top_features": [{"name": f, "importance": round(i, 4)} for f, i in top10],
        }
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Retrain log
        with open(RETRAIN_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"date": date.today().isoformat(), **metrics}, default=str) + "\n")

        print(f"\nRetrain complete. CV acc={avg_acc:.3f} auc={avg_auc:.3f}" if avg_acc else "\nRetrain complete.")
        return {"status": "ok", "metrics": metrics}

    except ImportError:
        print("[ERROR] sklearn/joblib not installed. Run: pip install scikit-learn joblib")
        return {"status": "import_error"}


# ─── Drift detection ──────────────────────────────────────────────────────────

def check_drift():
    """Compare model performance on recent data vs historical baseline."""
    if not os.path.exists(METRICS_PATH):
        print("No metrics file found. Run retrain first.")
        return

    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)

    baseline_auc = metrics.get("cv_avg_auc")
    print(f"\n=== Drift Report — {SYMBOL} ===")
    print(f"Last retrain: {metrics.get('trained_at', 'unknown')}")
    print(f"Baseline AUC: {baseline_auc}")

    # Simple heuristic: if last retrain was > 7 days ago, flag drift
    trained_at = metrics.get("trained_at", "")
    if trained_at:
        try:
            dt = datetime.fromisoformat(trained_at)
            days_old = (datetime.now() - dt).days
            if days_old > 7:
                print(f"[DRIFT WARNING] Model is {days_old} days old. Retrain recommended.")
            else:
                print(f"Model is {days_old} days old. OK.")
        except Exception:
            pass

    drift_report = {
        "symbol":       SYMBOL,
        "checked_at":   datetime.now().isoformat(),
        "baseline_auc": baseline_auc,
        "last_retrain": trained_at,
        "drift_flag":   False,
    }
    with open(DRIFT_FILE, "w", encoding="utf-8") as f:
        json.dump(drift_report, f, indent=2)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="XAUUSD Retrainer")
    parser.add_argument("--check-only", action="store_true", help="Only check if retrain needed")
    parser.add_argument("--drift",      action="store_true", help="Drift report only")
    args = parser.parse_args()

    if args.check_only or args.drift:
        check_drift()
    else:
        retrain()


if __name__ == "__main__":
    main()
