"""UK100 — Walk-Forward Retrainer"""

import os, sys, json, argparse, warnings
from datetime import datetime
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in [_HERE, _MVP_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

from config import SYMBOL, TIMEFRAME, N_BARS_LIVE, ARTIFACTS_DIR
from mt5_bridge import initialize as mt5_init, fetch_mt5_bars, is_connected as mt5_connected
from microstructure import build_intraday_features

N_SPLITS  = 5
EMBARGO   = 24
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "models")


def fetch_training_bars(n=5000):
    if not mt5_connected():
        if not mt5_init():
            raise RuntimeError("MT5 connection failed")
    bars = fetch_mt5_bars(TIMEFRAME, n, SYMBOL)
    if bars is None or len(bars) < 200:
        raise ValueError(f"Not enough bars: {len(bars) if bars is not None else 0}")
    return bars


def build_features(bars):
    feats = build_intraday_features(bars, TIMEFRAME)
    X_cols = [c for c in feats.columns if c not in ("signal","label","target")]
    X = feats[X_cols].fillna(0).replace([np.inf,-np.inf], 0)
    # Build target: 1=long profitable, -1=short profitable, 0=flat
    close = bars["close"].values
    target = []
    fwd = 6
    for i in range(len(close)):
        if i + fwd >= len(close):
            target.append(0)
        else:
            ret = (close[i+fwd] - close[i]) / close[i]
            if ret > 0.001:   target.append(1)
            elif ret < -0.001: target.append(-1)
            else:              target.append(0)
    y = pd.Series(target, index=feats.index)
    mask = X.index.isin(feats.index)
    return X[mask], y[mask]


def walk_forward_cv(X, y):
    n = len(X)
    fold_size = n // (N_SPLITS + 1)
    results = []
    for i in range(N_SPLITS):
        train_end = fold_size * (i + 1)
        test_start = train_end + EMBARGO
        test_end   = test_start + fold_size
        if test_end > n: break
        X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
        X_te, y_te = X.iloc[test_start:test_end], y.iloc[test_start:test_end]
        sc  = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        clf.fit(X_tr_s, y_tr)
        preds = clf.predict(X_te_s)
        acc   = accuracy_score(y_te, preds)
        results.append({"fold": i+1, "train_n": train_end, "test_n": len(y_te), "accuracy": round(acc,4)})
        print(f"  Fold {i+1}: acc={acc:.4f} | train={train_end} | test={len(y_te)}")
    return results


def train_final(X, y):
    os.makedirs(MODEL_DIR, exist_ok=True)
    sc  = StandardScaler()
    X_s = sc.fit_transform(X)
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    clf.fit(X_s, y)
    mpath = os.path.join(MODEL_DIR, f"{SYMBOL}_model.pkl")
    spath = os.path.join(MODEL_DIR, f"{SYMBOL}_scaler.pkl")
    joblib.dump(clf, mpath)
    joblib.dump(sc,  spath)
    print(f"  Model saved: {mpath}")
    print(f"  Scaler saved: {spath}")
    return clf, sc


def main():
    parser = argparse.ArgumentParser(description="UK100 Retrainer")
    parser.add_argument("--cv-only", action="store_true", help="Cross-validation only, no final fit")
    parser.add_argument("--bars",    type=int, default=5000)
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  UK100 Walk-Forward Retrainer")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*55)

    print("\nFetching bars...")
    bars = fetch_training_bars(args.bars)
    print(f"  Got {len(bars)} bars of {TIMEFRAME}")

    print("\nBuilding features...")
    X, y = build_features(bars)
    print(f"  X shape: {X.shape} | Classes: {dict(y.value_counts())}")

    print(f"\nWalk-forward CV ({N_SPLITS} splits, embargo={EMBARGO})...")
    cv_results = walk_forward_cv(X, y)
    accs = [r["accuracy"] for r in cv_results]
    print(f"  Mean acc: {np.mean(accs):.4f} | Std: {np.std(accs):.4f}")

    if not args.cv_only:
        print("\nTraining final model on full dataset...")
        train_final(X, y)

    rpath = os.path.join(ARTIFACTS_DIR, "retrain_results.json")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(rpath,"w",encoding="utf-8") as f:
        json.dump({"symbol":SYMBOL,"timestamp":datetime.now().isoformat(),
                   "n_bars":len(bars),"cv_results":cv_results,
                   "mean_acc":round(np.mean(accs),4) if accs else None}, f, indent=2)
    print(f"\nResults saved: {rpath}")
    print("\nDone.")

if __name__ == "__main__":
    main()
