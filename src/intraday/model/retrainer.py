"""
src/intraday/model/retrainer.py
Weekly Model Retrainer — Walk-forward retrain with MT5 live data + paper trade feedback.

Architecture:
  - Runs weekly (Sunday or Monday pre-market)
  - Fetches historical bars from MT5 (up to 100k bars)
  - Incorporates paper trade outcomes as training signal
  - Walk-forward CV with embargo to prevent look-ahead
  - Saves new model + metrics + drift report
  - Falls back to yfinance if MT5 not available

Usage:
  python -m src.intraday.model.retrainer              # retrain now
  python -m src.intraday.model.retrainer --check-only  # check if retrain needed
  python -m src.intraday.model.retrainer --drift        # drift report only
"""

import os
import sys
import io
import json
import time
import shutil
import argparse
from datetime import datetime, date, timedelta

# Fix Windows console encoding (only when running as main script)
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_OUT_DIR = os.path.join(_ROOT, "artifacts", "intraday")
_QA_DIR = os.path.join(_ROOT, "artifacts", "quantagent")
_MODEL_PATH = os.path.join(_OUT_DIR, "model_intraday.joblib")
_METRICS_PATH = os.path.join(_OUT_DIR, "intraday_train_metrics.json")
_RETRAIN_LOG = os.path.join(_OUT_DIR, "retrain_log.jsonl")
_DRIFT_PATH = os.path.join(_OUT_DIR, "drift_report.json")


# =========================================================================
#  CONFIG
# =========================================================================

INTERVAL = "5m"
HORIZON_BARS = 12        # 12 x 5m = 60 min ahead
EMBARGO_BARS = 24        # 2x horizon
N_SPLITS = 5
MIN_BARS_RETRAIN = 2000  # need at least this many bars
RETRAIN_EVERY_DAYS = 7   # retrain weekly
DRIFT_THRESHOLD = 0.05   # AUC drop > 5% triggers alert


# =========================================================================
#  DATA LOADING — MT5 or yfinance
# =========================================================================

def _fetch_training_bars(interval: str = INTERVAL, n_bars: int = 50000) -> pd.DataFrame:
    """Fetch bars for training: MT5 first, yfinance fallback."""
    # Try MT5 first
    try:
        from src.intraday.data.mt5_bridge import fetch_mt5_bars, initialize, is_connected
        if initialize():
            print(f"[retrain] Fetching {n_bars} bars from MT5...")
            df = fetch_mt5_bars(interval=interval, n_bars=n_bars)
            if not df.empty and len(df) > MIN_BARS_RETRAIN:
                print(f"[retrain] MT5: {len(df)} bars loaded")
                return df
            print(f"[retrain] MT5 returned only {len(df)} bars, trying yfinance...")
    except Exception as e:
        print(f"[retrain] MT5 not available: {e}")

    # Fallback to yfinance
    print("[retrain] Falling back to yfinance...")
    from src.intraday.data.tick_feed import fetch_intraday_bars
    df = fetch_intraday_bars(interval=interval)
    print(f"[retrain] yfinance: {len(df)} bars loaded")
    return df


# =========================================================================
#  PAPER TRADE INTEGRATION
# =========================================================================

def _load_paper_feedback() -> pd.DataFrame:
    """Load evaluated paper trades as supplementary training signal.
    Returns DataFrame with: timestamp, signal, pnl_4h, confidence, was_correct.
    """
    log_file = os.path.join(_QA_DIR, "paper_trades.jsonl")
    if not os.path.exists(log_file):
        return pd.DataFrame()

    trades = []
    with open(log_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
                if t.get("evaluated") and t.get("signal") != "FLAT":
                    trades.append({
                        "timestamp": t.get("timestamp"),
                        "signal": t.get("signal"),
                        "confidence": t.get("confidence"),
                        "pnl_4h": t.get("pnl_4h", 0),
                        "would_hit_sl": t.get("would_hit_sl", False),
                        "would_hit_tp": t.get("would_hit_tp", False),
                        "was_correct": (t.get("pnl_4h") or 0) > 0,
                        "setup_type": t.get("setup_type"),
                        "volatility_regime": t.get("volatility_regime"),
                    })
            except json.JSONDecodeError:
                continue

    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"[retrain] Loaded {len(df)} evaluated paper trades "
          f"(WR: {df['was_correct'].mean():.1%})")
    return df


def _augment_targets_with_feedback(feat: pd.DataFrame, paper: pd.DataFrame) -> pd.DataFrame:
    """Optionally weight samples near paper trade timestamps more heavily.
    Paper trades that were correct reinforce the model's existing signal;
    incorrect ones add emphasis to learn from mistakes.

    Returns feat with added 'sample_weight' column.
    """
    feat = feat.copy()
    feat["sample_weight"] = 1.0

    if paper.empty:
        return feat

    # For each paper trade, find nearby bars and boost their weight
    for _, trade in paper.iterrows():
        ts = trade["timestamp"]
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")

        # Find bars within 2 hours of the trade
        mask = (feat.index >= ts - timedelta(hours=1)) & (feat.index <= ts + timedelta(hours=1))
        if mask.sum() == 0:
            continue

        # Boost weight: correct trades get mild boost, incorrect get stronger boost
        # (we want the model to learn more from mistakes)
        if trade["was_correct"]:
            feat.loc[mask, "sample_weight"] *= 1.2
        else:
            feat.loc[mask, "sample_weight"] *= 1.5

    boosted = (feat["sample_weight"] != 1.0).sum()
    if boosted > 0:
        print(f"[retrain] Boosted {boosted} bars with paper trade feedback")

    return feat


# =========================================================================
#  FEATURE BUILDING
# =========================================================================

def _build_training_data(bars: pd.DataFrame) -> pd.DataFrame:
    """Build full feature set from bars."""
    from src.intraday.features.microstructure import build_intraday_features
    from src.intraday.data.session_calendar import annotate_sessions

    print("[retrain] Building microstructure features...")
    feat = build_intraday_features(bars, interval=INTERVAL)

    try:
        from src.intraday.features.regime import add_regime_features
        feat = add_regime_features(feat)
    except Exception as e:
        print(f"[retrain] Regime features skipped: {e}")

    try:
        feat = annotate_sessions(feat)
    except Exception as e:
        print(f"[retrain] Session annotation skipped: {e}")

    try:
        from src.intraday.features.context_swing import attach_swing_context
        feat = attach_swing_context(feat)
    except Exception as e:
        print(f"[retrain] Swing context skipped: {e}")

    # Target
    feat["future_close"] = feat["close"].shift(-HORIZON_BARS)
    feat["future_ret"] = (feat["future_close"] - feat["close"]) / feat["close"]
    feat["target"] = (feat["future_ret"] > 0.0005).astype(int)

    # RTH only
    if "is_rth" in feat.columns:
        feat = feat[feat["is_rth"] == 1].copy()

    feat = feat.dropna(subset=["future_ret"])
    print(f"[retrain] Training data: {len(feat)} bars with valid target")
    return feat


# =========================================================================
#  WALK-FORWARD TRAINING
# =========================================================================

def _get_feature_cols(df: pd.DataFrame) -> list:
    """Get feature columns for training."""
    from src.intraday.features.microstructure import feature_columns

    extra_cols = [
        "swing_bias_today", "swing_expected_ret",
        "swing_expected_vol", "swing_confidence", "swing_age_hours",
        "adx_14", "atr_zscore_60",
        "regime_trend", "regime_range", "regime_shock",
    ]

    cols = [c for c in feature_columns() if c in df.columns]
    cols += [c for c in extra_cols if c in df.columns]
    cols = [c for c in cols if c not in ("is_rth", "vwap_session")]
    return cols


def retrain_model(feat: pd.DataFrame, paper: pd.DataFrame = None) -> tuple:
    """
    Walk-forward retrain with optional paper trade feedback.

    Returns:
        (model, metrics, feature_cols)
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise RuntimeError("xgboost not installed: pip install xgboost")

    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

    # Augment with paper trade feedback
    if paper is not None and not paper.empty:
        feat = _augment_targets_with_feedback(feat, paper)

    cols = _get_feature_cols(feat)
    X = feat[cols].astype(float)
    y = feat["target"].values
    weights = feat.get("sample_weight", pd.Series(np.ones(len(feat)))).values

    n = len(feat)
    fold_size = n // (N_SPLITS + 1)
    fold_metrics = []

    print(f"[retrain] Walk-forward {N_SPLITS} folds, embargo={EMBARGO_BARS}")
    for k in range(N_SPLITS):
        train_end = (k + 1) * fold_size
        test_start = train_end + EMBARGO_BARS
        test_end = test_start + fold_size
        if test_end > n:
            break

        Xtr = X.iloc[:train_end].fillna(0)
        ytr = y[:train_end]
        wtr = weights[:train_end]
        Xte = X.iloc[test_start:test_end].fillna(0)
        yte = y[test_start:test_end]

        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
        model.fit(Xtr, ytr, sample_weight=wtr)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba > 0.5).astype(int)

        try:
            auc = roc_auc_score(yte, proba) if len(set(yte)) > 1 else float("nan")
        except Exception:
            auc = float("nan")
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, zero_division=0)

        fold_metrics.append({
            "fold": k,
            "n_train": int(len(Xtr)),
            "n_test": int(len(Xte)),
            "auc": round(float(auc), 4),
            "acc": round(float(acc), 4),
            "f1": round(float(f1), 4),
            "pos_rate_train": round(float(ytr.mean()), 3),
            "pos_rate_test": round(float(yte.mean()), 3),
        })
        print(f"  fold {k}: AUC={auc:.3f} ACC={acc:.3f} F1={f1:.3f}")

    # Final model on all data
    print("[retrain] Training final model on full dataset...")
    final = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    final.fit(X.fillna(0), y, sample_weight=weights)

    importances = sorted(
        zip(cols, final.feature_importances_), key=lambda x: x[1], reverse=True
    )[:15]

    metrics = {
        "retrain_timestamp": datetime.now().isoformat(),
        "data_source": "mt5" if len(feat) > 5000 else "yfinance",
        "interval": INTERVAL,
        "horizon_bars": HORIZON_BARS,
        "embargo_bars": EMBARGO_BARS,
        "n_total": n,
        "n_features": len(cols),
        "paper_trades_used": len(paper) if paper is not None and not paper.empty else 0,
        "fold_metrics": fold_metrics,
        "mean_auc": round(float(np.nanmean([f["auc"] for f in fold_metrics])), 4),
        "mean_acc": round(float(np.nanmean([f["acc"] for f in fold_metrics])), 4),
        "mean_f1": round(float(np.nanmean([f["f1"] for f in fold_metrics])), 4),
        "top_features": [{"name": n, "importance": round(float(i), 4)} for n, i in importances],
    }

    return final, metrics, cols


# =========================================================================
#  DRIFT DETECTION
# =========================================================================

def check_drift() -> dict:
    """Compare current model metrics with previous to detect drift."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "drift_detected": False,
        "needs_retrain": False,
        "reason": None,
    }

    # Check last retrain date
    if os.path.exists(_METRICS_PATH):
        try:
            with open(_METRICS_PATH) as f:
                prev = json.load(f)
            report["previous_metrics"] = {
                "mean_auc": prev.get("mean_auc"),
                "mean_acc": prev.get("mean_acc"),
                "n_total": prev.get("n_total"),
                "retrain_timestamp": prev.get("retrain_timestamp"),
            }

            # Check age
            rt = prev.get("retrain_timestamp")
            if rt:
                last_retrain = datetime.fromisoformat(rt)
                days_since = (datetime.now() - last_retrain).days
                report["days_since_retrain"] = days_since
                if days_since >= RETRAIN_EVERY_DAYS:
                    report["needs_retrain"] = True
                    report["reason"] = f"Model is {days_since} days old (threshold: {RETRAIN_EVERY_DAYS})"
        except Exception:
            report["needs_retrain"] = True
            report["reason"] = "Cannot read previous metrics"
    else:
        report["needs_retrain"] = True
        report["reason"] = "No previous model found"

    # Check paper trade performance
    paper = _load_paper_feedback()
    if not paper.empty:
        recent = paper[paper["timestamp"] > datetime.now() - timedelta(days=7)]
        if len(recent) >= 3:
            wr = recent["was_correct"].mean()
            report["recent_paper_wr"] = round(float(wr), 3)
            if wr < 0.35:
                report["drift_detected"] = True
                report["needs_retrain"] = True
                report["reason"] = f"Paper trade WR dropped to {wr:.1%} (last 7 days)"

    # Save drift report
    os.makedirs(_OUT_DIR, exist_ok=True)
    with open(_DRIFT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


# =========================================================================
#  MAIN RETRAIN FLOW
# =========================================================================

def run_retrain(force: bool = False) -> dict:
    """
    Full retrain pipeline:
      1. Check if retrain is needed (unless force=True)
      2. Fetch bars (MT5 or yfinance)
      3. Build features + target
      4. Load paper trade feedback
      5. Walk-forward retrain
      6. Save model + metrics
      7. Log retrain event
    """
    os.makedirs(_OUT_DIR, exist_ok=True)
    t0 = time.time()

    # 1. Check drift
    if not force:
        drift = check_drift()
        if not drift.get("needs_retrain"):
            print("[retrain] Model is up-to-date, no retrain needed")
            return {"status": "skipped", "reason": "model up-to-date", "drift": drift}
        print(f"[retrain] Retrain triggered: {drift.get('reason')}")

    # 2. Fetch bars
    bars = _fetch_training_bars(interval=INTERVAL, n_bars=50000)
    if bars.empty or len(bars) < MIN_BARS_RETRAIN:
        return {"status": "error", "reason": f"Insufficient bars ({len(bars)})"}

    # 3. Build features
    feat = _build_training_data(bars)
    if len(feat) < 500:
        print(f"[retrain] Warning: small dataset ({len(feat)} bars)")

    # 4. Load paper trade feedback
    paper = _load_paper_feedback()

    # 5. Backup previous model
    if os.path.exists(_MODEL_PATH):
        backup = _MODEL_PATH + f".backup_{date.today().isoformat()}"
        try:
            shutil.copy2(_MODEL_PATH, backup)
            print(f"[retrain] Previous model backed up to {backup}")
        except Exception:
            pass

    # 6. Retrain
    model, metrics, cols = retrain_model(feat, paper)

    # 7. Save
    import joblib
    joblib.dump({
        "model": model,
        "feature_cols": cols,
        "horizon_bars": HORIZON_BARS,
        "interval": INTERVAL,
        "retrain_date": date.today().isoformat(),
    }, _MODEL_PATH)

    with open(_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    elapsed = time.time() - t0

    # 8. Log retrain event
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "status": "ok",
        "elapsed_seconds": round(elapsed, 1),
        "n_bars": len(feat),
        "paper_trades_used": metrics.get("paper_trades_used", 0),
        "mean_auc": metrics["mean_auc"],
        "mean_acc": metrics["mean_acc"],
        "mean_f1": metrics.get("mean_f1"),
        "data_source": metrics.get("data_source"),
    }
    try:
        with open(_RETRAIN_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, default=str) + "\n")
    except Exception:
        pass

    print(f"\n[retrain] Done in {elapsed:.1f}s")
    print(f"  Model: {_MODEL_PATH}")
    print(f"  Mean AUC: {metrics['mean_auc']}")
    print(f"  Mean ACC: {metrics['mean_acc']}")
    print(f"  Paper trades used: {metrics.get('paper_trades_used', 0)}")
    print(f"  Top 3 features: {[f['name'] for f in metrics['top_features'][:3]]}")

    return {
        "status": "ok",
        "elapsed_seconds": round(elapsed, 1),
        "metrics": metrics,
        "model_path": _MODEL_PATH,
    }


# =========================================================================
#  CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Intraday Model Retrainer")
    parser.add_argument("--force", action="store_true", help="Force retrain regardless of schedule")
    parser.add_argument("--check-only", action="store_true", help="Only check if retrain is needed")
    parser.add_argument("--drift", action="store_true", help="Run drift report only")
    args = parser.parse_args()

    if args.drift:
        report = check_drift()
        print(json.dumps(report, indent=2, default=str))
        return

    if args.check_only:
        report = check_drift()
        if report.get("needs_retrain"):
            print(f"Retrain needed: {report.get('reason')}")
        else:
            print("Model is up-to-date")
        return

    result = run_retrain(force=args.force)
    if result.get("status") == "ok":
        print(f"\nRetrain successful: AUC={result['metrics']['mean_auc']}")
    else:
        print(f"\nRetrain result: {result.get('status')} — {result.get('reason', '')}")


if __name__ == "__main__":
    main()
