"""
notebooks/intraday_tweak_comparison.py
Compara variantes del modelo intradía sobre la misma data, en una sola corrida.

Variantes:
  V0  baseline                      (h=12, thr 0.62/0.38, swing on, all RTH)
  V1  microstructure pura  (1.c)    (h=12, thr 0.62/0.38, SIN swing+regime)
  V2  V0 + horizonte 2h    (1.b)    (h=24, thr 0.62/0.38, swing on)
  V3  V2 + thresholds más laxos (1.a) (h=24, thr 0.55/0.45)
  V4  V3 + filtro primeros 90min RTH (1.d)

Salida:
  artifacts/intraday/tweak_comparison.csv  ← tabla comparativa
  stdout                                    ← reporte legible
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime
from copy import deepcopy

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import joblib
import numpy as np
import pandas as pd

from src.intraday.data.tick_feed import fetch_intraday_bars
from src.intraday.data.session_calendar import annotate_sessions
from src.intraday.features.microstructure import build_intraday_features, feature_columns
from src.intraday.features.regime import add_regime_features
from src.intraday.features.context_swing import attach_swing_context
from src.intraday.execution.signal_router import route_signal, RouterConfig
from src.intraday.execution.slippage_model import MZS
from src.intraday.execution.risk_intraday import RiskConfig
from src.intraday.backtest.replay_engine import replay
from src.intraday.backtest.metrics import compute_metrics

INTERVAL = "5m"
INITIAL_CAPITAL = 10_000

# Feature pools
SWING_FEATURES = [
    "swing_bias_today", "swing_expected_ret",
    "swing_expected_vol", "swing_confidence", "swing_age_hours",
]
REGIME_FEATURES = [
    "adx_14", "atr_zscore_60",
    "regime_trend", "regime_range", "regime_shock",
]


def build_full_dataset() -> pd.DataFrame:
    """Carga + features completos UNA sola vez. Las variantes filtran sobre esto."""
    print("[setup] cargando barras 5m + features completos ...")
    bars = fetch_intraday_bars(interval=INTERVAL, use_cache=True)
    feat = build_intraday_features(bars, interval=INTERVAL)
    feat = add_regime_features(feat)
    feat = annotate_sessions(feat)
    feat = attach_swing_context(feat)
    return bars, feat


def prep_target(feat: pd.DataFrame, horizon: int,
                first_90min_only: bool = False) -> pd.DataFrame:
    df = feat.copy()
    df["future_close"] = df["close"].shift(-horizon)
    df["future_ret"]   = (df["future_close"] - df["close"]) / df["close"]
    df["target"]       = (df["future_ret"] > 0.0005).astype(int)
    df = df[df["is_rth"] == 1].copy()
    df = df.dropna(subset=["future_ret"])
    if first_90min_only:
        df = df[(df["minute_of_session"] >= 0) & (df["minute_of_session"] <= 90)]
    return df


def train_walkforward(df: pd.DataFrame, feature_cols: list[str],
                      horizon: int, embargo: int, n_splits: int = 5):
    from xgboost import XGBClassifier
    X = df[feature_cols].astype(float).fillna(0)
    y = df["target"].values
    n = len(df)
    fold_size = n // (n_splits + 1)
    aucs, accs = [], []
    from sklearn.metrics import accuracy_score, roc_auc_score
    for k in range(n_splits):
        train_end = (k + 1) * fold_size
        test_start = train_end + embargo
        test_end = test_start + fold_size
        if test_end > n: break
        Xtr, ytr = X.iloc[:train_end], y[:train_end]
        Xte, yte = X.iloc[test_start:test_end], y[test_start:test_end]
        model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              tree_method="hist", eval_metric="logloss",
                              random_state=42, n_jobs=-1)
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        try:
            aucs.append(roc_auc_score(yte, proba) if len(set(yte)) > 1 else np.nan)
        except Exception:
            aucs.append(np.nan)
        accs.append(accuracy_score(yte, (proba > 0.5).astype(int)))
    final = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          tree_method="hist", eval_metric="logloss",
                          random_state=42, n_jobs=-1)
    final.fit(X, y)
    return final, {
        "mean_auc": round(float(np.nanmean(aucs)), 4),
        "mean_acc": round(float(np.nanmean(accs)), 4),
        "fold_aucs": [round(float(a), 4) for a in aucs],
    }


def run_variant(name: str, feat: pd.DataFrame, bars: pd.DataFrame, *,
                use_swing: bool = True, use_regime: bool = True,
                horizon: int = 12, first_90min_only: bool = False,
                p_buy: float = 0.62, p_sell: float = 0.38) -> dict:
    print(f"\n{'='*72}")
    print(f"  {name}  (h={horizon}, thr={p_buy}/{p_sell}, "
          f"swing={'on' if use_swing else 'OFF'}, "
          f"regime={'on' if use_regime else 'OFF'}, "
          f"first90={first_90min_only})")
    print('='*72)

    df = prep_target(feat, horizon=horizon, first_90min_only=first_90min_only)
    print(f"  dataset: {len(df)} barras RTH con target")

    # Selección de features
    base = [c for c in feature_columns() if c in df.columns]
    base = [c for c in base if c not in ("is_rth", "vwap_session")]
    cols = list(base)
    if use_swing:
        cols += [c for c in SWING_FEATURES if c in df.columns]
    if use_regime:
        cols += [c for c in REGIME_FEATURES if c in df.columns]

    embargo = horizon * 2
    model, train_m = train_walkforward(df, cols, horizon=horizon, embargo=embargo)
    print(f"  AUC walk-fwd mean: {train_m['mean_auc']}  folds={train_m['fold_aucs']}")

    # Predict sobre TODO feat (no solo RTH filtrado, para que el router decida)
    X_all = feat.reindex(columns=cols).astype(float).fillna(0)
    feat_pred = feat.copy()
    feat_pred["prob_up"] = model.predict_proba(X_all)[:, 1]

    # Routing
    cfg = RouterConfig(p_buy_threshold=p_buy, p_sell_threshold=p_sell)
    sides = []
    for _, row in feat_pred.iterrows():
        sides.append(route_signal(float(row["prob_up"]), row.to_dict(), cfg).side)
    feat_pred["side_router"] = sides
    side_dist = pd.Series(sides).value_counts().to_dict()
    print(f"  side dist: {side_dist}")

    # Backtest
    keep = ["datetime", "close", "atr_14", "vol_zscore_30",
            "prob_up", "side_router", "swing_bias_today",
            "swing_expected_vol", "swing_age_hours", "no_trade", "is_rth"]
    keep = [c for c in keep if c in feat_pred.columns]
    signals = feat_pred[keep].copy()
    trades, summary = replay(signals_df=signals, bars_df=bars,
                             initial_capital=INITIAL_CAPITAL,
                             contract=MZS, horizon_bars=horizon,
                             cfg=RiskConfig())
    metrics = compute_metrics(trades, initial_capital=INITIAL_CAPITAL)

    out = {
        "variant": name,
        "horizon": horizon,
        "p_buy": p_buy, "p_sell": p_sell,
        "use_swing": use_swing, "use_regime": use_regime,
        "first_90min_only": first_90min_only,
        "n_features": len(cols),
        "n_train_rows": len(df),
        "auc_mean": train_m["mean_auc"],
        "fold_aucs": train_m["fold_aucs"],
        "n_signals_buy": side_dist.get("BUY", 0),
        "n_signals_sell": side_dist.get("SELL", 0),
        "n_trades": metrics.get("n_trades", 0),
        "win_rate": metrics.get("win_rate", 0),
        "profit_factor": metrics.get("profit_factor"),
        "expectancy": metrics.get("expectancy_per_trade", 0),
        "sharpe": metrics.get("sharpe_annualized", 0),
        "return_pct": metrics.get("return_pct", 0),
        "max_dd_pct": metrics.get("max_drawdown_pct", 0),
        "gate": metrics.get("fase0_gate", "?"),
    }
    print(f"  → trades={out['n_trades']} WR={out['win_rate']:.1%} "
          f"PF={out['profit_factor']} Sharpe={out['sharpe']} "
          f"Return={out['return_pct']}% Gate={out['gate']}")
    return out


def main():
    print("=" * 72)
    print("  AGROCAST INTRADAY — Comparación de tweaks")
    print(f"  Run: {datetime.now().isoformat()}")
    print("=" * 72)

    bars, feat = build_full_dataset()

    variants = []
    # V0 — baseline (todo on, defaults)
    variants.append(run_variant("V0_baseline", feat, bars,
                                use_swing=True, use_regime=True,
                                horizon=12, p_buy=0.62, p_sell=0.38))
    # V1 — micrestructura pura (1.c diagnóstico)
    variants.append(run_variant("V1_no_swing_regime", feat, bars,
                                use_swing=False, use_regime=False,
                                horizon=12, p_buy=0.62, p_sell=0.38))
    # V2 — horizonte 2h (1.b)
    variants.append(run_variant("V2_horizon_24", feat, bars,
                                use_swing=True, use_regime=True,
                                horizon=24, p_buy=0.62, p_sell=0.38))
    # V3 — V2 + thresholds laxos (1.a)
    variants.append(run_variant("V3_h24_thr_055", feat, bars,
                                use_swing=True, use_regime=True,
                                horizon=24, p_buy=0.55, p_sell=0.45))
    # V4 — V3 + first 90min RTH (1.d)
    variants.append(run_variant("V4_h24_thr_055_first90", feat, bars,
                                use_swing=True, use_regime=True,
                                horizon=24, p_buy=0.55, p_sell=0.45,
                                first_90min_only=True))

    # Tabla
    df = pd.DataFrame(variants)
    cols_show = ["variant", "auc_mean", "n_trades", "win_rate",
                 "profit_factor", "sharpe", "return_pct", "max_dd_pct", "gate"]
    print("\n" + "=" * 72)
    print("  TABLA COMPARATIVA")
    print("=" * 72)
    print(df[cols_show].to_string(index=False))

    out_path = os.path.join(_ROOT, "artifacts", "intraday", "tweak_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Tabla guardada en: {out_path}")

    # Veredicto
    best = df.sort_values("sharpe", ascending=False).iloc[0]
    print(f"\n  Mejor variante por Sharpe: {best['variant']}")
    print(f"    AUC={best['auc_mean']}  trades={best['n_trades']}  "
          f"WR={best['win_rate']:.1%}  PF={best['profit_factor']}  "
          f"Sharpe={best['sharpe']}  Return={best['return_pct']}%  "
          f"Gate={best['gate']}")


if __name__ == "__main__":
    main()
