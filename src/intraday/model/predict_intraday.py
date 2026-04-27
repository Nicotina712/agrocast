"""
src/intraday/model/predict_intraday.py
Inferencia: aplica el modelo entrenado a barras intradía y genera prob_up.

Genera artifacts/intraday/intraday_signals.csv con:
  datetime, close, prob_up, side_router, swing_bias_today, no_trade
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import joblib
import pandas as pd

from src.intraday.data.tick_feed import fetch_intraday_bars
from src.intraday.data.session_calendar import annotate_sessions
from src.intraday.features.microstructure import build_intraday_features
from src.intraday.features.regime import add_regime_features
from src.intraday.features.context_swing import attach_swing_context
from src.intraday.execution.signal_router import route_signal, RouterConfig


_OUT_DIR     = os.path.join(_ROOT, "artifacts", "intraday")
_MODEL_PATH  = os.path.join(_OUT_DIR, "model_intraday.joblib")
_SIGNALS_OUT = os.path.join(_OUT_DIR, "intraday_signals.csv")


def predict_all(interval: str = "5m", router_cfg: RouterConfig | None = None) -> pd.DataFrame:
    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(f"modelo no encontrado: {_MODEL_PATH}. Corre train_intraday.py primero.")
    bundle = joblib.load(_MODEL_PATH)
    model = bundle["model"]
    cols  = bundle["feature_cols"]

    bars = fetch_intraday_bars(interval=interval)
    feat = build_intraday_features(bars, interval=interval)
    feat = add_regime_features(feat)
    feat = annotate_sessions(feat)
    feat = attach_swing_context(feat)

    X = feat.reindex(columns=cols).astype(float).fillna(0)
    feat["prob_up"] = model.predict_proba(X)[:, 1]

    cfg = router_cfg or RouterConfig()
    sides, reasons = [], []
    for _, row in feat.iterrows():
        sig = route_signal(float(row["prob_up"]), row.to_dict(), cfg)
        sides.append(sig.side)
        reasons.append(sig.reason)
    feat["side_router"] = sides
    feat["router_reason"] = reasons

    keep = ["datetime", "close", "atr_14", "vol_zscore_30",
            "prob_up", "side_router", "router_reason",
            "swing_bias_today", "swing_age_hours", "no_trade", "is_rth", "session"]
    keep = [c for c in keep if c in feat.columns]
    out = feat[keep].copy()
    out.to_csv(_SIGNALS_OUT, index=False)
    print(f"[predict_intraday] {len(out)} barras → {_SIGNALS_OUT}")
    print(f"  side dist: {out['side_router'].value_counts().to_dict()}")
    return out


if __name__ == "__main__":
    predict_all()
