"""
src/intraday/features/regime.py
Detector simple de régimen de mercado intradía.

3 regímenes:
  TREND : ADX > 25 + atr_zscore > 0
  RANGE : ADX < 18
  SHOCK : atr_zscore > 2 (vol muy elevada)

El régimen se incluye como feature one-hot. Útil para que el modelo aprenda
que las mismas señales tienen distinto edge según el régimen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    up   = h.diff()
    down = -l.diff()
    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(alpha=1/n, adjust=False, min_periods=n).mean() / (atr + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False, min_periods=n).mean() / (atr + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade:
      adx_14
      atr_zscore_60
      regime_trend / regime_range / regime_shock  (one-hot)
    Requiere columnas: open, high, low, close, atr_14
    """
    out = df.copy()
    out["adx_14"] = _adx(out, 14)

    if "atr_14" in out.columns:
        atr_mean = out["atr_14"].rolling(60, min_periods=20).mean()
        atr_std  = out["atr_14"].rolling(60, min_periods=20).std()
        out["atr_zscore_60"] = (out["atr_14"] - atr_mean) / (atr_std + 1e-9)
    else:
        out["atr_zscore_60"] = 0.0

    is_shock = out["atr_zscore_60"] > 2.0
    is_trend = (out["adx_14"] > 25) & (out["atr_zscore_60"] > 0) & ~is_shock
    is_range = (out["adx_14"] < 18) & ~is_shock & ~is_trend

    out["regime_trend"] = is_trend.fillna(False).astype(int)
    out["regime_range"] = is_range.fillna(False).astype(int)
    out["regime_shock"] = is_shock.fillna(False).astype(int)
    return out
