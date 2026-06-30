"""
XAUUSD Gold — Feature Engine
36 technical indicators adapted for Gold's London-NY trading session.

Only difference from soja system: session hours use Gold's RTH window
(07:00–11:30 CT = 08:00–12:30 ET = London-NY overlap).

All indicator logic is identical — indicators are universal.
"""

from __future__ import annotations
from datetime import time as dt_time

import numpy as np
import pandas as pd

from config import RTH_OPEN_CT, RTH_CLOSE_CT, CT_OFFSET_HOURS

_RTH_OPEN_CT  = RTH_OPEN_CT   # 07:00 CT
_RTH_CLOSE_CT = RTH_CLOSE_CT  # 11:30 CT
_CT_OFFSET    = CT_OFFSET_HOURS  # -5 (CDT) or -6 (CST)


# ── helpers ──────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs    = gain / (loss + 1e-12)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l),
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP resets each Gold session (07:00 CT opening)."""
    vwap = np.full(len(df), np.nan)
    price = (df["high"] + df["low"] + df["close"]) / 3.0
    vol   = df["volume"].fillna(0).values

    cum_pv = 0.0
    cum_v  = 0.0
    prev_session = None

    for i, (idx, row) in enumerate(df.iterrows()):
        t = idx.time() if hasattr(idx, "time") else dt_time(0)
        session_date = idx.date() if hasattr(idx, "date") else None

        # Reset at each new session open
        is_open = (t >= _RTH_OPEN_CT) if t is not None else False
        new_session = (session_date != prev_session) and is_open

        if new_session or prev_session is None:
            cum_pv = 0.0
            cum_v  = 0.0
            prev_session = session_date

        cum_pv += price.iloc[i] * vol[i]
        cum_v  += vol[i]

        if cum_v > 0:
            vwap[i] = cum_pv / cum_v

    return pd.Series(vwap, index=df.index)


# ── session timing features ───────────────────────────────────────────────────

def _session_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time-based features for Gold's London-NY session.
    CT timestamps expected (or naive, treated as CT).
    """
    feats = pd.DataFrame(index=df.index)

    open_minutes  = _RTH_OPEN_CT.hour  * 60 + _RTH_OPEN_CT.minute   # 420
    close_minutes = _RTH_CLOSE_CT.hour * 60 + _RTH_CLOSE_CT.minute  # 690

    # MT5 returns bars in UTC. Convert to CT before comparing with RTH window.
    hour_ct   = (df.index.hour + _CT_OFFSET) % 24
    minute_ct = df.index.minute
    bar_minutes = hour_ct * 60 + minute_ct

    # Cyclical encoding of hour (0–23)
    feats["hour_sin"] = np.sin(2 * np.pi * hour_ct / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour_ct / 24)

    # Minutes since Gold RTH open (negative = pre-session)
    feats["minute_of_session"] = bar_minutes - open_minutes
    feats["mins_to_close"]     = np.maximum(0, close_minutes - bar_minutes)

    # Day of week (0=Mon)
    feats["dow"] = df.index.dayofweek

    # Session flags
    in_rth = (bar_minutes >= open_minutes) & (bar_minutes <= close_minutes)
    feats["is_rth"]         = in_rth.astype(int)
    feats["is_first_30min"] = ((bar_minutes >= open_minutes) & (bar_minutes < open_minutes + 30)).astype(int)
    feats["is_last_30min"]  = ((bar_minutes > close_minutes - 30) & (bar_minutes <= close_minutes)).astype(int)

    return feats


# ── main feature builder ──────────────────────────────────────────────────────

def build_intraday_features(df_bars: pd.DataFrame, interval: str = "60m") -> pd.DataFrame:
    """
    Build 36 technical indicators for XAUUSD from OHLCV bars.

    Args:
        df_bars: DataFrame with columns [open, high, low, close, volume],
                 index = DatetimeIndex in CT (or naive treated as CT).
        interval: bar interval string (unused but kept for API compatibility).

    Returns:
        DataFrame with 36 features. NaN rows appear at the beginning
        until indicators have enough history.
    """
    df = df_bars.copy()
    df.columns = [c.lower() for c in df.columns]

    feats = pd.DataFrame(index=df.index)

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"].fillna(0)

    # ── Momentum ──────────────────────────────────────────────────────────────
    feats["ret_1"]  = np.log(close / close.shift(1))
    feats["ret_3"]  = np.log(close / close.shift(3))
    feats["ret_12"] = np.log(close / close.shift(12))
    feats["session_return"] = np.log(close / close.shift(8))  # ~8h lookback for Gold

    feats["ema_fast"]  = _ema(close, 9)
    feats["ema_slow"]  = _ema(close, 26)
    feats["ema_cross"] = (feats["ema_fast"] - feats["ema_slow"]) / close
    feats["rsi_14"]    = _rsi(close, 14)
    feats["momentum_k"] = close - close.shift(10)

    # ── Volatility ────────────────────────────────────────────────────────────
    feats["atr_14"]        = _atr(df, 14)
    feats["atr_mean_30"]   = feats["atr_14"].rolling(30, min_periods=10).mean()
    feats["realized_vol_30"] = feats["ret_1"].rolling(30, min_periods=10).std()
    feats["range_pct"]     = (high - low) / close
    vol_mean = vol.rolling(30, min_periods=10).mean()
    vol_std  = vol.rolling(30, min_periods=10).std().replace(0, np.nan)
    feats["vol_zscore_30"] = (vol - vol_mean) / vol_std

    # ── Price structure / flow proxies ────────────────────────────────────────
    rng = (high - low).replace(0, np.nan)
    feats["body_pct"]    = (df["close"] - df["open"]) / rng
    feats["upper_wick"]  = (high - np.maximum(df["open"], close)) / rng
    feats["lower_wick"]  = (np.minimum(df["open"], close) - low)  / rng

    sign_flow = np.sign(close - df["open"])
    feats["cum_delta_5"]  = (sign_flow * vol).rolling(5,  min_periods=1).sum()
    feats["cum_delta_20"] = (sign_flow * vol).rolling(20, min_periods=1).sum()

    # ── VWAP ──────────────────────────────────────────────────────────────────
    feats["vwap_session"] = _session_vwap(df)
    feats["vwap_dist"]    = (close - feats["vwap_session"]) / feats["vwap_session"]

    # ── Session time ──────────────────────────────────────────────────────────
    time_feats = _session_time_features(df)
    feats = pd.concat([feats, time_feats], axis=1)

    return feats


def summarize_bars(df_bars: pd.DataFrame, df_feats: pd.DataFrame) -> dict:
    """
    Build a concise summary dict for the LLM agents.
    Returns current state + indicator snapshot + recent bars.
    """
    if df_bars.empty or df_feats.empty:
        return {}

    df_bars.columns = [c.lower() for c in df_bars.columns]
    last_idx  = df_feats.index[-1]
    last_bar  = df_bars.iloc[-1]
    last_feat = df_feats.iloc[-1]

    current_state = {
        "close":        round(float(last_bar["close"]), 2),
        "open":         round(float(last_bar["open"]),  2),
        "high":         round(float(last_bar["high"]),  2),
        "low":          round(float(last_bar["low"]),   2),
        "session_date": str(last_idx.date()) if hasattr(last_idx, "date") else str(last_idx),
        "hour_ct":      last_idx.hour if hasattr(last_idx, "hour") else 0,
        "mins_to_close": int(last_feat.get("mins_to_close", 0)),
        "is_rth":       bool(last_feat.get("is_rth", 0)),
        "is_first_30min": bool(last_feat.get("is_first_30min", 0)),
    }

    def _f(key, digits=4):
        v = last_feat.get(key, float("nan"))
        return round(float(v), digits) if not (isinstance(v, float) and np.isnan(v)) else None

    indicators = {
        "rsi_14":         _f("rsi_14", 1),
        "atr_14":         _f("atr_14", 2),
        "atr_mean_30":    _f("atr_mean_30", 2),
        "ema_fast":       _f("ema_fast", 2),
        "ema_slow":       _f("ema_slow", 2),
        "ema_cross":      _f("ema_cross"),
        "realized_vol_30": _f("realized_vol_30"),
        "vol_zscore_30":  _f("vol_zscore_30", 2),
        "vwap_session":   _f("vwap_session", 2),
        "vwap_dist":      _f("vwap_dist"),
        "body_pct":       _f("body_pct", 3),
        "upper_wick":     _f("upper_wick", 3),
        "lower_wick":     _f("lower_wick", 3),
        "cum_delta_5":    _f("cum_delta_5", 0),
        "cum_delta_20":   _f("cum_delta_20", 0),
        "momentum_k":     _f("momentum_k", 2),
    }

    returns = {
        "ret_1":          _f("ret_1"),
        "ret_3":          _f("ret_3"),
        "ret_12":         _f("ret_12"),
        "session_return": _f("session_return"),
    }

    # Last 6 bars summary
    recent = []
    for i in range(max(0, len(df_bars) - 6), len(df_bars)):
        b = df_bars.iloc[i]
        recent.append({
            "time":  str(df_bars.index[i]),
            "open":  round(float(b["open"]),  2),
            "high":  round(float(b["high"]),  2),
            "low":   round(float(b["low"]),   2),
            "close": round(float(b["close"]), 2),
        })

    return {
        "current_state": current_state,
        "indicators":    indicators,
        "returns":       returns,
        "recent_bars":   recent,
    }
