"""
WTI_N6 Crude Oil Robot — Microstructure Features
tick_volume — commodity CFD.
Round numbers: $5 and $10 steps.
Prime session: 08:00-13:30 CT (Mon-Fri).
"""

import os, sys
import numpy as np
import pandas as pd
from datetime import time

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _MVP_ROOT)

from config import CT_OFFSET_HOURS, PRIME_OPEN_CT, PRIME_CLOSE_CT

_PRIME_OPEN_CT  = PRIME_OPEN_CT
_PRIME_CLOSE_CT = PRIME_CLOSE_CT


def _utc_to_ct(ts):
    return ts + pd.Timedelta(hours=CT_OFFSET_HOURS)


def _is_prime(ts):
    ct = _utc_to_ct(ts)
    t  = ct.hour * 60 + ct.minute
    s  = _PRIME_OPEN_CT.hour  * 60 + _PRIME_OPEN_CT.minute
    e  = _PRIME_CLOSE_CT.hour * 60 + _PRIME_CLOSE_CT.minute
    return s <= t <= e


def _session_vwap(df):
    df = df.copy()
    df["_date"] = df.index.normalize()
    vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    df["_tpv"] = typical * df[vol_col]
    def _gvwap(g):
        return g["_tpv"].cumsum() / g[vol_col].cumsum().replace(0, np.nan)
    return df.groupby("_date", group_keys=False).apply(_gvwap)


def build_intraday_features(df_bars: pd.DataFrame, interval: str = "5m") -> pd.DataFrame:
    df    = df_bars.copy()
    feats = pd.DataFrame(index=df.index)
    close = df["close"]; high = df["high"]; low = df["low"]
    vol   = df["tick_volume"] if "tick_volume" in df.columns else df.get("volume", pd.Series(1, index=df.index))

    feats["ret_1"]  = close.pct_change(1)
    feats["ret_3"]  = close.pct_change(3)
    feats["ret_6"]  = close.pct_change(6)
    feats["ret_12"] = close.pct_change(12)

    feats["ema9"]       = close.ewm(span=9,  adjust=False).mean()
    feats["ema21"]      = close.ewm(span=21, adjust=False).mean()
    feats["ema50"]      = close.ewm(span=50, adjust=False).mean()
    feats["ema9_x_21"] = (feats["ema9"] - feats["ema21"]) / close

    d    = close.diff()
    gain = d.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(span=14, adjust=False).mean()
    feats["rsi14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    feats["atr14"]      = tr.ewm(span=14, adjust=False).mean()
    feats["atr14_norm"] = feats["atr14"] / close

    try:
        feats["vwap"]      = _session_vwap(df)
        feats["vwap_dist"] = (close - feats["vwap"]) / feats["vwap"]
    except Exception:
        feats["vwap"]      = close.rolling(20).mean()
        feats["vwap_dist"] = (close - feats["vwap"]) / feats["vwap"]

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feats["bb_upper"] = bb_mid + 2 * bb_std
    feats["bb_lower"] = bb_mid - 2 * bb_std
    feats["bb_pct"]   = (close - feats["bb_lower"]) / (feats["bb_upper"] - feats["bb_lower"] + 1e-8)
    feats["bb_width"] = (feats["bb_upper"] - feats["bb_lower"]) / bb_mid

    feats["vol_ma20"]  = vol.rolling(20).mean()
    feats["vol_ratio"] = vol / feats["vol_ma20"].replace(0, np.nan)
    feats["vol_trend"] = vol.rolling(5).mean() / vol.rolling(20).mean()

    rng = (high - low).replace(0, np.nan)
    feats["body_pct"]   = (close - df["open"]).abs() / rng
    feats["upper_wick"] = (high - df[["open","close"]].max(axis=1)) / rng
    feats["lower_wick"] = (df[["open","close"]].min(axis=1) - low) / rng
    feats["bull_candle"]= (close > df["open"]).astype(float)

    lr = np.log(close / close.shift(1))
    feats["realized_vol_12"] = lr.rolling(12).std() * np.sqrt(12)
    feats["realized_vol_24"] = lr.rolling(24).std() * np.sqrt(24)

    feats["mom_6"]  = close / close.shift(6)  - 1
    feats["mom_12"] = close / close.shift(12) - 1
    feats["mom_24"] = close / close.shift(24) - 1

    up_vol   = vol * feats["bull_candle"]
    down_vol = vol * (1 - feats["bull_candle"])
    feats["cum_delta_12"]   = (up_vol - down_vol).rolling(12).sum()
    feats["cum_delta_norm"] = feats["cum_delta_12"] / (vol.rolling(12).sum().replace(0, np.nan))

    feats["is_prime"]   = pd.Series([_is_prime(ts) for ts in df.index], index=df.index).astype(float)
    feats["ct_hour"]    = pd.Series([_utc_to_ct(ts).hour for ts in df.index], index=df.index).astype(float)
    feats["is_weekend"] = pd.Series([_utc_to_ct(ts).weekday() >= 5 for ts in df.index], index=df.index).astype(float)
    feats["day_frac"]   = pd.Series([(ts.hour*60+ts.minute)/1440.0 for ts in df.index], index=df.index)

    # WTI-specific: distance to $5 and $10 round numbers
    round_5  = (close / 5).round() * 5
    feats["dist_round5"]  = (close - round_5) / 5
    round_10 = (close / 10).round() * 10
    feats["dist_round10"] = (close - round_10) / 10
    feats["dist_round5k"] = feats["dist_round5"]  # compatibility alias

    return feats.ffill().bfill()


def summarize_bars(df_bars: pd.DataFrame, df_feats: pd.DataFrame) -> dict:
    latest = df_feats.iloc[-1]
    bar    = df_bars.iloc[-1]

    lookback = min(24, len(df_bars))
    recent   = df_bars.iloc[-lookback:]
    current_price = float(bar["close"])
    session_high  = float(recent["high"].max())
    session_low   = float(recent["low"].min())

    ts_utc = df_bars.index[-1]
    ts_ct  = _utc_to_ct(ts_utc)
    is_prime = bool(latest.get("is_prime", 0))
    is_wknd  = bool(latest.get("is_weekend", 0))

    mins_to_close = 0
    if is_prime:
        end_t = _PRIME_CLOSE_CT.hour * 60 + _PRIME_CLOSE_CT.minute
        now_t = ts_ct.hour * 60 + ts_ct.minute
        mins_to_close = max(0, end_t - now_t)

    recent_bars = []
    for i in range(min(6, len(df_bars))):
        b = df_bars.iloc[-(i+1)]
        f = df_feats.iloc[-(i+1)]
        vol_val = int(b["tick_volume"] if "tick_volume" in b.index else b.get("volume", 0))
        recent_bars.append({
            "i":      -i,
            "open":   round(float(b["open"]),  3),
            "high":   round(float(b["high"]),  3),
            "low":    round(float(b["low"]),   3),
            "close":  round(float(b["close"]), 3),
            "vol":    vol_val,
            "rsi":    round(float(f.get("rsi14", 50)), 1),
            "vwap_d": round(float(f.get("vwap_dist", 0))*100, 3),
        })

    return {
        "symbol":       "WTI_N6",
        "current_state": {
            "close":         current_price,
            "ct_time":       ts_ct.strftime("%H:%M"),
            "is_prime":      is_prime,
            "is_weekend":    is_wknd,
            "mins_to_close": mins_to_close,
        },
        "session_range": {
            "high":  round(session_high, 3),
            "low":   round(session_low,  3),
            "range": round(session_high - session_low, 3),
        },
        "indicators": {
            "rsi14":          round(float(latest.get("rsi14", 50)), 1),
            "atr14_usd":      round(float(latest.get("atr14", 0)),  3),
            "atr14_pct":      round(float(latest.get("atr14_norm", 0))*100, 3),
            "vwap_dist_pct":  round(float(latest.get("vwap_dist", 0))*100, 3),
            "bb_pct":         round(float(latest.get("bb_pct", 0.5)), 3),
            "bb_width_pct":   round(float(latest.get("bb_width", 0))*100, 3),
            "ema9":           round(float(latest.get("ema9", current_price)), 3),
            "ema21":          round(float(latest.get("ema21", current_price)), 3),
            "ema50":          round(float(latest.get("ema50", current_price)), 3),
            "vol_ratio":      round(float(latest.get("vol_ratio", 1)), 2),
            "realized_vol_12":round(float(latest.get("realized_vol_12", 0))*100, 3),
            "mom_6_pct":      round(float(latest.get("mom_6", 0))*100, 3),
            "mom_24_pct":     round(float(latest.get("mom_24", 0))*100, 3),
            "cum_delta_norm": round(float(latest.get("cum_delta_norm", 0)), 3),
            "dist_round5":    round(float(latest.get("dist_round5", 0))*100, 2),
        },
        "returns": {
            "ret_1":  round(float(latest.get("ret_1",  0))*100, 4),
            "ret_3":  round(float(latest.get("ret_3",  0))*100, 4),
            "ret_6":  round(float(latest.get("ret_6",  0))*100, 4),
            "ret_12": round(float(latest.get("ret_12", 0))*100, 4),
        },
        "recent_bars": recent_bars,
    }
