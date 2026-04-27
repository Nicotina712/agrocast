"""
src/intraday/features/microstructure.py
Features intradía sobre OHLCV (sin DOM).

Sin order book disponible en Fase 0, derivamos *proxies* de microestructura
a partir de OHLCV (yfinance):

  Precio / momentum:
    ret_1, ret_3, ret_12       — log-returns (1, 3, 12 barras)
    ema_fast, ema_slow         — EMAs y cruce (signed magnitude)
    rsi_14                     — Relative Strength Index
    momentum_k                 — P_t − P_{t-k}

  Volatilidad:
    atr_14                     — Average True Range (Wilder)
    realized_vol_30            — std de log-returns últimas 30 barras
    range_pct                  — (high-low)/close de la barra (rango relativo)

  Flujo / proxy de presión (sin DOM):
    body_pct                   — (close-open)/(high-low)  ∈ [-1, 1]
    upper_wick                 — (high - max(open,close))/(high-low)
    lower_wick                 — (min(open,close) - low)/(high-low)
    cum_delta_proxy            — Σ sign(close-open)·volume últimas N barras
                                 (proxy de cumulative delta sin tick rule)
    vol_zscore_30              — (vol_t − μ_30)/σ_30  (filtro de actividad)

  VWAP:
    vwap_session               — VWAP acumulado intra-sesión RTH (CT)
    vwap_dist                  — (close − vwap)/vwap

  Estacionalidad:
    hour_sin, hour_cos         — encoding cíclico de hora CT
    minute_of_session          — minutos desde apertura RTH (08:30 CT)
    mins_to_close              — minutos hasta 13:20 CT
    dow                        — día de semana (0=Mon)
    is_first_30min             — flag: primeros 30 min sesión RTH
    is_last_30min              — flag: últimos 30 min sesión RTH

API:
    build_intraday_features(df_bars, interval) -> pd.DataFrame
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd


_RTH_OPEN_CT  = time(8, 30)
_RTH_CLOSE_CT = time(13, 20)


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs    = gain / (loss + 1e-12)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ATR sobre alto/bajo/cierre."""
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l),
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


# ──────────────────────────────────────────────────────────────────────────
# main builder
# ──────────────────────────────────────────────────────────────────────────

def build_intraday_features(df: pd.DataFrame, interval: str = "5m") -> pd.DataFrame:
    """
    Construye features intradía sobre barras OHLCV.

    Args:
      df: salida de tick_feed.fetch_intraday_bars (cols: datetime, open, high, low, close, volume)
      interval: "1m", "5m", "15m", "60m" (afecta ventanas de ATR/EMA si querés escalar)

    Returns:
      DataFrame con todas las columnas originales + features. Index preservado.
      Las primeras N filas tendrán NaN en lookback features (esperado).
    """
    if df.empty:
        return df.copy()

    out = df.copy().reset_index(drop=True)
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
    out = out.sort_values("datetime").reset_index(drop=True)

    # Tiempo en CT (Chicago) para sesiones
    dt_ct = out["datetime"].dt.tz_convert("America/Chicago")
    out["date_ct"]   = dt_ct.dt.date
    out["time_ct"]   = dt_ct.dt.time
    out["hour_ct"]   = dt_ct.dt.hour
    out["minute_ct"] = dt_ct.dt.minute
    out["dow"]       = dt_ct.dt.dayofweek

    # ── Returns ────────────────────────────────────────────────────────
    log_close = np.log(out["close"].replace(0, np.nan))
    out["ret_1"]  = log_close.diff(1)
    out["ret_3"]  = log_close.diff(3)
    out["ret_12"] = log_close.diff(12)

    # ── Momentum / EMA ─────────────────────────────────────────────────
    out["ema_fast"]   = _ema(out["close"], 9)
    out["ema_slow"]   = _ema(out["close"], 26)
    out["ema_cross"]  = (out["ema_fast"] - out["ema_slow"]) / out["close"]
    out["rsi_14"]     = _rsi(out["close"], 14)
    out["momentum_5"] = out["close"] - out["close"].shift(5)

    # ── Volatilidad ────────────────────────────────────────────────────
    out["atr_14"]           = _atr(out, 14)
    out["realized_vol_30"]  = out["ret_1"].rolling(30, min_periods=10).std()
    rng = (out["high"] - out["low"])
    out["range_pct"]        = rng / out["close"]

    # ── Anatomía de vela (proxies de flujo) ────────────────────────────
    rng_safe = rng.replace(0, np.nan)
    out["body_pct"]    = (out["close"] - out["open"]) / rng_safe
    out["upper_wick"]  = (out["high"]  - out[["open","close"]].max(axis=1)) / rng_safe
    out["lower_wick"]  = (out[["open","close"]].min(axis=1) - out["low"])  / rng_safe

    # ── Cumulative delta proxy ─────────────────────────────────────────
    sign = np.sign(out["close"] - out["open"]).fillna(0)
    signed_vol = sign * out["volume"].fillna(0)
    out["cum_delta_5"]  = signed_vol.rolling(5,  min_periods=1).sum()
    out["cum_delta_20"] = signed_vol.rolling(20, min_periods=1).sum()

    # ── Volumen z-score (filtro de actividad) ──────────────────────────
    vol_mean = out["volume"].rolling(30, min_periods=10).mean()
    vol_std  = out["volume"].rolling(30, min_periods=10).std()
    out["vol_zscore_30"] = (out["volume"] - vol_mean) / (vol_std + 1e-9)

    # ── VWAP intra-sesión (RTH 08:30–13:20 CT) ─────────────────────────
    in_rth = (out["time_ct"] >= _RTH_OPEN_CT) & (out["time_ct"] < _RTH_CLOSE_CT)
    out["is_rth"] = in_rth.astype(int)

    typical = (out["high"] + out["low"] + out["close"]) / 3
    pv = typical * out["volume"].fillna(0)
    # Reset acumulado al inicio de cada sesión RTH
    rth_only = out[in_rth].copy()
    rth_only["cum_pv"]  = rth_only.groupby("date_ct")["cum_pv_seed"] if False else \
                         pv[in_rth].groupby(out.loc[in_rth, "date_ct"]).cumsum()
    rth_only["cum_vol"] = out.loc[in_rth, "volume"].fillna(0) \
                            .groupby(out.loc[in_rth, "date_ct"]).cumsum()
    vwap = rth_only["cum_pv"] / (rth_only["cum_vol"] + 1e-9)

    out["vwap_session"] = np.nan
    out.loc[in_rth, "vwap_session"] = vwap.values
    # Evitar div/0 en primera barra de sesión (vwap puede ser 0 si volume=0)
    vwap_safe = out["vwap_session"].replace(0, np.nan)
    out["vwap_dist"]    = (out["close"] - vwap_safe) / vwap_safe
    out["vwap_dist"]    = out["vwap_dist"].replace([np.inf, -np.inf], np.nan)

    # ── Estacionalidad ────────────────────────────────────────────────
    hours_dec = out["hour_ct"] + out["minute_ct"] / 60
    out["hour_sin"] = np.sin(2 * np.pi * hours_dec / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hours_dec / 24)

    open_min  = _RTH_OPEN_CT.hour  * 60 + _RTH_OPEN_CT.minute   # 510
    close_min = _RTH_CLOSE_CT.hour * 60 + _RTH_CLOSE_CT.minute  # 800
    abs_min   = out["hour_ct"] * 60 + out["minute_ct"]
    out["minute_of_session"] = (abs_min - open_min).where(in_rth, np.nan)
    out["mins_to_close"]     = (close_min - abs_min).where(in_rth, np.nan)
    out["is_first_30min"]    = ((in_rth) & (out["minute_of_session"] <  30)).astype(int)
    out["is_last_30min"]     = ((in_rth) & (out["mins_to_close"]    <= 30)).astype(int)

    # Cleanup
    out = out.drop(columns=["time_ct", "minute_ct"])
    return out


def feature_columns() -> list[str]:
    """Lista canónica de columnas-feature (excluye OHLCV y metadata)."""
    return [
        "ret_1", "ret_3", "ret_12",
        "ema_cross", "rsi_14", "momentum_5",
        "atr_14", "realized_vol_30", "range_pct",
        "body_pct", "upper_wick", "lower_wick",
        "cum_delta_5", "cum_delta_20",
        "vol_zscore_30",
        "vwap_session", "vwap_dist",
        "hour_sin", "hour_cos",
        "minute_of_session", "mins_to_close",
        "is_first_30min", "is_last_30min",
        "is_rth", "dow",
    ]


def quick_summary(feat: pd.DataFrame) -> pd.DataFrame:
    """Resumen estadístico de los features (para revisar sanidad)."""
    cols = [c for c in feature_columns() if c in feat.columns]
    summ = feat[cols].describe(percentiles=[.01, .5, .99]).T
    summ["nan_pct"] = feat[cols].isna().mean().mul(100).round(2).values
    return summ[["count", "mean", "std", "1%", "50%", "99%", "nan_pct"]]


if __name__ == "__main__":
    from src.intraday.data.tick_feed import fetch_intraday_bars
    df = fetch_intraday_bars("5m")
    feat = build_intraday_features(df, "5m")
    print(f"\nfeatures shape: {feat.shape}")
    print(quick_summary(feat).to_string())
