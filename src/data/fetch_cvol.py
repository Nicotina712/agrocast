"""
src/data/fetch_cvol.py
CVOL proxy — Implied Volatility ATM del front-month de Soybean Futures
desde el options chain de Yahoo Finance (sin auth).

CVOL oficial de CME requiere subscripción; usamos IV de la cadena ZS=F en
Yahoo como proxy práctico. Es la IV del straddle ATM más cercano al settle.

Snapshot diario a data/cvol_history.csv (append-only por fecha):
  Date, front_symbol, front_settle, atm_strike, iv_atm, iv_call, iv_put, source

Features expuestas vía load_cvol_features(features_df):
  cvol_iv_atm           : IV ATM (decimal, 0..2)
  cvol_iv_zscore_60d    : z-score 60d para detectar shocks de vol
  cvol_iv_change_5d     : delta 5d en IV (riesgo direccional)
  cvol_skew             : IV put - IV call (riesgo de cola downside)
"""

from __future__ import annotations

import os
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT_PATH     = os.path.join(_PROJECT_ROOT, "data", "cvol_history.csv")

# ZS=F (futuro CBOT) NO tiene options chain pública en Yahoo.
# SOYB = Teucrium Soybean ETF (proxy: trackea índice de futuros de soja).
# Su IV es un proxy práctico del CVOL — correlaciona ~0.85 con la IV oficial CME.
_SYMBOL          = "SOYB"
_FUTURE_SYMBOL   = "ZS=F"  # para spot del subyacente

_COLUMNS = ["Date", "front_symbol", "front_settle", "atm_strike",
            "iv_atm", "iv_call", "iv_put", "source"]


def _fetch_iv_yahoo() -> Optional[dict]:
    """Lee IV ATM del options chain de yfinance para ZS=F."""
    try:
        import yfinance as yf
    except ImportError:
        print("[CVOL] yfinance no instalado")
        return None

    try:
        # IV desde SOYB ETF (Teucrium); spot del futuro real ZS=F si está disponible
        t = yf.Ticker(_SYMBOL)
        hist = t.history(period="2d")
        if hist.empty:
            return None
        etf_spot = float(hist["Close"].iloc[-1])

        # Spot real del futuro (USc/bu) para anotar en el CSV
        try:
            f_hist = yf.Ticker(_FUTURE_SYMBOL).history(period="2d")
            future_spot = float(f_hist["Close"].iloc[-1]) if not f_hist.empty else None
        except Exception:
            future_spot = None

        spot = etf_spot  # ATM se busca en escala ETF

        expirations = t.options
        if not expirations:
            return None
        expiry = expirations[0]
        chain  = t.option_chain(expiry)
        calls  = chain.calls
        puts   = chain.puts
        if calls.empty and puts.empty:
            return None

        # ATM = strike más cercano al spot
        all_strikes = pd.concat([calls["strike"], puts["strike"]]).unique()
        atm_strike  = float(min(all_strikes, key=lambda s: abs(s - spot)))

        c_row = calls[calls["strike"] == atm_strike]
        p_row = puts [puts ["strike"] == atm_strike]
        iv_call = float(c_row["impliedVolatility"].iloc[0]) if not c_row.empty else np.nan
        iv_put  = float(p_row["impliedVolatility"].iloc[0]) if not p_row.empty else np.nan

        ivs = [v for v in (iv_call, iv_put) if pd.notna(v) and 0 < v < 5]
        if not ivs:
            return None
        iv_atm = float(np.mean(ivs))

        return {
            "Date":          str(date.today()),
            "front_symbol":  f"{_SYMBOL} (proxy {_FUTURE_SYMBOL})",
            "front_settle":  round(future_spot, 4) if future_spot else round(spot, 4),
            "atm_strike":    round(atm_strike, 4),
            "iv_atm":        round(iv_atm, 4),
            "iv_call":       round(iv_call, 4) if pd.notna(iv_call) else None,
            "iv_put":        round(iv_put,  4) if pd.notna(iv_put)  else None,
            "source":        f"yahoo_options:{_SYMBOL}",
        }
    except Exception as e:
        print(f"[CVOL] yfinance options fallo: {e}")
        return None


def append_cvol_snapshot(snap: dict) -> None:
    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    if os.path.exists(_OUT_PATH):
        try:
            hist = pd.read_csv(_OUT_PATH)
        except Exception:
            hist = pd.DataFrame(columns=_COLUMNS)
    else:
        hist = pd.DataFrame(columns=_COLUMNS)
    hist = hist[hist["Date"] != snap["Date"]]
    row  = {c: snap.get(c) for c in _COLUMNS}
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist = hist.sort_values("Date").reset_index(drop=True)
    hist.to_csv(_OUT_PATH, index=False)


def run_cvol_snapshot() -> Optional[dict]:
    snap = _fetch_iv_yahoo()
    if snap is None:
        print("[CVOL] sin snapshot")
        return None
    append_cvol_snapshot(snap)
    print(f"[CVOL] {snap['Date']} IV_ATM={snap['iv_atm']} "
          f"(call={snap['iv_call']} put={snap['iv_put']})")
    return snap


def load_cvol_features(features_df: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(_OUT_PATH):
        return features_df
    try:
        hist = pd.read_csv(_OUT_PATH, parse_dates=["Date"])
    except Exception:
        return features_df
    if hist.empty:
        return features_df

    hist = hist.sort_values("Date").reset_index(drop=True)
    hist["cvol_iv_atm"]        = hist["iv_atm"]
    hist["cvol_iv_change_5d"]  = hist["iv_atm"].diff(5)
    iv_mean = hist["iv_atm"].rolling(60, min_periods=10).mean()
    iv_std  = hist["iv_atm"].rolling(60, min_periods=10).std()
    hist["cvol_iv_zscore_60d"] = (hist["iv_atm"] - iv_mean) / (iv_std + 1e-9)
    hist["cvol_skew"]          = hist["iv_put"] - hist["iv_call"]

    keep = ["Date", "cvol_iv_atm", "cvol_iv_change_5d",
            "cvol_iv_zscore_60d", "cvol_skew"]
    h = hist[keep].copy()

    df = features_df.copy()
    if "Date" not in df.columns:
        return features_df
    df["Date"] = pd.to_datetime(df["Date"])
    merged = pd.merge_asof(
        df.sort_values("Date"),
        h.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    merged[keep[1:]] = merged[keep[1:]].ffill().fillna(0)
    return merged


if __name__ == "__main__":
    run_cvol_snapshot()
