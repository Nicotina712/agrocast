"""
src/intraday/features/context_swing.py
Bridge swing → intradía. Lee artifacts/signals.csv (read-only) y agrega
features de contexto macro a las barras intradía.

Contrato:
  attach_swing_context(intraday_df) -> DataFrame con columnas extra:
    swing_bias_today    : -1 / 0 / +1
    swing_expected_ret  : float
    swing_expected_vol  : float
    swing_confidence    : float [0,1]
    swing_age_hours     : float

Si signals.csv no existe, todas las columnas se llenan con valores neutros
(0, 0.0) y se loggea un warning — el modelo intradía sigue funcionando
sin el prior macro.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_SIGNALS_PATH = os.path.join(_PROJECT_ROOT, "artifacts", "signals.csv")

_SIGNAL_MAP = {"BUY": 1, "HOLD": 0, "SELL": -1}


def _load_swing_signals(path: Optional[str] = None) -> pd.DataFrame:
    path = path or _SIGNALS_PATH
    if not os.path.exists(path):
        print(f"[context_swing] signals.csv no encontrado en {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
    except Exception as e:
        print(f"[context_swing] error leyendo signals.csv: {e}")
        return pd.DataFrame()
    return df.sort_values("Date").reset_index(drop=True)


def attach_swing_context(
    intraday_df: pd.DataFrame,
    signals_path: Optional[str] = None,
    dt_col: str = "datetime",
) -> pd.DataFrame:
    """
    Une el último signal swing disponible (D-1 al menos) a cada barra intradía
    vía merge_asof backward → garantía point-in-time.

    Si signal_age > 5 días, se considera stale y se neutraliza.
    """
    out = intraday_df.copy()
    out[dt_col] = pd.to_datetime(out[dt_col], utc=True)

    swing = _load_swing_signals(signals_path)
    if swing.empty:
        # Modo neutro: features a 0 para que el modelo siga corriendo
        out["swing_bias_today"]   = 0
        out["swing_expected_ret"] = 0.0
        out["swing_expected_vol"] = 0.0
        out["swing_confidence"]   = 0.0
        out["swing_age_hours"]    = 0.0
        return out

    # signals.csv tiene Date diaria → asumir release 16:00 CT (cierre RTH ~+3h)
    # para evitar leakage usamos shift de 1 día
    swing = swing.copy()
    if swing["Date"].dt.tz is None:
        swing["Date"] = swing["Date"].dt.tz_localize("UTC")
    swing["available_at"] = swing["Date"] + pd.Timedelta(hours=21)  # ~16:00 CT next-day safe

    cols_needed = ["available_at"]
    if "signal" in swing.columns:
        swing["swing_bias_today"] = swing["signal"].map(_SIGNAL_MAP).fillna(0).astype(int)
        cols_needed.append("swing_bias_today")
    if "expected_return" in swing.columns:
        swing["swing_expected_ret"] = swing["expected_return"].astype(float)
        cols_needed.append("swing_expected_ret")
    if "expected_vol" in swing.columns:
        swing["swing_expected_vol"] = swing["expected_vol"].astype(float)
        cols_needed.append("swing_expected_vol")
    if "confidence" in swing.columns:
        swing["swing_confidence"] = swing["confidence"].astype(float).clip(0, 1)
        cols_needed.append("swing_confidence")

    swing_lite = swing[cols_needed].sort_values("available_at").reset_index(drop=True)

    merged = pd.merge_asof(
        out.sort_values(dt_col),
        swing_lite,
        left_on=dt_col, right_on="available_at",
        direction="backward",
    )

    # Edad del signal en horas
    merged["swing_age_hours"] = (
        (merged[dt_col] - merged["available_at"]).dt.total_seconds() / 3600
    ).fillna(999)

    # Stale si > 120h (5 días)
    stale = merged["swing_age_hours"] > 120
    for col in ("swing_bias_today", "swing_expected_ret",
                "swing_expected_vol", "swing_confidence"):
        if col in merged.columns:
            merged.loc[stale, col] = 0

    # Defaults para nulos
    for col, default in (("swing_bias_today", 0),
                         ("swing_expected_ret", 0.0),
                         ("swing_expected_vol", 0.0),
                         ("swing_confidence", 0.0)):
        if col not in merged.columns:
            merged[col] = default
        merged[col] = merged[col].fillna(default)

    merged = merged.drop(columns=["available_at"], errors="ignore")
    return merged


if __name__ == "__main__":
    from src.intraday.data.tick_feed import fetch_intraday_bars
    df = fetch_intraday_bars("5m")
    enr = attach_swing_context(df)
    print(enr[["datetime", "close", "swing_bias_today",
               "swing_expected_vol", "swing_age_hours"]].tail(10))
