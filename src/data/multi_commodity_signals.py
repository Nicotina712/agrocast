"""
src/data/multi_commodity_signals.py
Señales técnicas para Soja, Maíz y Trigo (CBOT).

Por cada commodity calcula:
  - Precio actual y variación diaria / semanal
  - MA20, MA50, MA200 y señal de cruce
  - RSI(14)
  - Bandas de Bollinger (20d, ±2σ)
  - Señal compuesta: BUY / SELL / HOLD

Cache: data/multi_commodity.json (TTL 4h)
"""

import json
import os
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_PATH   = os.path.join(_PROJECT_ROOT, "data", "multi_commodity.json")
_TTL_HOURS    = 4

COMMODITIES = {
    "Soybeans": {"label": "Soja",  "unit": "USc/bu", "ticker": "ZS=F", "emoji": "🌱"},
    "Maize":    {"label": "Maíz",  "unit": "USc/bu", "ticker": "ZC=F", "emoji": "🌽"},
    "Wheat":    {"label": "Trigo", "unit": "USc/bu", "ticker": "ZW=F", "emoji": "🌾"},
}


def _cache_valid() -> bool:
    if not os.path.exists(_CACHE_PATH):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(_CACHE_PATH))
    return age < timedelta(hours=_TTL_HOURS)


def _rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-8)
    rsi   = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


def _compute_signals(series: pd.Series) -> dict:
    """Calcula indicadores técnicos sobre una serie de precios."""
    s = series.dropna()
    if len(s) < 21:
        return {}

    price = float(s.iloc[-1])
    prev1 = float(s.iloc[-2]) if len(s) > 1 else price
    prev5 = float(s.iloc[-6]) if len(s) > 5 else price

    ma20  = float(s.rolling(20).mean().iloc[-1])
    ma50  = float(s.rolling(50).mean().iloc[-1]) if len(s) >= 50 else None
    ma200 = float(s.rolling(200).mean().iloc[-1]) if len(s) >= 200 else None

    # Bollinger Bands (20d, ±2σ)
    bb_mid = ma20
    bb_std = float(s.rolling(20).std().iloc[-1])
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pct   = round((price - bb_lower) / (bb_upper - bb_lower + 1e-8) * 100, 1)

    rsi = _rsi(s)

    # Señal compuesta (scoring)
    score = 0
    signals_list = []

    # MA crossover
    if ma50 and price > ma50:
        score += 1
        signals_list.append("precio > MA50")
    elif ma50 and price < ma50:
        score -= 1
        signals_list.append("precio < MA50")

    if ma200 and ma50:
        if ma50 > ma200:
            score += 1
            signals_list.append("MA50 > MA200 (golden cross)")
        else:
            score -= 1
            signals_list.append("MA50 < MA200 (death cross)")

    # RSI
    if rsi < 35:
        score += 2
        signals_list.append(f"RSI sobrevendido ({rsi:.0f})")
    elif rsi > 65:
        score -= 2
        signals_list.append(f"RSI sobrecomprado ({rsi:.0f})")

    # Bollinger
    if bb_pct < 20:
        score += 1
        signals_list.append("precio en banda inferior Bollinger")
    elif bb_pct > 80:
        score -= 1
        signals_list.append("precio en banda superior Bollinger")

    # Tendencia semanal
    weekly_ret = (price - prev5) / (prev5 + 1e-8) * 100
    if weekly_ret > 2:
        score += 1
    elif weekly_ret < -2:
        score -= 1

    if score >= 2:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "price":       round(price, 2),
        "chg_1d_pct":  round((price - prev1) / (prev1 + 1e-8) * 100, 2),
        "chg_5d_pct":  round((price - prev5) / (prev5 + 1e-8) * 100, 2),
        "ma20":        round(ma20, 2),
        "ma50":        round(ma50, 2) if ma50 else None,
        "ma200":       round(ma200, 2) if ma200 else None,
        "rsi":         round(rsi, 1),
        "bb_upper":    round(bb_upper, 2),
        "bb_lower":    round(bb_lower, 2),
        "bb_pct":      bb_pct,
        "signal":      signal,
        "score":       score,
        "signals_list": signals_list,
    }


def _fetch_wheat_live() -> pd.Series | None:
    """Descarga Wheat (ZW=F) en tiempo real si no está en raw_market.csv."""
    try:
        import yfinance as yf
        raw = yf.download("ZW=F", period="2y", interval="1d", progress=False)
        if raw.empty:
            return None
        if hasattr(raw.columns, "get_level_values"):
            raw.columns = raw.columns.get_level_values(0)
        s = raw["Close"].squeeze()
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception as e:
        print(f"   [MultiComm] Wheat yfinance error: {e}")
        return None


def get_multi_commodity_signals() -> dict:
    """
    Retorna señales técnicas para Soja, Maíz y Trigo.

    Retorna dict con:
      commodities: dict por commodity con price, signal, indicators
      as_of: fecha
    """
    if _cache_valid():
        try:
            with open(_CACHE_PATH) as f:
                return json.load(f)
        except Exception:
            pass

    print("   [MultiComm] Calculando señales multi-commodity...")

    raw_path = os.path.join(_PROJECT_ROOT, "data", "raw_market.csv")
    mkt = pd.read_csv(raw_path, parse_dates=["Date"]).sort_values("Date") \
        if os.path.exists(raw_path) else pd.DataFrame()

    result = {}

    for col, meta in COMMODITIES.items():
        sig = {}
        if not mkt.empty and col in mkt.columns:
            sig = _compute_signals(mkt[col])
        elif col == "Wheat":
            # Wheat no está en raw_market.csv aún — descarga en vivo
            wheat_series = _fetch_wheat_live()
            if wheat_series is not None:
                sig = _compute_signals(wheat_series)

        if sig:
            result[col] = {**meta, **sig}
        else:
            result[col] = {**meta, "signal": "N/D", "price": None}

    output = {
        "ok":          True,
        "commodities": result,
        "as_of":       date.today().isoformat(),
    }

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    for col, v in result.items():
        print(f"   [MultiComm] {v['label']}: {v.get('price','?')} {v.get('unit','')} -> {v.get('signal','?')}")

    return output
