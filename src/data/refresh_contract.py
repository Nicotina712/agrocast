"""
src/data/refresh_contract.py

Regenera data/current_contract.json en cada corrida del pipeline.

Motivo: el archivo lo generaba SOLO el endpoint /api/current_contract de
news_server.py (fetch live CME vía yfinance, cache 1h). Al migrar de servidor
el endpoint dejó de llamarse y el precio quedó congelado (bug de precio stale:
generated_at viejo con price desfasado ~$70/bu respecto al cierre real).

Consumidores del archivo:
  - src/intel/intelligence_engine.py  (contexto del debate multi-agente)
  - src/intel/market_synthesis.py     (síntesis de mercado)
  - src/producer/producer_brief.py    (ya mitigado: ignora cache >36h)

Estrategia (root-fix):
  1. Intentar fetch live del front-month CBOT vía yfinance.
  2. Si el live falla, caer al ÚLTIMO CIERRE de data/raw_market.csv.
  En ambos casos se escribe generated_at fresco para que los consumidores
  (y el guard de 36h del producer) vean un precio actual.

La lógica de selección del contrato front-month replica la de news_server.py
(get_current_contract_data) — es determinística (calendario CBOT).
"""

from __future__ import annotations

import os
import json
from datetime import date, timedelta

import pandas as pd

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
CONTRACT_PATH = os.path.join(DATA_DIR, "current_contract.json")
RAW_MARKET    = os.path.join(DATA_DIR, "raw_market.csv")

# Mapa de mes → código de contrato CBOT
_MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}
_MONTH_NAMES = {
    "F": "ENE", "G": "FEB", "H": "MAR", "J": "ABR", "K": "MAY", "M": "JUN",
    "N": "JUL", "Q": "AGO", "U": "SEP", "V": "OCT", "X": "NOV", "Z": "DIC",
}
# Contratos de soja CBOT más líquidos: ENE,MAR,MAY,JUL,AGO,SEP,NOV
_ACTIVE_MONTHS = [1, 3, 5, 7, 8, 9, 11]
_ROLL_DAYS_BEFORE_FND = 5  # rotar cuando quedan ≤5 días hábiles para FND


def _next_active(y: int, m: int) -> tuple[int, int]:
    for _ in range(24):
        if m in _ACTIVE_MONTHS:
            return y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return y, m


def _business_days_until(target: date, today: date) -> int:
    delta, count = (target - today).days, 0
    for i in range(delta):
        if (today + timedelta(days=i + 1)).weekday() < 5:
            count += 1
    return count


def _first_notice_day(cy: int, cm: int) -> date:
    """FND CBOT = último día hábil del mes anterior al contrato."""
    import calendar
    if cm == 1:
        fy, fm = cy - 1, 12
    else:
        fy, fm = cy, cm - 1
    d = date(fy, fm, calendar.monthrange(fy, fm)[1])
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _resolve_contracts(today: date) -> dict:
    front_y, front_m = _next_active(today.year, today.month)
    fnd = _first_notice_day(front_y, front_m)
    if _business_days_until(fnd, today) <= _ROLL_DAYS_BEFORE_FND:
        nm, ny = front_m + 1, front_y
        if nm > 12:
            nm, ny = 1, front_y + 1
        front_y, front_m = _next_active(ny, nm)

    nm, ny = front_m + 1, front_y
    if nm > 12:
        nm, ny = 1, front_y + 1
    next_y, next_m = _next_active(ny, nm)

    def ticker(y, m):
        return f"ZS{_MONTH_CODES[m]}{str(y)[-2:]}.CBT"

    def label(y, m):
        return f"{_MONTH_NAMES[_MONTH_CODES[m]]}{y}"

    front_fnd = _first_notice_day(front_y, front_m)
    return {
        "front_ticker":  ticker(front_y, front_m),
        "front_contract": label(front_y, front_m),
        "next_ticker":   ticker(next_y, next_m),
        "next_contract": label(next_y, next_m),
        "days_to_fnd":   _business_days_until(front_fnd, today),
        "fnd_date":      front_fnd.isoformat(),
    }


def _fetch_last_price(ticker: str) -> float | None:
    try:
        import yfinance as _yf
        raw = _yf.download(ticker, period="5d", interval="1d", progress=False)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return float(raw["Close"].dropna().iloc[-1])
    except Exception as e:
        print(f"   [WARN] refresh_contract: no se pudo descargar {ticker}: {e}")
        return None


def _last_raw_close() -> float | None:
    try:
        mkt = pd.read_csv(RAW_MARKET)
        return float(mkt["Soybeans"].iloc[-1])
    except Exception as e:
        print(f"   [WARN] refresh_contract: no se pudo leer raw_market.csv: {e}")
        return None


def refresh_current_contract() -> dict:
    """Regenera data/current_contract.json. Devuelve el dict escrito."""
    today = date.today()
    meta = _resolve_contracts(today)

    front_price = _fetch_last_price(meta["front_ticker"])
    next_price  = _fetch_last_price(meta["next_ticker"])

    source = "live_cme"
    if front_price is None:
        # Fallback: último cierre histórico (consistente con forecast/tendencia)
        front_price = _last_raw_close()
        source = "raw_market_close"

    spread = None
    if front_price is not None and next_price is not None:
        spread = round(next_price - front_price, 2)

    result = {
        "ok":             front_price is not None,
        "ticker":         "ZS=F",
        "front_contract": meta["front_contract"],
        "front_ticker":   meta["front_ticker"],
        "price":          round(front_price, 2) if front_price is not None else None,
        "next_contract":  meta["next_contract"],
        "next_ticker":    meta["next_ticker"],
        "next_price":     round(next_price, 2) if next_price is not None else None,
        "spread_usc":     spread,
        "days_to_fnd":    meta["days_to_fnd"],
        "fnd_date":       meta["fnd_date"],
        "source":         source,
        "generated_at":   pd.Timestamp.now().isoformat(timespec="seconds"),
    }

    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CONTRACT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
    except Exception as e:
        print(f"   [WARN] refresh_contract: no se pudo guardar current_contract.json: {e}")

    return result


if __name__ == "__main__":
    out = refresh_current_contract()
    print(json.dumps(out, ensure_ascii=False, indent=2))
