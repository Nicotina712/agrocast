"""
src/trader/term_structure.py
Curva de futuros ZS (Soybeans) — term structure contango/backwardation.

Descarga los contratos activos de ZS desde Yahoo Finance y construye
la curva de precios por vencimiento.

Contratos disponibles en Yahoo Finance:
  ZSF → Enero
  ZSH → Marzo
  ZSK → Mayo
  ZSN → Julio
  ZSQ → Agosto
  ZSU → Septiembre
  ZSX → Noviembre

Se descarga el precio del día (cierre más reciente) para cada contrato.
"""

import os
from datetime import date, datetime

import pandas as pd

_MONTH_CODES = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}
_MONTH_NAMES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
}

# Contratos de soja en CBOT disponibles en Yahoo Finance
_ZS_CODES = ["F", "H", "K", "N", "Q", "U", "X"]


def _next_expiry(month_num: int) -> date:
    """Retorna la fecha de expiración del próximo contrato en ese mes."""
    today = date.today()
    year  = today.year
    # Si ya pasó ese mes este año, ir al año siguiente
    if month_num <= today.month:
        year += 1
    return date(year, month_num, 1)


def fetch_term_structure() -> list:
    """
    Descarga los precios actuales de los contratos ZS activos.

    Retorna lista de dicts con:
      code, month, month_name, expiry, price_usd_bu, price_usd_ton
    """
    try:
        import yfinance as yf
    except ImportError:
        return []

    today = date.today()
    year2 = str(today.year)[-2:]
    next_year2 = str(today.year + 1)[-2:]

    results = []
    seen_months = set()

    for code in _ZS_CODES:
        month_num = _MONTH_CODES[code]

        # Si el contrato del año actual ya expiró, empezar directo con el siguiente año
        start_years = [next_year2] if month_num <= today.month else [year2, next_year2]
        for y2 in start_years:
            ticker_sym = f"ZS{code}{y2}.CBT"
            try:
                t = yf.Ticker(ticker_sym)
                info = t.fast_info
                price = getattr(info, "last_price", None)
                if price is None or price <= 0:
                    hist = t.history(period="5d")
                    if hist.empty:
                        continue
                    price = float(hist["Close"].iloc[-1])

                if price and price > 0 and month_num not in seen_months:
                    expiry = _next_expiry(month_num)
                    results.append({
                        "ticker":       ticker_sym,
                        "code":         code,
                        "month":        month_num,
                        "month_name":   _MONTH_NAMES[month_num],
                        "year":         int("20" + y2),
                        "expiry":       str(expiry),
                        "price_usd_bu": round(float(price) / 100, 4),  # USc → USD
                        "price_usd_ton": round(float(price) / 100 * 36.744, 2),
                        "days_to_expiry": (expiry - today).days,
                    })
                    seen_months.add(month_num)
                    break
            except Exception:
                continue

    if not results:
        return []

    # Ordenar por fecha de expiración
    results.sort(key=lambda x: x["days_to_expiry"])

    # Añadir spread vs. front month
    if results:
        front_price = results[0]["price_usd_bu"]
        for r in results:
            r["spread_vs_front"] = round(r["price_usd_bu"] - front_price, 4)

    # Detectar estructura
    if len(results) >= 2:
        spreads = [r["spread_vs_front"] for r in results[1:]]
        avg_spread = sum(spreads) / len(spreads)
        structure = "CONTANGO" if avg_spread > 0.005 else (
            "BACKWARDATION" if avg_spread < -0.005 else "FLAT"
        )
        interpretation = {
            "CONTANGO":      "El mercado paga prima a futuros lejanos — exceso de oferta inmediata.",
            "BACKWARDATION": "El mercado paga prima al contrato cercano — escasez de oferta inmediata.",
            "FLAT":          "Estructura plana — equilibrio entre oferta y demanda.",
        }
    else:
        structure = "UNKNOWN"
        interpretation = {"UNKNOWN": "Datos insuficientes."}

    return {
        "contracts":      results,
        "structure":      structure,
        "interpretation": interpretation.get(structure, ""),
        "as_of":          str(today),
        "n_contracts":    len(results),
    }
