"""
src/data/event_calendar.py
Calendario maestro de eventos USDA + ventanas de roll de futuros.

Cada evento tiene:
  - date            (YYYY-MM-DD)
  - kind            WASDE | ACREAGE | STOCKS | CROP_PROGRESS | ROLL_WINDOW
  - cost_multiplier multiplicador para el cost model (más alto = peor liquidez/spread)

build_event_features(df) añade columnas al features.csv:
  - days_to_next_event
  - in_event_window (±2 días alrededor de un evento mayor)
  - post_event_drift (1 si estamos 1-3 días después de un evento)
  - event_cost_mult (multiplicador de costo de la fecha, default 1.0)
  - is_roll_window (1 si estamos en últimas 2 semanas del front)
"""

import os
from datetime import date, datetime, timedelta
from typing import List, Dict

import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# WASDE — segundo MARTES de cada mes (publicación 12:00 ET)
#   Nota: hasta ~2013 USDA publicaba el segundo viernes; desde entonces
#   el calendario oficial es el segundo martes (a veces miércoles si feriado).
#   Otros módulos del proyecto (wasde_dates, telegram_bot, session_calendar)
#   ya usan martes — este archivo era el outlier.
# Acreage Report — último día hábil de junio
# Quarterly Stocks — último día hábil de marzo, junio, septiembre, diciembre
# Crop Progress — todos los lunes durante temporada (abril-noviembre)


def _second_tuesday(year: int, month: int) -> date:
    """Segundo martes del mes (fecha estándar de WASDE desde 2013)."""
    d = date(year, month, 1)
    tuesdays = []
    while d.month == month:
        if d.weekday() == 1:  # martes
            tuesdays.append(d)
        d += timedelta(days=1)
    return tuesdays[1] if len(tuesdays) >= 2 else tuesdays[0]


def _last_business_day(year: int, month: int) -> date:
    """Último día hábil del mes (lun-vie)."""
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def build_event_calendar(start_year: int = 2014, end_year: int | None = None) -> List[Dict]:
    """Genera lista completa de eventos."""
    if end_year is None:
        end_year = date.today().year + 1

    events: List[Dict] = []
    for y in range(start_year, end_year + 1):
        # WASDE mensual
        for m in range(1, 13):
            try:
                events.append({
                    "date": _second_tuesday(y, m).isoformat(),
                    "kind": "WASDE",
                    "cost_multiplier": 1.5,
                })
            except Exception:
                pass

        # Acreage Report (junio)
        try:
            events.append({
                "date": _last_business_day(y, 6).isoformat(),
                "kind": "ACREAGE",
                "cost_multiplier": 2.0,
            })
        except Exception:
            pass

        # Quarterly Stocks
        for m in (3, 6, 9, 12):
            try:
                events.append({
                    "date": _last_business_day(y, m).isoformat(),
                    "kind": "STOCKS",
                    "cost_multiplier": 1.8,
                })
            except Exception:
                pass

    return sorted(events, key=lambda e: e["date"])


def build_event_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Añade columnas de eventos al features_df."""
    if "Date" not in features_df.columns:
        return features_df

    df = features_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    cal = build_event_calendar()
    cal_df = pd.DataFrame(cal)
    cal_df["date"] = pd.to_datetime(cal_df["date"])
    # Solo eventos mayores ponderan en cost_multiplier
    major = cal_df[cal_df["kind"].isin(["WASDE", "ACREAGE", "STOCKS"])].copy()
    event_dates = sorted(major["date"].tolist())
    cost_map = dict(zip(major["date"], major["cost_multiplier"]))

    # Listas por tipo de evento (Fix #5: días-a-evento continuo por tipo)
    by_kind = {
        kind: sorted(cal_df[cal_df["kind"] == kind]["date"].tolist())
        for kind in ("WASDE", "ACREAGE", "STOCKS")
    }

    days_to_next  = []
    days_since    = []
    in_window     = []
    post_drift    = []
    event_mult    = []
    is_roll_win   = []
    days_to_wasde = []
    days_since_wasde = []
    days_to_acreage  = []
    days_to_stocks   = []
    event_intensity  = []  # decae exponencial con días al evento más cercano

    import math

    for d in df["Date"]:
        # próximo evento (cualquiera mayor)
        future = [ed for ed in event_dates if ed >= d]
        past   = [ed for ed in event_dates if ed < d]
        nxt    = (future[0] - d).days if future else 999
        prev   = (d - past[-1]).days  if past else 999

        days_to_next.append(nxt)
        days_since.append(prev)
        in_window.append(1 if (nxt <= 2 or prev <= 2) else 0)
        post_drift.append(1 if 1 <= prev <= 3 else 0)

        # Por tipo
        def _next_diff(dates):
            f = [ed for ed in dates if ed >= d]
            return (f[0] - d).days if f else 999

        def _prev_diff(dates):
            p = [ed for ed in dates if ed < d]
            return (d - p[-1]).days if p else 999

        d_to_w  = _next_diff(by_kind["WASDE"])
        d_sn_w  = _prev_diff(by_kind["WASDE"])
        d_to_a  = _next_diff(by_kind["ACREAGE"])
        d_to_s  = _next_diff(by_kind["STOCKS"])

        days_to_wasde.append(d_to_w)
        days_since_wasde.append(d_sn_w)
        days_to_acreage.append(d_to_a)
        days_to_stocks.append(d_to_s)

        # event_intensity: decae con τ=5d desde el evento más cercano (pre o post)
        nearest = min(nxt, prev)
        event_intensity.append(round(math.exp(-nearest / 5.0), 4))

        # cost multiplier: si estamos en ±1 día del evento, aplicar
        mult = 1.0
        for ed, m in cost_map.items():
            if abs((d - ed).days) <= 1:
                mult = max(mult, float(m))
        event_mult.append(mult)

        # ventana de roll: días 15-31 del mes anterior al vencimiento
        # heurística: días 18-30 del mes (la mayoría de fondos rola en el "Goldman roll" days 5-9 del mes)
        is_roll_win.append(1 if 5 <= d.day <= 9 else 0)

    df["days_to_next_event"]  = days_to_next
    df["days_since_event"]    = days_since
    df["in_event_window"]     = in_window
    df["post_event_drift"]    = post_drift
    df["event_cost_mult"]     = event_mult
    df["is_roll_window"]      = is_roll_win
    df["days_to_wasde"]       = days_to_wasde
    df["days_since_wasde"]    = days_since_wasde
    df["days_to_acreage"]     = days_to_acreage
    df["days_to_stocks"]      = days_to_stocks
    df["event_intensity"]     = event_intensity
    return df


if __name__ == "__main__":
    cal = build_event_calendar(2024, 2026)
    print(f"{len(cal)} eventos generados")
    for e in cal[:6]:
        print(e)
