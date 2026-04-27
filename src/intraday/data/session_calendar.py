"""
src/intraday/data/session_calendar.py
Calendario de sesiones CME Globex para soja (ZS/MZS).

Horario oficial CME (todos en CT — Chicago):
  Domingo 19:00 → Viernes 13:20
  Break diario 07:45 – 08:30 CT (mantenimiento)

Sesiones lógicas:
  RTH    08:30 – 13:20 CT  (Regular Trading Hours, mayor liquidez)
  ETH-EU 02:00 – 07:45 CT  (Europa activa)
  ETH-AS 19:00 – 02:00 CT  (Asia, baja liquidez)

WASDE: segundo martes de cada mes, release 12:00 ET = 11:00 CT.
       Regla del piloto: NO operar entre 10:30 y 11:30 CT en día WASDE.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Iterable

import pandas as pd


RTH_OPEN_CT  = time(8, 30)
RTH_CLOSE_CT = time(13, 20)
BREAK_START_CT = time(7, 45)
BREAK_END_CT   = time(8, 30)
GLOBEX_OPEN_CT = time(19, 0)   # domingo
GLOBEX_CLOSE_FRI_CT = time(13, 20)

WASDE_RELEASE_CT = time(11, 0)  # 12:00 ET
WASDE_NO_TRADE_WINDOW = (time(10, 30), time(11, 30))


def is_rth(dt: pd.Timestamp) -> bool:
    """¿La marca temporal cae dentro de RTH (08:30–13:20 CT)?"""
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    t = dt.tz_convert("America/Chicago").time()
    return RTH_OPEN_CT <= t < RTH_CLOSE_CT


def is_globex_open(dt: pd.Timestamp) -> bool:
    """¿Globex abierto? (excluye sábados y break diario)."""
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    ct = dt.tz_convert("America/Chicago")
    dow, t = ct.dayofweek, ct.time()
    # Sábado todo cerrado
    if dow == 5:
        return False
    # Domingo: abre 19:00
    if dow == 6:
        return t >= GLOBEX_OPEN_CT
    # Viernes: cierra 13:20
    if dow == 4:
        if t >= GLOBEX_CLOSE_FRI_CT:
            return False
    # Break diario 07:45–08:30
    if BREAK_START_CT <= t < BREAK_END_CT:
        return False
    return True


def second_tuesday(year: int, month: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != 1:
        d += timedelta(days=1)
    return d + timedelta(weeks=1)


def wasde_dates(start_year: int, end_year: int) -> list[date]:
    out = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            out.append(second_tuesday(y, m))
    return out


def in_wasde_no_trade_window(dt: pd.Timestamp) -> bool:
    """¿Estamos en la ventana 10:30-11:30 CT de un día WASDE?"""
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    ct = dt.tz_convert("America/Chicago")
    if ct.date() not in wasde_dates(ct.year, ct.year):
        return False
    t = ct.time()
    return WASDE_NO_TRADE_WINDOW[0] <= t < WASDE_NO_TRADE_WINDOW[1]


def annotate_sessions(df: pd.DataFrame, dt_col: str = "datetime") -> pd.DataFrame:
    """
    Añade columnas:
      session     : 'RTH' | 'ETH_EU' | 'ETH_AS' | 'CLOSED'
      is_rth      : 1/0
      is_wasde_day: 1/0
      no_trade    : 1/0  (regla pre-WASDE)
    """
    out = df.copy()
    out[dt_col] = pd.to_datetime(out[dt_col], utc=True)
    ct = out[dt_col].dt.tz_convert("America/Chicago")
    t = ct.dt.time
    dow = ct.dt.dayofweek

    in_rth = (t >= RTH_OPEN_CT) & (t < RTH_CLOSE_CT)
    in_eu  = (t >= time(2, 0)) & (t < BREAK_START_CT)
    in_as  = ((t >= GLOBEX_OPEN_CT) | (t < time(2, 0))) & ~in_rth & ~in_eu

    out["is_rth"] = in_rth.astype(int)
    out["session"] = "CLOSED"
    out.loc[in_rth, "session"] = "RTH"
    out.loc[in_eu & ~in_rth, "session"] = "ETH_EU"
    out.loc[in_as & ~in_rth & ~in_eu, "session"] = "ETH_AS"

    yrs = sorted(set(ct.dt.year.unique().tolist()))
    wasde = set(wasde_dates(min(yrs), max(yrs)))
    out["is_wasde_day"] = ct.dt.date.isin(wasde).astype(int)

    in_window = (t >= WASDE_NO_TRADE_WINDOW[0]) & (t < WASDE_NO_TRADE_WINDOW[1])
    out["no_trade"] = ((out["is_wasde_day"] == 1) & in_window).astype(int)

    # Sábado siempre no_trade
    out.loc[dow == 5, "no_trade"] = 1
    return out


if __name__ == "__main__":
    print("WASDE 2026:", wasde_dates(2026, 2026))
