"""
src/data/wasde_dates.py
Genera fechas históricas de reportes USDA WASDE y features de proximidad.

Los reportes WASDE (World Agricultural Supply and Demand Estimates) son
publicados mensualmente por el USDA y son los mayores movedores de precio
en los mercados de granos. Se publican el segundo martes de cada mes.
"""

import pandas as pd
from datetime import date, timedelta


def _second_tuesday(year: int, month: int) -> date:
    """Retorna el segundo martes de un mes dado."""
    d = date(year, month, 1)
    while d.weekday() != 1:  # 1 = Tuesday
        d += timedelta(days=1)
    return d + timedelta(weeks=1)


def get_wasde_dates(start_year: int = 2015) -> list[pd.Timestamp]:
    """Lista de fechas históricas de reportes WASDE hasta hoy."""
    today = date.today()
    dates = []
    for year in range(start_year, today.year + 1):
        for month in range(1, 13):
            d = _second_tuesday(year, month)
            if d <= today:
                dates.append(pd.Timestamp(d))
    return sorted(dates)


def add_wasde_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Añade features de proximidad a reportes WASDE — point-in-time correctas
    (Fix #12): el contenido del reporte sólo es observable POST-release (12:00 ET),
    así que `wasde_days_behind` empieza desde el día siguiente al release; el día
    mismo se trata como aún en ventana pre-release.
      - wasde_days_ahead   : días hasta el próximo reporte (0 = día del reporte aún sin release)
      - wasde_days_behind  : días desde el último release (>=1 si ya está disponible)
      - wasde_window       : 1 si estamos dentro de ±3 días de un reporte
      - wasde_freshness    : 1.0 al día siguiente del release, decae exponencial (τ=15d)
    """
    import math
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    wasde = get_wasde_dates(start_year=int(df[date_col].dt.year.min()))

    def _proximity(dt: pd.Timestamp) -> tuple[int, int]:
        diffs = [(w - dt).days for w in wasde]
        future = [d for d in diffs if d >= 0]
        # PIT: sólo cuentan releases pasados con d <= -1 (al menos 1 día post-release)
        past   = [abs(d) for d in diffs if d <= -1]
        to_next    = min(future) if future else 999
        from_last  = min(past)   if past   else 999
        return to_next, from_last

    results = df[date_col].apply(_proximity)
    df["wasde_days_ahead"]  = results.apply(lambda x: x[0]).astype(int)
    df["wasde_days_behind"] = results.apply(lambda x: x[1]).astype(int)
    df["wasde_window"]      = (
        (df["wasde_days_ahead"] <= 3) | (df["wasde_days_behind"] <= 3)
    ).astype(int)
    df["wasde_freshness"]   = df["wasde_days_behind"].apply(
        lambda d: round(math.exp(-d / 15.0), 4) if d < 999 else 0.0
    )

    return df
