"""
src/data/wasde_stress_test.py
WASDE Stress Test — analiza las reacciones de precio en los 5 reports más volátiles.

Los reportes WASDE se publican el 2do martes de cada mes (aproximadamente).
Este módulo:
  1. Genera la lista histórica de fechas WASDE aproximadas (2016–hoy)
  2. Busca la variación de precio en el día posterior a cada reporte
  3. Identifica los 5 reports con mayor impacto absoluto
  4. Muestra señal del modelo vs. resultado real para cada uno

Cache: data/wasde_stress.json (TTL 24h)
"""

import json
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_PATH   = os.path.join(_PROJECT_ROOT, "data", "wasde_stress.json")
_TTL_HOURS    = 24


def _cache_valid() -> bool:
    if not os.path.exists(_CACHE_PATH):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(_CACHE_PATH))
    return age < timedelta(hours=_TTL_HOURS)


def _get_wasde_dates(start_year: int = 2016, end_year: int | None = None) -> list[date]:
    """
    Genera fechas WASDE aproximadas: 2do martes de cada mes.
    USDA publica entre el 8 y el 14 del mes.
    """
    if end_year is None:
        end_year = date.today().year

    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Encontrar el 2do martes del mes
            first_day = date(year, month, 1)
            # weekday(): Monday=0, Tuesday=1
            days_until_tue = (1 - first_day.weekday()) % 7
            first_tue = first_day + timedelta(days=days_until_tue)
            second_tue = first_tue + timedelta(weeks=1)
            if second_tue.month == month and second_tue <= date.today():
                dates.append(second_tue)
    return dates


def _compute_wasde_impacts(mkt: pd.DataFrame, wasde_dates: list[date]) -> pd.DataFrame:
    """
    Para cada fecha WASDE, calcula la variación de precio al día siguiente
    (o el primer día hábil disponible).
    """
    mkt = mkt.sort_values("Date").reset_index(drop=True)
    price_series = mkt.set_index("Date")["Soybeans"]

    rows = []
    for wasde_date in wasde_dates:
        # Buscar precio en la fecha o el día hábil más cercano
        ts = pd.Timestamp(wasde_date)
        window_pre  = price_series[price_series.index <= ts].tail(1)
        window_post = price_series[price_series.index >  ts].head(3)

        if window_pre.empty or window_post.empty:
            continue

        price_pre  = float(window_pre.iloc[-1])
        price_post = float(window_post.iloc[0])
        date_post  = window_post.index[0].date()
        move_usc   = price_post - price_pre
        move_pct   = move_usc / price_pre * 100

        rows.append({
            "wasde_date":  wasde_date.isoformat(),
            "date_post":   date_post.isoformat(),
            "price_pre":   round(price_pre, 2),
            "price_post":  round(price_post, 2),
            "move_usc":    round(move_usc, 2),
            "move_pct":    round(move_pct, 2),
            "direction":   "SUBIDA" if move_usc > 0 else "CAÍDA",
            "abs_move":    abs(move_usc),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _match_model_signal(wasde_date_str: str, signals_path: str) -> str | None:
    """Busca la señal del modelo más cercana a la fecha WASDE."""
    try:
        sig_df = pd.read_csv(signals_path, parse_dates=["Date"])
        sig_df = sig_df.sort_values("Date")
        ts = pd.Timestamp(wasde_date_str)
        # Señal en los 3 días previos al reporte
        window = sig_df[sig_df["Date"] <= ts].tail(1)
        if not window.empty:
            return str(window["signal"].iloc[-1])
    except Exception:
        pass
    return None


def get_wasde_stress_test(n_top: int = 5) -> dict:
    """
    Retorna análisis de los N reports WASDE más volátiles.

    Retorna dict con:
      top_events     : lista de los N events más volátiles
      all_stats      : estadísticas del universo completo
      as_of          : fecha de cálculo
    """
    if _cache_valid():
        try:
            with open(_CACHE_PATH) as f:
                return json.load(f)
        except Exception:
            pass

    print("   [WASDE Stress] Calculando impactos históricos...")

    raw_path = os.path.join(_PROJECT_ROOT, "data", "raw_market.csv")
    if not os.path.exists(raw_path):
        return {"ok": False, "error": "raw_market.csv no encontrado"}

    mkt = pd.read_csv(raw_path, parse_dates=["Date"])
    wasde_dates = _get_wasde_dates()
    impacts_df  = _compute_wasde_impacts(mkt, wasde_dates)

    if impacts_df.empty:
        return {"ok": False, "error": "Sin datos de impacto WASDE"}

    signals_path = os.path.join(_PROJECT_ROOT, "artifacts", "signals.csv")

    # Top N por magnitud absoluta
    top_df = impacts_df.nlargest(n_top, "abs_move").copy()

    top_events = []
    for _, row in top_df.iterrows():
        signal = _match_model_signal(row["wasde_date"], signals_path)
        # ¿El modelo habría acertado la dirección?
        model_dir = None
        model_correct = None
        if signal == "BUY":
            model_dir = "SUBIDA"
            model_correct = row["direction"] == "SUBIDA"
        elif signal == "SELL":
            model_dir = "CAÍDA"
            model_correct = row["direction"] == "CAÍDA"

        top_events.append({
            "wasde_date":     row["wasde_date"],
            "price_pre":      row["price_pre"],
            "price_post":     row["price_post"],
            "move_usc":       row["move_usc"],
            "move_pct":       row["move_pct"],
            "direction":      row["direction"],
            "model_signal":   signal or "N/D",
            "model_correct":  model_correct,
        })

    # Estadísticas generales
    avg_abs  = float(impacts_df["abs_move"].mean())
    std_abs  = float(impacts_df["abs_move"].std())
    pct_up   = float((impacts_df["move_usc"] > 0).mean() * 100)
    n_events = len(impacts_df)

    # Distribución: ¿cuántos events > 10, 20, 30 USc/bu?
    buckets = {
        "gt_10usc": int((impacts_df["abs_move"] > 10).sum()),
        "gt_20usc": int((impacts_df["abs_move"] > 20).sum()),
        "gt_30usc": int((impacts_df["abs_move"] > 30).sum()),
    }

    result = {
        "ok":          True,
        "top_events":  top_events,
        "all_stats": {
            "n_events":        n_events,
            "avg_move_usc":    round(avg_abs, 2),
            "std_move_usc":    round(std_abs, 2),
            "pct_up":          round(pct_up, 1),
            "pct_down":        round(100 - pct_up, 1),
            "buckets":         buckets,
        },
        "as_of": date.today().isoformat(),
    }

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"   [WASDE Stress] {n_events} events analizados. Top move: {top_events[0]['move_usc']:+.1f} USc/bu")
    return result
