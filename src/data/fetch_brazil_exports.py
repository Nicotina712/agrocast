"""
src/data/fetch_brazil_exports.py
Brazil Export Pace Tracker — soja (HS 1201).

Brasil es el mayor exportador mundial de soja (~60% del comercio global).
El pace de exportaciones semanales vs. año anterior y vs. proyección USDA
es uno de los mejores indicadores de oferta disponible en el mercado.

Fuentes (orden de prioridad):
  1. ComexStat MDIC API  — subdominio api.comexstat no resuelve en algunas redes
  2. World Bank API      — datos anuales de exportación agrícola Brasil
  3. Estimación estacional — porcentaje histórico de la proyección USDA

Cache: data/brazil_exports.json (TTL 24h)
"""

import json
import os
from datetime import datetime, timedelta, date

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_PATH   = os.path.join(_PROJECT_ROOT, "data", "brazil_exports.json")
_TTL_HOURS    = 24

# Proyecciones USDA estáticas MY 2025/26 (fallback si API no disponible)
_USDA_BRAZIL_PROJECTION_MMT = 103.0


def _cache_valid() -> bool:
    if not os.path.exists(_CACHE_PATH):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(_CACHE_PATH))
    return age < timedelta(hours=_TTL_HOURS)


def _fetch_comexstat_monthly(year: int) -> list:
    """
    Exportaciones mensuales de soja de Brasil vía ComexStat/MDIC.
    Intenta con la URL pública (no el subdominio API).
    """
    try:
        import requests
        # El subdominio api.comexstat.mdic.gov.br no resuelve en algunas redes
        # Intentamos el endpoint REST alternativo
        url = "https://api.comexstat.mdic.gov.br/general"
        body = {
            "flow":   "export",
            "year":   year,
            "months": list(range(1, 13)),
            "hs":     ["12010090", "12019000"],
        }
        r = requests.post(url, json=body, timeout=15)
        if r.status_code != 200:
            return []
        rows = r.json().get("data", {}).get("list", [])
        monthly = {}
        for row in rows:
            m  = row.get("month")
            kg = row.get("metricTon") or (row.get("netKg", 0) / 1000)
            if m:
                monthly[m] = monthly.get(m, 0) + (kg or 0)
        return [{"month": m, "exported_tons": round(t, 0)} for m, t in sorted(monthly.items())]
    except Exception:
        return []


def _fetch_worldbank_brazil(year: int) -> float | None:
    """
    World Bank API — exportaciones agrícolas Brasil en millones USD.
    Usamos como proxy de volumen (conversión aproximada por precio medio).
    Indicador: TX.VAL.AGRI.ZS.UN (Agricultural raw materials exports, % merchandise)
    Mejor: NV.AGR.TOTL.ZS o datos de soja específicos vía Comtrade API
    """
    try:
        import requests
        # Usamos el indicator de valor de exportaciones totales de Brasil
        # para el año más reciente disponible
        url = (f"https://api.worldbank.org/v2/country/BR/indicator/"
               f"TX.VAL.AGRI.ZS.UN?format=json&mrv=3&date={year-2}:{year}")
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or len(data) < 2:
            return None
        records = [d for d in data[1] if d.get("value") is not None]
        if not records:
            return None
        # Retorna el % más reciente — lo usamos como señal de tendencia, no volumen
        return float(records[0]["value"])
    except Exception:
        return None


def _seasonal_ytd_estimate(week: int, annual_mmt: float) -> float:
    """
    Estimación de exportaciones YTD basada en estacionalidad histórica de Brasil.
    Brasil concentra ~75% de sus exportaciones en Q1-Q2 (cosecha marzo-julio).
    Curva ajustada por semana del año.
    """
    # Tabla de % acumulado histórico por semana (aproximación)
    seasonal_curve = {
        1: 0.01, 4: 0.03, 8: 0.07, 13: 0.14, 17: 0.25,
        22: 0.40, 26: 0.54, 30: 0.64, 35: 0.72, 39: 0.79,
        44: 0.86, 48: 0.92, 52: 0.97,
    }
    # Interpolación lineal entre puntos de la curva
    weeks = sorted(seasonal_curve.keys())
    pct = 0.0
    for i, w in enumerate(weeks):
        if week <= w:
            if i == 0:
                pct = seasonal_curve[w] * (week / w)
            else:
                w0, w1 = weeks[i-1], w
                p0, p1 = seasonal_curve[w0], seasonal_curve[w]
                pct = p0 + (p1 - p0) * (week - w0) / (w1 - w0)
            break
    else:
        pct = seasonal_curve[52]
    return round(annual_mmt * pct, 2)


def _compute_pace_signal(pct_done: float, week: int) -> str:
    """RAPIDO / NORMAL / LENTO vs. estacionalidad histórica."""
    expected_pct = _seasonal_ytd_estimate(week, 100.0)  # en % del anual
    delta = pct_done - expected_pct
    if delta > 5:
        return "RAPIDO"
    elif delta < -5:
        return "LENTO"
    return "NORMAL"


def get_brazil_export_pace() -> dict:
    """
    Retorna el pace de exportaciones de soja de Brasil.

    Retorna dict con:
      usda_projection_mmt  : proyección USDA para el año agrícola
      exported_ytd_mmt     : exportado hasta la fecha
      pct_completed        : % del objetivo completado
      yoy_pct              : variación vs. año anterior (si disponible)
      weekly_pace_mmt      : pace semanal promedio
      weeks_to_complete    : semanas restantes al pace actual
      signal               : RAPIDO / NORMAL / LENTO
      interpretation       : texto descriptivo
      data_source          : comexstat / seasonal_estimate
      monthly_breakdown    : [{month, exported_tons}]
      as_of                : fecha
    """
    if _cache_valid():
        try:
            with open(_CACHE_PATH) as f:
                data = json.load(f)
            print(f"   [BrazilExp] Cache valido ({data.get('as_of', '?')})")
            return data
        except Exception:
            pass

    print("   [BrazilExp] Descargando pace exportaciones Brasil...")
    today = date.today()
    year  = today.year
    week  = today.isocalendar()[1]

    usda_proj   = _USDA_BRAZIL_PROJECTION_MMT
    data_source = "seasonal_estimate"

    # ── Intento 1: ComexStat ─────────────────────────────────────────────
    monthly      = _fetch_comexstat_monthly(year)
    monthly_prev = _fetch_comexstat_monthly(year - 1)

    total_mmt_ytd  = round(sum(m["exported_tons"] for m in monthly) / 1_000_000, 2)
    total_mmt_prev = round(sum(m["exported_tons"] for m in monthly_prev[:len(monthly)]) / 1_000_000, 2)

    if total_mmt_ytd > 0:
        data_source = "comexstat"
        print(f"   [BrazilExp] ComexStat OK: {total_mmt_ytd} MMT YTD")
    else:
        # ── Fallback: estimación estacional ─────────────────────────────
        total_mmt_ytd = _seasonal_ytd_estimate(week, usda_proj)
        monthly = []
        print(f"   [BrazilExp] ComexStat no disponible — estimacion estacional: {total_mmt_ytd} MMT")

        # Año anterior: misma estimación con proyección histórica (~98 MMT en 2024)
        total_mmt_prev = _seasonal_ytd_estimate(week, 98.0)

    pct_done  = round((total_mmt_ytd / usda_proj) * 100, 1) if usda_proj else None
    yoy_pct   = (round((total_mmt_ytd - total_mmt_prev) / total_mmt_prev * 100, 1)
                 if total_mmt_prev > 0 else None)

    weeks_elapsed = max(week, 1)
    weekly_pace   = round(total_mmt_ytd / weeks_elapsed, 3) if total_mmt_ytd > 0 else None
    remaining_mmt = max(usda_proj - total_mmt_ytd, 0) if usda_proj else None
    weeks_to_complete = (round(remaining_mmt / weekly_pace, 0)
                         if (weekly_pace and remaining_mmt) else None)

    signal = _compute_pace_signal(pct_done or 0, week)

    signal_text = {"RAPIDO": "adelantado — presión bajista sobre precios",
                   "LENTO":  "retrasado — oferta menos disponible, potencialmente alcista",
                   "NORMAL": "en línea con expectativas estacionales"}
    interp = (f"Brasil exportó {total_mmt_ytd:.1f} MMT ({pct_done}% de proyección USDA {usda_proj} MMT). "
              f"Pace {signal_text[signal]}.")
    if yoy_pct is not None:
        direction = "+" if yoy_pct >= 0 else ""
        interp += f" YoY: {direction}{yoy_pct:.1f}%."

    result = {
        "usda_projection_mmt":  usda_proj,
        "exported_ytd_mmt":     total_mmt_ytd,
        "pct_completed":        pct_done,
        "yoy_pct":              yoy_pct,
        "weekly_pace_mmt":      weekly_pace,
        "weeks_to_complete":    weeks_to_complete,
        "week_of_year":         week,
        "signal":               signal,
        "interpretation":       interp,
        "data_source":          data_source,
        "monthly_breakdown":    monthly,
        "as_of":                today.isoformat(),
    }

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"   [BrazilExp] {total_mmt_ytd} MMT ({pct_done}%) | Signal: {signal} | Fuente: {data_source}")
    return result
