"""
src/data/wasde_api.py
WASDE Surprise Score — multi-source con fallback progresivo.

Fuentes (en orden de prioridad):
  1. USDA FAS PSD API  — bloqueada desde algunas redes (404)
  2. USDA NASS QuickStats API — requiere USDA_NASS_API_KEY en .env (gratis)
  3. Estimaciones estáticas del último WASDE conocido (siempre disponible)

Cache: data/wasde_official.json (TTL 6h)
"""

import json
import os
from datetime import datetime, timedelta, date

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_PATH   = os.path.join(_PROJECT_ROOT, "data", "wasde_official.json")
_HIST_PATH    = os.path.join(_PROJECT_ROOT, "data", "wasde_history.json")
_TTL_HOURS    = 6

_FAS_BASE = "https://apps.fas.usda.gov/psdonline/api/psd/commodity/{commodity}/country/{country}/year/{year}"
_SOY_CODE = "2222000"

_COUNTRIES = {
    "world":     "0",
    "argentina": "2020",
    "brazil":    "2024",
    "usa":       "2840",
    "china":     "5910",
}

_KEY_ATTRS = {
    "Production":            "production_mmt",
    "Exports":               "exports_mmt",
    "Imports":               "imports_mmt",
    "Domestic Consumption":  "consumption_mmt",
    "Ending Stocks":         "ending_stocks_mmt",
    "Beginning Stocks":      "beginning_stocks_mmt",
}

# Últimas estimaciones USDA conocidas (WASDE abril 2026, MY 2025/26)
# Se usan cuando la API no está disponible — actualizar con cada WASDE mensual.
_STATIC_ESTIMATES = {
    "marketing_year": "2025/26",
    "report_date":    "2026-04-10",
    "world": {
        "production_mmt":      422.0,
        "exports_mmt":         175.0,
        "consumption_mmt":     328.0,
        "ending_stocks_mmt":   122.4,
    },
    "brazil": {
        "production_mmt":      169.0,
        "exports_mmt":         103.0,
    },
    "argentina": {
        "production_mmt":       49.0,
        "exports_mmt":           5.0,
    },
    "usa": {
        "production_mmt":       120.7,
        "exports_mmt":           54.4,
    },
    "china": {
        "imports_mmt":          109.0,
    },
}


def _cache_valid() -> bool:
    if not os.path.exists(_CACHE_PATH):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(_CACHE_PATH))
    return age < timedelta(hours=_TTL_HOURS)


def _fetch_fas_psd(country_code: str, year: int) -> dict:
    """Intenta USDA FAS PSD API. Retorna {} si no disponible."""
    try:
        import requests
        url = _FAS_BASE.format(commodity=_SOY_CODE, country=country_code, year=year)
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return {}
        result = {}
        for item in r.json():
            attr = item.get("attributeDescription", "")
            for key, col in _KEY_ATTRS.items():
                if key.lower() in attr.lower():
                    val = item.get("value")
                    try:
                        result[col] = round(float(val), 2) if val is not None else None
                    except (TypeError, ValueError):
                        pass
        return result
    except Exception:
        return {}


def _fetch_nass_quickstats(nass_key: str) -> dict:
    """
    USDA NASS QuickStats API — US soybean production & stocks.
    Key gratuita en: https://quickstats.nass.usda.gov/api
    Solo retorna datos de EE.UU.; datos globales siguen siendo estáticos.
    """
    try:
        import requests
        params = {
            "key":               nass_key,
            "commodity_desc":    "SOYBEANS",
            "statisticcat_desc": "PRODUCTION",
            "unit_desc":         "BU",
            "agg_level_desc":    "NATIONAL",
            "year__GE":          date.today().year - 1,
            "format":            "JSON",
        }
        r = requests.get("https://quickstats.nass.usda.gov/api/api_GET/",
                         params=params, timeout=15)
        if r.status_code != 200:
            return {}
        rows = r.json().get("data", [])
        if not rows:
            return {}
        latest = sorted(rows, key=lambda x: x.get("year", 0), reverse=True)[0]
        bu = float(latest.get("Value", "0").replace(",", ""))
        mmt = round(bu * 0.0000272155, 2)  # bushels → MMT
        return {"usa": {"production_mmt": mmt}}
    except Exception:
        return {}


def _load_history() -> list:
    try:
        with open(_HIST_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def _save_history(history: list, new_snapshot: dict) -> list:
    """
    Agrega `new_snapshot` a la historia SOLO si el report_date es distinto
    al último guardado. Evita duplicar el mismo reporte en cada pipeline run.
    El score de sorpresa se calcula ANTES de agregar el snapshot actual
    para comparar "este WASDE vs el anterior", no vs sí mismo.
    """
    os.makedirs(os.path.dirname(_HIST_PATH), exist_ok=True)
    # Deduplicar por report_date: si el último entry tiene el mismo report_date, no agregar
    new_date = new_snapshot.get("report_date")
    if history:
        last_date = history[-1].get("report_date")
        if last_date and new_date and last_date == new_date:
            return history  # mismo WASDE, no duplicar
    history.append(new_snapshot)
    with open(_HIST_PATH, "w") as f:
        json.dump(history[-36:], f, indent=2)  # conservar últimos 36 reportes (~3 años)
    return history


def _compute_surprise(current_stocks: float, current_report_date: str, history: list) -> dict:
    """
    Calcula el surprise score como desviación del actual vs el reporte ANTERIOR.
    Usa la historia de reportes únicos (deduplicados por report_date).
    - expected = ending_stocks del reporte inmediatamente anterior
    - surprise_pct = (expected - actual) / expected × 100
      positivo = stocks menores de lo esperado → BULLISH (escasez sorpresiva)
      negativo = stocks mayores de lo esperado → BEARISH (abundancia sorpresiva)
    """
    if current_stocks is None:
        return {"score": None, "expected": None, "direction": "NEUTRAL", "n_history": 0}

    # Filtrar entradas válidas excluyendo el reporte actual
    prior = [
        h for h in history
        if h.get("world_ending_stocks") is not None
        and h.get("report_date") != current_report_date
    ]
    if not prior:
        return {"score": None, "expected": None, "direction": "NEUTRAL", "n_history": 0}

    # Expected = reporte inmediatamente anterior (no promedio, para capturar MoM real)
    expected = float(prior[-1]["world_ending_stocks"])
    if expected == 0:
        return {"score": None, "expected": None, "direction": "NEUTRAL", "n_history": len(prior)}

    surprise_pct = (expected - current_stocks) / expected * 100

    # Normalizar por std histórico de cambios MoM para calibrar la magnitud
    stocks_series = [h["world_ending_stocks"] for h in prior if h.get("world_ending_stocks")]
    if len(stocks_series) >= 3:
        deltas = [stocks_series[i] - stocks_series[i-1] for i in range(1, len(stocks_series))]
        delta_std = (sum(d**2 for d in deltas) / len(deltas)) ** 0.5 or 1.0
        surprise_z = round((expected - current_stocks) / delta_std, 2)
    else:
        surprise_z = None

    direction = "BULLISH" if surprise_pct > 1 else "BEARISH" if surprise_pct < -1 else "NEUTRAL"
    return {
        "score":      round(surprise_pct, 2),
        "surprise_z": surprise_z,
        "expected":   round(expected, 2),
        "actual":     round(current_stocks, 2),
        "direction":  direction,
        "n_history":  len(prior),
    }


def get_wasde_official() -> dict:
    """
    Obtiene estimaciones WASDE más recientes con fallback progresivo.

    Retorna dict con: report_year, world, argentina, brazil, usa, china,
    surprise, signal, note, data_source, as_of
    """
    if _cache_valid():
        try:
            with open(_CACHE_PATH) as f:
                return json.load(f)
        except Exception:
            pass

    print("   [WASDE] Obteniendo datos de oferta/demanda global...")
    year = date.today().year
    data_source = "static_estimates"

    # ── Intento 1: USDA FAS PSD API ──────────────────────────────────────
    world = _fetch_fas_psd(_COUNTRIES["world"], year)
    if world.get("ending_stocks_mmt"):
        data_source = "usda_fas_psd"
        arg   = _fetch_fas_psd(_COUNTRIES["argentina"], year)
        bra   = _fetch_fas_psd(_COUNTRIES["brazil"],    year)
        usa   = _fetch_fas_psd(_COUNTRIES["usa"],       year)
        china = _fetch_fas_psd(_COUNTRIES["china"],     year)
        print(f"   [WASDE] FAS PSD API OK — ending stocks mundiales: {world.get('ending_stocks_mmt')} MMT")
    else:
        # ── Intento 2: USDA NASS (solo si hay key) ──────────────────────
        nass_key = os.getenv("USDA_NASS_API_KEY", "")
        nass_data = {}
        if nass_key:
            nass_data = _fetch_nass_quickstats(nass_key)
            if nass_data:
                data_source = "usda_nass"
                print(f"   [WASDE] NASS QuickStats OK")

        # ── Fallback: estimaciones estáticas ────────────────────────────
        print(f"   [WASDE] Usando estimaciones WASDE {_STATIC_ESTIMATES['report_date']} (API no disponible)")
        world = _STATIC_ESTIMATES["world"].copy()
        arg   = _STATIC_ESTIMATES["argentina"].copy()
        bra   = _STATIC_ESTIMATES["brazil"].copy()
        usa   = {**_STATIC_ESTIMATES["usa"], **(nass_data.get("usa", {}))}
        china = _STATIC_ESTIMATES["china"].copy()

    # report_date del WASDE actual (estático o de la API)
    static_report_date = (
        _STATIC_ESTIMATES.get("report_date")
        if data_source in ("static_estimates", "usda_nass")
        else datetime.now().strftime("%Y-%m-%d")
    )
    released_at_iso = None
    if static_report_date:
        released_at_iso = f"{static_report_date}T17:00:00+00:00"

    # ── Surprise score ────────────────────────────────────────────────────
    # IMPORTANTE: cargar historia ANTES de agregar el snapshot actual,
    # para comparar este WASDE contra el reporte anterior (no contra sí mismo).
    # El historial se deduplicó por report_date → cada entrada es un WASDE único.
    history = _load_history()
    world_stocks = world.get("ending_stocks_mmt")
    surprise = _compute_surprise(world_stocks, static_report_date, history)

    snapshot = {
        "timestamp":           datetime.now().isoformat(),
        "report_date":         static_report_date,
        "world_ending_stocks": world_stocks,
        "world_production":    world.get("production_mmt"),
        "world_exports":       world.get("exports_mmt"),
        "brazil_exports":      bra.get("exports_mmt"),
        "argentina_exports":   arg.get("exports_mmt"),
    }
    history = _save_history(history, snapshot)

    score = surprise.get("score")
    n_hist = surprise.get("n_history", 0)
    if score is not None:
        if score > 2:
            note = f"Stocks globales {score:+.1f}% por debajo de expectativas → presión ALCISTA."
        elif score < -2:
            note = f"Stocks globales {score:+.1f}% por encima de expectativas → presión BAJISTA."
        else:
            note = f"Stocks globales en línea con expectativas (sorpresa: {score:+.1f}%) → NEUTRAL."
    else:
        my = _STATIC_ESTIMATES["marketing_year"]
        ws = world_stocks
        n_needed = max(0, 2 - n_hist)
        note = (f"Estimación MY {my}: {ws} MMT ending stocks mundiales. "
                f"Acumulando historial ({n_hist} reportes únicos, necesita {n_needed} más).")

    result = {
        "report_year":        year,
        "marketing_year":     _STATIC_ESTIMATES["marketing_year"],
        "report_date":        static_report_date,
        "released_at":        released_at_iso,
        "world":              world,
        "argentina":          arg,
        "brazil":             bra,
        "usa":                usa,
        "china":              china,
        "surprise":           surprise,
        "surprise_z":         surprise.get("surprise_z"),
        "signal":             surprise.get("direction", "NEUTRAL"),
        "note":               note,
        "data_source":        data_source,
        "history_n":          len(history),
        "history_n_unique":   n_hist,
        "as_of":              datetime.now().isoformat(),
    }

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"   [WASDE] Stocks mundiales: {world_stocks} MMT | "
          f"Signal: {result['signal']} | Fuente: {data_source}")
    return result
