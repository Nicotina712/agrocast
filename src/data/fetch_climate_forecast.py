"""
src/data/fetch_climate_forecast.py
Fetcher de PRONÓSTICOS climáticos forward (no realized).

Diferencia vs los climate features actuales:
  - Hoy tenemos clima REALIZADO (NASA POWER + ENSO ONI histórico)
  - Esto es clima FORWARD: outlooks 30/60/90d publicados oficialmente

Fuentes:
  1. NOAA CPC ENSO Forecast (texto/JSON via IRI ENSO Quick Look)
     Frecuencia mensual, latencia ~5d, gratis.

  2. CPC Seasonal Outlooks (Temperature, Precipitation 1-3 month)
     URL pública: https://www.cpc.ncep.noaa.gov/products/predictions/
     Formato: shapefile / texto, latencia 0d.

  3. NMME (North American Multi-Model Ensemble) — temp/precip 1-9 meses
     Disponible via IRI Data Library, formato NetCDF, gratis.

Implementación inicial: SOLO IRI ENSO Quick Look (la fuente más simple y la
que más impacta soja vía cosecha BR/AR). El resto queda como TODO.

Output: data/climate_forecast.csv con columnas:
  Date, enso_forecast_3m, enso_forecast_6m, enso_phase_forecast,
  cpc_temp_outlook_30d, cpc_precip_outlook_30d (cuando se implementen)

Cache: 24h.
"""
from __future__ import annotations
import os, json, time
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_PATH   = os.path.join(_PROJECT_ROOT, "data", "climate_forecast.csv")
_CACHE_META   = os.path.join(_PROJECT_ROOT, "data", "climate_forecast_meta.json")
_TTL_HOURS    = 24


def _cache_valid() -> bool:
    if not os.path.exists(_CACHE_META):
        return False
    try:
        with open(_CACHE_META, "r") as f:
            meta = json.load(f)
        ts = float(meta.get("ts", 0))
        return (time.time() - ts) < _TTL_HOURS * 3600
    except Exception:
        return False


def fetch_cpc_oni(timeout_s: int = 20) -> dict:
    """Descarga la tabla oficial ONI (Oceanic Niño Index) del CPC.

    URL canonical (texto plano, actualizado semanal/mensual):
    https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt

    Format: SEAS YR TOTAL ANOM (3-month seasonal averages).
    Esta es la fuente OFICIAL de NOAA para ONI — la usan en publicaciones.

    Returns dict con: oni_recent (lista últimos 12 valores con fecha),
                      oni_value_now (último), oni_3m_trend.
    """
    import requests
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    try:
        r = requests.get(url, timeout=timeout_s)
        if r.status_code != 200:
            return {"ok": False, "error": f"CPC ONI HTTP {r.status_code}"}
        # Format columnas: SEAS YR TOTAL ANOM
        lines = [ln.strip() for ln in r.text.splitlines() if ln.strip() and not ln.startswith("SEAS")]
        rows = []
        seas_to_month = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
                         "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}
        for ln in lines:
            parts = ln.split()
            if len(parts) != 4:
                continue
            seas, yr, total, anom = parts
            month = seas_to_month.get(seas)
            if month is None:
                continue
            try:
                rows.append({
                    "seas": seas, "year": int(yr), "month": month,
                    "total": float(total), "anom": float(anom),
                })
            except Exception:
                continue
        if not rows:
            return {"ok": False, "error": "no rows parsed"}

        # Ordenar cronológicamente
        rows.sort(key=lambda r: (r["year"], r["month"]))
        recent = rows[-12:]
        now    = rows[-1]
        trend_3m = rows[-1]["anom"] - rows[-4]["anom"] if len(rows) >= 4 else 0.0

        return {
            "ok": True,
            "source": url,
            "oni_value_now": now["anom"],
            "oni_seas_now":  now["seas"],
            "oni_year_now":  now["year"],
            "oni_3m_trend":  round(trend_3m, 3),
            "history_last_12m": recent,
        }
    except Exception as e:
        return {"ok": False, "error": f"fetch_cpc_oni: {e}"}


def fetch_iri_enso_forecast(timeout_s: int = 20) -> dict:
    """Wrapper: intenta CPC ONI oficial primero (texto canonical, fiable),
    luego cae a heurística si falla.

    El CPC ONI es la base autoritativa de NOAA — usamos eso como ground truth
    y proyectamos a 3-6 meses con persistencia + decay (ONI tiene autocorr alta
    a horizontes <6m, ~0.7-0.85 en literatura)."""
    cpc = fetch_cpc_oni(timeout_s=timeout_s)
    if not cpc.get("ok"):
        return {"ok": False, "error": cpc.get("error")}

    # Forecast simple desde ONI oficial:
    # ONI tiene autocorrelación lag-3 ≈ 0.85 en estado neutral, baja a ~0.6 en transición.
    # Usamos persistencia con leve decay hacia 0.
    oni_now = cpc["oni_value_now"]
    trend_3m = cpc["oni_3m_trend"]
    persist_3m = 0.85
    persist_6m = 0.65
    enso_3m = oni_now * persist_3m + trend_3m * 0.3
    enso_6m = oni_now * persist_6m + trend_3m * 0.15

    def _phase(v):
        if v > 0.5:  return "el_nino"
        if v < -0.5: return "la_nina"
        return "neutral"

    return {
        "ok": True,
        "source": "cpc_oni_official",
        "oni_value_now":   round(oni_now, 3),
        "oni_seas_now":    cpc["oni_seas_now"],
        "oni_year_now":    cpc["oni_year_now"],
        "enso_value_3m":   round(enso_3m, 3),
        "enso_value_6m":   round(enso_6m, 3),
        "enso_phase_now":  _phase(oni_now),
        "enso_phase_3m":   _phase(enso_3m),
        "enso_phase_6m":   _phase(enso_6m),
        "trend_3m":        trend_3m,
        "history_last_12m": cpc["history_last_12m"],
    }


def _heuristic_enso_from_history(df_climate: pd.DataFrame | None) -> dict:
    """Fallback: estima fase ENSO forward 3-6 meses a partir de la tendencia
    del ONI más reciente. Heurística simple: si ONI viene subiendo > 0.5 en
    3 meses, asumir fase Niño en 3-6 meses; si bajando < -0.5, Niña; resto neutral.

    Esto NO es un forecast oficial pero da una señal direccional cuando los
    endpoints externos fallan."""
    try:
        if df_climate is None or "enso_oni" not in df_climate.columns:
            return {"ok": False, "error": "no enso_oni in features"}
        oni = df_climate["enso_oni"].dropna().tail(120)
        if len(oni) < 30:
            return {"ok": False, "error": "ONI history too short"}
        recent     = float(oni.iloc[-1])
        trend_3m   = float(oni.iloc[-1] - oni.iloc[-min(63, len(oni))])
        if recent > 0.5 or (recent > 0 and trend_3m > 0.3):
            phase_3m, phase_6m = "el_nino", "el_nino"
        elif recent < -0.5 or (recent < 0 and trend_3m < -0.3):
            phase_3m, phase_6m = "la_nina", "la_nina"
        else:
            phase_3m, phase_6m = "neutral", "neutral"

        # Forecast numérico naive: persistencia + decay
        decay = 0.7
        enso_3m = recent + trend_3m * 0.5 * decay
        enso_6m = recent + trend_3m * 0.3 * decay
        return {
            "ok": True, "source": "heuristic_from_oni",
            "enso_value_now": round(recent, 3),
            "enso_value_3m":  round(enso_3m, 3),
            "enso_value_6m":  round(enso_6m, 3),
            "enso_phase_3m":  phase_3m,
            "enso_phase_6m":  phase_6m,
            "trend_3m":       round(trend_3m, 3),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_climate_forecast(force: bool = False) -> dict:
    """Punto de entrada: devuelve forecast climático (con cache 24h)."""
    if not force and _cache_valid():
        try:
            df = pd.read_csv(_CACHE_PATH, parse_dates=["Date"])
            with open(_CACHE_META, "r") as f:
                meta = json.load(f)
            return {"ok": True, "cached": True, "meta": meta,
                    "latest": df.iloc[-1].to_dict() if not df.empty else {}}
        except Exception:
            pass

    # Intentar fetch oficial; si falla, fallback heurístico
    iri = fetch_iri_enso_forecast()
    if not iri.get("ok"):
        # Cargar history para heurística
        try:
            climate_path = os.path.join(_PROJECT_ROOT, "data", "climate_macro.csv")
            df_climate = pd.read_csv(climate_path) if os.path.exists(climate_path) else None
        except Exception:
            df_climate = None
        result = _heuristic_enso_from_history(df_climate)
        result["fallback"] = True
    else:
        result = iri
        result["fallback"] = False

    # Persistir como CSV (un row por día)
    today = pd.Timestamp.now().normalize()
    row = {"Date": today, **{k: v for k, v in result.items()
                              if k not in ("ok", "raw", "source")}}
    df_new = pd.DataFrame([row])
    if os.path.exists(_CACHE_PATH):
        try:
            df_old = pd.read_csv(_CACHE_PATH, parse_dates=["Date"])
            df_old = df_old[df_old["Date"] != today]   # remplazar si ya hay row de hoy
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            pass
    df_new.to_csv(_CACHE_PATH, index=False)

    with open(_CACHE_META, "w") as f:
        json.dump({"ts": time.time(), "source": result.get("source", "heuristic"),
                   "fallback": result.get("fallback", False)}, f, indent=2)
    return result


def load_climate_forecast_features(features: pd.DataFrame) -> pd.DataFrame:
    """Mergea forecast climático al features DataFrame.
    El forecast actual aplica a TODOS los días (es un estado forward); usamos
    el último valor disponible.
    """
    out = features.copy()
    if not os.path.exists(_CACHE_PATH):
        return out
    try:
        fc = pd.read_csv(_CACHE_PATH, parse_dates=["Date"])
        if fc.empty:
            return out
        latest = fc.iloc[-1]
        # Aplicar valores como columna constante (último forecast vigente)
        for col in ("enso_value_3m", "enso_value_6m", "trend_3m"):
            if col in fc.columns:
                out[f"clim_fwd_{col}"] = float(latest[col]) if pd.notna(latest[col]) else 0.0
        # Phase como categórico → encoded
        if "enso_phase_3m" in fc.columns:
            phase = str(latest["enso_phase_3m"]) if pd.notna(latest["enso_phase_3m"]) else "neutral"
            out["clim_fwd_phase_3m_nino"]    = 1.0 if phase == "el_nino" else 0.0
            out["clim_fwd_phase_3m_nina"]    = 1.0 if phase == "la_nina" else 0.0
            out["clim_fwd_phase_3m_neutral"] = 1.0 if phase == "neutral" else 0.0
    except Exception as e:
        print(f"   [WARN] climate_forecast load: {e}")
    return out
