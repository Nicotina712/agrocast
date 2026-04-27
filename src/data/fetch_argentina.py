"""
src/data/fetch_argentina.py
Señal de oferta argentina — Dashboard ampliado.

Componentes:
  1. Blue dollar / cepo cambiario (Bluelytics API)
  2. Retenciones (derechos de exportación) — alícuota actual
  3. CIARA-CEC pace de liquidaciones semanales (divisas del complejo oleaginoso)
  4. Composición del signal: Argentina Supply Pressure Score (0-100)

Impacto en precio de soja:
  - Cepo activo + retenciones altas → retención de grano → ALCISTA
  - Sin cepo + retenciones bajas → liquidación acelerada → BAJISTA
  - CIARA-CEC liquidaciones rápidas → oferta disponible → BAJISTA

Cache: data/argentina_supply.json (TTL 12h)
"""

import json
import os
import time
from datetime import date, datetime

import requests

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_PATH   = os.path.join(_PROJECT_ROOT, "data", "argentina_supply.json")
_CACHE_TTL    = 43200  # 12 horas

# Retenciones históricas de referencia (alícuota soja grano)
# Actualizadas manualmente cuando cambia la política
_RETENCIONES_REFERENCIA = {
    "soja_grano":  33.0,   # % — alícuota base 2024-2025
    "soja_harina": 31.0,
    "soja_aceite": 31.0,
}


def _load_cache() -> dict:
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if time.time() - data.get("_ts", 0) < _CACHE_TTL:
                return data
        except Exception:
            pass
    return {}


def _save_cache(data: dict) -> None:
    data["_ts"] = time.time()
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _fetch_blue_dollar() -> dict:
    """
    Obtiene tipo de cambio oficial/blue desde Bluelytics API.
    Retorna dict con cepo_activo, spread_pct, oficial, blue.
    """
    try:
        r = requests.get(
            "https://api.bluelytics.com.ar/v2/latest",
            timeout=5,
            headers={"User-Agent": "AgroCastPRO/1.0"},
        )
        if r.status_code == 200:
            data = r.json()
            oficial = data.get("oficial", {}).get("value_sell", 0)
            blue    = data.get("blue",    {}).get("value_sell", 0)
            # También intentar dólar MEP/CCL como señal adicional
            mep     = data.get("blue", {}).get("value_buy", blue)  # aproximación

            if oficial > 0 and blue > 0:
                spread_pct = (blue - oficial) / oficial * 100
                return {
                    "oficial":    round(oficial, 1),
                    "blue":       round(blue, 1),
                    "mep":        round(mep, 1),
                    "spread_pct": round(spread_pct, 1),
                    "cepo_activo": spread_pct > 15,
                    "source":     "bluelytics",
                }
    except Exception as e:
        print(f"   [AR] Bluelytics error: {e}")

    # Fallback: dolarito.ar API
    try:
        r = requests.get("https://dolarapi.com/v1/dolares/blue", timeout=5)
        if r.status_code == 200:
            d = r.json()
            blue = float(d.get("venta", 0))
            r2 = requests.get("https://dolarapi.com/v1/dolares/oficial", timeout=5)
            if r2.status_code == 200:
                oficial = float(r2.json().get("venta", 0))
                if oficial > 0 and blue > 0:
                    spread = (blue - oficial) / oficial * 100
                    return {
                        "oficial": round(oficial, 1), "blue": round(blue, 1),
                        "mep": round(blue, 1), "spread_pct": round(spread, 1),
                        "cepo_activo": spread > 15, "source": "dolarapi",
                    }
    except Exception:
        pass

    return {"oficial": None, "blue": None, "mep": None, "spread_pct": None,
            "cepo_activo": True, "source": "fallback"}


def _fetch_retenciones() -> dict:
    """
    Intenta obtener la alícuota actual de retenciones (derechos de exportación).
    Fuente: Infocampo / AFIP (scraping simple). Fallback: tabla hardcodeada.
    """
    try:
        r = requests.get(
            "https://www.afip.gob.ar/df/derechos-exportacion.asp",
            timeout=8, headers={"User-Agent": "Mozilla/5.0"},
        )
        if r.status_code == 200:
            # Buscar "soja" y porcentaje en texto
            text = r.text.lower()
            idx = text.find("soja")
            if idx > 0:
                snippet = text[idx:idx+100]
                import re
                m = re.search(r"(\d{1,2}(?:[,\.]\d+)?)\s*%", snippet)
                if m:
                    pct = float(m.group(1).replace(",", "."))
                    if 0 < pct < 50:
                        return {"soja_grano": pct, "source": "AFIP", "nota": "Scraping AFIP"}
    except Exception:
        pass

    # Fallback: valores de referencia actualizados manualmente
    return dict(**_RETENCIONES_REFERENCIA, source="reference_table",
                nota="Alícuota de referencia 2024-2025 — verificar cambios regulatorios")


def _fetch_ciara_pace() -> dict:
    """
    Estima el pace de liquidaciones del complejo oleaginoso argentino (CIARA-CEC).
    CIARA publica semanalmente las divisas liquidadas (USD millones).
    Intentamos scrapear su sitio; fallback: estimación desde precio + estacionalidad.
    """
    try:
        r = requests.get(
            "https://www.ciara.com.ar/estadisticas.html",
            timeout=8, headers={"User-Agent": "Mozilla/5.0"},
        )
        if r.status_code == 200:
            import re
            text = r.text
            # Buscar monto en millones de dólares en texto
            matches = re.findall(r"USD\s*([\d\.]+)\s*[Mm]illones", text)
            if matches:
                monto = float(matches[0].replace(".", ""))
                return {"liquidaciones_usd_mm": monto, "source": "CIARA",
                        "nota": "Liquidaciones semanales complejo soja-girasol"}
    except Exception:
        pass

    # Estimación: en cosecha (Mar-Jun) Argentina liquida ~$1.500-2.000MM/semana
    # Fuera de cosecha: ~$300-600MM/semana
    month = date.today().month
    if 3 <= month <= 6:
        est = 1500
        season = "cosecha"
    elif month in [7, 8]:
        est = 800
        season = "post-cosecha"
    else:
        est = 400
        season = "entrecosecha"

    return {"liquidaciones_usd_mm": est, "source": "seasonal_estimate",
            "nota": f"Estimación estacional ({season}) — sin datos CIARA disponibles"}


def _compute_argentina_score(blue_data: dict, retenciones: dict, ciara: dict) -> dict:
    """
    Calcula el Argentina Supply Pressure Score (0-100).
    100 = máxima retención (alcista para precio)
    0 = máxima liquidación (bajista para precio)
    """
    score = 50   # base neutral
    factors = []

    # Factor 1: Cepo/spread (peso 40%)
    spread = blue_data.get("spread_pct")
    if spread is not None:
        if spread > 80:
            cepo_score = 90; factors.append(f"Brecha {spread:.0f}%: retención masiva")
        elif spread > 40:
            cepo_score = 75; factors.append(f"Brecha {spread:.0f}%: retención alta")
        elif spread > 15:
            cepo_score = 60; factors.append(f"Brecha {spread:.0f}%: retención moderada")
        else:
            cepo_score = 25; factors.append(f"Brecha {spread:.0f}%: sin cepo relevante")
        score = score * 0.6 + cepo_score * 0.4

    # Factor 2: Retenciones (peso 30%) — altas retenciones = retención de grano
    ret = retenciones.get("soja_grano", 33)
    if ret > 35:
        ret_score = 75; factors.append(f"Retenciones {ret:.0f}%: presión a retener")
    elif ret > 28:
        ret_score = 55; factors.append(f"Retenciones {ret:.0f}%: nivel normal")
    else:
        ret_score = 30; factors.append(f"Retenciones {ret:.0f}%: estimula liquidación")
    score = score * 0.7 + ret_score * 0.3

    # Factor 3: Pace CIARA (peso 30%) — liquidaciones rápidas = presión bajista
    liq = ciara.get("liquidaciones_usd_mm", 500)
    if liq > 1500:
        liq_score = 20; factors.append(f"CIARA {liq:.0f}MM/sem: liquidación acelerada")
    elif liq > 700:
        liq_score = 45; factors.append(f"CIARA {liq:.0f}MM/sem: pace normal")
    else:
        liq_score = 70; factors.append(f"CIARA {liq:.0f}MM/sem: liquidación lenta")
    score = score * 0.7 + liq_score * 0.3

    score = round(score)
    if score >= 65:
        signal = "ALCISTA"
        summary = "Retención argentina elevada — oferta global restringida."
    elif score >= 40:
        signal = "NEUTRAL"
        summary = "Oferta argentina dentro de parámetros normales."
    else:
        signal = "BAJISTA"
        summary = "Liquidación acelerada en Argentina — presión sobre precio global."

    return {"score": score, "signal": signal, "summary": summary, "factors": factors}


def get_argentina_supply_signal() -> dict:
    """
    Retorna señal completa de oferta argentina.

    Retorna dict con:
      cepo_activo        : bool
      spread_pct         : brecha blue/oficial
      usd_oficial / usd_blue : tipos de cambio
      retenciones        : alícuota soja grano (%)
      ciara_liquidaciones: divisas liquidadas USD millones/semana
      supply_score       : 0-100 (retención → 100, liquidación → 0)
      impacto_precio     : ALCISTA / NEUTRAL / BAJISTA
      razonamiento       : texto explicativo
      factores           : lista de factores
      ultima_actualizacion: fecha
    """
    cached = _load_cache()
    if cached:
        print(f"   [AR] Caché válido ({cached.get('ultima_actualizacion', '?')})")
        return cached

    print("   [AR] Actualizando señal Argentina (cepo + retenciones + CIARA)…")

    blue_data   = _fetch_blue_dollar()
    retenciones = _fetch_retenciones()
    ciara       = _fetch_ciara_pace()
    pressure    = _compute_argentina_score(blue_data, retenciones, ciara)

    cepo_activo = blue_data.get("cepo_activo", True)
    spread_pct  = blue_data.get("spread_pct")

    # Razonamiento compuesto
    razones = []
    if spread_pct is not None:
        razones.append(f"Brecha cambiaria {spread_pct:.0f}% (oficial ${blue_data.get('oficial')} / blue ${blue_data.get('blue')})")
    razones.append(f"Retenciones soja {retenciones.get('soja_grano', 33):.0f}%")
    razones.append(f"Liquidaciones CIARA ~{ciara.get('liquidaciones_usd_mm', '?')} MM USD/sem")
    razones.extend(pressure.get("factors", []))

    data = {
        "cepo_activo":          cepo_activo,
        "spread_pct":           spread_pct,
        "usd_oficial":          blue_data.get("oficial"),
        "usd_blue":             blue_data.get("blue"),
        "usd_mep":              blue_data.get("mep"),
        "retenciones_soja":     retenciones.get("soja_grano", 33.0),
        "retenciones_source":   retenciones.get("source"),
        "ciara_liquidaciones_usd_mm": ciara.get("liquidaciones_usd_mm"),
        "ciara_source":         ciara.get("source"),
        "supply_score":         pressure["score"],
        "impacto_precio":       pressure["signal"],
        "razonamiento":         pressure["summary"],
        "factores":             razones,
        "fuente":               "Bluelytics + AFIP + CIARA",
        "ultima_actualizacion": str(date.today()),
        "nota":                 "Argentina + Brasil = ~80% exportaciones mundiales de soja",
    }
    _save_cache(data)
    print(f"   [AR] Score: {pressure['score']} | Impacto: {pressure['signal']} | "
          f"Cepo: {cepo_activo} | Brecha: {spread_pct}%")
    return data
