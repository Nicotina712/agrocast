"""
src/trader/signal_breakdown.py
Breakdown explicado del score de confianza de la señal AgroCast.

Desglosa la señal final en sus componentes:
  1. Intelligence Engine — 5-agent debate (fuente primaria, 35%)
  2. Demanda China       — crush margin + importaciones (20%)
  3. WASDE               — sorpresa de stocks mundiales (15%)
  4. ML (XGBoost 7d)     — probabilidad de retorno positivo (15%)
  5. Técnico             — RSI, Bollinger, MA crossover (10%)
  6. Forecast 30d        — dirección del modelo Ridge+ensemble (0%, informativo)
  7. Estacionalidad      — patrón histórico del mes actual (0%, informativo)

Retorna un score compuesto 0–100 con contribución por factor.
Cache: data/signal_breakdown.json (TTL 4h)
"""

import json
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_PATH   = os.path.join(_PROJECT_ROOT, "data", "signal_breakdown.json")
_TTL_HOURS    = 4


def _cache_valid() -> bool:
    if not os.path.exists(_CACHE_PATH):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(_CACHE_PATH))
    return age < timedelta(hours=_TTL_HOURS)


def _factor_bar(score: float) -> str:
    """Convierte un score -1..1 en descripción textual."""
    if score > 0.5:   return "MUY ALCISTA"
    if score > 0.15:  return "ALCISTA"
    if score > -0.15: return "NEUTRO"
    if score > -0.5:  return "BAJISTA"
    return "MUY BAJISTA"


def _ie_factor() -> dict:
    """Factor Intelligence Engine: veredicto del debate multi-agente (5 agentes Claude Sonnet).

    Esta es la fuente primaria de verdad del sistema. El IE integra fundamentales,
    técnicos, análogos históricos y sentimiento en un debate estructurado.
    """
    try:
        ie_path = os.path.join(_PROJECT_ROOT, "data", "intelligence_engine_verdict.json")
        if not os.path.exists(ie_path):
            raise FileNotFoundError("IE verdict not found")
        with open(ie_path, "r", encoding="utf-8") as f:
            ie = json.load(f)

        verdict_data = ie.get("verdict", {})
        verdict = verdict_data.get("verdict", "HOLD")
        confidence = verdict_data.get("confidence", 0.5)
        reasoning = verdict_data.get("reasoning", "")

        # Map verdict to score: BUY → +confidence, SELL → -confidence, HOLD → 0
        if verdict in ("BUY", "STRONG_BUY"):
            score = float(confidence)
        elif verdict in ("SELL", "STRONG_SELL"):
            score = -float(confidence)
        else:
            score = 0.0

        # Clamp to [-1, 1]
        score = float(np.clip(score, -1, 1))

        # Extract price range
        range_7d = verdict_data.get("price_range_7d", {})
        range_str = ""
        if range_7d:
            range_str = f" · Rango 7d: {range_7d.get('low','?')}-{range_7d.get('high','?')}"

        # Check freshness (IE should be from today)
        ts = ie.get("timestamp", "")
        age_str = ""
        if ts:
            try:
                ie_dt = datetime.fromisoformat(ts)
                age_hours = (datetime.now() - ie_dt).total_seconds() / 3600
                if age_hours > 24:
                    age_str = f" (hace {int(age_hours)}h)"
            except Exception:
                pass

        detail = (f"Veredicto: {verdict} · Confianza: {confidence:.0%} · "
                  f"Sizing: {verdict_data.get('position_sizing', '?')}%{range_str}{age_str}")

        return {
            "name":      "Intelligence Engine (5-Agent Debate)",
            "score":     round(score, 3),
            "direction": _factor_bar(score),
            "detail":    detail,
            "weight":    0.35,
            "raw":       {"verdict": verdict, "confidence": confidence,
                         "price_range_7d": range_7d, "reasoning": reasoning[:200]},
        }
    except Exception:
        return {"name": "Intelligence Engine (5-Agent Debate)", "score": 0, "direction": "NEUTRO",
                "detail": "Sin datos (debate no ejecutado hoy)", "weight": 0.35, "raw": {}}


def _ml_factor() -> dict:
    """Factor ML: probabilidad XGBoost de retorno positivo (7d)."""
    try:
        sig = pd.read_csv(os.path.join(_PROJECT_ROOT, "artifacts", "signals.csv"))
        exp_ret  = float(sig["expected_return"].iloc[-1])
        conf     = float(sig["confidence"].iloc[-1])
        signal   = str(sig["signal"].iloc[-1])
        # expected_return positivo = bullish, negativo = bearish
        score = float(np.clip(exp_ret * 5, -1, 1))  # escalar a -1..1
        return {
            "name":        "Modelo ML (XGBoost 7d)",
            "score":       round(score, 3),
            "direction":   _factor_bar(score),
            "detail":      f"Retorno esperado: {exp_ret*100:+.2f}% · Señal: {signal} · Confianza: {conf:.1%}",
            "weight":      0.15,
            "raw":         {"expected_return": round(exp_ret, 4), "confidence": round(conf, 4), "signal": signal},
        }
    except Exception:
        return {"name": "Modelo ML (XGBoost 7d)", "score": 0, "direction": "NEUTRO",
                "detail": "Sin datos", "weight": 0.15, "raw": {}}


def _forecast_factor() -> dict:
    """
    Factor Forecast 30d: dirección del ensemble Ridge+XGBoost.

    NOTA: peso = 0 (informativo). El forecast 30d satura sistemáticamente
    por el cap diario ±1% acumulado a 30 días, llegando a +12% típico y
    siendo scoreado como "MUY ALCISTA" (+1.0). Eso secuestraba el composite
    e invertía la señal de los demás factores. Se mantiene visible en UI
    pero no pondera hasta que se calibre el modelo subyacente.
    """
    try:
        from src.model.predict_multihorizon import get_multihorizon_forecast
        mh = get_multihorizon_forecast()
        h30 = next((h for h in mh.get("horizons", []) if h["horizon"] == "30d"), None)
        if not h30:
            raise ValueError("Sin forecast 30d")
        ret_pct = h30.get("return_pct", 0)
        score   = float(np.clip(ret_pct / 10, -1, 1))
        return {
            "name":      "Forecast 30d (Ridge+XGB Ensemble)",
            "score":     round(score, 3),
            "direction": _factor_bar(score),
            "detail":    f"Precio objetivo: {h30.get('forecast',0):.1f} USc/bu · Retorno: {ret_pct:+.1f}% (informativo, no pondera)",
            "weight":    0.00,
            "raw":       {"forecast": h30.get("forecast"), "return_pct": ret_pct},
        }
    except Exception:
        return {"name": "Forecast 30d", "score": 0, "direction": "NEUTRO",
                "detail": "Sin datos", "weight": 0.00, "raw": {}}


def _china_factor() -> dict:
    """Factor Demanda China: crush margin + score de importaciones."""
    try:
        china_path = os.path.join(_PROJECT_ROOT, "data", "china_demand.json")
        if not os.path.exists(china_path):
            raise FileNotFoundError
        with open(china_path) as f:
            china = json.load(f)
        demand_score = china.get("demand_score", 50)
        score = (demand_score - 50) / 50  # 50=neutro → 0; 100=fuerte → +1; 0=débil → -1
        crush = china.get("crush_margin", {})
        crush_sig = crush.get("signal", "NEUTRAL")
        detail = f"Score demanda: {demand_score}/100 · Crush: {crush_sig}"
        if china.get("imports_yoy_pct") is not None:
            detail += f" · Importaciones YoY: {china['imports_yoy_pct']:+.1f}%"
        return {
            "name":      "Demanda China",
            "score":     round(score, 3),
            "direction": _factor_bar(score),
            "detail":    detail,
            "weight":    0.20,
            "raw":       {"demand_score": demand_score, "crush_signal": crush_sig},
        }
    except Exception:
        return {"name": "Demanda China", "score": 0, "direction": "NEUTRO",
                "detail": "Sin datos", "weight": 0.20, "raw": {}}


def _wasde_factor() -> dict:
    """Factor WASDE: sorpresa de stocks mundiales."""
    try:
        wasde_path = os.path.join(_PROJECT_ROOT, "data", "wasde_official.json")
        if not os.path.exists(wasde_path):
            raise FileNotFoundError
        with open(wasde_path) as f:
            wasde = json.load(f)
        signal = wasde.get("signal", "NEUTRAL")
        surp   = (wasde.get("surprise") or {}).get("score", 0) or 0
        score_map = {"BULLISH": 0.6, "BEARISH": -0.6, "NEUTRAL": 0.0}
        score = float(score_map.get(signal, 0))
        if abs(surp) > 0:
            score = float(np.clip(score + surp / 10, -1, 1))
        detail = f"Señal WASDE: {signal}"
        world = wasde.get("world") or {}
        if world.get("ending_stocks_mmt"):
            detail += f" · Stocks finales: {world['ending_stocks_mmt']:.1f} MMT"
        return {
            "name":      "WASDE / Stocks Mundiales",
            "score":     round(score, 3),
            "direction": _factor_bar(score),
            "detail":    detail,
            "weight":    0.15,
            "raw":       {"signal": signal, "surprise_score": surp},
        }
    except Exception:
        return {"name": "WASDE / Stocks Mundiales", "score": 0, "direction": "NEUTRO",
                "detail": "Sin datos", "weight": 0.15, "raw": {}}


def _technical_factor() -> dict:
    """Factor técnico: RSI, Bollinger, MA crossover para soja."""
    try:
        multi_path = os.path.join(_PROJECT_ROOT, "data", "multi_commodity.json")
        if os.path.exists(multi_path):
            with open(multi_path) as f:
                mc = json.load(f)
            soy = mc.get("commodities", {}).get("Soybeans", {})
        else:
            from src.data.multi_commodity_signals import get_multi_commodity_signals
            mc  = get_multi_commodity_signals()
            soy = mc.get("commodities", {}).get("Soybeans", {})

        tech_score = soy.get("score", 0)
        score = float(np.clip(tech_score / 5, -1, 1))  # score max ±5 → ±1
        rsi   = soy.get("rsi", 50)
        bb    = soy.get("bb_pct", 50)
        detail = f"Score técnico: {tech_score:+d} · RSI: {rsi:.0f} · Bollinger%: {bb:.0f}%"
        factors = soy.get("signals_list", [])
        if factors:
            detail += f" · {'; '.join(factors[:2])}"
        return {
            "name":      "Análisis Técnico (Soja)",
            "score":     round(score, 3),
            "direction": _factor_bar(score),
            "detail":    detail,
            "weight":    0.15,
            "raw":       {"tech_score": tech_score, "rsi": rsi, "bb_pct": bb},
        }
    except Exception:
        return {"name": "Análisis Técnico", "score": 0, "direction": "NEUTRO",
                "detail": "Sin datos", "weight": 0.15, "raw": {}}


def _seasonal_factor() -> dict:
    """Factor estacional: patrón histórico del mes actual vs. precio actual."""
    try:
        # Calcular estacionalidad directamente desde raw_market.csv
        raw_path = os.path.join(_PROJECT_ROOT, "data", "raw_market.csv")
        df = pd.read_csv(raw_path, parse_dates=["Date"]).sort_values("Date")
        df["ret_1m"] = df["Soybeans"].pct_change(21)
        df["month"]  = df["Date"].dt.month
        month = datetime.now().month
        subset = df[df["month"] == month]["ret_1m"].dropna()
        if subset.empty:
            raise ValueError("Sin datos mensuales")
        avg_pct  = float(subset.mean()) * 100
        std_pct  = float(subset.std()) * 100 if len(subset) > 1 else 0
        month_data = {
            "avg_pct":   round(avg_pct, 2),
            "ci_upper":  round(avg_pct + std_pct, 2),
            "ci_lower":  round(avg_pct - std_pct, 2),
        }
        avg_ret = month_data.get("avg_pct", 0) or 0
        score   = float(np.clip(avg_ret / 5, -1, 1))  # ±5% → ±1
        month_names = ["","Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
        detail = (f"{month_names[month]}: retorno histórico promedio {avg_ret:+.1f}% · "
                  f"±1σ: [{month_data.get('ci_lower',0):+.1f}%, {month_data.get('ci_upper',0):+.1f}%]")
        return {
            "name":      "Estacionalidad Histórica",
            "score":     round(score, 3),
            "direction": _factor_bar(score),
            "detail":    detail,
            "weight":    0.00,  # informativo, no pesa en el score final
            "raw":       {"month": month, "avg_pct": avg_ret},
        }
    except Exception:
        return {"name": "Estacionalidad Histórica", "score": 0, "direction": "NEUTRO",
                "detail": "Sin datos", "weight": 0.00, "raw": {}}


def get_signal_breakdown() -> dict:
    """
    Retorna el breakdown completo del score de confianza.

    Retorna dict con:
      factors       : lista de factores con score, dirección y detalle
      composite_score : score compuesto ponderado 0–100
      composite_signal: BUY / SELL / HOLD
      as_of         : fecha
    """
    if _cache_valid():
        try:
            with open(_CACHE_PATH) as f:
                return json.load(f)
        except Exception:
            pass

    print("   [Breakdown] Calculando breakdown de señal...")

    factors = [
        _ie_factor(),
        _china_factor(),
        _wasde_factor(),
        _ml_factor(),
        _technical_factor(),
        _forecast_factor(),
        _seasonal_factor(),
    ]

    # Score compuesto ponderado (solo factores con weight > 0)
    total_weight  = sum(f["weight"] for f in factors if f["weight"] > 0)
    weighted_sum  = sum(f["score"] * f["weight"] for f in factors if f["weight"] > 0)
    composite_raw = weighted_sum / total_weight if total_weight > 0 else 0
    composite_score = int(round((composite_raw + 1) / 2 * 100))  # -1..1 → 0..100

    if composite_raw > 0.15:
        composite_signal = "BUY"
    elif composite_raw < -0.15:
        composite_signal = "SELL"
    else:
        composite_signal = "HOLD"

    result = {
        "ok":               True,
        "factors":          factors,
        "composite_score":  composite_score,
        "composite_raw":    round(composite_raw, 3),
        "composite_signal": composite_signal,
        "as_of":            date.today().isoformat(),
    }

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"   [Breakdown] Score: {composite_score}/100 -> {composite_signal}")
    return result
