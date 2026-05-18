"""
src/producer/sell_signal.py
Módulo Productor — Semáforo de Venta para productores de soja en Uruguay.

El productor agropecuario necesita saber:
  1. ¿A cuánto está el precio hoy en USD/ton? (referencia Chicago + conversión)
  2. ¿Es este un buen momento para vender?
  3. ¿Cuánto me cuesta esperar X días en almacenamiento?

Precio local estimado (Uruguay):
  - Base: Chicago Soybeans (USD/bushel) × 36.744 = USD/tonelada métrica
  - Basis Uruguay: -20 a -35 USD/ton (descuento por distancia a puertos,
    calidad, logística interna). Se usa -25 USD/ton como estimación conservadora.
  - El mercado agro uruguayo opera en USD, no en UYU.
  - UYU se muestra como referencia usando el tipo de cambio del BCU.

Semáforo de venta — lógica:
  - Score compuesto 0-100 (mayor = más favorable para vender ahora)
  - Componentes:
      1. Señal del clasificador (BUY=0, HOLD=50, SELL=100)
      2. Estacionalidad (mes actual vs. histórico)
      3. Precio relativo (precio actual vs. promedio 90 días)
      4. WASDE proximity (próximo reporte en ≤7 días → incertidumbre alta)
"""

import os
import sys
import json
from datetime import date, timedelta

import pandas as pd
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Conversión: 1 tonelada = 36.744 bushels
BUSHELS_PER_TON = 36.744

# Basis estimado Uruguay (descuento vs. Chicago FOB)
# Se puede configurar en .env como URUGUAY_BASIS_USD_TON
_DEFAULT_BASIS = -25.0  # USD/ton

# Costo de almacenamiento anual en silo/bolsa (% del valor del grano/año)
# Uruguay: ~5-7% anual (silo propio) o ~8-10% (tercerizado)
STORAGE_COST_PCT_ANNUAL = float(os.getenv("STORAGE_COST_PCT", "6.0"))


def get_uyu_usd_rate() -> float:
    """
    Obtiene el tipo de cambio USD/UYU desde open.er-api.com (gratuito, sin clave).
    Fallback: 42.0 si el servicio no está disponible.
    """
    try:
        import requests
        r = requests.get(
            "https://open.er-api.com/v6/latest/USD",
            timeout=5,
            headers={"User-Agent": "AgroCastPRO/1.0"},
        )
        if r.status_code == 200:
            data = r.json()
            uyu = data.get("rates", {}).get("UYU")
            if uyu and 35 < float(uyu) < 80:
                return round(float(uyu), 2)
    except Exception:
        pass
    # Fallback conservador (actualizar si el mercado cambia mucho)
    return 42.0


def compute_price_uyu(price_usc_bu: float, basis: float = _DEFAULT_BASIS) -> dict:
    """
    Convierte precio Chicago en USc/bu (centavos) a USD/ton y UYU/ton.
    Los precios en raw_market.csv están en USc/bu (cotización estándar CBOT).
    """
    price_usd_bu = price_usc_bu / 100.0          # USc → USD
    usd_ton_chicago = price_usd_bu * BUSHELS_PER_TON
    usd_ton_local   = usd_ton_chicago + basis     # basis negativo = descuento
    uyu_rate        = get_uyu_usd_rate()
    uyu_ton         = usd_ton_local * uyu_rate
    return {
        "chicago_usc_bu":   round(price_usc_bu, 2),
        "chicago_usd_bu":   round(price_usd_bu, 4),
        "chicago_usd_ton":  round(usd_ton_chicago, 2),
        "basis_usd_ton":    round(basis, 2),
        "local_usd_ton":    round(usd_ton_local, 2),
        "uyu_per_usd":      round(uyu_rate, 2),
        "local_uyu_ton":    round(uyu_ton, 0),
    }


def compute_storage_cost(price_usd_ton: float, days_list: list = None) -> list:
    """
    Calcula el costo acumulado de almacenar la soja por N días.
    Retorna lista de {days, cost_usd_ton, cost_pct, break_even_needed_pct}.
    """
    if days_list is None:
        days_list = [7, 15, 30, 45, 60, 90]
    daily_rate = STORAGE_COST_PCT_ANNUAL / 100 / 365
    result = []
    for d in days_list:
        cost_pct     = daily_rate * d * 100
        cost_usd_ton = price_usd_ton * daily_rate * d
        result.append({
            "days":                 d,
            "cost_usd_ton":         round(cost_usd_ton, 2),
            "cost_pct":             round(cost_pct, 3),
            "break_even_needed_pct": round(cost_pct, 2),
        })
    return result


def compute_storage_roi(
    price_usd_ton: float,
    forecast_return_pct: float | None,
    days_list: list = None,
    financing_rate_annual_pct: float | None = None,
    quality_risk_pct_per_month: float = 0.3,
) -> list:
    """
    ROI completo de almacenamiento: compara el beneficio esperado vs. costo total.

    Costo total = costo operativo silo + costo financiero + riesgo calidad
    Beneficio esperado = apreciación de precio proyectada por el modelo

    Parámetros
    ----------
    price_usd_ton          : precio actual local (USD/ton)
    forecast_return_pct    : retorno proyectado por el modelo (%) — puede ser None
    financing_rate_annual_pct : tasa de financiamiento del productor (% anual)
                               Si None, se usa PRODUCER_FINANCING_RATE_PCT del .env o 0
    quality_risk_pct_per_month: pérdida % por mes por humedad/fumigación (default 0.3%)

    Retorna lista de dicts por horizonte con:
      days, storage_cost_usd, financing_cost_usd, quality_risk_usd,
      total_cost_usd, expected_gain_usd, net_roi_usd, recommendation
    """
    if days_list is None:
        days_list = [15, 30, 45, 60, 90]

    financing_rate = financing_rate_annual_pct
    if financing_rate is None:
        financing_rate = float(os.getenv("PRODUCER_FINANCING_RATE_PCT", "0"))

    daily_storage   = STORAGE_COST_PCT_ANNUAL / 100 / 365
    daily_financing = financing_rate / 100 / 365

    result = []
    for d in days_list:
        # Costos
        storage_cost    = price_usd_ton * daily_storage * d
        financing_cost  = price_usd_ton * daily_financing * d
        quality_risk    = price_usd_ton * (quality_risk_pct_per_month / 100) * (d / 30)
        total_cost      = storage_cost + financing_cost + quality_risk

        # Beneficio esperado del modelo (lineal proporcional al horizonte)
        if forecast_return_pct is not None:
            # Asumimos que el modelo predice retorno a 30 días; escalar linealmente
            scaled_return = forecast_return_pct * (d / 30)
            expected_gain = price_usd_ton * scaled_return / 100
        else:
            scaled_return = None
            expected_gain = None

        net_roi = round(expected_gain - total_cost, 2) if expected_gain is not None else None

        if net_roi is not None:
            if net_roi > 2:
                rec = "ALMACENAR"
                rec_note = f"Ganancia neta esperada ${net_roi:.1f}/ton"
            elif net_roi > -2:
                rec = "EVALUAR"
                rec_note = f"Beneficio marginal (${net_roi:.1f}/ton) — depende de liquidez"
            else:
                rec = "VENDER"
                rec_note = f"Costo supera beneficio esperado (${net_roi:.1f}/ton)"
        else:
            rec = "SIN_SEÑAL"
            rec_note = "Sin proyección de precio disponible"

        result.append({
            "days":               d,
            "storage_cost_usd":   round(storage_cost, 2),
            "financing_cost_usd": round(financing_cost, 2),
            "quality_risk_usd":   round(quality_risk, 2),
            "total_cost_usd":     round(total_cost, 2),
            "expected_gain_usd":  round(expected_gain, 2) if expected_gain is not None else None,
            "net_roi_usd":        net_roi,
            "scaled_return_pct":  round(scaled_return, 2) if scaled_return is not None else None,
            "recommendation":     rec,
            "rec_note":           rec_note,
        })
    return result


def _get_seasonality_score(month: int, seasonality_data: list | None) -> dict:
    """
    Retorna el score estacional del mes actual (0-100, mayor = mes históricamente fuerte).
    """
    if not seasonality_data:
        return {"score": 50, "avg_pct": None, "win_rate": None, "rank": None}

    months = sorted(seasonality_data, key=lambda x: x.get("avg_pct", 0))
    n = len(months)
    for i, m in enumerate(months):
        if m.get("month") == month:
            score    = round((i / max(n - 1, 1)) * 100)
            return {
                "score":    score,
                "avg_pct":  m.get("avg_pct"),
                "win_rate": m.get("win_rate"),
                "rank":     i + 1,          # 1 = peor mes, 12 = mejor mes
                "n_months": n,
            }
    return {"score": 50, "avg_pct": None, "win_rate": None, "rank": None}


def compute_sell_signal(
    current_price_usd_bu: float,
    model_signal: dict | None,
    price_history: list | None,
    wasde_dates: list | None,
    seasonality_data: list | None,
    basis: float = _DEFAULT_BASIS,
) -> dict:
    """
    Calcula el semáforo de venta para el productor.

    Retorna:
        signal:      "VENDER" | "ESPERAR" | "RETENER"
        color:       "green"  | "yellow"  | "red"
        score:       0-100 (100 = muy favorable para vender)
        components:  desglose por componente
        reasoning:   lista de frases explicativas
        prices:      precios en USD/ton y UYU/ton
        storage:     tabla de costos de almacenamiento
    """
    today = date.today()
    scores = {}
    reasoning = []

    # ── Componente 1: Señal del Intelligence Engine (o modelo ML fallback)
    # SELL = buen momento para vender (score alto)
    # HOLD = incierto (score medio)
    # BUY  = precio va a subir, mejor esperar (score bajo)
    sig_map = {"SELL": 80, "HOLD": 50, "BUY": 20}
    if model_signal:
        sig       = model_signal.get("signal", "HOLD")
        conf      = round(model_signal.get("confidence", 0) * 100)
        is_ie     = model_signal.get("source") == "intelligence_engine"
        source_name = "El análisis de inteligencia (5 analistas)" if is_ie else "El modelo"
        scores["model"] = sig_map.get(sig, 50)
        if sig == "SELL":
            reasoning.append(
                f"{source_name} anticipa baja de precios (confianza {conf}%) "
                f"— favorable para vender ahora."
            )
        elif sig == "BUY":
            reasoning.append(
                f"{source_name} anticipa suba de precios (confianza {conf}%) "
                f"— considere esperar antes de vender."
            )
        else:
            reasoning.append(
                f"{source_name} no tiene una dirección clara (confianza {conf}%)."
            )
    else:
        scores["model"] = 50

    # ── Componente 2: Precio relativo (vs. promedio 90 días) ──────
    price_score = 50
    price_vs_avg = None
    if price_history and len(price_history) >= 30:
        hist_prices = [p["Soybeans"] for p in price_history
                       if p.get("Soybeans") and p["Soybeans"] > 0]
        if hist_prices:
            avg_90 = np.mean(hist_prices[-90:]) if len(hist_prices) >= 90 else np.mean(hist_prices)
            price_vs_avg = (current_price_usd_bu - avg_90) / avg_90 * 100
            if price_vs_avg > 5:
                price_score = 85
                reasoning.append(
                    f"Precio actual {price_vs_avg:+.1f}% sobre el promedio de 90 días "
                    f"— precio relativamente alto, momento favorable."
                )
            elif price_vs_avg > 1:
                price_score = 65
                reasoning.append(
                    f"Precio {price_vs_avg:+.1f}% sobre promedio 90 días — ligeramente favorable.")
            elif price_vs_avg < -5:
                price_score = 20
                reasoning.append(
                    f"Precio {price_vs_avg:+.1f}% bajo el promedio 90 días — precio deprimido, "
                    f"evalúe retener hasta recuperación."
                )
            else:
                price_score = 45
                reasoning.append(f"Precio en rango normal ({price_vs_avg:+.1f}% vs. promedio 90 días).")
    scores["price_vs_avg"] = price_score

    # ── Componente 3: Estacionalidad del mes actual ───────────────
    month = today.month
    seas  = _get_seasonality_score(month, seasonality_data)
    seas_score = seas["score"]
    scores["seasonality"] = seas_score
    month_names = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    mn = month_names[month - 1]
    if seas.get("avg_pct") is not None:
        wr = seas.get("win_rate", 0)
        avg = seas.get("avg_pct", 0)
        if seas_score >= 70:
            reasoning.append(
                f"{mn} es históricamente uno de los mejores meses para vender "
                f"(retorno promedio {avg:+.1f}%, tasa de éxito {wr:.0f}%)."
            )
        elif seas_score <= 30:
            reasoning.append(
                f"{mn} es históricamente débil para la soja "
                f"(retorno promedio {avg:+.1f}%, tasa de éxito {wr:.0f}%)."
            )
        else:
            reasoning.append(f"{mn}: comportamiento estacional neutro (promedio {avg:+.1f}%).")

    # ── Componente 4: WASDE proximity ────────────────────────────
    wasde_score = 50
    wasde_note  = None
    if wasde_dates:
        nearest = wasde_dates[0] if isinstance(wasde_dates[0], dict) else None
        if nearest:
            days_to = nearest.get("days_to", 99)
            if days_to <= 5:
                wasde_score = 30
                wasde_note  = f"Próximo WASDE en {days_to} días — incertidumbre alta, espere el reporte."
                reasoning.append(wasde_note)
            elif days_to <= 14:
                wasde_score = 45
                wasde_note  = f"WASDE en {days_to} días — el mercado puede moverse con el reporte."
            else:
                wasde_score = 60
    scores["wasde"] = wasde_score

    # ── Score final ponderado ─────────────────────────────────────
    weights = {"model": 0.40, "price_vs_avg": 0.30, "seasonality": 0.20, "wasde": 0.10}
    total_score = sum(scores.get(k, 50) * w for k, w in weights.items())
    total_score = round(total_score)

    # ── Semáforo ──────────────────────────────────────────────────
    if total_score >= 65:
        signal = "VENDER"
        color  = "green"
        summary = "Las condiciones actuales son favorables para vender."
    elif total_score >= 42:
        signal = "ESPERAR"
        color  = "yellow"
        summary = "Condiciones mixtas — evalúe según su necesidad de liquidez."
    else:
        signal = "RETENER"
        color  = "red"
        summary = "Las condiciones sugieren retener la cosecha si puede financiar el almacenamiento."

    # ── Precios ───────────────────────────────────────────────────
    prices  = compute_price_uyu(current_price_usd_bu, basis)
    storage = compute_storage_cost(prices["local_usd_ton"])

    return {
        "signal":         signal,
        "color":          color,
        "score":          total_score,
        "summary":        summary,
        "reasoning":      reasoning,
        "components":     scores,
        "prices":         prices,
        "storage_costs":  storage,
        "seasonality":    seas,
        "price_vs_avg_pct": round(price_vs_avg, 2) if price_vs_avg is not None else None,
        "wasde_note":     wasde_note,
        "as_of":          str(today),
    }
