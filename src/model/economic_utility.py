"""
src/model/economic_utility.py
Modelo de utilidad económica para el productor: esperar vs vender hoy.

Cambia el paradigma:
  - Antes: "predigo precio en 30d = $1188" (frágil, casi siempre se equivoca).
  - Ahora: "P(esperar 30d > vender hoy) = 47%, ahorro esperado = -$3, peor caso = -$140".

La decisión real del productor es Bayesiana:
    Acción ∈ {SELL_NOW, WAIT_30d, FORWARD_30d}
    Costo de esperar = storage_cost + financing_cost + riesgo_baja
    Beneficio de esperar = E[price_t30] - price_t  (sólo si > costos)

Inputs (todos por TON):
    current_price       : precio CBOT hoy (USD/bu, convertimos)
    storage_cost_month  : costo de silo (USD/ton/mes) — default 6 USD/ton/mes
    financing_rate      : tasa anual costo de oportunidad — default 8%
    horizon_days        : 30 default
    n_paths             : Monte Carlo paths (default 5000)

Output:
    expected_value_wait, q10_wait, q50_wait, q90_wait    (en USD/ton)
    expected_value_sell                                   (= current_price)
    diff_wait_minus_sell                                  (positivo → esperar conviene en promedio)
    prob_wait_better                                      (P(wait > sell))
    var_5pct                                              (Value at Risk 5%: peor caso 1-en-20)
    cvar_5pct                                             (Conditional VaR: media de la cola 5%)
    decision                                              (recomendación SELL_NOW | WAIT | INDIFFERENT)

Conversión USD/bu → USD/ton: 1 bushel soja = 27.2155 kg → 1 ton = 36.7437 bu.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

BU_PER_TON = 36.7437  # bushels en una tonelada métrica de soja
# CBOT cotiza soja en centavos de USD por bushel (cents/bu) — el feature
# `Soybeans` los preserva. Para USD/ton hay que dividir por 100 antes
# de multiplicar por BU_PER_TON.
CENTS_TO_USD = 0.01

DEFAULT_STORAGE_COST_PER_TON_MONTH = 6.0   # USD/ton/mes (silo + seguro + manipuleo)
DEFAULT_FINANCING_RATE_ANNUAL      = 0.08  # 8% anual costo de oportunidad


def utility_wait_vs_sell(
    df_features: pd.DataFrame,
    storage_cost_per_ton_month: float = DEFAULT_STORAGE_COST_PER_TON_MONTH,
    financing_rate_annual:      float = DEFAULT_FINANCING_RATE_ANNUAL,
    horizon_days:               int   = 30,
    n_paths:                    int   = 5000,
    artifacts_dir:              str | None = None,
) -> dict:
    """Calcula utilidad esperada de WAIT vs SELL_NOW para una tonelada.
    Todos los valores se expresan en USD/ton para la decisión real del productor.
    """
    from src.model.predict_horizons import forecast_paths, _gaussian_anchor_sample, forecast_anchors

    anchors = forecast_anchors(df_features, artifacts_dir=artifacts_dir)
    if not anchors.get("horizons"):
        return {"ok": False, "reason": "no horizons model available"}

    p_t_bu_cents = float(anchors["current_price"])           # cents/bu (CBOT convention)
    p_t_bu  = p_t_bu_cents * CENTS_TO_USD                    # USD/bu
    p_t_ton = p_t_bu * BU_PER_TON                            # USD/ton hoy

    # Sample del precio terminal usando ancla más cercana al horizonte
    h_keys = sorted(anchors["horizons"].keys())
    nearest = min(h_keys, key=lambda x: abs(x - horizon_days))
    rng = np.random.default_rng(42)
    terminal_bu_cents = _gaussian_anchor_sample(p_t_bu_cents, anchors["horizons"][nearest], n_paths, rng)
    terminal_ton      = (terminal_bu_cents * CENTS_TO_USD) * BU_PER_TON

    # Costos de esperar (USD/ton, prorrateado a horizonte)
    months = horizon_days / 30.0
    storage_cost = storage_cost_per_ton_month * months
    financing_cost = p_t_ton * financing_rate_annual * (horizon_days / 365.0)
    total_wait_cost = storage_cost + financing_cost

    # Utilidad de cada acción (en USD/ton)
    # SELL_NOW: cobra p_t hoy, libre de costos posteriores
    util_sell = p_t_ton
    # WAIT: cobra terminal en t+30, paga storage + financing
    util_wait = terminal_ton - total_wait_cost

    diff = util_wait - util_sell
    prob_wait_better = float((util_wait > util_sell).mean() * 100)
    expected_diff    = float(diff.mean())

    # Risk metrics — útiles para productor adverso al riesgo
    sorted_diff = np.sort(diff)
    var_5      = float(sorted_diff[int(0.05 * n_paths)])     # peor 5%
    cvar_5     = float(sorted_diff[:int(0.05 * n_paths)].mean()) if int(0.05 * n_paths) > 0 else var_5

    # Decisión (regla simple — el productor puede ajustar su umbral)
    if expected_diff > total_wait_cost * 0.5 and prob_wait_better >= 55:
        decision = "WAIT"
        decision_reason = (f"Esperar tiene retorno esperado de +${expected_diff:.0f}/ton "
                           f"con {prob_wait_better:.0f}% de chances de ganar.")
    elif expected_diff < -total_wait_cost * 0.5 or prob_wait_better < 40:
        decision = "SELL_NOW"
        decision_reason = (f"Esperar tiene retorno esperado de ${expected_diff:.0f}/ton "
                           f"y solo {prob_wait_better:.0f}% de chances de ganar.")
    else:
        decision = "INDIFFERENT"
        decision_reason = (f"Retorno esperado pequeño (${expected_diff:.0f}/ton). "
                           "Decisión depende de tu costo financiero y aversión al riesgo.")

    return {
        "ok":             True,
        "horizon_days":   horizon_days,
        "current_price_cents_bu": round(p_t_bu_cents, 2),
        "current_price_usd_bu":   round(p_t_bu, 4),
        "current_price_usd_ton":  round(p_t_ton, 2),
        "wait_costs_usd_ton": {
            "storage":   round(storage_cost, 2),
            "financing": round(financing_cost, 2),
            "total":     round(total_wait_cost, 2),
            "rate_annual_pct":   round(financing_rate_annual * 100, 2),
            "storage_per_month": round(storage_cost_per_ton_month, 2),
        },
        "expected_value_sell_now_usd_ton": round(util_sell, 2),
        "expected_value_wait_usd_ton":     round(float(util_wait.mean()), 2),
        "wait_quantiles_usd_ton": {
            "q05": round(float(np.quantile(util_wait, 0.05)), 2),
            "q25": round(float(np.quantile(util_wait, 0.25)), 2),
            "q50": round(float(np.quantile(util_wait, 0.50)), 2),
            "q75": round(float(np.quantile(util_wait, 0.75)), 2),
            "q95": round(float(np.quantile(util_wait, 0.95)), 2),
        },
        "diff_wait_minus_sell_usd_ton": round(expected_diff, 2),
        "prob_wait_better_pct":         round(prob_wait_better, 1),
        "var_5pct_usd_ton":              round(var_5, 2),
        "cvar_5pct_usd_ton":             round(cvar_5, 2),
        "decision":         decision,
        "decision_reason":  decision_reason,
        "n_paths":          n_paths,
        "as_of":            pd.Timestamp.now().isoformat(timespec="seconds"),
    }
