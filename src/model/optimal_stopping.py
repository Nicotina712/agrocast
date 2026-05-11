"""
src/model/optimal_stopping.py
Política óptima de venta para el productor vía Dynamic Programming
(backward induction). Resuelve el optimal stopping problem:

    V_t(p) = max( p,                                  # vender ahora a p
                  E[V_{t+1}(p_{t+1}) | p_t = p] - cost_per_day )

Donde V_t(p) es el valor de tener el grano si quedan t días para
decidir, dado que el precio actual es p.

Inputs:
    - paths: matriz (n_paths × T) de trayectorias de precio simuladas
             (ya las generamos en predict_horizons.forecast_paths)
    - storage_cost_per_day, financing_per_day_rate
    - price_grid: grilla discreta de precios para tabular V_t

Output:
    - reservation_price[t]: precio mínimo a partir del cual conviene vender
                             cuando quedan `t` días al horizonte
    - decision_now: SELL_NOW si p_t > reservation_price[T], WAIT si no
    - expected_optimal_value: E[V_T(p)] al inicio del horizonte
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.model.economic_utility import BU_PER_TON, CENTS_TO_USD


def solve_optimal_stopping(
    paths_cents_bu: np.ndarray,
    horizon_days: int,
    storage_cost_per_ton_month: float = 6.0,
    financing_rate_annual: float = 0.08,
    n_grid: int = 50,
) -> dict:
    """Resuelve backward induction sobre las trayectorias simuladas.

    paths_cents_bu : (n_paths, horizon_days) precios diarios simulados (cents/bu)
    Returns dict con la política óptima.
    """
    n_paths, T = paths_cents_bu.shape
    if T != horizon_days:
        T = min(T, horizon_days)
        paths_cents_bu = paths_cents_bu[:, :T]

    # Convertir a USD/ton
    to_ton = BU_PER_TON * CENTS_TO_USD
    paths_ton = paths_cents_bu * to_ton

    # Costos diarios (USD/ton/día)
    storage_per_day = storage_cost_per_ton_month / 30.0

    # Grilla de precios (en USD/ton) para discretizar la función valor
    p_min = float(paths_ton.min()) * 0.95
    p_max = float(paths_ton.max()) * 1.05
    grid  = np.linspace(p_min, p_max, n_grid)

    # V[t, i] = valor óptimo si quedan (T - t) días y precio actual = grid[i]
    # Caso terminal: t = T - 1 (último día) → forzosamente vendés
    V = np.zeros((T, n_grid))
    V[T - 1, :] = grid                           # vender al precio final del horizonte

    # Construir transiciones empíricas: para cada t y precio actual, cuál es la
    # distribución del precio en t+1 a través de los paths.
    # Aproximación: para cada par (t, t+1), modelar el cambio relativo log
    # como mezcla empírica (uso paths directamente).
    for t in range(T - 2, -1, -1):
        # Precios reales en t y t+1 a través de paths
        p_t   = paths_ton[:, t]
        p_t1  = paths_ton[:, t + 1]
        # Costo de "esperar un día" en USD/ton:
        # storage + financing sobre el precio actual
        for i, p in enumerate(grid):
            financing_per_day = p * financing_rate_annual / 365.0
            cost_wait = storage_per_day + financing_per_day
            # Mapear paths más cercanos al precio actual p (kernel discreto)
            weights = np.exp(-((p_t - p) / max(grid[1] - grid[0], 1e-6)) ** 2)
            weights = weights / max(weights.sum(), 1e-9)
            # E[V_{t+1}(p_{t+1}) | p_t = p] vía paths ponderados
            # Para cada path k, V_{t+1}(p_{t+1,k}) se interpola sobre la grilla
            v_t1_paths = np.interp(p_t1, grid, V[t + 1, :])
            ev = float((v_t1_paths * weights).sum())
            v_wait = ev - cost_wait
            V[t, i] = max(p, v_wait)

    # Reservation price por día (precio mínimo para vender)
    reservation = np.zeros(T)
    for t in range(T):
        # En cada t, el reservation price = mínimo p donde sell ≥ wait
        # Calculamos wait_value = V[t,i] - grid[i] (excedente de esperar)
        if t == T - 1:
            reservation[t] = grid[0]   # forzosamente se vende
            continue
        sell_dominates = grid >= (V[t, :] - 1e-6)
        # Buscar el primer índice donde vender domina (V[t,i] = grid[i])
        if sell_dominates.any():
            reservation[t] = float(grid[sell_dominates][0])
        else:
            reservation[t] = float(grid[-1] + 1)   # nunca conviene vender

    # Decisión hoy (t=0): comparar precio inicial con reservation[0]
    p_now = float(paths_ton[:, 0].mean())
    decision = "SELL_NOW" if p_now >= reservation[0] else "WAIT"

    # Valor esperado óptimo al inicio
    initial_idx = int(np.argmin(np.abs(grid - p_now)))
    expected_v0 = float(V[0, initial_idx])

    # Comparación: utilidad de vender hoy vs valor óptimo
    util_sell_now    = p_now
    util_optimal     = expected_v0
    util_terminal    = float(paths_ton[:, -1].mean()) - (storage_per_day + p_now * financing_rate_annual / 365) * T

    return {
        "ok": True,
        "horizon_days":  T,
        "n_paths":       int(n_paths),
        "decision_now":  decision,
        "p_now_usd_ton": round(p_now, 2),
        "reservation_today_usd_ton":   round(reservation[0], 2),
        "reservation_curve":           [round(float(r), 2) for r in reservation.tolist()],
        "expected_value_optimal_usd_ton":   round(util_optimal, 2),
        "expected_value_sell_now_usd_ton":  round(util_sell_now, 2),
        "expected_value_always_wait_usd_ton": round(util_terminal, 2),
        "lift_optimal_vs_sell_now_pct": round((util_optimal / util_sell_now - 1) * 100, 3),
        "lift_optimal_vs_wait_pct":     round((util_optimal / util_terminal - 1) * 100, 3),
    }


def optimal_stopping_decision(
    df_features: pd.DataFrame,
    storage_cost_per_ton_month: float = 6.0,
    financing_rate_annual: float = 0.08,
    horizon_days: int = 30,
    n_paths: int = 1000,
    artifacts_dir: str | None = None,
) -> dict:
    """Wrapper de alto nivel: genera paths y aplica backward induction."""
    from src.model.predict_horizons import forecast_paths
    paths_out = forecast_paths(df_features, n_paths=n_paths, horizon_days=horizon_days,
                                artifacts_dir=artifacts_dir)
    if not paths_out.get("ok"):
        return {"ok": False, "error": paths_out.get("error", "no paths")}

    paths = paths_out.get("paths")
    # Si forecast_paths no devuelve la matriz, la reconstruimos rápido
    if paths is None:
        # Fallback: muestrear paths internos (no implementado ahora)
        return {"ok": False, "error": "paths matrix not in output"}

    return solve_optimal_stopping(
        paths_cents_bu=paths,
        horizon_days=horizon_days,
        storage_cost_per_ton_month=storage_cost_per_ton_month,
        financing_rate_annual=financing_rate_annual,
    )
