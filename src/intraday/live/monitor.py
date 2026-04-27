"""
src/intraday/live/monitor.py
Monitor de drift y kill switch (Fase 2 — STUB).

Responsabilidades futuras:
  - PSI (Population Stability Index) sobre features clave
  - Detección de degradación de win rate vs backtest
  - Kill switch automático: detiene trading si PSI > 0.25 en 3+ features
  - Logging continuo de PnL live
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    PSI compara distribución 'expected' (training) vs 'actual' (live recent).
    Reglas heurísticas:
      PSI < 0.10 → estable
      PSI 0.10-0.25 → drift moderado, vigilar
      PSI > 0.25 → drift severo, retraining urgente
    """
    expected = np.asarray(expected, dtype=float); expected = expected[~np.isnan(expected)]
    actual   = np.asarray(actual,   dtype=float); actual   = actual[~np.isnan(actual)]
    if len(expected) < 50 or len(actual) < 20:
        return 0.0
    edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
    edges[0]  -= 1e-9
    edges[-1] += 1e-9
    e_hist, _ = np.histogram(expected, edges)
    a_hist, _ = np.histogram(actual,   edges)
    e_pct = e_hist / e_hist.sum()
    a_pct = a_hist / a_hist.sum()
    eps = 1e-6
    psi = np.sum((a_pct - e_pct) * np.log((a_pct + eps) / (e_pct + eps)))
    return float(round(psi, 4))


def kill_switch_check(state: dict) -> tuple[bool, str]:
    """Devuelve (should_kill, reason). Stub Fase 0."""
    if state.get("daily_dd_pct", 0) <= -2.0:
        return True, "daily DD <= -2%"
    if state.get("losses_streak", 0) >= 3:
        return True, "3 SLs consecutivos"
    return False, "OK"
