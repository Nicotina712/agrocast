"""
src/model/regime_switching.py
Markov-Switching Regression de retornos diarios para soja.

Idea: los parámetros del proceso de retornos cambian entre regímenes
(bajo / alto vol). Un solo modelo lineal no captura: en régimen tendencial
domina momentum; en lateral, mean-reversion.

Implementación: statsmodels MarkovRegression sobre log-returns con K=2 estados.
Output: prob de cada estado hoy + parámetros condicionales + smoothed states
históricos.

Uso para forecast: combinar α blending del modelo horizons CON la prob de
estado actual:
  - Si P(estado=high_vol) > 0.7 → reducir α (más peso al RW), ampliar bandas
  - Si P(estado=low_vol)  > 0.7 → mantener α normal
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import json, os


def fit_regime_switching(df: pd.DataFrame, n_states: int = 2,
                          lookback_days: int = 1500) -> dict:
    """Fit Markov-Switching Regression sobre log-returns.

    Args:
        df: DataFrame con columnas Date y Soybeans
        n_states: número de regímenes (2 default)
        lookback_days: ventana de entrenamiento

    Returns:
        dict con prob de estado actual + parámetros + smoothed history
    """
    if df is None or df.empty or "Soybeans" not in df.columns:
        return {"ok": False, "error": "no data"}

    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except ImportError:
        return {"ok": False, "error": "statsmodels MarkovRegression not available"}

    f = df.sort_values("Date").tail(lookback_days).copy()
    f["log_ret"] = np.log(f["Soybeans"] / f["Soybeans"].shift(1))
    series = f["log_ret"].dropna()
    if len(series) < 200:
        return {"ok": False, "error": "serie muy corta"}

    try:
        model = MarkovRegression(
            series.values,
            k_regimes=n_states,
            trend="c",
            switching_variance=True,   # cada régimen tiene su σ
        )
        res = model.fit(disp=False, maxiter=200)
    except Exception as e:
        return {"ok": False, "error": f"fit error: {e}"}

    # Smoothed probabilities (P(estado | toda la serie))
    smoothed_probs = res.smoothed_marginal_probabilities
    # Acceso robusto: smoothed_probs puede ser DataFrame o ndarray
    if hasattr(smoothed_probs, "iloc"):
        # DataFrame: columnas = estados, filas = tiempo
        state_probs_today = {f"state_{i}": float(smoothed_probs.iloc[-1, i]) for i in range(n_states)}
    else:
        # ndarray (n_obs, n_states)
        state_probs_today = {f"state_{i}": float(smoothed_probs[-1, i]) for i in range(n_states)}

    # Parámetros por régimen — usar res.params como pandas Series indexada por nombre
    params_series = pd.Series(res.params, index=res.model.param_names) if not hasattr(res.params, "index") else res.params
    means  = []
    sigmas = []
    for i in range(n_states):
        # Buscar nombres con formato "const[i]" o "regime_i.const"
        const_key = next((k for k in params_series.index if f"[{i}]" in k and "const" in k), None)
        sig_key   = next((k for k in params_series.index if f"[{i}]" in k and "sigma" in k), None)
        if const_key is None:
            const_key = next((k for k in params_series.index if "const" in k.lower()), None)
        means.append(float(params_series[const_key]) if const_key else 0.0)
        sigmas.append(float(np.sqrt(params_series[sig_key])) if sig_key else 0.01)

    # Etiquetar: estado con menor σ = "low_vol"
    sigma_order = sorted(range(n_states), key=lambda i: sigmas[i])
    label_map = {sigma_order[0]: "low_vol"}
    if n_states == 2:
        label_map[sigma_order[1]] = "high_vol"
    elif n_states == 3:
        label_map[sigma_order[1]] = "medium_vol"
        label_map[sigma_order[2]] = "high_vol"

    state_probs_labeled = {label_map[i]: state_probs_today[f"state_{i}"] for i in range(n_states)}
    current_state_idx = max(range(n_states), key=lambda i: state_probs_today[f"state_{i}"])
    current_state = label_map[current_state_idx]

    # Matriz de transición (estados sin etiquetar)
    try:
        regime_trans = res.regime_transition.tolist()
        # Etiquetar
        labels_in_order = [label_map[i] for i in range(n_states)]
    except Exception:
        regime_trans = None
        labels_in_order = list(label_map.values())

    # Ajuste sugerido para α blending del modelo horizons
    # Si estamos en high_vol con alta convicción, bajamos α (más RW, más cauto)
    p_high = state_probs_labeled.get("high_vol", 0.0)
    if p_high > 0.7:
        alpha_adjustment = -0.20
        alpha_advice     = "reducir peso del modelo (régimen volátil)"
    elif p_high > 0.4:
        alpha_adjustment = -0.10
        alpha_advice     = "reducir peso del modelo levemente"
    else:
        alpha_adjustment = 0.0
        alpha_advice     = "mantener α nominal (régimen estable)"

    return {
        "ok":               True,
        "n_states":         n_states,
        "current_state":    current_state,
        "current_state_prob": round(state_probs_today[f"state_{current_state_idx}"], 3),
        "state_probs_today": {k: round(v, 3) for k, v in state_probs_labeled.items()},
        "regime_means":     [round(float(m), 6) for m in means],
        "regime_sigmas":    [round(s, 5) for s in sigmas],
        "regime_labels_order": labels_in_order,
        "transition_matrix": regime_trans,
        "alpha_adjustment": round(alpha_adjustment, 2),
        "alpha_advice":     alpha_advice,
        "n_obs_fitted":     int(len(series)),
        "log_likelihood":   round(float(res.llf), 2),
        "aic":              round(float(res.aic), 2),
        "bic":              round(float(res.bic), 2),
    }


def save_regime_switching(features: pd.DataFrame, artifacts_dir: str) -> dict:
    out = fit_regime_switching(features, n_states=2)
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(os.path.join(artifacts_dir, "regime_switching.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out
