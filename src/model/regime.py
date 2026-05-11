"""
src/model/regime.py
Detector de régimen de mercado para soja.

Régimen explica MUCHO sobre cuándo el modelo es útil:
  - calm     → momentum/carry domina, modelo aporta lift moderado
  - choppy   → ruido domina, modelo cerca del RW (α bajo)
  - shock    → cambio estructural, modelo incierto pero direccional sí

Implementación simple y honesta basada en señales que ya tenemos en features.csv:
  - vol_30d (volatilidad realizada)
  - cvol_iv_atm (IV ATM si disponible)
  - news_velocity_7d (picos de cobertura preceden volatilidad)
  - drift de la MA90 (tendencial vs lateral)

No hay HMM aquí — es un router determinista entrenado por percentiles históricos.
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd

REGIMES = ["calm", "trend", "choppy", "shock"]


def detect_regime(features: pd.DataFrame, lookback_days: int = 252) -> dict:
    """Clasifica el régimen actual usando los últimos `lookback_days` para
    fijar los umbrales (percentiles 33/66 sobre vol y velocity).

    Retorna dict con régimen, scores y explicación legible.
    """
    if features is None or features.empty:
        return {"regime": "unknown", "ok": False, "reason": "no features"}

    f = features.sort_values("Date").copy()
    last = f.iloc[-1]

    # ── Inputs (con fallback a 0 si la columna no existe) ──
    # Usamos vol_60d en lugar de vol_30d: H1 mostró que la ventana de 60d es
    # mejor predictor de vol futura (+25 % MAE vs 30d). Mantenemos vol_30d
    # como referencia secundaria.
    vol_col   = "vol_60d" if "vol_60d" in f.columns else "vol_30d"
    vol_now    = float(last.get(vol_col, 0) or 0)
    velocity   = float(last.get("news_velocity_7d", 1) or 1)
    iv         = float(last.get("cvol_iv_atm", 0) or 0)
    ma90_slope = float(last.get("ma90_slope", 0) or 0)
    pvma90     = float(last.get("price_vs_ma90", 0) or 0)

    # ── Umbrales históricos sobre lookback ──
    hist = f.tail(lookback_days)
    vol_p33 = float(hist[vol_col].quantile(0.33)) if vol_col in hist.columns else 0.01
    vol_p66 = float(hist[vol_col].quantile(0.66)) if vol_col in hist.columns else 0.02
    vel_p66 = float(hist["news_velocity_7d"].quantile(0.66)) if "news_velocity_7d" in hist.columns else 1.5

    # ── Clasificación ──
    is_high_vol     = vol_now >= vol_p66
    is_news_spike   = velocity >= vel_p66
    is_trending     = abs(pvma90) > 0.04 and abs(ma90_slope) > 0.001
    is_low_vol      = vol_now <= vol_p33

    if is_high_vol and is_news_spike:
        regime = "shock"
        explanation = "Volatilidad alta + cobertura de noticias acelerada → cambio estructural probable."
        confidence_hint = "low"
    elif is_high_vol:
        regime = "choppy"
        explanation = "Alta volatilidad sin spike de noticias → mercado errático, predicciones poco confiables."
        confidence_hint = "low"
    elif is_trending:
        regime = "trend"
        explanation = f"Tendencia {'alcista' if ma90_slope > 0 else 'bajista'} sostenida — modelo aporta dirección."
        confidence_hint = "medium"
    elif is_low_vol:
        regime = "calm"
        explanation = "Volatilidad baja, mercado lateral — predicciones más estables."
        confidence_hint = "medium"
    else:
        regime = "calm"
        explanation = "Régimen normal sin extremos."
        confidence_hint = "medium"

    return {
        "ok": True,
        "regime":         regime,
        "explanation":    explanation,
        "confidence_hint": confidence_hint,
        "signals": {
            "vol_window":        vol_col,
            "vol_now":           round(vol_now, 4),
            "vol_p33":           round(vol_p33, 4),
            "vol_p66":           round(vol_p66, 4),
            "news_velocity_7d":  round(velocity, 2),
            "iv_atm":            round(iv, 4),
            "ma90_slope":        round(ma90_slope, 5),
            "price_vs_ma90":     round(pvma90, 4),
        },
        "as_of": pd.Timestamp.now().isoformat(timespec="seconds"),
    }


def confidence_score(alpha: float | None, regime: str | None,
                     coverage_pct: float | None) -> dict:
    """Combina α del modelo, régimen y cobertura empírica en un semáforo
    de confianza para la UI del productor.

    Salida:
        level   : "high" | "medium" | "low"
        label   : texto legible
        reason  : razón principal
    """
    a = alpha if alpha is not None else 0.0
    cov = coverage_pct if coverage_pct is not None else 0.0
    reg = regime or "unknown"

    # Reglas explicables — no overfittear acá, simple es mejor
    if reg == "shock":
        return {"level": "low",
                "label": "Confianza baja — cambio de régimen detectado",
                "reason": "El mercado está en shock. Bandas más anchas; tomar la dirección como hipótesis, no certeza."}
    if a >= 0.40 and 70 <= cov <= 90 and reg in ("trend", "calm"):
        return {"level": "high",
                "label": "Confianza alta — modelo aporta dirección con margen",
                "reason": f"α={a:.2f} (modelo pesa fuerte) y banda calibrada en {cov:.0f}%."}
    if a >= 0.15 and 60 <= cov <= 95:
        return {"level": "medium",
                "label": "Confianza media — usar como una señal entre varias",
                "reason": f"α={a:.2f} (aporte moderado) y cobertura {cov:.0f}%."}
    if a < 0.15:
        return {"level": "low",
                "label": "Confianza baja — modelo casi neutral",
                "reason": (f"α={a:.2f}: el modelo apenas le gana al baseline en este régimen. "
                           "Confiar más en banda y régimen que en el precio puntual.")}
    return {"level": "medium", "label": "Confianza media", "reason": ""}


def save_regime(features: pd.DataFrame, artifacts_dir: str) -> dict:
    """Persiste el régimen actual (heurístico + HMM) a `artifacts/regime.json`."""
    out = detect_regime(features)
    # Anexar régimen probabilistico HMM (no bloqueante)
    try:
        out["hmm"] = hmm_regime(features)
    except Exception as _e:
        out["hmm"] = {"ok": False, "error": str(_e)}
    os.makedirs(artifacts_dir, exist_ok=True)
    path = os.path.join(artifacts_dir, "regime.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    return out


# ─────────────────────────────────────────────────────────────────
# Modelo HMM 2-estados (Gaussian Mixture + transitions)
# ─────────────────────────────────────────────────────────────────
def hmm_regime(features: pd.DataFrame, lookback_days: int = 1500,
               n_states: int = 3) -> dict:
    """Régimen probabilistico vía Gaussian Mixture (proxy de HMM Gaussiano).

    Aprende `n_states` clusters sobre (retorno, volatilidad) en una ventana
    larga (~6 años) y clasifica el período actual. Útil cuando el detector
    heurístico es ambiguo y queremos una probabilidad de pertenencia.

    n_states=2 suele dar low_vol / high_vol;
    n_states=3 separa low_vol / medium / crisis.
    """
    if features is None or features.empty or "Soybeans" not in features.columns:
        return {"ok": False, "reason": "no features"}

    f = features.sort_values("Date").tail(lookback_days).copy()
    if len(f) < 200:
        return {"ok": False, "reason": "lookback insuficiente"}

    # Features de régimen: retorno log y vol rolling
    f["log_ret"] = np.log(f["Soybeans"] / f["Soybeans"].shift(1))
    f["vol_20"] = f["log_ret"].rolling(20).std()
    f["abs_drift_60"] = f["log_ret"].rolling(60).mean().abs()
    df = f.dropna(subset=["log_ret", "vol_20", "abs_drift_60"]).copy()
    if len(df) < 100:
        return {"ok": False, "reason": "dropna dejó muestra muy chica"}

    X = df[["vol_20", "abs_drift_60"]].values

    try:
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=n_states, covariance_type="full",
                              random_state=42, n_init=3, max_iter=200)
        gmm.fit(X)
        labels = gmm.predict(X)
        probs  = gmm.predict_proba(X[-1:])[0]
    except Exception as _e:
        return {"ok": False, "reason": f"GMM falló: {_e}"}

    # Etiquetar estados por nivel de vol_20 medio
    state_vol_mean = []
    for s in range(n_states):
        mask = labels == s
        state_vol_mean.append((s, float(X[mask, 0].mean()) if mask.any() else 0.0))
    # Ordenar por vol creciente → state_0 = low_vol, state_n = high_vol
    state_vol_mean.sort(key=lambda x: x[1])
    state_label_map = {}
    if n_states == 2:
        names = ["low_vol", "high_vol"]
    elif n_states == 3:
        names = ["low_vol", "medium_vol", "high_vol"]
    else:
        names = [f"s{i}" for i in range(n_states)]
    for new_idx, (orig_idx, _v) in enumerate(state_vol_mean):
        state_label_map[orig_idx] = names[new_idx]

    # Estado actual = argmax de prob
    current_state = int(np.argmax(probs))
    current_label = state_label_map[current_state]
    current_proba = float(probs[current_state])

    # Matriz de transición empírica (frecuencia de cambios estado_t → estado_t+1)
    trans = np.zeros((n_states, n_states))
    for i in range(len(labels) - 1):
        trans[labels[i], labels[i + 1]] += 1
    trans = trans / np.maximum(trans.sum(axis=1, keepdims=True), 1)

    # Distribución estable de estados
    state_freq = {state_label_map[s]: int((labels == s).sum()) for s in range(n_states)}

    return {
        "ok":              True,
        "n_states":        n_states,
        "lookback_days":   int(len(df)),
        "current_state":   current_label,
        "current_proba":   round(current_proba, 3),
        "state_probs":     {state_label_map[s]: round(float(probs[s]), 3) for s in range(n_states)},
        "state_freq":      state_freq,
        "transition_matrix": [[round(float(trans[i, j]), 3) for j in range(n_states)]
                              for i in range(n_states)],
        "transition_labels": [state_label_map[s] for s in range(n_states)],
        "explanation": (
            f"Mercado en estado '{current_label}' con probabilidad "
            f"{current_proba*100:.0f}%. "
            f"En la última ventana ({len(df)}d), "
            f"{state_freq[current_label]} días estuvieron en este régimen."
        ),
    }
