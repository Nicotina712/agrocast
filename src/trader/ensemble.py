"""
src/trader/ensemble.py
Bayesian-style ensemble: combina señal del modelo XGB con stance del LLM,
ponderando por la accuracy histórica de cada uno.

Idea (sin sobre-ingeniería estadística):
  - El modelo emite p_model = P(precio sube en 14d) ∈ [0,1].
  - El LLM emite stance ∈ {BULLISH, NEUTRAL, BEARISH} → mapeado a p_llm:
        BULLISH=0.65, NEUTRAL=0.50, BEARISH=0.35.
    (los valores se modulan por la fuerza de convicción si está disponible)
  - Cada fuente tiene un *peso* derivado de su hit-rate histórico:
        w_i = max(hit_rate_i - 0.5, 0)    ∈ [0, 0.5]
    Si el modelo tiene 57% hit y el LLM 53%, → w_model=0.07, w_llm=0.03.
    Si una fuente tiene <50% hit, su peso es 0 (se ignora).
  - p_ensemble = (w_model·p_model + w_llm·p_llm) / (w_model + w_llm)
    Si ambos pesos son 0, devuelve p_model como fallback.

  Disagreement: |p_model - p_llm|. Si > 0.25 → ABSTAIN (HOLD forzado),
  porque las dos fuentes se contradicen y no hay convicción agregable.

Salida: dict con p_model, p_llm, p_ensemble, w_model, w_llm,
        signal_ensemble ∈ {BUY, SELL, HOLD, ABSTAIN}, agreement_score.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LLM_ACCT_PATH     = os.path.join(_PROJECT_ROOT, "data", "llm_accountability_summary.json")
_LLM_SNAPSHOTS     = os.path.join(_PROJECT_ROOT, "data", "llm_snapshots.json")
_MODEL_HISTORY     = os.path.join(_PROJECT_ROOT, "artifacts", "model_oos_history.json")

_BUY_THRESH    = 0.58
_SELL_THRESH   = 0.42
_DISAGREE_GATE = 0.25

_STANCE_MAP = {
    "BULLISH": 0.65,
    "NEUTRAL": 0.50,
    "BEARISH": 0.35,
}


def _model_hit_rate() -> Optional[float]:
    if os.path.exists(_MODEL_HISTORY):
        try:
            j = json.loads(open(_MODEL_HISTORY).read())
            hr = j.get("recent_12m_hit") or j.get("oos_hit_rate")
            return float(hr) if hr is not None else None
        except Exception:
            pass
    artif = os.path.join(_PROJECT_ROOT, "artifacts", "returns_model.joblib")
    if os.path.exists(artif):
        try:
            import joblib
            saved = joblib.load(artif)
            return float(saved.get("val_acc")) if saved.get("val_acc") else None
        except Exception:
            return None
    return None


def _llm_hit_rate() -> Optional[float]:
    if os.path.exists(_LLM_ACCT_PATH):
        try:
            j = json.loads(open(_LLM_ACCT_PATH).read())
            return float(j.get("hit_rate")) if j.get("hit_rate") is not None else None
        except Exception:
            pass
    if os.path.exists(_LLM_SNAPSHOTS):
        try:
            snaps = json.loads(open(_LLM_SNAPSHOTS).read())
            verified = [s for s in snaps if s.get("hit") in (True, False)]
            if not verified:
                return None
            return sum(1 for s in verified if s["hit"]) / len(verified)
        except Exception:
            return None
    return None


def _llm_stance_to_prob(stance: str, conviction: Optional[float] = None) -> float:
    base = _STANCE_MAP.get((stance or "NEUTRAL").upper(), 0.50)
    if conviction is None:
        return base
    # conviction ∈ [0,1] amplifica desvío de 0.5
    delta = (base - 0.5) * float(np.clip(conviction, 0, 1))
    return float(0.5 + 2 * delta) if abs(delta) > 0 else base


def _weight(hit_rate: Optional[float]) -> float:
    if hit_rate is None:
        return 0.0
    return max(0.0, float(hit_rate) - 0.50)


def ensemble_signal(p_model: float,
                    llm_stance: Optional[str] = None,
                    llm_conviction: Optional[float] = None) -> dict:
    """Combina probabilidad del modelo con stance del LLM."""
    p_model = float(np.clip(p_model, 0.0, 1.0))
    hr_m    = _model_hit_rate()
    hr_l    = _llm_hit_rate()
    w_m     = _weight(hr_m)
    w_l     = _weight(hr_l)

    if llm_stance is None:
        p_llm = None
        w_l   = 0.0
    else:
        p_llm = _llm_stance_to_prob(llm_stance, llm_conviction)

    if (w_m + w_l) <= 1e-9:
        p_ens = p_model
    elif p_llm is None:
        p_ens = p_model
    else:
        p_ens = (w_m * p_model + w_l * p_llm) / (w_m + w_l)

    disagreement = abs(p_model - p_llm) if p_llm is not None else 0.0
    abstain      = (p_llm is not None) and (disagreement > _DISAGREE_GATE)

    if abstain:
        signal = "ABSTAIN"
    elif p_ens > _BUY_THRESH:
        signal = "BUY"
    elif p_ens < _SELL_THRESH:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "p_model":          round(p_model, 4),
        "p_llm":             round(p_llm, 4) if p_llm is not None else None,
        "p_ensemble":        round(float(p_ens), 4),
        "w_model":           round(w_m, 4),
        "w_llm":             round(w_l, 4),
        "model_hit_rate":    round(hr_m, 4) if hr_m is not None else None,
        "llm_hit_rate":      round(hr_l, 4) if hr_l is not None else None,
        "disagreement":      round(disagreement, 4),
        "abstain":           bool(abstain),
        "signal_ensemble":   signal,
        "agreement_score":   round(1.0 - disagreement, 4),
    }


if __name__ == "__main__":
    print(json.dumps(ensemble_signal(0.62, "BULLISH", 0.7), indent=2))
    print(json.dumps(ensemble_signal(0.62, "BEARISH", 0.7), indent=2))  # disagreement
