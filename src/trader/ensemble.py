"""
src/trader/ensemble.py
Bayesian-style ensemble: combina señal del modelo XGB con el Intelligence Engine,
ponderando por la accuracy histórica de cada uno.

Jerarquía de señales (post-consolidación):
  1. Intelligence Engine (5-agent debate): fuente primaria
     - p_ie: BUY=0.5+conf/2, SELL=0.5-conf/2, HOLD=0.5
     - Peso base: 0.20 (antes de hit-rate adjustment)
  2. Modelo XGB: señal cuantitativa secundaria
     - p_model = P(precio sube en 14d) ∈ [0,1]
     - Peso ajustado por hit-rate histórico

  Si el IE tiene veredicto claro (conf > 0.5), domina la señal.
  Si no hay IE disponible, fallback al modelo como antes.

Salida: dict con p_model, p_ie, p_ensemble, signal_ensemble, etc.
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
_IE_BASE_WEIGHT = 0.20  # IE gets base weight even without hit-rate history

_STANCE_MAP = {
    "BULLISH": 0.65,
    "NEUTRAL": 0.50,
    "BEARISH": 0.35,
}

_IE_VERDICT_PATH = os.path.join(_PROJECT_ROOT, "data", "intelligence_engine_verdict.json")


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


def _get_ie_probability() -> Optional[tuple]:
    """
    Lee el IE verdict y lo convierte a probabilidad.
    BUY  → 0.5 + confidence/2  (ej: conf=0.7 → p=0.85)
    SELL → 0.5 - confidence/2  (ej: conf=0.7 → p=0.15)
    HOLD → 0.5
    Returns (p_ie, confidence, verdict) or None if unavailable/stale.
    """
    if not os.path.exists(_IE_VERDICT_PATH):
        return None
    try:
        from datetime import datetime, timedelta
        data = json.loads(open(_IE_VERDICT_PATH, "r", encoding="utf-8").read())
        # Check freshness (48h max)
        ts = data.get("timestamp", "")
        if ts:
            ie_dt = datetime.fromisoformat(ts)
            if (datetime.now() - ie_dt) > timedelta(hours=48):
                return None
        v = data.get("verdict", {})
        verdict = v.get("verdict", "HOLD")
        conf = float(v.get("confidence", 0.5))
        if verdict == "BUY":
            p_ie = 0.5 + conf / 2
        elif verdict == "SELL":
            p_ie = 0.5 - conf / 2
        else:
            p_ie = 0.5
        return (float(np.clip(p_ie, 0.0, 1.0)), conf, verdict)
    except Exception:
        return None


def ensemble_signal(p_model: float,
                    llm_stance: Optional[str] = None,
                    llm_conviction: Optional[float] = None) -> dict:
    """Combina probabilidad del modelo con IE verdict (primario) y LLM stance."""
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

    # ── IE verdict integration (primary signal) ──
    ie_result = _get_ie_probability()
    p_ie = None
    w_ie = 0.0
    ie_verdict = None
    ie_confidence = None

    if ie_result is not None:
        p_ie, ie_confidence, ie_verdict = ie_result
        # IE gets base weight + any hit-rate bonus
        w_ie = _IE_BASE_WEIGHT
        # TODO: add IE hit-rate from debate_repository when enough data

    # ── Ensemble calculation ──
    total_w = w_m + w_l + w_ie
    if total_w <= 1e-9:
        # No weights established: IE verdict dominates if available, else model
        if p_ie is not None:
            p_ens = p_ie
        else:
            p_ens = p_model
    else:
        # Weighted average of all available signals
        numerator = w_m * p_model
        if p_llm is not None:
            numerator += w_l * p_llm
        else:
            total_w -= w_l
        if p_ie is not None:
            numerator += w_ie * p_ie
        else:
            total_w -= w_ie
        p_ens = numerator / total_w if total_w > 0 else p_model

    # Disagreement: between model and IE (primary comparison)
    if p_ie is not None:
        disagreement = abs(p_model - p_ie)
    elif p_llm is not None:
        disagreement = abs(p_model - p_llm)
    else:
        disagreement = 0.0

    # No longer ABSTAIN on disagreement — IE verdict is authoritative
    # Only abstain if there's no IE and model/LLM disagree strongly
    abstain = (p_ie is None) and (p_llm is not None) and (abs(p_model - p_llm) > _DISAGREE_GATE)

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
        "p_llm":            round(p_llm, 4) if p_llm is not None else None,
        "p_ie":             round(p_ie, 4) if p_ie is not None else None,
        "p_ensemble":       round(float(p_ens), 4),
        "w_model":          round(w_m, 4),
        "w_llm":            round(w_l, 4),
        "w_ie":             round(w_ie, 4),
        "ie_verdict":       ie_verdict,
        "ie_confidence":    round(ie_confidence, 4) if ie_confidence is not None else None,
        "model_hit_rate":   round(hr_m, 4) if hr_m is not None else None,
        "llm_hit_rate":     round(hr_l, 4) if hr_l is not None else None,
        "disagreement":     round(disagreement, 4),
        "abstain":          bool(abstain),
        "signal_ensemble":  signal,
        "signal_source":    "intelligence_engine" if (p_ie is not None and w_ie > 0) else "ml_model",
        "agreement_score":  round(1.0 - disagreement, 4),
    }


if __name__ == "__main__":
    print(json.dumps(ensemble_signal(0.62, "BULLISH", 0.7), indent=2))
    print(json.dumps(ensemble_signal(0.62, "BEARISH", 0.7), indent=2))  # disagreement
