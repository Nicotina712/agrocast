"""
src/intel/hybrid_model.py
Modelo híbrido ML + Narrativa — combina predicciones del decision classifier
con inteligencia de eventos narrativos.

La hipótesis: el modelo ML (decision classifier, XGBClassifier sobre features
técnicos/macro) captura tendencia estructural, pero puede fallar en shocks
narrativos de corto plazo (oil/biofuel, geopolítica, etc.). El modelo híbrido
ajusta la señal ML con el "régimen narrativo" actual.

NO es un modelo aprendido (insuficientes datos para optimizar pesos).
Es una regla simple y transparente:

  hybrid_score = ml_score + α × narrative_adjustment
  narrative_adjustment = direction × strength × (1 - fade_risk)

Donde α (blending weight) se fija en 0.15 (conservador).

Backtest comparativo:
  always_sell   — baseline
  always_wait   — espera siempre
  ml_only       — decision classifier binario
  narrative_only — sigue la narrativa
  hybrid        — combinación ML + narrativa
  oracle        — información perfecta
"""
from __future__ import annotations
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.model.decision_classifier import (
    _build_decision_features,
    _compute_cost_pct,
    _resolve_profile,
    train_decision_classifier,
    HORIZONS_MULTI,
    PRODUCER_PROFILES,
)
from src.intel.event_intelligence import (
    build_event_memory,
    _classify_event,
    _safe_zscore,
    _compute_fade_risk,
)

ALPHA = 0.15   # blending weight (conservador)


def _narrative_adjustment(event_info: dict) -> float:
    """Calcula el ajuste narrativo en [-1, 1].
    Positivo = bullish (esperar puede pagar), negativo = bearish (vender ya).
    """
    dir_map = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
    direction = dir_map.get(event_info.get("direction", "neutral"), 0.0)
    strength  = event_info.get("narrative_strength", 0.0)
    fade      = event_info.get("fade_risk", 0.5)
    return direction * strength * (1.0 - fade)


def hybrid_verdict(
    ml_p_wait: float,
    event_info: dict,
    analogs: dict | None = None,
    alpha: float = ALPHA,
) -> dict:
    """Combina señal ML + narrativa en un veredicto híbrido.

    Parámetros:
        ml_p_wait:   P(WAIT pague) del decision classifier [0, 1]
        event_info:  dict de event_intelligence.detect_current_event()
        analogs:     dict de narrative_analogs (opcional)
        alpha:       peso del ajuste narrativo

    Returns dict con:
        hybrid_p_wait, ml_p_wait, narrative_adj, verdict, explanation
    """
    adj = _narrative_adjustment(event_info)
    hybrid_p = float(np.clip(ml_p_wait + alpha * adj, 0.0, 1.0))

    # Veredicto
    if hybrid_p >= 0.60:
        verdict = "WAIT"
        verdict_es = "ESPERAR"
    elif hybrid_p <= 0.40:
        verdict = "SELL"
        verdict_es = "VENDER"
    else:
        verdict = "INDIFFERENT"
        verdict_es = "SIN CONVICCION"

    # Confianza
    delta_p = abs(hybrid_p - 0.5)
    confidence = "high" if delta_p >= 0.20 else "medium" if delta_p >= 0.10 else "low"

    # Agreement / contradiction
    ml_dir = "bullish" if ml_p_wait >= 0.55 else "bearish" if ml_p_wait <= 0.45 else "neutral"
    nar_dir = event_info.get("direction", "neutral")

    if ml_dir == nar_dir:
        alignment = "agree"
    elif ml_dir == "neutral" or nar_dir == "neutral":
        alignment = "partial"
    else:
        alignment = "contradict"

    # Explanation
    dir_es = {"bullish": "alcista", "bearish": "bajista", "neutral": "neutral"}.get(nar_dir, nar_dir)
    etype = event_info.get("event_type", "?")

    if alignment == "contradict":
        expl = (f"El modelo ML indica {ml_dir} (P={ml_p_wait:.2f}) pero "
                f"el flujo narrativo ({etype}) es {dir_es} "
                f"(strength={event_info.get('narrative_strength', 0):.2f}, "
                f"fade_risk={event_info.get('fade_risk', 0):.2f}). "
                f"Ajuste: {adj:+.3f} -> P(WAIT) hibrido={hybrid_p:.2f}.")
    elif alignment == "agree":
        expl = (f"ML y narrativa coinciden en direccion {dir_es}. "
                f"P(WAIT) ML={ml_p_wait:.2f}, ajuste narrativo={adj:+.3f}, "
                f"hibrido={hybrid_p:.2f}. Conviccion reforzada.")
    else:
        expl = (f"ML neutral (P={ml_p_wait:.2f}), narrativa {dir_es} ({etype}). "
                f"Ajuste narrativo={adj:+.3f}, hibrido={hybrid_p:.2f}.")

    result = {
        "ok":                   True,
        "hybrid_p_wait":        round(hybrid_p, 3),
        "ml_p_wait":            round(ml_p_wait, 3),
        "narrative_adjustment": round(adj, 4),
        "alpha":                alpha,
        "verdict":              verdict,
        "verdict_es":           verdict_es,
        "confidence":           confidence,
        "alignment":            alignment,
        "event_type":           etype,
        "event_direction":      nar_dir,
        "narrative_strength":   event_info.get("narrative_strength", 0),
        "fade_risk":            event_info.get("fade_risk", 0),
        "explanation":          expl,
    }

    if analogs and analogs.get("ok"):
        result["analog_bullish_7d_pct"] = analogs.get("bullish_pct_7d")
        result["analog_fade_rate_pct"]  = analogs.get("fade_rate_7d_pct")
        result["analog_narrative"]      = analogs.get("narrative")

    return result


def backtest_hybrid(
    df: pd.DataFrame,
    profile_name: str = "default",
    horizon_days: int = 15,
    test_months: int = 12,
    alpha: float = ALPHA,
    price_per_ton: float | None = None,
) -> dict:
    """Backtest comparativo: ML-only vs Narrative-only vs Hybrid vs baselines.

    Entrena ML en datos anteriores al test period (OOS limpio).
    En cada ventana no solapada del test period:
      1. Predice P(WAIT) con ML
      2. Calcula narrative_adjustment del estado de ese dia
      3. Combina en hybrid_p_wait
      4. Compara 6 estrategias vs outcome real.
    """
    d = _build_decision_features(df).sort_values("Date").reset_index(drop=True)
    profile = _resolve_profile(profile_name)

    if price_per_ton is None:
        last_p = float(d.iloc[-1]["Soybeans"])
        price_per_ton = last_p * 0.01 * 36.7437

    cost_pct = _compute_cost_pct(
        profile["storage"], profile["financing"],
        price_per_ton, profile.get("quality_risk_per_month", 0.0),
        horizon_days=horizon_days,
    )

    cutoff = d["Date"].max() - pd.DateOffset(months=test_months)

    # Forward returns
    ret_col = f"ret_{horizon_days}d_fwd"
    if ret_col not in d.columns:
        d[ret_col] = d["Soybeans"].pct_change(horizon_days).shift(-horizon_days)

    # Train ML model on data before test period
    train_df = d[d["Date"] < cutoff].copy()
    bundle = train_decision_classifier(train_df, cost_pct=cost_pct, horizon_days=horizon_days)
    if not bundle.get("ok"):
        return {"ok": False, "error": "ML training failed"}

    model = bundle["model"]
    calibrator = bundle.get("calibrator")
    feats = bundle["features"]

    # Z-scores for narrative state (computed over full history for consistency)
    for col in ["Oil_chg7", "Oil_chg1", "Dollar_chg7", "Dollar_chg1",
                "mom_5d", "mom_20d", "news_sentiment", "news_velocity_7d"]:
        if col in d.columns:
            d[f"z_{col}"] = _safe_zscore(d[col])

    # Test period
    test_mask = (d["Date"] >= cutoff) & d[ret_col].notna()
    test_df = d[test_mask].copy().reset_index(drop=True)

    if len(test_df) < 3:
        return {"ok": False, "error": "insufficient test data"}

    # ML predictions
    X_test = test_df[feats].fillna(0).replace([np.inf, -np.inf], 0)
    p_raw = model.predict_proba(X_test)[:, 1].astype(float)
    p_cal = np.clip(calibrator.transform(p_raw), 0, 1) if calibrator else p_raw

    # Non-overlapping windows
    selected = []
    i = 0
    while i < len(test_df):
        selected.append(i)
        curr = test_df.iloc[i]["Date"]
        nxt = curr + pd.Timedelta(days=horizon_days)
        j = i + 1
        while j < len(test_df) and test_df.iloc[j]["Date"] < nxt:
            j += 1
        i = j

    records = []
    for i in selected:
        row = test_df.iloc[i]
        ret_actual = float(row[ret_col])
        delta_actual = ret_actual - cost_pct

        ml_p = float(p_cal[i])

        # Narrative state at this date
        row_dict = row.to_dict()
        z_scores = {}
        for col in ["Oil_chg7", "Oil_chg1", "Dollar_chg7", "Dollar_chg1",
                     "mom_5d", "mom_20d", "news_sentiment", "news_velocity_7d"]:
            z_col = f"z_{col}"
            if z_col in row_dict:
                z_scores[col] = float(row_dict[z_col])

        event_info = _classify_event(row_dict, z_scores)
        adj = _narrative_adjustment(event_info)
        hybrid_p = float(np.clip(ml_p + alpha * adj, 0.0, 1.0))

        # Narrative-only decision: direction + strength based
        nar_dir = 1.0 if event_info["direction"] == "bullish" else (
            -1.0 if event_info["direction"] == "bearish" else 0.0)
        nar_strength = event_info["narrative_strength"]
        nar_wait = nar_dir > 0 and nar_strength > 0.3

        # PnL vs always-sell (0 = baseline)
        strategies = {
            "always_sell":     0.0,
            "always_wait":     delta_actual,
            "ml_only":         delta_actual if ml_p >= 0.5 else 0.0,
            "narrative_only":  delta_actual if nar_wait else 0.0,
            "hybrid":          delta_actual if hybrid_p >= 0.5 else 0.0,
            "oracle":          max(0.0, delta_actual),
        }

        records.append({
            "date":           str(row["Date"])[:10],
            "ret_actual_pct": round(ret_actual * 100, 3),
            "delta_pct":      round(delta_actual * 100, 3),
            "ml_p_wait":      round(ml_p, 3),
            "nar_adj":        round(adj, 4),
            "hybrid_p_wait":  round(hybrid_p, 3),
            "event_type":     event_info["event_type"],
            "direction":      event_info["direction"],
            "nar_strength":   event_info["narrative_strength"],
            "fade_risk":      event_info["fade_risk"],
            "alignment":      "agree" if (ml_p >= 0.5) == nar_wait else "contradict",
            **{f"pnl_{s}": round(v * price_per_ton, 3) for s, v in strategies.items()},
        })

    n = len(records)
    if n < 2:
        return {"ok": False, "error": "insufficient decisions"}

    # Aggregate strategies
    strat_names = ["always_sell", "always_wait", "ml_only", "narrative_only", "hybrid", "oracle"]
    test_span = (pd.Timestamp(records[-1]["date"]) - pd.Timestamp(records[0]["date"])).days or 1
    dec_per_year = n / (test_span / 365.25)

    summary = {}
    for s in strat_names:
        vals = [r[f"pnl_{s}"] for r in records]
        mean_v = float(np.mean(vals))
        summary[s] = {
            "n_decisions":              n,
            "mean_pnl_usd_ton":         round(mean_v, 2),
            "total_pnl_usd_ton":        round(float(np.sum(vals)), 2),
            "win_rate_pct":             round(float(np.mean([v > 0 for v in vals])) * 100, 1),
            "annualized_excess_usd_ton": round(mean_v * dec_per_year, 2),
            "pnl_q10_usd_ton":          round(float(np.quantile(vals, 0.10)), 2),
            "pnl_q90_usd_ton":          round(float(np.quantile(vals, 0.90)), 2),
        }

    # Agreement analysis: when hybrid disagrees with ML, who wins?
    disagree = [r for r in records if r["alignment"] == "contradict"]
    agree    = [r for r in records if r["alignment"] == "agree"]

    agreement_analysis = {
        "n_agree":     len(agree),
        "n_disagree":  len(disagree),
        "disagree_hybrid_mean": round(float(np.mean([r["pnl_hybrid"] for r in disagree])), 2) if disagree else None,
        "disagree_ml_mean":     round(float(np.mean([r["pnl_ml_only"] for r in disagree])), 2) if disagree else None,
    }

    return {
        "ok":               True,
        "as_of":            datetime.now().isoformat(timespec="seconds"),
        "profile_name":     profile_name,
        "horizon_days":     horizon_days,
        "test_months":      test_months,
        "cutoff_date":      str(cutoff)[:10],
        "n_decisions":      n,
        "price_per_ton":    round(price_per_ton, 2),
        "cost_pct":         round(cost_pct * 100, 3),
        "alpha":            alpha,
        "strategies":       summary,
        "agreement":        agreement_analysis,
        "sample_records":   records[:10],
        "context_note": (
            "Backtest OOS hibrido ML+Narrativa. El modelo hibrido ajusta la "
            "P(WAIT) del clasificador con un narrative_adjustment basado en "
            "evento detectado, strength y fade_risk. Alpha=" + str(alpha) +
            " (conservador). Panel INFORMATIVO, no decisor."
        ),
    }


def save_hybrid_backtest(
    df: pd.DataFrame,
    artifacts_dir: str,
    profile_name: str = "default",
    horizons: list[int] | None = None,
    test_months: int = 12,
) -> dict:
    """Ejecuta backtest hibrido para cada horizonte y persiste."""
    if horizons is None:
        horizons = HORIZONS_MULTI

    results = {"ok": True, "horizons": {}, "as_of": datetime.now().isoformat(timespec="seconds"),
               "profile_name": profile_name}

    for h in horizons:
        try:
            r = backtest_hybrid(df, profile_name=profile_name,
                                horizon_days=h, test_months=test_months)
            results["horizons"][f"{h}d"] = r
        except Exception as e:
            results["horizons"][f"{h}d"] = {"ok": False, "error": str(e)}

    out_dir = os.path.join(artifacts_dir, "hybrid_backtest")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{profile_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    return results
