"""
src/intel/narrative_forecast.py
Narrative-Only Forecast — rango esperado por horizonte dado el evento actual.

Usa análogos narrativos del event_memory para producir, por cada horizonte
(1d, 7d, 15d, 30d), la distribución empírica de outcomes:
  - Rango esperado (Q10, Q25, median, Q75, Q90)
  - P(up) / P(down)
  - Mean expected outcome (%)
  - Dirección dominante y confianza

El horizonte 1d es el más relevante para el productor porque responde
"qué esperar hoy/mañana dado el evento actual".

Hallazgo clave del análisis diario:
- Los eventos ESTRECHAN el rango diario (std 0.701% vs 0.847% sin evento)
- La direccionalidad a 1d es ~50-53% (apenas sobre coin flip)
- El valor informativo está en el RANGO, no en la dirección
"""
from __future__ import annotations
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.intel.event_intelligence import (
    build_event_memory,
    detect_current_event,
    _safe_zscore,
    _classify_event,
    NARRATIVE_STATE_FEATURES,
)


HORIZONS_NARRATIVE = [1, 7, 15, 30]


def narrative_range_forecast(
    current_event: dict,
    event_memory: pd.DataFrame,
    k: int = 20,
    min_gap_days: int = 7,
    price_usd_ton: float | None = None,
) -> dict:
    """Produce range forecast por horizonte usando análogos narrativos.

    Para cada horizonte, busca los k análogos más similares al estado actual
    y reporta la distribución empírica de sus outcomes.

    Returns dict con forecasts por horizonte + metadata.
    """
    if event_memory.empty or len(event_memory) < k:
        return {"ok": False, "error": "insufficient event memory"}

    state_cols = ["oil_chg7", "dollar_chg7", "mom_5d", "mom_20d", "rsi_14",
                  "vol_30d", "news_sentiment", "cot_noncomm_long_pct"]

    avail = [c for c in state_cols if c in event_memory.columns and c in current_event]
    if len(avail) < 3:
        return {"ok": False, "error": "insufficient state features"}

    em = event_memory.copy()
    em["date_dt"] = pd.to_datetime(em["date"])

    as_of = pd.Timestamp(current_event.get("as_of", datetime.now().strftime("%Y-%m-%d")))
    gap_date = as_of - pd.Timedelta(days=min_gap_days)
    em = em[em["date_dt"] < gap_date].copy()

    if len(em) < k:
        return {"ok": False, "error": "insufficient events after gap filter"}

    # Normalize features and find k nearest neighbors
    M = em[avail].fillna(0).replace([np.inf, -np.inf], 0).values
    means = M.mean(axis=0)
    stds = M.std(axis=0)
    stds[stds < 1e-9] = 1.0
    M_norm = (M - means) / stds

    v = np.array([float(current_event.get(c, 0)) for c in avail])
    v_norm = (v - means) / stds

    dists = np.sqrt(((M_norm - v_norm) ** 2).sum(axis=1))
    top_k_idx = np.argsort(dists)[:k]
    neighbors = em.iloc[top_k_idx]

    # Also find type-filtered analogs (same event_type)
    event_type = current_event.get("event_type", "mixed")
    type_mask = em["event_type"] == event_type
    type_filtered = em[type_mask]

    # Build forecasts per horizon
    forecasts = {}
    horizon_cols = {1: "outcome_1d_pct", 7: "outcome_7d_pct", 30: "outcome_30d_pct"}

    for h in HORIZONS_NARRATIVE:
        col = f"outcome_{h}d_pct" if f"outcome_{h}d_pct" in neighbors.columns else None
        if col is None:
            # Try closest available
            if h == 15 and "outcome_7d_pct" in neighbors.columns:
                col = "outcome_7d_pct"
            else:
                forecasts[f"{h}d"] = {"available": False}
                continue

        outcomes = neighbors[col].dropna().values
        if len(outcomes) < 5:
            forecasts[f"{h}d"] = {"available": False, "n_samples": len(outcomes)}
            continue

        # Type-filtered outcomes (if enough)
        type_outcomes = None
        if len(type_filtered) >= 10 and col in type_filtered.columns:
            to = type_filtered[col].dropna().values
            if len(to) >= 10:
                type_outcomes = to

        # Compute range statistics
        q10 = float(np.quantile(outcomes, 0.10))
        q25 = float(np.quantile(outcomes, 0.25))
        median = float(np.median(outcomes))
        q75 = float(np.quantile(outcomes, 0.75))
        q90 = float(np.quantile(outcomes, 0.90))
        mean_val = float(outcomes.mean())
        std_val = float(outcomes.std())
        p_up = float((outcomes > 0).mean())

        # Convert to USD/ton if price available
        usd_range = None
        if price_usd_ton:
            usd_range = {
                "q10": round(price_usd_ton * q10 / 100, 2),
                "q25": round(price_usd_ton * q25 / 100, 2),
                "median": round(price_usd_ton * median / 100, 2),
                "q75": round(price_usd_ton * q75 / 100, 2),
                "q90": round(price_usd_ton * q90 / 100, 2),
                "mean": round(price_usd_ton * mean_val / 100, 2),
            }

        # Direction and confidence
        if p_up >= 0.65:
            direction = "bullish"
            confidence = "medium" if p_up < 0.75 else "high"
        elif p_up <= 0.35:
            direction = "bearish"
            confidence = "medium" if p_up > 0.25 else "high"
        else:
            direction = "neutral"
            confidence = "low"

        fc = {
            "available": True,
            "n_samples": len(outcomes),
            "range_pct": {
                "q10": round(q10, 3),
                "q25": round(q25, 3),
                "median": round(median, 3),
                "q75": round(q75, 3),
                "q90": round(q90, 3),
            },
            "mean_pct": round(mean_val, 3),
            "std_pct": round(std_val, 3),
            "p_up": round(p_up, 3),
            "p_down": round(1 - p_up, 3),
            "direction": direction,
            "confidence": confidence,
            "usd_ton": usd_range,
        }

        # Add type-specific stats if available
        if type_outcomes is not None:
            fc["type_filtered"] = {
                "n_samples": len(type_outcomes),
                "mean_pct": round(float(type_outcomes.mean()), 3),
                "std_pct": round(float(type_outcomes.std()), 3),
                "p_up": round(float((type_outcomes > 0).mean()), 3),
                "q10": round(float(np.quantile(type_outcomes, 0.10)), 3),
                "q90": round(float(np.quantile(type_outcomes, 0.90)), 3),
            }

        forecasts[f"{h}d"] = fc

    # Build 1d narrative for the producer
    daily_fc = forecasts.get("1d", {})
    narrative = _build_daily_narrative(current_event, daily_fc, price_usd_ton)

    return {
        "ok": True,
        "as_of": current_event.get("as_of"),
        "price_usd_ton": round(price_usd_ton, 2) if price_usd_ton else None,
        "event_active": current_event.get("is_event", False),
        "event_type": event_type,
        "event_direction": current_event.get("direction", "neutral"),
        "narrative_strength": current_event.get("narrative_strength", 0),
        "fade_risk": current_event.get("fade_risk", 0),
        "forecasts": forecasts,
        "daily_narrative": narrative,
        "n_analogs_used": len(neighbors),
        "dist_median": round(float(np.median(dists[top_k_idx])), 3),
    }


def _build_daily_narrative(event: dict, daily_fc: dict, price_usd_ton: float | None) -> str:
    """Construye narrativa en español para el productor sobre qué esperar hoy/mañana."""
    if not daily_fc or not daily_fc.get("available"):
        return "Sin suficientes datos para estimar el rango diario."

    is_event = event.get("is_event", False)
    etype = event.get("event_type", "?")
    direction = event.get("direction", "neutral")
    dir_es = {"bullish": "alcista", "bearish": "bajista", "neutral": "neutral"}.get(direction, direction)

    type_labels = {
        "oil_energy": "energia/biofuel", "china_demand": "demanda China",
        "weather": "clima", "supply_surprise": "supply/WASDE",
        "policy_shift": "politica", "macro_fx": "macro/FX",
        "geopolitical": "geopolitica", "speculative_mom": "momentum especulativo",
        "mixed": "mixto",
    }

    rng = daily_fc["range_pct"]
    p_up = daily_fc["p_up"]

    parts = []
    if is_event:
        parts.append(f"Evento activo: {type_labels.get(etype, etype)} ({dir_es}).")
        parts.append(f"En situaciones similares, el rango diario esperado es "
                     f"{rng['q10']:+.2f}% a {rng['q90']:+.2f}% "
                     f"(mediana {rng['median']:+.2f}%).")
        if price_usd_ton and daily_fc.get("usd_ton"):
            u = daily_fc["usd_ton"]
            parts.append(f"En USD/ton: {u['q10']:+.1f} a {u['q90']:+.1f} USD/ton.")
        parts.append(f"Probabilidad de suba: {p_up*100:.0f}%.")
        if daily_fc.get("std_pct", 0) < 0.80:
            parts.append("Los eventos tienden a estrechar el rango (menor volatilidad que dias sin evento).")
    else:
        parts.append("Sin evento narrativo activo.")
        parts.append(f"Rango tipico diario: {rng['q10']:+.2f}% a {rng['q90']:+.2f}%.")

    return " ".join(parts)


def backtest_narrative_only(
    df: pd.DataFrame,
    horizon_days: int = 7,
    test_months: int = 12,
    price_per_ton: float | None = None,
    cost_pct: float = 0.0,
) -> dict:
    """Backtest de la estrategia narrative-only para un horizonte dado.

    Narrative-only: esperar cuando direction=bullish y narrative_strength>0.3.
    Incluye análisis diario (overlapping) para 1d.
    """
    from src.model.decision_classifier import _build_decision_features

    d = _build_decision_features(df).sort_values("Date").reset_index(drop=True)

    if price_per_ton is None:
        last_p = float(d.iloc[-1]["Soybeans"])
        price_per_ton = last_p * 0.01 * 36.7437

    cutoff = d["Date"].max() - pd.DateOffset(months=test_months)

    ret_col = f"ret_{horizon_days}d_fwd"
    if ret_col not in d.columns:
        d[ret_col] = d["Soybeans"].pct_change(horizon_days).shift(-horizon_days)

    # Z-scores for event detection
    for col in ["Oil_chg7", "Oil_chg1", "Dollar_chg7", "Dollar_chg1",
                "mom_5d", "mom_20d", "news_sentiment", "news_velocity_7d"]:
        if col in d.columns:
            d[f"z_{col}"] = _safe_zscore(d[col])

    # Test period (daily for 1d, non-overlapping for longer)
    test_mask = (d["Date"] >= cutoff) & d[ret_col].notna()
    test_df = d[test_mask].copy().reset_index(drop=True)

    if len(test_df) < 5:
        return {"ok": False, "error": "insufficient test data"}

    # For 1d, use all days (daily). For longer horizons, non-overlapping.
    if horizon_days <= 1:
        selected = list(range(len(test_df)))
    else:
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

        row_dict = row.to_dict()
        z_scores = {}
        for col in ["Oil_chg7", "Oil_chg1", "Dollar_chg7", "Dollar_chg1",
                    "mom_5d", "mom_20d", "news_sentiment", "news_velocity_7d"]:
            z_col = f"z_{col}"
            if z_col in row_dict:
                z_scores[col] = float(row_dict[z_col])

        event_info = _classify_event(row_dict, z_scores)
        max_z = max(abs(v) for v in z_scores.values()) if z_scores else 0
        is_event = max_z >= 1.5

        nar_bullish = event_info["direction"] == "bullish" and event_info["narrative_strength"] > 0.3
        nar_wait = is_event and nar_bullish

        records.append({
            "date": str(row["Date"])[:10],
            "ret_pct": round(ret_actual * 100, 3),
            "delta_pct": round(delta_actual * 100, 3),
            "is_event": is_event,
            "event_type": event_info["event_type"],
            "direction": event_info["direction"],
            "nar_strength": event_info["narrative_strength"],
            "nar_wait": nar_wait,
            "pnl_always_sell": 0.0,
            "pnl_always_wait": round(delta_actual * price_per_ton, 3),
            "pnl_narrative": round(delta_actual * price_per_ton, 3) if nar_wait else 0.0,
            "pnl_oracle": round(max(0.0, delta_actual) * price_per_ton, 3),
        })

    n = len(records)
    strats = ["always_sell", "always_wait", "narrative", "oracle"]
    summary = {}
    for s in strats:
        vals = [r[f"pnl_{s}"] for r in records]
        summary[s] = {
            "n_decisions": n,
            "mean_pnl_usd_ton": round(float(np.mean(vals)), 3),
            "total_pnl_usd_ton": round(float(np.sum(vals)), 2),
            "win_rate_pct": round(float(np.mean([v > 0 for v in vals])) * 100, 1),
        }

    # Event-active stats
    active_records = [r for r in records if r["is_event"]]
    inactive_records = [r for r in records if not r["is_event"]]

    range_stats = {
        "all_days_std_pct": round(float(np.std([r["ret_pct"] for r in records])), 3),
        "event_days_std_pct": round(float(np.std([r["ret_pct"] for r in active_records])), 3) if active_records else None,
        "no_event_days_std_pct": round(float(np.std([r["ret_pct"] for r in inactive_records])), 3) if inactive_records else None,
        "n_event_days": len(active_records),
        "n_no_event_days": len(inactive_records),
    }

    return {
        "ok": True,
        "horizon_days": horizon_days,
        "n_decisions": n,
        "test_months": test_months,
        "cutoff_date": str(cutoff)[:10],
        "price_per_ton": round(price_per_ton, 2),
        "strategies": summary,
        "range_stats": range_stats,
        "narrative_active_pct": round(len(active_records) / n * 100, 1) if n else 0,
    }


def save_narrative_forecast(
    df: pd.DataFrame,
    artifacts_dir: str,
    test_months: int = 12,
) -> dict:
    """Genera y persiste el forecast narrativo + backtest por horizonte."""
    from src.intel.event_intelligence import detect_current_event, save_event_memory

    # Ensure event_memory exists
    em_path = os.path.join(artifacts_dir, "event_memory.csv")
    if not os.path.exists(em_path):
        save_event_memory(df, artifacts_dir)

    em = pd.read_csv(em_path) if os.path.exists(em_path) else pd.DataFrame()

    # Detect current event
    event = detect_current_event(df)

    # Price
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    last_p = float(d.sort_values("Date").iloc[-1]["Soybeans"])
    price_usd_ton = last_p * 0.01 * 36.7437

    # Range forecast from analogs
    forecast = narrative_range_forecast(event, em, k=20, price_usd_ton=price_usd_ton)

    # Backtest per horizon
    backtests = {}
    for h in HORIZONS_NARRATIVE:
        try:
            bt = backtest_narrative_only(df, horizon_days=h, test_months=test_months,
                                         price_per_ton=price_usd_ton)
            backtests[f"{h}d"] = bt
        except Exception as e:
            backtests[f"{h}d"] = {"ok": False, "error": str(e)}

    result = {
        "ok": True,
        "as_of": datetime.now().isoformat(timespec="seconds"),
        "forecast": forecast,
        "backtests": backtests,
    }

    out_dir = os.path.join(artifacts_dir, "narrative_forecast")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "latest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    return result
