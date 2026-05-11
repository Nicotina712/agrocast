"""
src/intel/event_intelligence.py
Market Intelligence Engine — detección de eventos narrativos y event memory.

Detecta eventos de mercado a partir del estado narrativo (noticias, drivers,
técnico, cross-market) y construye una memoria persistente de eventos con
sus outcomes reales. Esto permite encontrar análogos narrativos y validar
si la capa de inteligencia aporta valor vs el modelo ML puro.

Event types:
  oil_energy       — shock energético transmitido via biofuel/crush
  china_demand     — cambio en demanda/sentimiento China
  weather          — clima en BR/AR/US
  supply_surprise  — WASDE, supply global
  policy_shift     — política comercial/agro
  macro_fx         — USD/macro driven
  geopolitical     — geopolítica
  speculative_mom  — momentum técnico + posicionamiento especulativo
  mixed            — múltiples drivers sin dominante claro
"""
from __future__ import annotations
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd


# ── Constantes ──────────────────────────────────────────────────────
EVENT_THRESHOLD_SIGMA = 1.5     # z-score mínimo para considerar un evento
FADE_RSI_HIGH = 70              # RSI sobrecomprado
FADE_RSI_LOW  = 30              # RSI sobrevendido
ANALOG_K      = 20              # vecinos por defecto

DRIVER_MAP = {
    "oil_energy":      ["Oil_chg7", "Oil_chg1"],
    "macro_fx":        ["Dollar_chg7", "Dollar_chg1"],
    "speculative_mom": ["mom_5d", "mom_20d"],
}

NEWS_DRIVER_MAP = {
    "oil_energy":    ["news_macro_oil_signal", "news_biofuels_signal"],
    "china_demand":  ["news_china_demand_signal"],
    "weather":       ["news_weather_br_signal", "news_weather_ar_signal", "news_weather_us_signal"],
    "supply_surprise": ["news_supply_global_signal", "news_usda_report_signal"],
    "policy_shift":  ["news_policy_us_signal", "news_policy_br_signal", "news_policy_ar_signal"],
    "macro_fx":      ["news_macro_usd_signal"],
    "geopolitical":  ["news_geopolitics_signal"],
}

# Features para el vector de estado narrativo (análogos)
NARRATIVE_STATE_FEATURES = [
    "Oil_chg7", "Dollar_chg7", "mom_5d", "mom_20d", "rsi_14",
    "vol_30d", "news_sentiment", "news_velocity_7d",
    "cot_noncomm_long_pct", "price_pct_in_12m_range",
]


def _safe_zscore(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    rm = series.rolling(window, min_periods=min_periods).mean().shift(1)
    rs = series.rolling(window, min_periods=min_periods).std().shift(1).replace(0, 1e-9)
    return ((series - rm) / rs).fillna(0).replace([np.inf, -np.inf], 0)


def _compute_fade_risk(row: dict) -> float:
    """Estima fade risk en [0, 1] basado en RSI + volatilidad + alignment."""
    rsi = row.get("rsi_14", 50)
    vol = row.get("vol_30d", 0)
    mom5 = row.get("mom_5d", 0)
    mom20 = row.get("mom_20d", 0)

    fade = 0.0
    # RSI extremo → mayor fade risk
    if rsi > FADE_RSI_HIGH:
        fade += 0.3 * min((rsi - FADE_RSI_HIGH) / 15, 1.0)
    elif rsi < FADE_RSI_LOW:
        fade += 0.3 * min((FADE_RSI_LOW - rsi) / 15, 1.0)

    # Momentum misalignment (5d vs 20d en distinta dirección)
    if mom5 * mom20 < 0:
        fade += 0.2

    # Alta volatilidad → mayor fade risk
    if vol > 0.02:
        fade += 0.15 * min(vol / 0.04, 1.0)

    # Movimiento reciente fuerte (overextension)
    if abs(mom5) > 0.03:
        fade += 0.2 * min(abs(mom5) / 0.06, 1.0)

    return float(np.clip(fade, 0.0, 0.95))


def _classify_event(row: dict, z_scores: dict) -> dict:
    """Clasifica el tipo de evento y driver dominante a partir de z-scores."""
    # Prioridad: el z-score más alto determina el driver primario
    driver_scores = {}

    # Price-based drivers (siempre disponibles)
    oil_z = max(abs(z_scores.get("Oil_chg7", 0)), abs(z_scores.get("Oil_chg1", 0)))
    fx_z  = max(abs(z_scores.get("Dollar_chg7", 0)), abs(z_scores.get("Dollar_chg1", 0)))
    mom_z = max(abs(z_scores.get("mom_5d", 0)), abs(z_scores.get("mom_20d", 0)))

    driver_scores["oil_energy"]      = oil_z
    driver_scores["macro_fx"]        = fx_z
    driver_scores["speculative_mom"] = mom_z

    # News-based drivers (disponibles solo si hay intel history)
    for etype, cols in NEWS_DRIVER_MAP.items():
        vals = [abs(row.get(c, 0)) for c in cols if c in row]
        if vals:
            driver_scores[etype] = max(driver_scores.get(etype, 0), max(vals) * 2.5)

    # Seleccionar primario y secundario
    sorted_drivers = sorted(driver_scores.items(), key=lambda x: x[1], reverse=True)
    primary   = sorted_drivers[0] if sorted_drivers else ("mixed", 0)
    secondary = sorted_drivers[1] if len(sorted_drivers) > 1 else ("mixed", 0)

    if primary[1] < 0.5:
        event_type = "mixed"
    else:
        event_type = primary[0]

    # Narrative strength = normalización del score dominante
    narrative_strength = float(np.clip(primary[1] / 3.0, 0.0, 1.0))

    # Speculation level (proxy: COT non-comm positioning extreme + momentum)
    cot = row.get("cot_noncomm_long_pct", 50)
    spec = 0.0
    if cot is not None and not np.isnan(cot):
        spec = abs(cot - 50) / 50  # 0 = neutral, 1 = extreme
    spec = float(np.clip((spec + min(mom_z / 3, 0.5)) / 1.5, 0.0, 1.0))

    # Data confirmation (inverse of speculation — si hay news high impact)
    n_high = row.get("intel_n_high_impact", 0)
    n_arts = row.get("intel_n_articles", 0)
    data_conf = float(np.clip(n_high / max(n_arts, 1) + 0.2 * min(n_arts / 20, 1), 0.0, 1.0))

    # Direction from the dominant driver
    if event_type == "oil_energy":
        direction = "bullish" if row.get("Oil_chg7", 0) > 0 else "bearish"
    elif event_type == "macro_fx":
        direction = "bearish" if row.get("Dollar_chg7", 0) > 0 else "bullish"
    elif event_type == "speculative_mom":
        direction = "bullish" if row.get("mom_5d", 0) > 0 else "bearish"
    else:
        ns = row.get("news_sentiment", 0)
        direction = "bullish" if ns > 0.02 else "bearish" if ns < -0.02 else "neutral"

    fade_risk = _compute_fade_risk(row)

    return {
        "event_type":         event_type,
        "primary_driver":     primary[0],
        "secondary_driver":   secondary[0],
        "direction":          direction,
        "narrative_strength":  round(narrative_strength, 3),
        "speculation_level":   round(spec, 3),
        "data_confirmation":   round(data_conf, 3),
        "fade_risk":           round(fade_risk, 3),
    }


def build_event_memory(df: pd.DataFrame, threshold_sigma: float = EVENT_THRESHOLD_SIGMA) -> pd.DataFrame:
    """Construye event memory retroactivo a partir de features.csv.

    Para cada día de trading, computa z-scores de drivers y detecta
    "eventos" donde algún z-score > threshold. Para cada evento,
    registra el estado narrativo + outcomes reales (ret_1d, 7d, 30d).

    Returns DataFrame con columnas:
        date, price, ret_1d, ret_7d, ret_30d, event_type, direction,
        primary_driver, secondary_driver, narrative_strength,
        speculation_level, data_confirmation, fade_risk,
        outcome_1d, outcome_7d, outcome_30d,
        fade_occurred_7d (bool: si revirtió >50% en 7d)
    """
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values("Date").reset_index(drop=True)

    # Z-scores de drivers principales
    z_cols = {}
    for col in ["Oil_chg7", "Oil_chg1", "Dollar_chg7", "Dollar_chg1",
                "mom_5d", "mom_20d", "news_sentiment", "news_velocity_7d"]:
        if col in d.columns:
            d[f"z_{col}"] = _safe_zscore(d[col])
            z_cols[col] = f"z_{col}"

    # Soy 1d return z-score (para detectar movimientos propios)
    if "Soybeans" in d.columns:
        d["soy_ret_1d"] = d["Soybeans"].pct_change()
        d["z_soy_ret_1d"] = _safe_zscore(d["soy_ret_1d"])
        z_cols["soy_ret_1d"] = "z_soy_ret_1d"

    # SoybeanOil change (biofuel proxy)
    if "SoybeanOil" in d.columns:
        d["soyoil_chg7"] = d["SoybeanOil"].pct_change(7)
        d["z_soyoil_chg7"] = _safe_zscore(d["soyoil_chg7"])
        z_cols["soyoil_chg7"] = "z_soyoil_chg7"

    # Detectar eventos: cualquier z-score > threshold
    z_col_names = list(z_cols.values())
    if not z_col_names:
        return pd.DataFrame()

    d["max_abs_z"] = d[z_col_names].abs().max(axis=1)
    events_mask = d["max_abs_z"] >= threshold_sigma

    # Necesitamos outcomes (ret_fwd)
    for col in ["ret_1d_fwd", "ret_7d_fwd", "ret_15d_fwd", "ret_30d_fwd"]:
        if col not in d.columns:
            h = int(col.split("_")[1].replace("d", ""))
            d[col] = d["Soybeans"].pct_change(h).shift(-h)

    # Filtrar: necesitamos al menos ret_1d_fwd para outcome
    events_mask = events_mask & d["ret_1d_fwd"].notna()

    records = []
    for idx in d.index[events_mask]:
        row = d.iloc[idx]
        row_dict = row.to_dict()
        zs = {k: float(row.get(v, 0)) for k, v in z_cols.items()}

        event_info = _classify_event(row_dict, zs)

        ret_1d  = row.get("ret_1d_fwd")
        ret_7d  = row.get("ret_7d_fwd")
        ret_15d = row.get("ret_15d_fwd")
        ret_30d = row.get("ret_30d_fwd")

        # Fade detection: ¿el movimiento de 1d se revirtió >50% en 7d?
        fade_7d = False
        if ret_1d is not None and ret_7d is not None and not np.isnan(ret_1d) and not np.isnan(ret_7d):
            if abs(ret_1d) > 0.002:
                fade_7d = (ret_1d * ret_7d < 0) or (abs(ret_7d) < abs(ret_1d) * 0.5)

        records.append({
            "date":               str(row["Date"])[:10],
            "price":              round(float(row.get("Soybeans", 0)), 2),
            "soy_ret_1d":         round(float(row.get("soy_ret_1d", 0)) * 100, 3),
            "event_type":         event_info["event_type"],
            "direction":          event_info["direction"],
            "primary_driver":     event_info["primary_driver"],
            "secondary_driver":   event_info["secondary_driver"],
            "narrative_strength":  event_info["narrative_strength"],
            "speculation_level":   event_info["speculation_level"],
            "data_confirmation":   event_info["data_confirmation"],
            "fade_risk":           event_info["fade_risk"],
            # State features for analog search
            "oil_chg7":           round(float(row.get("Oil_chg7", 0)), 4),
            "dollar_chg7":        round(float(row.get("Dollar_chg7", 0)), 4),
            "rsi_14":             round(float(row.get("rsi_14", 50)), 1),
            "vol_30d":            round(float(row.get("vol_30d", 0)), 4),
            "mom_5d":             round(float(row.get("mom_5d", 0)), 4),
            "mom_20d":            round(float(row.get("mom_20d", 0)), 4),
            "news_sentiment":     round(float(row.get("news_sentiment", 0)), 4),
            "cot_noncomm_long_pct": round(float(row.get("cot_noncomm_long_pct", 50)), 1),
            # Outcomes
            "outcome_1d_pct":     round(float(ret_1d) * 100, 3) if ret_1d is not None and not np.isnan(ret_1d) else None,
            "outcome_7d_pct":     round(float(ret_7d) * 100, 3) if ret_7d is not None and not np.isnan(ret_7d) else None,
            "outcome_15d_pct":    round(float(ret_15d) * 100, 3) if ret_15d is not None and not np.isnan(ret_15d) else None,
            "outcome_30d_pct":    round(float(ret_30d) * 100, 3) if ret_30d is not None and not np.isnan(ret_30d) else None,
            "fade_occurred_7d":   fade_7d,
        })

    em = pd.DataFrame(records)
    return em


def detect_current_event(df: pd.DataFrame, news_intel: dict | None = None) -> dict:
    """Detecta y caracteriza el evento actual (última fila de df).

    Parámetros:
        df: features DataFrame
        news_intel: dict con señales de news_intel.json (opcional, para enriquecer)

    Returns dict con event_info + analogs_ready flag.
    """
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values("Date").reset_index(drop=True)

    last = d.iloc[-1]
    row_dict = last.to_dict()

    # Enriquecer con news_intel si disponible
    if news_intel and isinstance(news_intel, dict):
        drivers = news_intel.get("drivers", {})
        for driver_name, driver_data in drivers.items():
            col = f"news_{driver_name}_signal"
            if isinstance(driver_data, dict):
                row_dict[col] = driver_data.get("signal", 0)

    # Z-scores al momento actual
    z_scores = {}
    for col in ["Oil_chg7", "Oil_chg1", "Dollar_chg7", "Dollar_chg1",
                "mom_5d", "mom_20d", "news_sentiment", "news_velocity_7d"]:
        if col in d.columns:
            zs = _safe_zscore(d[col])
            z_scores[col] = float(zs.iloc[-1])

    # Soy ret 1d
    if "Soybeans" in d.columns:
        ret = d["Soybeans"].pct_change()
        z_scores["soy_ret_1d"] = float(_safe_zscore(ret).iloc[-1])

    event_info = _classify_event(row_dict, z_scores)

    max_z = max(abs(v) for v in z_scores.values()) if z_scores else 0
    is_event = max_z >= EVENT_THRESHOLD_SIGMA

    # Include raw state features for analog search
    state_features = {}
    for col in ["Oil_chg7", "Dollar_chg7", "mom_5d", "mom_20d", "rsi_14",
                "vol_30d", "news_sentiment", "cot_noncomm_long_pct"]:
        if col in row_dict:
            v = row_dict[col]
            state_features[col.lower() if col[0].isupper() else col] = float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0.0
    # Normalize key names to match event_memory columns
    state_features["oil_chg7"] = state_features.pop("oil_chg7", float(row_dict.get("Oil_chg7", 0)))
    state_features["dollar_chg7"] = state_features.pop("dollar_chg7", float(row_dict.get("Dollar_chg7", 0)))

    return {
        "ok":              True,
        "is_event":        is_event,
        "as_of":           str(last["Date"])[:10],
        "price":           round(float(last.get("Soybeans", 0)), 2),
        "max_z_score":     round(max_z, 2),
        "z_scores":        {k: round(v, 2) for k, v in z_scores.items()},
        **event_info,
        **state_features,
    }


def find_narrative_analogs(
    current_state: dict,
    event_memory: pd.DataFrame,
    k: int = ANALOG_K,
    min_gap_days: int = 30,
) -> dict:
    """Busca análogos narrativos en event_memory basado en el estado actual.

    Usa distancia euclídea en z-score-normalized narrative state features.
    Filtra eventos demasiado recientes (min_gap_days).
    """
    if event_memory.empty or len(event_memory) < k:
        return {"ok": False, "n": 0, "error": "insufficient event memory"}

    state_cols = ["oil_chg7", "dollar_chg7", "mom_5d", "mom_20d", "rsi_14",
                  "vol_30d", "news_sentiment", "cot_noncomm_long_pct"]

    avail = [c for c in state_cols if c in event_memory.columns and c in current_state]
    if len(avail) < 3:
        return {"ok": False, "n": 0, "error": "insufficient state features"}

    em = event_memory.copy()
    em["date_dt"] = pd.to_datetime(em["date"])

    # Filtrar eventos muy recientes
    as_of = pd.Timestamp(current_state.get("as_of", datetime.now().strftime("%Y-%m-%d")))
    gap_date = as_of - pd.Timedelta(days=min_gap_days)
    em = em[em["date_dt"] < gap_date].copy()

    if len(em) < k:
        return {"ok": False, "n": 0, "error": "insufficient historical events after gap filter"}

    # Construir matriz de features + normalizar
    M = em[avail].fillna(0).replace([np.inf, -np.inf], 0).values
    means = M.mean(axis=0)
    stds  = M.std(axis=0)
    stds[stds < 1e-9] = 1.0
    M_norm = (M - means) / stds

    # Vector actual
    v = np.array([float(current_state.get(c, 0)) for c in avail])
    v_norm = (v - means) / stds

    # Distancias
    dists = np.sqrt(((M_norm - v_norm) ** 2).sum(axis=1))
    top_k_idx = np.argsort(dists)[:k]

    neighbors = em.iloc[top_k_idx]

    # Outcomes
    out_1d = neighbors["outcome_1d_pct"].dropna().values
    out_7d = neighbors["outcome_7d_pct"].dropna().values
    out_30d = neighbors["outcome_30d_pct"].dropna().values
    fade_rates = neighbors["fade_occurred_7d"].mean() if "fade_occurred_7d" in neighbors.columns else None

    # Event type distribution en análogos
    type_dist = neighbors["event_type"].value_counts().to_dict()

    # Dirección dominante en outcomes
    bullish_1d = float((out_1d > 0).mean()) if len(out_1d) else None
    bullish_7d = float((out_7d > 0).mean()) if len(out_7d) else None

    return {
        "ok":               True,
        "n":                len(neighbors),
        "k":                k,
        "dist_median":      round(float(np.median(dists[top_k_idx])), 3),
        "event_types":      type_dist,
        "outcome_1d_mean":  round(float(out_1d.mean()), 3) if len(out_1d) else None,
        "outcome_7d_mean":  round(float(out_7d.mean()), 3) if len(out_7d) else None,
        "outcome_30d_mean": round(float(out_30d.mean()), 3) if len(out_30d) else None,
        "outcome_1d_q10":   round(float(np.quantile(out_1d, 0.10)), 3) if len(out_1d) >= 5 else None,
        "outcome_1d_q90":   round(float(np.quantile(out_1d, 0.90)), 3) if len(out_1d) >= 5 else None,
        "outcome_7d_q10":   round(float(np.quantile(out_7d, 0.10)), 3) if len(out_7d) >= 5 else None,
        "outcome_7d_q90":   round(float(np.quantile(out_7d, 0.90)), 3) if len(out_7d) >= 5 else None,
        "bullish_pct_1d":   round(bullish_1d * 100, 1) if bullish_1d is not None else None,
        "bullish_pct_7d":   round(bullish_7d * 100, 1) if bullish_7d is not None else None,
        "fade_rate_7d_pct": round(float(fade_rates) * 100, 1) if fade_rates is not None else None,
        "sample_dates":     neighbors["date"].tolist()[:10],
        "narrative": _build_narrative(
            len(neighbors), out_1d, out_7d, out_30d, fade_rates,
            current_state.get("event_type", "?"),
            current_state.get("direction", "?"),
        ),
    }


def _build_narrative(n, out_1d, out_7d, out_30d, fade_rate, event_type, direction):
    dir_es = {"bullish": "alcista", "bearish": "bajista", "neutral": "neutral"}.get(direction, direction)

    parts = [f"En {n} eventos narrativos similares (tipo dominante: {event_type}, dir: {dir_es}):"]
    if len(out_1d):
        pct_pos = (out_1d > 0).mean() * 100
        parts.append(f"  1d: continuidad en {pct_pos:.0f}% de casos, media {out_1d.mean():+.2f}%.")
    if len(out_7d):
        pct_pos = (out_7d > 0).mean() * 100
        parts.append(f"  7d: alcista en {pct_pos:.0f}%, media {out_7d.mean():+.2f}%.")
    if len(out_30d):
        pct_pos = (out_30d > 0).mean() * 100
        parts.append(f"  30d: alcista en {pct_pos:.0f}%, media {out_30d.mean():+.2f}%.")
    if fade_rate is not None:
        parts.append(f"  Fade rate a 7d: {fade_rate*100:.0f}% de los casos.")
    return " ".join(parts)


def save_event_memory(df: pd.DataFrame, artifacts_dir: str) -> dict:
    """Construye event memory y persiste en artifacts/event_memory.csv + .json."""
    em = build_event_memory(df)
    if em.empty:
        return {"ok": False, "n": 0}

    out_dir = artifacts_dir
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "event_memory.csv")
    em.to_csv(csv_path, index=False)

    # Summary JSON
    summary = {
        "ok":           True,
        "n_events":     len(em),
        "date_range":   [em["date"].min(), em["date"].max()],
        "event_types":  em["event_type"].value_counts().to_dict(),
        "direction_dist": em["direction"].value_counts().to_dict(),
        "mean_narrative_strength": round(float(em["narrative_strength"].mean()), 3),
        "mean_fade_risk": round(float(em["fade_risk"].mean()), 3),
        "fade_rate_7d_pct": round(float(em["fade_occurred_7d"].mean()) * 100, 1)
                            if "fade_occurred_7d" in em.columns else None,
        "as_of":        datetime.now().isoformat(timespec="seconds"),
    }
    json_path = os.path.join(out_dir, "event_memory.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary
