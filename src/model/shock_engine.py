"""
src/model/shock_engine.py
Shock Catalog & Analog Engine.

Premisa: no podemos predecir SI ocurrirá un shock, pero SÍ podemos aprender
de cada shock pasado:
  - su magnitud y co-driver
  - su trayectoria post-shock (peak day, half-fade, supervivencia 30d)
  - su efecto promedio condicional al tipo

Cuando el sistema detecta un shock activo, busca análogos históricos y
reporta estadísticas agregadas. La salida co-existe con el modelo XGB:
  - Mercado normal → manda el modelo (predicción puntual + bandas)
  - Shock activo  → manda el shock engine (recomendación condicional)

Validación empírica (scripts/shock_half_life_analysis.py, 126 shocks 10y):
  - oil_driven   N=43: peak day 15d, peak gain +13.79%, ret 30d +8.03%, sup 84%
  - soyoil_driven N=53: peak day 11d, peak gain +10.46%, ret 30d +4.77%, sup 79%
  - speculative  N=30: peak day  8d, peak gain  +9.11%, ret 30d +2.64%, sup 80%

Para el shock actual (oil +5.96%/7d, soja +4.42%/7d):
  - 16 análogos históricos
  - ret 30d esperado: +6.76% (Q25=+2.89%, Q75=+15.61%)
  - Supervivencia 30d: 87.5%
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd

# ── Constantes ─────────────────────────────────────────────────────
SPIKE_QUANTILE       = 0.95   # shock fuerte = ret > p95
NEAR_SHOCK_QUANTILE  = 0.85   # near-shock = ret > p85 (movimiento notable pero no extremo)
OIL_CODRIVER_Q       = 0.75
SOYOIL_CODRIVER_Q    = 0.75
DROP_QUANTILE        = 0.05
NEAR_DROP_QUANTILE   = 0.15
ANALOG_TOLERANCE     = 0.30
MIN_ANALOGS          = 5


# ── Detección ──────────────────────────────────────────────────────
def detect_shocks_history(df: pd.DataFrame) -> pd.DataFrame:
    """Detecta TODOS los shocks históricos en ventanas 5d Y 10d.
    Cada fecha aparece como máximo una vez (la primera ventana en disparar).
    Returns DataFrame con shocks clasificados."""
    if "Soybeans" not in df.columns:
        return pd.DataFrame()

    d = df.sort_values("Date").reset_index(drop=True).copy()
    d["ret_5d"]     = d["Soybeans"].pct_change(5)
    d["ret_10d"]    = d["Soybeans"].pct_change(10)
    d["oil_5d"]     = d["Oil"].pct_change(5)        if "Oil"        in d.columns else 0.0
    d["oil_10d"]    = d["Oil"].pct_change(10)       if "Oil"        in d.columns else 0.0
    d["soyoil_5d"]  = d["SoybeanOil"].pct_change(5)  if "SoybeanOil" in d.columns else 0.0
    d["soyoil_10d"] = d["SoybeanOil"].pct_change(10) if "SoybeanOil" in d.columns else 0.0

    # Incluimos near-shocks (top 15%) en el catálogo para tener más análogos disponibles
    # cuando el shock activo es notable pero no extremo.
    thr_spike_5  = float(d["ret_5d"].quantile(NEAR_SHOCK_QUANTILE))   # p85
    thr_spike_10 = float(d["ret_10d"].quantile(NEAR_SHOCK_QUANTILE))
    thr_drop_5   = float(d["ret_5d"].quantile(NEAR_DROP_QUANTILE))
    thr_drop_10  = float(d["ret_10d"].quantile(NEAR_DROP_QUANTILE))
    thr_oil_5    = float(d["oil_5d"].quantile(OIL_CODRIVER_Q))      if (d["oil_5d"]  != 0).any() else 1e9
    thr_oil_10   = float(d["oil_10d"].quantile(OIL_CODRIVER_Q))     if (d["oil_10d"] != 0).any() else 1e9
    thr_soyoil_5  = float(d["soyoil_5d"].quantile(SOYOIL_CODRIVER_Q))  if (d["soyoil_5d"]  != 0).any() else 1e9
    thr_soyoil_10 = float(d["soyoil_10d"].quantile(SOYOIL_CODRIVER_Q)) if (d["soyoil_10d"] != 0).any() else 1e9

    is_spike_5  = d["ret_5d"]  >= thr_spike_5
    is_drop_5   = d["ret_5d"]  <= thr_drop_5
    is_spike_10 = d["ret_10d"] >= thr_spike_10
    is_drop_10  = d["ret_10d"] <= thr_drop_10

    is_event = is_spike_5 | is_drop_5 | is_spike_10 | is_drop_10
    events   = d[is_event].copy()
    # Etiquetar fortaleza del evento (top 5% = "strong", solo top 15% = "near")
    thr_strong_spike_5  = float(d["ret_5d"].quantile(SPIKE_QUANTILE))
    thr_strong_spike_10 = float(d["ret_10d"].quantile(SPIKE_QUANTILE))
    thr_strong_drop_5   = float(d["ret_5d"].quantile(DROP_QUANTILE))
    thr_strong_drop_10  = float(d["ret_10d"].quantile(DROP_QUANTILE))

    def classify(row):
        # Elegir la ventana de mayor magnitud absoluta entre las que dispararon
        v5  = row["ret_5d"]  if (row["ret_5d"]  >= thr_spike_5  or row["ret_5d"]  <= thr_drop_5)  else 0.0
        v10 = row["ret_10d"] if (row["ret_10d"] >= thr_spike_10 or row["ret_10d"] <= thr_drop_10) else 0.0
        if abs(v10) > abs(v5):
            ret, oil, soyoil, thr_oil, thr_soyoil = row["ret_10d"], row["oil_10d"], row["soyoil_10d"], thr_oil_10, thr_soyoil_10
            window = "10d"
            is_strong_event = (ret >= thr_strong_spike_10) or (ret <= thr_strong_drop_10)
        else:
            ret, oil, soyoil, thr_oil, thr_soyoil = row["ret_5d"], row["oil_5d"], row["soyoil_5d"], thr_oil_5, thr_soyoil_5
            window = "5d"
            is_strong_event = (ret >= thr_strong_spike_5) or (ret <= thr_strong_drop_5)
        direction = "up" if ret > 0 else "down"
        if direction == "up":
            if oil >= thr_oil:         stype = "oil_driven"
            elif soyoil >= thr_soyoil: stype = "soyoil_driven"
            else:                       stype = "speculative_up"
        else:
            stype = "oil_collapse" if oil <= -thr_oil else "speculative_down"
        return pd.Series({"shock_type": stype, "shock_direction": direction,
                           "window_active": window, "ret_active": ret, "oil_active": oil,
                           "soyoil_active": soyoil, "is_strong": bool(is_strong_event)})

    classification = events.apply(classify, axis=1)
    events = pd.concat([events, classification], axis=1)

    return events[["Date", "Soybeans",
                    "ret_5d", "ret_10d", "oil_5d", "oil_10d", "soyoil_5d", "soyoil_10d",
                    "ret_active", "oil_active", "soyoil_active",
                    "shock_type", "shock_direction", "window_active",
                    "is_strong"]].reset_index(drop=True)


def measure_trajectory(df_full: pd.DataFrame, shock_idx: int, days: int = 30) -> dict:
    """Trayectoria post-shock para el evento en df_full[shock_idx]."""
    if shock_idx + days >= len(df_full) or shock_idx - 5 < 0:
        return {"ok": False}

    p0    = float(df_full.iloc[shock_idx]["Soybeans"])
    p_pre = float(df_full.iloc[shock_idx - 5]["Soybeans"])
    spike_size = (p0 - p_pre) / p_pre if p_pre > 0 else 0.0

    future = df_full.iloc[shock_idx:shock_idx + days + 1]["Soybeans"].values
    if len(future) < days:
        return {"ok": False}

    rels = (future - p_pre) / p_pre if p_pre > 0 else np.zeros_like(future)
    if spike_size > 0:                       # shock alcista: mido peak (max)
        peak_idx  = int(np.argmax(rels))
        peak_rel  = float(rels[peak_idx])
        # Half-fade: cuándo cae al 50% del peak
        target = peak_rel * 0.5
        half_fade = next((i for i in range(peak_idx + 1, len(rels)) if rels[i] <= target), None)
        full_fade = next((i for i in range(peak_idx + 1, len(rels)) if rels[i] <= 0),       None)
    else:                                    # shock bajista: mido valle (min)
        peak_idx  = int(np.argmin(rels))
        peak_rel  = float(rels[peak_idx])
        target = peak_rel * 0.5
        half_fade = next((i for i in range(peak_idx + 1, len(rels)) if rels[i] >= target), None)
        full_fade = next((i for i in range(peak_idx + 1, len(rels)) if rels[i] >= 0),       None)

    persistence_30d = float(rels[-1] / peak_rel) if abs(peak_rel) > 1e-6 else 0.0
    survived_30d    = bool((rels[-1] > 0) if spike_size > 0 else (rels[-1] < 0))

    return {
        "ok":               True,
        "spike_size_pct":   round(spike_size * 100, 2),
        "peak_day":         int(peak_idx),
        "peak_gain_pct":    round(peak_rel * 100, 2),
        "half_fade_day":    half_fade,
        "full_fade_day":    full_fade,
        "ret_30d_pct":      round(float(rels[-1]) * 100, 2),
        "persistence_30d_pct": round(persistence_30d * 100, 1),
        "survived_30d":     survived_30d,
    }


# ── Catálogo persistido ────────────────────────────────────────────
def build_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Construye el catálogo completo: detecta shocks + mide trayectorias."""
    shocks = detect_shocks_history(df)
    rows = []
    df_idx = df.sort_values("Date").reset_index(drop=True)
    for _, sh in shocks.iterrows():
        idx_arr = df_idx.index[df_idx["Date"] == sh["Date"]]
        if len(idx_arr) == 0:
            continue
        idx = int(idx_arr[0])
        traj = measure_trajectory(df_idx, idx, days=30)
        if not traj.get("ok"):
            continue
        rows.append({"Date": sh["Date"],
                      "Soybeans":        float(sh["Soybeans"]),
                      "shock_type":      sh["shock_type"],
                      "shock_direction": sh["shock_direction"],
                      "window_active":   sh["window_active"],
                      "is_strong":       bool(sh.get("is_strong", False)),
                      "ret_5d_at_shock":     float(sh["ret_5d"]),
                      "ret_10d_at_shock":    float(sh["ret_10d"]),
                      "ret_active_at_shock": float(sh["ret_active"]),
                      "oil_5d_at_shock":      float(sh["oil_5d"]),
                      "oil_10d_at_shock":     float(sh["oil_10d"]),
                      "oil_active_at_shock":  float(sh["oil_active"]),
                      "soyoil_5d_at_shock":   float(sh["soyoil_5d"]),
                      "soyoil_10d_at_shock":  float(sh["soyoil_10d"]),
                      **traj})
    return pd.DataFrame(rows)


def save_catalog(df: pd.DataFrame, artifacts_dir: str) -> str:
    catalog = build_catalog(df)
    os.makedirs(artifacts_dir, exist_ok=True)
    path = os.path.join(artifacts_dir, "shock_catalog.csv")
    catalog.to_csv(path, index=False)
    return path


# ── Detección "shock activo HOY" ───────────────────────────────────
def detect_current_shock(df: pd.DataFrame) -> dict:
    """Mira la última fila de df y dice si HOY hay un shock activo.
    Usa ventana DOBLE (5d y 10d) para capturar shocks de distinta velocidad:
      - 5d: shocks rápidos / agudos
      - 10d: shocks que se desplegaron en 1-2 semanas y todavía son relevantes."""
    if "Soybeans" not in df.columns or len(df) < 11:
        return {"ok": False, "is_shock": False}

    d = df.sort_values("Date").reset_index(drop=True)
    last = d.iloc[-1]
    # Ventana 5d trading
    ret_5d     = float((last["Soybeans"] / d.iloc[-6]["Soybeans"] - 1))
    oil_5d     = float((last["Oil"] / d.iloc[-6]["Oil"] - 1)) if "Oil" in d.columns else 0.0
    soyoil_5d  = float((last["SoybeanOil"] / d.iloc[-6]["SoybeanOil"] - 1)) if "SoybeanOil" in d.columns else 0.0
    # Ventana 10d trading
    ret_10d    = float((last["Soybeans"] / d.iloc[-11]["Soybeans"] - 1))
    oil_10d    = float((last["Oil"] / d.iloc[-11]["Oil"] - 1)) if "Oil" in d.columns else 0.0
    soyoil_10d = float((last["SoybeanOil"] / d.iloc[-11]["SoybeanOil"] - 1)) if "SoybeanOil" in d.columns else 0.0

    # Umbrales históricos por ventana
    rets_5  = d["Soybeans"].pct_change(5)
    rets_10 = d["Soybeans"].pct_change(10)
    thr_spike_5  = float(rets_5.quantile(SPIKE_QUANTILE))
    thr_drop_5   = float(rets_5.quantile(DROP_QUANTILE))
    thr_spike_10 = float(rets_10.quantile(SPIKE_QUANTILE))
    thr_drop_10  = float(rets_10.quantile(DROP_QUANTILE))

    # Thresholds adicionales: near-shock (top 15% — movimiento notable)
    thr_near_spike_5  = float(rets_5.quantile(NEAR_SHOCK_QUANTILE))
    thr_near_spike_10 = float(rets_10.quantile(NEAR_SHOCK_QUANTILE))
    thr_near_drop_5   = float(rets_5.quantile(NEAR_DROP_QUANTILE))
    thr_near_drop_10  = float(rets_10.quantile(NEAR_DROP_QUANTILE))

    is_spike_5  = ret_5d  >= thr_spike_5
    is_drop_5   = ret_5d  <= thr_drop_5
    is_spike_10 = ret_10d >= thr_spike_10
    is_drop_10  = ret_10d <= thr_drop_10

    is_spike = is_spike_5 or is_spike_10
    is_drop  = is_drop_5  or is_drop_10

    # Near-shock: dispara aunque no sea extremo
    is_near_spike_5  = ret_5d  >= thr_near_spike_5  and not is_spike_5
    is_near_spike_10 = ret_10d >= thr_near_spike_10 and not is_spike_10
    is_near_drop_5   = ret_5d  <= thr_near_drop_5   and not is_drop_5
    is_near_drop_10  = ret_10d <= thr_near_drop_10  and not is_drop_10
    is_near_event = (is_near_spike_5 or is_near_spike_10 or is_near_drop_5 or is_near_drop_10)

    # Elegir ventana que disparó (la que tenga mayor magnitud absoluta)
    triggered_10 = is_spike_10 or is_drop_10 or is_near_spike_10 or is_near_drop_10
    triggered_5  = is_spike_5  or is_drop_5  or is_near_spike_5  or is_near_drop_5
    if abs(ret_10d) > abs(ret_5d) and triggered_10:
        ret_active, oil_active, soyoil_active = ret_10d, oil_10d, soyoil_10d
        window_active = "10d"
        thr_spike_active, thr_drop_active = thr_spike_10, thr_drop_10
    else:
        ret_active, oil_active, soyoil_active = ret_5d, oil_5d, soyoil_5d
        window_active = "5d"
        thr_spike_active, thr_drop_active = thr_spike_5, thr_drop_5

    if not (is_spike or is_drop or is_near_event):
        return {"ok": True, "is_shock": False, "is_near_shock": False,
                "ret_5d_pct":  round(ret_5d*100, 2),  "ret_10d_pct": round(ret_10d*100, 2),
                "thr_spike_5d_pct":     round(thr_spike_5*100, 2),
                "thr_spike_10d_pct":    round(thr_spike_10*100, 2),
                "thr_drop_5d_pct":      round(thr_drop_5*100, 2),
                "thr_drop_10d_pct":     round(thr_drop_10*100, 2),
                "thr_near_spike_5d_pct":  round(thr_near_spike_5*100, 2),
                "thr_near_spike_10d_pct": round(thr_near_spike_10*100, 2)}

    # Clasificar usando la ventana activa (5d o 10d)
    # Direction se determina por el SIGNO del ret_active (no por is_spike vs is_drop,
    # porque también podemos estar en near-shock que no dispara is_spike/is_drop pero
    # tiene dirección clara)
    win = 5 if window_active == "5d" else 10
    direction = "up" if ret_active > 0 else "down"
    if direction == "up":
        oils = d["Oil"].pct_change(win) if "Oil" in d.columns else None
        soys = d["SoybeanOil"].pct_change(win) if "SoybeanOil" in d.columns else None
        thr_oil    = float(oils.quantile(OIL_CODRIVER_Q))    if oils is not None else 1e9
        thr_soyoil = float(soys.quantile(SOYOIL_CODRIVER_Q)) if soys is not None else 1e9
        if oil_active >= thr_oil:         shock_type = "oil_driven"
        elif soyoil_active >= thr_soyoil: shock_type = "soyoil_driven"
        else:                              shock_type = "speculative_up"
    else:
        oils = d["Oil"].pct_change(win) if "Oil" in d.columns else None
        thr_oil_neg = -float(oils.quantile(OIL_CODRIVER_Q)) if oils is not None else -1e9
        shock_type = "oil_collapse" if oil_active <= thr_oil_neg else "speculative_down"

    is_strong = (is_spike or is_drop)   # top 5%
    return {
        "ok":             True,
        "is_shock":       True,
        "is_strong_shock": is_strong,        # True = top 5%, False = top 15% (near-shock)
        "is_near_shock":  not is_strong,
        "shock_type":     shock_type,
        "shock_direction": direction,
        "window_active":  window_active,
        "ret_5d_pct":     round(ret_5d * 100, 2),
        "ret_10d_pct":    round(ret_10d * 100, 2),
        "oil_5d_pct":     round(oil_5d * 100, 2),
        "oil_10d_pct":    round(oil_10d * 100, 2),
        "soyoil_5d_pct":  round(soyoil_5d * 100, 2),
        "soyoil_10d_pct": round(soyoil_10d * 100, 2),
        "thr_spike_5d_pct":  round(thr_spike_5 * 100, 2),
        "thr_spike_10d_pct": round(thr_spike_10 * 100, 2),
        "thr_drop_5d_pct":   round(thr_drop_5 * 100, 2),
        "thr_drop_10d_pct":  round(thr_drop_10 * 100, 2),
        "thr_near_spike_5d_pct":  round(thr_near_spike_5 * 100, 2),
        "thr_near_spike_10d_pct": round(thr_near_spike_10 * 100, 2),
        "shock_date":     str(last["Date"])[:10],
        "shock_price":    float(last["Soybeans"]),
        "ret_active_pct": round(ret_active * 100, 2),
        "oil_active_pct": round(oil_active * 100, 2),
    }


# ── Análogos ───────────────────────────────────────────────────────
def find_analogs(catalog: pd.DataFrame, current: dict, tolerance: float = ANALOG_TOLERANCE) -> pd.DataFrame:
    """Filtra el catálogo por shocks similares al actual.
    Mismo tipo + ret en rango ±tolerance + co-driver en rango ±tolerance.
    Usa la ventana activa (5d o 10d) que disparó el shock."""
    if catalog.empty or not current.get("is_shock"):
        return pd.DataFrame()

    same_type = catalog[catalog["shock_type"] == current["shock_type"]].copy()
    if same_type.empty:
        return same_type

    # Usar ret y oil de la ventana ACTIVA (la que disparó)
    target_ret = current.get("ret_active_pct", current.get("ret_5d_pct", 0)) / 100.0
    target_oil = current.get("oil_active_pct", current.get("oil_5d_pct", 0)) / 100.0
    lo_ret = target_ret * (1 - tolerance)
    hi_ret = target_ret * (1 + tolerance)
    if target_ret < 0:
        lo_ret, hi_ret = hi_ret, lo_ret  # invertir bounds para negativos

    # Comparar contra ret_active del catálogo (la ventana que disparó cada shock)
    ret_col = "ret_active_at_shock" if "ret_active_at_shock" in same_type.columns else "ret_5d_at_shock"
    analogs = same_type[same_type[ret_col].between(min(lo_ret, hi_ret),
                                                    max(lo_ret, hi_ret))]
    if "oil" in current["shock_type"]:
        lo_oil = target_oil * (1 - tolerance)
        hi_oil = target_oil * (1 + tolerance)
        if target_oil < 0:
            lo_oil, hi_oil = hi_oil, lo_oil
        oil_col = "oil_active_at_shock" if "oil_active_at_shock" in analogs.columns else "oil_5d_at_shock"
        analogs = analogs[analogs[oil_col].between(min(lo_oil, hi_oil),
                                                    max(lo_oil, hi_oil))]
    return analogs


def aggregate_outcomes(analogs: pd.DataFrame) -> dict:
    """Calcula stats agregados sobre los análogos para reportar al productor."""
    if analogs.empty:
        return {"n": 0}

    ret_30d = analogs["ret_30d_pct"].dropna()
    peak_d  = analogs["peak_day"].dropna()
    peak_g  = analogs["peak_gain_pct"].dropna()
    hf      = analogs["half_fade_day"].dropna()
    survived = analogs["survived_30d"].mean() * 100

    return {
        "n":                    int(len(analogs)),
        "ret_30d_q25_pct":      float(ret_30d.quantile(0.25)) if not ret_30d.empty else None,
        "ret_30d_med_pct":      float(ret_30d.median())       if not ret_30d.empty else None,
        "ret_30d_q75_pct":      float(ret_30d.quantile(0.75)) if not ret_30d.empty else None,
        "peak_day_med":         int(peak_d.median())          if not peak_d.empty else None,
        "peak_day_q25":         int(peak_d.quantile(0.25))    if not peak_d.empty else None,
        "peak_day_q75":         int(peak_d.quantile(0.75))    if not peak_d.empty else None,
        "peak_gain_med_pct":    float(peak_g.median())        if not peak_g.empty else None,
        "half_fade_med":        int(hf.median())              if not hf.empty     else None,
        "survived_30d_pct":     float(survived),
        "confidence":           ("high"   if len(analogs) >= 15
                                  else "medium" if len(analogs) >= MIN_ANALOGS
                                  else "low"),
    }


def build_recommendation(current: dict, stats: dict) -> dict:
    """Construye payload INFORMATIVO sobre el shock activo.

    IMPORTANTE — esta función NO emite recomendaciones direccionales tipo
    "WAIT_FOR_PEAK" o "WAIT_RECOVERY". El backtest sobre 971 shocks históricos
    (artifacts_eval/backtest_shock_engine.csv) mostró que esas acciones
    pierden estadísticamente vs always-sell (lift -1.28 %, p<0.0001).

    El módulo queda como capa INFORMATIVA: describe los análogos históricos
    y deja que el modelo regular (Shock Engine NO sustituye al modelo) tome
    la decisión. La narrativa es útil para el productor pero NO accionable
    como signal único.
    """
    if not current.get("is_shock") or stats.get("n", 0) == 0:
        return {"action": "NO_SHOCK", "advisory": "info_only",
                "message": "Sin shock activo — el modelo regular gobierna la decisión."}

    n = stats["n"]
    if n < MIN_ANALOGS:
        return {
            "action": "INSUFFICIENT_ANALOGS", "advisory": "info_only",
            "message": f"Shock detectado pero solo {n} análogos en histórico. Confianza baja.",
            "tactical": "Mantenga la decisión del modelo regular.",
        }

    direction = current.get("shock_direction", "up")
    ret_med  = stats.get("ret_30d_med_pct", 0)
    ret_q25  = stats.get("ret_30d_q25_pct", 0)
    ret_q75  = stats.get("ret_30d_q75_pct", 0)
    survived = stats.get("survived_30d_pct", 50)
    peak_d   = stats.get("peak_day_med")
    peak_q25 = stats.get("peak_day_q25")
    peak_q75 = stats.get("peak_day_q75")
    peak_g   = stats.get("peak_gain_med_pct")

    # Mensaje descriptivo (sin recomendación direccional)
    direction_label = "alcista" if direction == "up" else "bajista"
    msg_parts = [f"Shock {direction_label} detectado con {n} análogos históricos."]
    if peak_d is not None and peak_g is not None:
        msg_parts.append(f"Peak típico: día {peak_d} post-shock (rango Q25-Q75: día {peak_q25}-{peak_q75}, ganancia mediana {peak_g:+.1f}%).")
    msg_parts.append(f"Retorno mediano a 30d: {ret_med:+.1f}% (banda Q25-Q75: {ret_q25:+.1f}% a {ret_q75:+.1f}%).")
    msg_parts.append(f"Sobrevive 30d: {survived:.0f}% de los casos.")
    message = " ".join(msg_parts)

    # Calificación cualitativa del shock (descriptiva, NO recomendación)
    if direction == "up" and ret_med > 3 and survived > 70:
        action = "INFO_BULLISH_PERSISTENT"
        tactical = (f"Históricamente este tipo de shock alcista persistió en {survived:.0f}% de los casos. "
                    "Esto NO es recomendación — el backtest mostró que actuar sobre estos análogos pierde plata. "
                    "Tome el dato como contexto y siga la decisión del modelo regular.")
    elif direction == "up" and ret_med < -2:
        action = "INFO_BULLISH_FADES"
        tactical = (f"Históricamente este tipo de shock se diluye (ret 30d med {ret_med:+.1f}%). "
                    "Información de contexto — no actúe sólo en base a esto.")
    elif direction == "down" and ret_med < -3:
        action = "INFO_BEARISH_DEEPENS"
        tactical = (f"Históricamente la caída suele profundizarse antes de recuperar. "
                    "Información de contexto — no actúe sólo en base a esto.")
    else:
        action = "INFO_AMBIGUOUS"
        tactical = (f"Análogos con outcomes mixtos. La señal no es lo suficientemente clara "
                    "para ningún tipo de acción direccional.")

    return {
        "action":     action,           # ahora todas son INFO_*
        "advisory":   "info_only",      # marca explícita: nunca accionable como decisión única
        "message":    message,
        "tactical":   tactical,
        "confidence": stats.get("confidence", "medium"),
        "note":       "Backtest 2016-2026 (N=971): acciones direccionales basadas en análogos pierden vs always-sell con p<0.0001. Este panel es informativo, no decisor.",
    }


# ── API principal ──────────────────────────────────────────────────
def assess_shock(df: pd.DataFrame, artifacts_dir: str | None = None) -> dict:
    """Endpoint principal: ¿hay shock hoy? Si sí, busca análogos y recomienda.
    Returns dict completo listo para serializar a JSON."""
    current = detect_current_shock(df)
    out = {"as_of": pd.Timestamp.now().isoformat(timespec="seconds"),
           "current": current}

    if not current.get("is_shock"):
        out["analogs_found"] = 0
        out["recommendation"] = {"action": "NO_SHOCK",
                                  "message": "Sin shock activo. Use el modelo regular."}
        return out

    catalog = build_catalog(df)
    if artifacts_dir:
        os.makedirs(artifacts_dir, exist_ok=True)
        catalog.to_csv(os.path.join(artifacts_dir, "shock_catalog.csv"), index=False)

    analogs = find_analogs(catalog, current)
    stats   = aggregate_outcomes(analogs)
    rec     = build_recommendation(current, stats)

    out["analogs_found"] = stats.get("n", 0)
    out["analog_stats"]  = stats
    out["recommendation"] = rec
    return out


def save_active_shock(features: pd.DataFrame, artifacts_dir: str) -> dict:
    """Persiste el assessment a artifacts/active_shock.json para consumo por la API/UI."""
    out = assess_shock(features, artifacts_dir=artifacts_dir)
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(os.path.join(artifacts_dir, "active_shock.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    return out
