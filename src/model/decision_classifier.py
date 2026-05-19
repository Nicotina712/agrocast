"""
src/model/decision_classifier.py
Cost-aware Decision Classifier — panel INFORMATIVO sobre la conveniencia
de esperar 30d vs vender hoy, calibrado al costo de carrying del productor.

⚠️ IMPORTANTE — este módulo es INFORMATIVO, no decisor.
Backtest exhaustivo (artifacts_eval/test_decision_classifier_v3.py) mostró:
  - Best config (Idea A, asymmetric loss fp_weight=2.0): PnL −0.113 %/mes
  - Always-sell baseline: 0 %
  - Diferencia: −1.36 %/año vs always-sell
  - Es la mejor aproximación predictiva probada (de 21 configs en 7 enfoques)
    pero SIGUE PERDIENDO ligeramente al always-sell.

Por qué publicarlo igual:
  - Es una segunda opinión calibrada que el productor puede consultar
  - La probabilidad calibrada es información honesta sobre la convicción
  - En contextos donde el always-sell no aplica (productor sin liquidez
    inmediata, aversión específica al timing), la prob puede ser útil
  - NO se debe usar como signal único — solo como contexto

DESIGN
------
Target: ret_30d_fwd > cost_pct  (binario: ¿esperar pagó tu costo?)
Features: 17 indicadores compuestos (price-relative, momentum, news, macro,
          stagionalidad, percentil 12m)
Loss: asymmetric (sample_weight=2.0 en label=0 → penaliza falsos positivos)
Calibration: isotonic regression sobre val
Refit: cada 90 días (walk-forward)
Output:
  - prob_wait_pays_calibrated   ∈ [0, 1]
  - decision_advisory           "WAIT" / "SELL" / "INDIFFERENT"
  - threshold_active            0.5 default
  - confidence                  high / medium / low
  - context_note                 framing honesto

Target cost por defecto: 1.0 %/mes (productor low-cost). Configurable
via parámetro storage_cost_per_ton_month + financing_rate_annual.
"""
from __future__ import annotations
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.isotonic import IsotonicRegression

HORIZON_DAYS    = 30                  # default (legacy, single-horizon API)
HORIZONS_MULTI  = [7, 15, 30]         # Fase 1.2: multi-horizonte
TRAIN_YEARS     = 5
ASYMMETRIC_FP_WEIGHT = 2.0            # Idea A validada empíricamente

DEFAULT_STORAGE_USD_TON_MONTH = 6.0
DEFAULT_FINANCING_RATE_ANNUAL = 0.08
DEFAULT_PRICE_USD_TON         = 400.0

# ── Profiles del productor (Fase 1.3) ─────────────────────────────
PRODUCER_PROFILES = {
    # Default: productor típico Uruguay, costos medios
    "default":        {"storage": 6.0,  "financing": 0.08, "quality_risk_per_month": 0.0},
    # Bajo costo: silo propio + cash, productor capitalizado
    "low_cost":       {"storage": 0.0,  "financing": 0.05, "quality_risk_per_month": 0.0},
    # Alto costo: storage rentado + crédito caro
    "high_cost":      {"storage": 10.0, "financing": 0.15, "quality_risk_per_month": 0.5},
    # Necesidad de liquidez: el costo de oportunidad del cash es muy alto
    # → penaliza esperar incluso si retorno esperado es positivo
    "liquidity_need": {"storage": 6.0,  "financing": 0.18, "quality_risk_per_month": 0.0},
    # Sensibilidad a calidad: grano susceptible (alta humedad, riesgo de deterioro)
    "quality_aware":  {"storage": 6.0,  "financing": 0.08, "quality_risk_per_month": 1.5},
}

# Features decision-relevant
FEATURE_COLS = [
    "news_sentiment", "news_velocity_7d",
    "mom_5d", "mom_20d", "rsi_14",
    "vol_30d", "vol_60d",
    "Oil_chg7", "Dollar_chg7",
    "soy_corn_ratio", "enso_oni",
    "month_sin", "month_cos",
    "cot_noncomm_long_pct", "wasde_bull_bias",
    "price_pct_in_12m_range", "seasonal_ret_30d_med",
]


def _compute_cost_pct(storage_per_ton_month: float, financing_annual: float,
                      price_per_ton: float, quality_risk_per_month: float = 0.0,
                      horizon_days: int = 30) -> float:
    """Costo total para ESPERAR `horizon_days` días, como fracción del precio.
    Componentes:
      - storage_pct: prorrateado por meses
      - financing_pct: tasa anual prorrateada
      - quality_risk: degradación esperada (Fase 1.3 - quality_aware profile)
    """
    months = horizon_days / 30.0
    storage_pct  = (storage_per_ton_month / max(price_per_ton, 1.0)) * months
    financing_pct = financing_annual * (horizon_days / 365.0)
    quality_pct  = (quality_risk_per_month / max(price_per_ton, 1.0)) * months
    return storage_pct + financing_pct + quality_pct


def _resolve_profile(profile: str | dict | None) -> dict:
    """Resuelve un nombre de profile a sus parámetros, o devuelve dict directo."""
    if isinstance(profile, dict):
        return profile
    if profile is None:
        profile = "default"
    return PRODUCER_PROFILES.get(profile, PRODUCER_PROFILES["default"])


def _build_decision_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds price_pct_in_12m_range and seasonal_ret_30d_med if missing.
    These are decision-aware composite features."""
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    if "price_pct_in_12m_range" not in d.columns:
        p_min = d["Soybeans"].rolling(252, min_periods=60).min().shift(1)
        p_max = d["Soybeans"].rolling(252, min_periods=60).max().shift(1)
        d["price_pct_in_12m_range"] = (d["Soybeans"] - p_min) / (p_max - p_min + 1e-9)
    if "seasonal_ret_30d_med" not in d.columns:
        d["month"] = d["Date"].dt.month
        d["seasonal_ret_30d_med"] = d.groupby("month")["Soybeans"].transform(
            lambda s: s.pct_change(30).shift(-30).expanding(min_periods=12).median().shift(1)
        ).fillna(0)
    return d


def train_decision_classifier(df: pd.DataFrame, cost_pct: float,
                                train_years: int = TRAIN_YEARS,
                                fp_weight: float = ASYMMETRIC_FP_WEIGHT,
                                horizon_days: int = HORIZON_DAYS) -> dict:
    """Entrena el clasificador con asymmetric loss + isotonic calibration.

    horizon_days: ventana del target ret_Hd_fwd (Fase 1.2 multi-horizon).
    Returns dict con:
        model, calibrator, features, base_rate_wait, fp_weight,
        cost_pct, horizon_days, val_acc, val_brier, trained_at
    """
    d = _build_decision_features(df)
    target_col = f"ret_{horizon_days}d_fwd"
    if target_col not in d.columns:
        d[target_col] = d["Soybeans"].pct_change(horizon_days).shift(-horizon_days)
    d["label_wait_paid"] = (d[target_col] > cost_pct).astype(int)

    end = d["Date"].max()
    train_start = end - pd.DateOffset(years=train_years)
    tr = d[(d["Date"] >= train_start) & (d["Date"] < end)].dropna(subset=["label_wait_paid"]).copy()

    feats = [f for f in FEATURE_COLS if f in tr.columns]
    if len(tr) < 200 or not feats:
        return {"ok": False, "error": "insufficient training data"}

    X = tr[feats].fillna(0).replace([np.inf, -np.inf], 0)
    y = tr["label_wait_paid"]

    # Temporal split 80/20 with embargo gap >= horizon to prevent target leakage
    split = int(len(tr) * 0.8)
    _embargo_days = horizon_days + 5  # horizon + buffer
    if "Date" in tr.columns:
        _dates = pd.to_datetime(tr["Date"])
        _cut = _dates.iloc[split - 1]
        _embargo_end = _cut + pd.Timedelta(days=_embargo_days)
        _train_mask = _dates <= _cut
        _val_mask = _dates > _embargo_end
        X_train, X_val = X[_train_mask], X[_val_mask]
        y_train, y_val = y[_train_mask], y[_val_mask]
        _dropped = len(tr) - len(X_train) - len(X_val)
        print(f"   [DC] Embargo {_embargo_days}d: train={len(X_train)}, val={len(X_val)}, "
              f"dropped={_dropped} rows")
    else:
        # Fallback: skip rows equal to horizon in the gap
        X_train = X.iloc[:split]
        X_val = X.iloc[split + horizon_days:]
        y_train = y.iloc[:split]
        y_val = y.iloc[split + horizon_days:]

    # Asymmetric loss: peso fp_weight en muestras negativas (label=0)
    sample_w = np.where(y_train == 0, fp_weight, 1.0)
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw = max(n_neg / max(n_pos, 1), 1.0) / fp_weight   # compensar el extra peso

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="logloss", scale_pos_weight=spw,
    )
    model.fit(X_train, y_train, sample_weight=sample_w, verbose=False)

    # Calibration
    calibrator = None
    val_brier_raw = float("nan")
    val_brier_cal = float("nan")
    val_acc = float("nan")
    try:
        p_val_raw = model.predict_proba(X_val)[:, 1]
        val_brier_raw = float(np.mean((p_val_raw - y_val.values) ** 2))
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(p_val_raw, y_val.values.astype(float))
        p_val_cal = calibrator.transform(p_val_raw)
        val_brier_cal = float(np.mean((p_val_cal - y_val.values) ** 2))
        val_acc = float(np.mean((p_val_cal >= 0.5).astype(int) == y_val.values))
    except Exception as _e:
        pass

    # ── Fase 2.1: Delta regressor (magnitud económica) ────────────
    # Entrena un regresor de delta_wait_vs_sell_usd_ton (no solo signo).
    # Esto agrega información de MAGNITUD que el clasificador no provee.
    delta_model = None
    val_delta_mae = float("nan")
    try:
        # Target: delta económico real en %
        # delta = ret_Hd_fwd - cost_pct → puede ser positivo o negativo
        ret_col = f"ret_{horizon_days}d_fwd"
        delta_col = "delta_wait_vs_sell"
        d[delta_col] = d[ret_col] - cost_pct

        tr2 = d[(d["Date"] >= train_start) & (d["Date"] < end)].dropna(subset=[delta_col]).copy()
        if len(tr2) >= 200:
            X2 = tr2[feats].fillna(0).replace([np.inf, -np.inf], 0)
            y2 = tr2[delta_col]
            split2 = int(len(tr2) * 0.8)
            X2_train, X2_val = X2.iloc[:split2], X2.iloc[split2:]
            y2_train, y2_val = y2.iloc[:split2], y2.iloc[split2:]
            delta_model = XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                eval_metric="mae",
            )
            delta_model.fit(X2_train, y2_train, verbose=False)
            try:
                pred_val = delta_model.predict(X2_val)
                val_delta_mae = float(np.mean(np.abs(pred_val - y2_val.values)))
            except Exception:
                pass
    except Exception as _e:
        delta_model = None

    return {
        "ok": True,
        "model":       model,
        "calibrator":  calibrator,
        "delta_model": delta_model,        # Fase 2.1
        "features":    feats,
        "fp_weight":   fp_weight,
        "cost_pct":    cost_pct,
        "horizon_days": horizon_days,
        "base_rate_wait":  float(y_train.mean()),
        "n_train":     int(len(X_train)),
        "n_val":       int(len(X_val)),
        "val_acc":     val_acc,
        "val_brier_raw": val_brier_raw,
        "val_brier_cal": val_brier_cal,
        "val_delta_mae": val_delta_mae,
        "trained_at":  datetime.now().isoformat(timespec="seconds"),
    }


# ─────────────────────────────────────────────────────────────────
# Análogos explicativos (Fase 1.1) — informativo, NO decisor
# ─────────────────────────────────────────────────────────────────
ANALOG_FEATURES = [
    "price_pct_in_12m_range", "mom_5d", "mom_20d", "rsi_14",
    "vol_30d", "Oil_chg7", "news_sentiment",
]


def find_decision_analogs(df: pd.DataFrame, cost_pct: float,
                           horizon_days: int = HORIZON_DAYS,
                           k: int = 20, min_gap_days: int = 60) -> dict:
    """Para la fila más reciente de df, busca top-K casos históricos similares
    en espacio de features y reporta sus outcomes reales (delta_wait_vs_sell,
    win_rate, peor caso, mejor caso).

    NO se usa como predictor (validamos empíricamente que NN-analog regresor
    pierde). Se usa como capa NARRATIVA para el productor: "en 38 casos
    similares, esperar fue mejor en 42% — worst case −$18/ton, best +$31".
    """
    d = _build_decision_features(df)
    target_col = f"ret_{horizon_days}d_fwd"
    if target_col not in d.columns:
        d[target_col] = d["Soybeans"].pct_change(horizon_days).shift(-horizon_days)

    avail = [f for f in ANALOG_FEATURES if f in d.columns]
    if len(avail) < 3 or len(d) < 200:
        return {"ok": False, "n": 0}

    # Z-score rolling para distancia consistente
    Z = pd.DataFrame(index=d.index)
    for f in avail:
        s = d[f]
        rm = s.rolling(252, min_periods=60).mean().shift(1)
        rs = s.rolling(252, min_periods=60).std().shift(1).replace(0, 1e-6)
        Z[f] = (s - rm) / rs
    Z = Z.fillna(0).replace([np.inf, -np.inf], 0)

    last_idx = len(d) - 1
    last_date = d.iloc[last_idx]["Date"]
    v_t = Z.iloc[last_idx].values

    # Filtrar histórico: gap suficiente + outcome conocido
    gap_date = last_date - pd.Timedelta(days=min_gap_days)
    hist_mask = (d["Date"] < gap_date) & d[target_col].notna()
    hist_idx = d.index[hist_mask]
    if len(hist_idx) < k * 2:
        return {"ok": False, "n": 0}

    V_hist = Z.loc[hist_idx].values
    dists = np.sqrt(((V_hist - v_t) ** 2).sum(axis=1))
    top_k = np.argsort(dists)[:k]
    neighbor_idx = hist_idx[top_k]

    # Para cada vecino, calcular delta_wait_vs_sell con costos del horizonte
    neighbor_rets = d.loc[neighbor_idx, target_col].values
    deltas_pct = neighbor_rets - cost_pct  # delta vs always-sell, en %
    wait_won = deltas_pct > 0

    # Convertir a USD/ton aproximado (precio actual)
    p_now_cents = float(d.iloc[-1]["Soybeans"])
    p_now_ton = p_now_cents * 0.01 * 36.7437
    deltas_usd = deltas_pct * p_now_ton

    return {
        "ok":           True,
        "n":            int(k),
        "k":            k,
        "horizon_days": horizon_days,
        "wait_win_rate_pct":  round(float(wait_won.mean()) * 100, 1),
        "delta_avg_pct":      round(float(deltas_pct.mean()) * 100, 3),
        "delta_med_pct":      round(float(np.median(deltas_pct)) * 100, 3),
        "delta_avg_usd_ton":  round(float(deltas_usd.mean()), 2),
        "delta_med_usd_ton":  round(float(np.median(deltas_usd)), 2),
        "delta_q10_usd_ton":  round(float(np.quantile(deltas_usd, 0.10)), 2),
        "delta_q90_usd_ton":  round(float(np.quantile(deltas_usd, 0.90)), 2),
        "worst_case_usd_ton": round(float(deltas_usd.min()), 2),
        "best_case_usd_ton":  round(float(deltas_usd.max()), 2),
        "neighbor_dates":     [str(dt)[:10] for dt in d.loc[neighbor_idx, "Date"].tolist()[:10]],
        "neighbor_dist_med":  round(float(np.median(dists[top_k])), 3),
        "narrative": (
            f"En {k} situaciones históricas con patrón similar (a {horizon_days}d), "
            f"esperar fue mejor que vender hoy en {float(wait_won.mean())*100:.0f}% de los casos. "
            f"Delta medio de esperar: ${float(deltas_usd.mean()):+.1f}/ton. "
            f"Peor caso: ${float(deltas_usd.min()):+.1f}/ton. "
            f"Mejor caso: ${float(deltas_usd.max()):+.1f}/ton."
        ),
    }


def _combine_partial_sell(p_cal: float, delta_pct: float | None,
                            cost_pct: float) -> tuple[str, int, int, str]:
    """Fase 2.2: Combinar P(WAIT) + delta esperado en una decisión graduada.
    Returns (decision_label, sell_pct, hold_pct, reason)

    Lógica (regla simple, no aprendida):
      1. Si NO tenemos delta (regresor falló) → caer a clasificación binaria simple.
      2. Si delta esperado <= 0 (esperar destruye valor) → SELL_100 sin importar p.
      3. Si delta_neto = delta - cost_pct/2 < min_neto → SELL alto.
      4. Combinar p_cal y magnitud para grado de WAIT.

    Mín. delta neto para empezar a considerar WAIT: 0.5 % del precio.
    """
    if delta_pct is None or np.isnan(delta_pct):
        # Solo clasificación
        if p_cal >= 0.62:   return ("SELL_30",  30, 70, "P(WAIT) alta sin magnitud confirmada")
        if p_cal >= 0.55:   return ("SELL_50",  50, 50, "P(WAIT) moderada sin magnitud confirmada")
        if p_cal <= 0.40:   return ("SELL_100", 100, 0, "P(WAIT) baja → vender todo")
        return ("SELL_70", 70, 30, "Sin convicción direccional clara")

    # Con delta disponible
    # delta_pct ya viene en escala fracción (ej. 0.015 = +1.5%)
    # Mínimo delta neto para considerar WAIT (debe cubrir incertidumbre + costo de oportunidad)
    MIN_DELTA = 0.005   # 0.5% del precio

    # Si el modelo dice que delta es negativo o muy chico → SELL
    if delta_pct < -MIN_DELTA:
        return ("SELL_100", 100, 0,
                f"Delta esperado {delta_pct*100:+.2f}% < cero → esperar destruye valor")
    if delta_pct < MIN_DELTA:
        if p_cal >= 0.55:
            return ("SELL_70", 70, 30,
                    f"Delta marginal ({delta_pct*100:+.2f}%) aunque P(WAIT)={p_cal:.2f}")
        return ("SELL_100", 100, 0,
                f"Delta esperado {delta_pct*100:+.2f}% no compensa, P(WAIT)={p_cal:.2f}")

    # delta_pct >= MIN_DELTA → hay magnitud potencial
    # Combinar con probabilidad
    if p_cal >= 0.62 and delta_pct >= 0.015:
        return ("HOLD_70", 30, 70,
                f"Confianza alta ({p_cal:.2f}) y delta sólido ({delta_pct*100:+.2f}%)")
    if p_cal >= 0.55 and delta_pct >= 0.010:
        return ("SPLIT_50", 50, 50,
                f"Convicción moderada ({p_cal:.2f}) con delta razonable ({delta_pct*100:+.2f}%)")
    if p_cal >= 0.50 and delta_pct >= MIN_DELTA:
        return ("SELL_70", 70, 30,
                f"Probabilidad simétrica ({p_cal:.2f}) con delta marginal ({delta_pct*100:+.2f}%)")
    # P(WAIT) baja pero delta positivo → ambigüedad, vender más
    return ("SELL_70", 70, 30,
            f"P(WAIT)={p_cal:.2f} baja aunque delta {delta_pct*100:+.2f}% positivo")


def predict_decision(df: pd.DataFrame, bundle: dict, threshold: float = 0.5) -> dict:
    """Predicción para la fila más reciente. Output INFORMATIVO con magnitud
    económica (Fase 2.1) y recomendación PARTIAL_SELL (Fase 2.2)."""
    if not bundle.get("ok"):
        return {"ok": False, "error": bundle.get("error", "no model")}

    d = _build_decision_features(df)
    feats = bundle["features"]
    last_row = d.iloc[[-1]]
    X = last_row[feats].fillna(0).replace([np.inf, -np.inf], 0)
    p_raw = float(bundle["model"].predict_proba(X)[0, 1])
    p_cal = float(bundle["calibrator"].transform([p_raw])[0]) if bundle.get("calibrator") else p_raw
    p_cal = float(np.clip(p_cal, 0.0, 1.0))

    # Fase 2.1: predicción de delta (magnitud económica)
    delta_pct = None
    delta_usd_ton = None
    if bundle.get("delta_model") is not None:
        try:
            delta_pct = float(bundle["delta_model"].predict(X)[0])
            # Convertir a USD/ton aproximado
            current_price_cents = float(last_row["Soybeans"].iloc[0])
            current_price_ton = current_price_cents * 0.01 * 36.7437
            delta_usd_ton = round(delta_pct * current_price_ton, 2)
        except Exception:
            delta_pct = None

    # Decision advisory básica
    if p_cal >= threshold + 0.10:
        advisory = "WAIT"
    elif p_cal <= threshold - 0.10:
        advisory = "SELL"
    else:
        advisory = "INDIFFERENT"

    confidence = "high" if abs(p_cal - 0.5) >= 0.20 else "medium" if abs(p_cal - 0.5) >= 0.10 else "low"

    # Fase 2.2: PARTIAL_SELL recomendación combinada
    cost_pct = bundle["cost_pct"]
    partial_label, sell_pct, hold_pct, reason = _combine_partial_sell(
        p_cal, delta_pct, cost_pct
    )

    return {
        "ok":                          True,
        "as_of":                       str(last_row["Date"].iloc[0])[:10],
        "current_price":               float(last_row["Soybeans"].iloc[0]),
        "cost_pct_used":               round(cost_pct * 100, 3),
        "prob_wait_pays_raw":          round(p_raw, 3),
        "prob_wait_pays_calibrated":   round(p_cal, 3),
        "threshold_active":            threshold,
        "decision_advisory":           advisory,
        "confidence":                  confidence,
        # Fase 2.1: magnitud
        "delta_predicted_pct":         round(delta_pct * 100, 3) if delta_pct is not None else None,
        "delta_predicted_usd_ton":     delta_usd_ton,
        "delta_model_mae_val":         round(bundle.get("val_delta_mae", float("nan")), 4)
                                          if not np.isnan(bundle.get("val_delta_mae", float("nan"))) else None,
        # Fase 2.2: PARTIAL_SELL combinado
        "partial_decision":            partial_label,
        "sell_pct":                    sell_pct,
        "hold_pct":                    hold_pct,
        "partial_reason":              reason,
        # Metadata
        "base_rate_wait_train":        round(bundle.get("base_rate_wait", 0) * 100, 1),
        "fp_weight":                   bundle.get("fp_weight"),
        "context_note": (
            "Panel INFORMATIVO. Combinación P(WAIT) calibrada + delta esperado "
            "(Fase 2). El backtest histórico mostró que las predicciones direccionales "
            "no superan al always-sell con significancia. Use como segunda opinión."
        ),
        "trained_at": bundle.get("trained_at"),
        "n_train":    bundle.get("n_train"),
    }


def save_decision_classifier(features_df: pd.DataFrame, artifacts_dir: str,
                               storage_per_ton_month: float = DEFAULT_STORAGE_USD_TON_MONTH,
                               financing_annual: float = DEFAULT_FINANCING_RATE_ANNUAL,
                               quality_risk_per_month: float = 0.0,
                               price_per_ton: float | None = None,
                               profile_name: str = "default",
                               horizons: list[int] | None = None) -> dict:
    """Entrena clasificadores multi-horizonte (Fase 1.2) + análogos (Fase 1.1)
    + soporte de profile (Fase 1.3).
    Persiste a artifacts/decision_classifier.json (legacy, default profile)
    y a artifacts/decision_classifier/{profile}.json para multi-profile.
    """
    if price_per_ton is None:
        try:
            last_p = float(features_df.sort_values("Date").iloc[-1]["Soybeans"])
            price_per_ton = last_p * 0.01 * 36.7437   # cents/bu → USD/ton
        except Exception:
            price_per_ton = DEFAULT_PRICE_USD_TON

    if horizons is None:
        horizons = HORIZONS_MULTI

    out = {
        "ok": True,
        "as_of":             datetime.now().isoformat(timespec="seconds"),
        "profile_name":      profile_name,
        "storage_per_ton_month": storage_per_ton_month,
        "financing_annual":  financing_annual,
        "quality_risk_per_month": quality_risk_per_month,
        "price_per_ton_used": round(price_per_ton, 2),
        "horizons": {},
        "context_note": (
            "Panel INFORMATIVO multi-horizonte. Backtest 5y mostró que esta probabilidad "
            "calibrada es la mejor aproximación predictiva validada (PnL -1.36%/año vs "
            "always-sell, el menor gap entre 21 configs probadas), pero SIGUE PERDIENDO "
            "al always-sell. Use como segunda opinión, no como decisor único."
        ),
    }

    bundles = {}
    for h in horizons:
        cost_pct_h = _compute_cost_pct(storage_per_ton_month, financing_annual,
                                         price_per_ton, quality_risk_per_month, horizon_days=h)
        bundle = train_decision_classifier(features_df, cost_pct=cost_pct_h, horizon_days=h)
        if not bundle.get("ok"):
            out["horizons"][f"{h}d"] = {"ok": False, "error": bundle.get("error", "training failed")}
            continue
        bundles[h] = bundle
        pred = predict_decision(features_df, bundle, threshold=0.5)
        pred["cost_pct_used"] = round(cost_pct_h * 100, 3)
        pred["val_metrics"] = {
            "n_train":       bundle["n_train"],
            "n_val":         bundle["n_val"],
            "val_acc":       round(bundle["val_acc"], 3) if not np.isnan(bundle["val_acc"]) else None,
            "val_brier_raw": round(bundle["val_brier_raw"], 4) if not np.isnan(bundle["val_brier_raw"]) else None,
            "val_brier_cal": round(bundle["val_brier_cal"], 4) if not np.isnan(bundle["val_brier_cal"]) else None,
        }
        # Análogos para este horizonte (Fase 1.1)
        try:
            pred["analogs"] = find_decision_analogs(features_df, cost_pct_h,
                                                      horizon_days=h, k=20)
        except Exception as _e:
            pred["analogs"] = {"ok": False, "error": str(_e)}
        out["horizons"][f"{h}d"] = pred

    # Recomendar el horizonte donde el modelo tiene MÁS convicción
    best_h = None
    best_conf = -1
    for h_str, h_data in out["horizons"].items():
        if not h_data.get("ok", True):
            continue
        p = h_data.get("prob_wait_pays_calibrated", 0.5)
        conf = abs(p - 0.5)
        if conf > best_conf:
            best_conf = conf
            best_h = h_str
    out["best_horizon"] = best_h
    out["best_horizon_confidence"] = round(best_conf, 3)

    # Persistir
    os.makedirs(artifacts_dir, exist_ok=True)
    profile_dir = os.path.join(artifacts_dir, "decision_classifier")
    os.makedirs(profile_dir, exist_ok=True)

    # Default profile va al path legacy para no romper el endpoint actual
    if profile_name == "default":
        with open(os.path.join(artifacts_dir, "decision_classifier.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
    # Siempre persistir también en el subdir multi-profile
    with open(os.path.join(profile_dir, f"{profile_name}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    # Persistir bundles
    try:
        import joblib
        for h, bundle in bundles.items():
            joblib.dump(bundle, os.path.join(profile_dir, f"{profile_name}_h{h}d.joblib"))
    except Exception:
        pass

    return out


def save_all_profiles(features_df: pd.DataFrame, artifacts_dir: str,
                       price_per_ton: float | None = None) -> dict:
    """Entrena y persiste decision classifier para TODOS los profiles
    (Fase 1.3). Output: dict {profile_name: result}."""
    results = {}
    for prof_name, params in PRODUCER_PROFILES.items():
        try:
            r = save_decision_classifier(
                features_df, artifacts_dir,
                storage_per_ton_month=params["storage"],
                financing_annual=params["financing"],
                quality_risk_per_month=params.get("quality_risk_per_month", 0.0),
                price_per_ton=price_per_ton,
                profile_name=prof_name,
            )
            results[prof_name] = r
        except Exception as e:
            results[prof_name] = {"ok": False, "error": str(e)}
    return results
