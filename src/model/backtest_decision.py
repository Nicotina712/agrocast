"""
src/model/backtest_decision.py
Backtest formal OOS de estrategias de venta de soja — Fase 2.3.

Compara 6 estrategias en ventanas no-solapadas sobre el último año:
  always_sell   — vende hoy siempre (baseline, delta=0)
  always_wait   — espera H días siempre
  split_50      — 50% hoy, 50% en t+H
  oracle        — elige óptimo con información perfecta (techo teórico)
  model_binary  — usa P(WAIT) calibrada >= 0.5
  model_partial — usa sell_pct/hold_pct del _combine_partial_sell (Fase 2.2)

El modelo se entrena en datos ANTERIORES al periodo de test (OOS limpio).
No se reutilizan los bundles del pipeline para evitar leakage.

Output por horizonte:
  n_decisions, strategies: {mean_pnl_usd_ton, total_pnl_usd_ton,
  win_rate_pct, annualized_excess_usd_ton, pnl_q10/q90_usd_ton}
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
    _combine_partial_sell,
    train_decision_classifier,
    HORIZONS_MULTI,
    PRODUCER_PROFILES,
    FEATURE_COLS,
)

STRATEGY_NAMES = [
    "always_sell",
    "always_wait",
    "split_50",
    "oracle",
    "model_binary",
    "model_partial",
]


def _select_nonoverlapping(df: pd.DataFrame, date_col: str, step_days: int) -> list[int]:
    """Índices no solapados de df ordenado por date_col, avanzando step_days cada vez."""
    df = df.sort_values(date_col).reset_index(drop=True)
    selected = []
    i = 0
    while i < len(df):
        selected.append(i)
        curr_date = df.iloc[i][date_col]
        nxt = curr_date + pd.Timedelta(days=step_days)
        j = i + 1
        while j < len(df) and df.iloc[j][date_col] < nxt:
            j += 1
        i = j
    return selected


def _backtest_horizon(
    df: pd.DataFrame,
    horizon_days: int,
    cost_pct: float,
    price_per_ton: float,
    cutoff_date: pd.Timestamp,
) -> dict:
    """Backtest OOS para un horizonte dado.
    Entrena en df[Date < cutoff_date], evalúa en df[Date >= cutoff_date].
    """
    d = _build_decision_features(df).sort_values("Date").reset_index(drop=True)

    ret_col = f"ret_{horizon_days}d_fwd"
    if ret_col not in d.columns:
        d[ret_col] = d["Soybeans"].pct_change(horizon_days).shift(-horizon_days)

    # Train split (OOS limpio)
    train_df = d[d["Date"] < cutoff_date].copy()
    if len(train_df) < 300:
        return {"ok": False, "error": "insufficient train data"}

    bundle = train_decision_classifier(train_df, cost_pct=cost_pct, horizon_days=horizon_days)
    if not bundle.get("ok"):
        return {"ok": False, "error": bundle.get("error", "training failed")}

    model      = bundle["model"]
    calibrator = bundle.get("calibrator")
    delta_model = bundle.get("delta_model")
    feats      = bundle["features"]

    # Test split: necesita outcome real conocido
    test_df = d[(d["Date"] >= cutoff_date) & d[ret_col].notna()].copy()
    if len(test_df) < 3:
        return {"ok": False, "error": "insufficient test data"}

    test_df = test_df.sort_values("Date").reset_index(drop=True)
    X_test = test_df[feats].fillna(0).replace([np.inf, -np.inf], 0)

    p_raw = model.predict_proba(X_test)[:, 1].astype(float)
    p_cal = np.clip(calibrator.transform(p_raw), 0.0, 1.0) if calibrator else p_raw

    delta_pct_pred = None
    if delta_model is not None:
        try:
            delta_pct_pred = delta_model.predict(X_test).astype(float)
        except Exception:
            delta_pct_pred = None

    # Ventanas no solapadas
    selected = _select_nonoverlapping(test_df, "Date", horizon_days)
    if not selected:
        return {"ok": False, "error": "no non-overlapping windows"}

    records = []
    for i in selected:
        ret_actual = float(test_df.iloc[i][ret_col])
        p = float(p_cal[i])
        d_pct = float(delta_pct_pred[i]) if delta_pct_pred is not None else None

        partial_label, sell_pct, hold_pct, _ = _combine_partial_sell(p, d_pct, cost_pct)

        # Delta económico real de esperar vs vender hoy
        delta_actual = ret_actual - cost_pct

        pnls = {
            "always_sell":   0.0,
            "always_wait":   delta_actual,
            "split_50":      0.5 * delta_actual,
            "oracle":        max(0.0, delta_actual),
            "model_binary":  delta_actual if p >= 0.5 else 0.0,
            "model_partial": (hold_pct / 100.0) * delta_actual,
        }
        records.append({
            "date":          str(test_df.iloc[i]["Date"])[:10],
            "ret_actual":    round(ret_actual * 100, 3),
            "delta_actual":  round(delta_actual * 100, 3),
            "p_cal":         round(p, 3),
            "partial_label": partial_label,
            "sell_pct":      sell_pct,
            "hold_pct":      hold_pct,
            **{f"pnl_{s}": round(v * price_per_ton, 3) for s, v in pnls.items()},
        })

    n = len(records)
    # Decisiones por año aproximado (basado en ventanas del periodo de test real)
    if n >= 2:
        test_span_days = (
            pd.Timestamp(records[-1]["date"]) - pd.Timestamp(records[0]["date"])
        ).days or 1
        decisions_per_year = n / (test_span_days / 365.25)
    else:
        decisions_per_year = n

    summary = {}
    for s in STRATEGY_NAMES:
        col = f"pnl_{s}"
        vals = [r[col] for r in records]
        mean_v = float(np.mean(vals))
        summary[s] = {
            "n_decisions":              n,
            "mean_pnl_usd_ton":         round(mean_v, 2),
            "total_pnl_usd_ton":        round(float(np.sum(vals)), 2),
            "win_rate_pct":             round(float(np.mean([v > 0 for v in vals])) * 100, 1),
            "annualized_excess_usd_ton": round(mean_v * decisions_per_year, 2),
            "pnl_q10_usd_ton":          round(float(np.quantile(vals, 0.10)), 2),
            "pnl_q90_usd_ton":          round(float(np.quantile(vals, 0.90)), 2),
        }

    return {
        "ok":          True,
        "n_decisions": n,
        "cost_pct":    round(cost_pct * 100, 3),
        "horizon_days": horizon_days,
        "strategies":  summary,
        "sample_records": records[:5],
    }


def run_decision_backtest(
    df: pd.DataFrame,
    profile_name: str = "default",
    horizons: list[int] | None = None,
    test_months: int = 12,
    price_per_ton: float | None = None,
) -> dict:
    """Backtest comparativo OOS para todos los horizontes del profile dado.

    Parámetros:
        df:           features DataFrame con columna Date y Soybeans.
        profile_name: nombre del PRODUCER_PROFILES a usar.
        horizons:     lista de horizontes en días (default HORIZONS_MULTI).
        test_months:  meses finales del dataset usados como test OOS.
        price_per_ton: precio de referencia en USD/ton (default: último precio).
    """
    if horizons is None:
        horizons = HORIZONS_MULTI

    profile_params = _resolve_profile(profile_name)

    d = _build_decision_features(df).sort_values("Date").reset_index(drop=True)

    if price_per_ton is None:
        last_p = float(d.iloc[-1]["Soybeans"])
        price_per_ton = last_p * 0.01 * 36.7437  # cents/bu → USD/ton

    cutoff_date = d["Date"].max() - pd.DateOffset(months=test_months)

    result = {
        "ok":          True,
        "as_of":       datetime.now().isoformat(timespec="seconds"),
        "profile_name": profile_name,
        "test_months": test_months,
        "cutoff_date": str(cutoff_date)[:10],
        "price_per_ton_used": round(price_per_ton, 2),
        "horizons":    {},
        "context_note": (
            "Backtest OOS — modelo entrenado en datos previos al periodo de test. "
            "Delta siempre relativo a Always-Sell (baseline=0). Oracle = techo teórico "
            "con información perfecta. Panel INFORMATIVO."
        ),
    }

    for h in horizons:
        cost_pct = _compute_cost_pct(
            profile_params["storage"],
            profile_params["financing"],
            price_per_ton,
            profile_params.get("quality_risk_per_month", 0.0),
            horizon_days=h,
        )
        try:
            h_result = _backtest_horizon(d, h, cost_pct, price_per_ton, cutoff_date)
        except Exception as e:
            h_result = {"ok": False, "error": str(e)}
        result["horizons"][f"{h}d"] = h_result

    return result


def save_backtest_decision(
    df: pd.DataFrame,
    artifacts_dir: str,
    profile_name: str = "default",
    test_months: int = 12,
    price_per_ton: float | None = None,
) -> dict:
    """Ejecuta el backtest y persiste en artifacts/backtest_decision/{profile}.json."""
    result = run_decision_backtest(
        df, profile_name=profile_name,
        test_months=test_months, price_per_ton=price_per_ton,
    )
    out_dir = os.path.join(artifacts_dir, "backtest_decision")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{profile_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def save_all_backtest_profiles(
    df: pd.DataFrame,
    artifacts_dir: str,
    test_months: int = 12,
    price_per_ton: float | None = None,
) -> dict:
    """Persiste backtest para TODOS los profiles."""
    results = {}
    for prof_name in PRODUCER_PROFILES:
        try:
            results[prof_name] = save_backtest_decision(
                df, artifacts_dir,
                profile_name=prof_name,
                test_months=test_months,
                price_per_ton=price_per_ton,
            )
        except Exception as e:
            results[prof_name] = {"ok": False, "error": str(e)}
    return results
