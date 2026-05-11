"""
scripts/producer_strategy_backtest.py
Backtest walk-forward de la cadena COMPLETA del productor:
  features → entrenamiento → forecast → economic_utility → decisión → P&L real

Idea: simular que un productor con 100 ton/mes para vender hubiera seguido al
sistema durante el último año. Comparar contra estrategias naive.

Diseño honesto (sin leakage):
  - Para cada fecha D de decisión, entrenamos el modelo SÓLO con datos < D.
  - economic_utility(features_<D) → decisión SELL_NOW | WAIT | INDIFFERENT.
  - Si SELL_NOW → vendemos al precio de D.
  - Si WAIT     → esperamos 30d, vendemos al precio de D+30 menos costos.
  - Si INDIFFERENT → mitad y mitad (default conservador).

Estrategias comparadas:
  M  Modelo: sigue las decisiones del modelo
  H  Always sell: vende todo en cada D (productor neutral, sin opinión)
  W  Always wait: posterga 30d en cada D (apuesta direccional alcista pasiva)
  A  Average: vende al promedio del precio en [D, D+30]
  O  Oracle: vende cuando el precio fue más alto en [D, D+30] (cota superior)
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model.train_horizons   import train_horizon
from src.model.predict_horizons import forecast_anchors
from src.model.economic_utility import utility_wait_vs_sell, BU_PER_TON, CENTS_TO_USD

FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")

# Parámetros del productor simulado
TONS_PER_DECISION = 100   # vende 100 ton por decisión
STORAGE_USD_TON_MO = 6.0
FINANCING_RATE     = 0.08
HORIZON_DAYS       = 30

# Backtest config
DECISION_FREQ_DAYS = 21   # una decisión cada ~3 semanas (≈12 al año)
LOOKBACK_DAYS      = 365  # último año
TRAIN_YEARS        = 5
VAL_DAYS           = 180


def run_backtest():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    end = df["Date"].max()
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)

    decision_dates = []
    d = start
    while d <= end - pd.Timedelta(days=HORIZON_DAYS):
        # buscar primer trading day >= d
        future = df[df["Date"] >= d]
        if future.empty:
            break
        decision_dates.append(future.iloc[0]["Date"])
        d = future.iloc[0]["Date"] + pd.Timedelta(days=DECISION_FREQ_DAYS)

    print(f"Backtest productor: {len(decision_dates)} decisiones entre "
          f"{decision_dates[0].date()} y {decision_dates[-1].date()}\n")

    rows = []
    for D in decision_dates:
        # Walk-forward: features hasta D-1 (sin ver D)
        df_train = df[df["Date"] < D].copy()
        if len(df_train) < TRAIN_YEARS * 252:
            continue

        # Fecha del precio de venta si esperamos: D + HORIZON
        future = df[df["Date"] >= D + pd.Timedelta(days=HORIZON_DAYS)]
        if future.empty:
            continue
        D30 = future.iloc[0]["Date"]

        price_D    = float(df.loc[df["Date"] == D, "Soybeans"].iloc[0])
        price_D30  = float(df.loc[df["Date"] == D30, "Soybeans"].iloc[0])
        # Precio promedio en [D, D30] para estrategia "Average"
        price_avg  = float(df[(df["Date"] >= D) & (df["Date"] <= D30)]["Soybeans"].mean())
        # Oracle: máximo en [D, D30]
        price_max  = float(df[(df["Date"] >= D) & (df["Date"] <= D30)]["Soybeans"].max())

        # Entrenar modelo con datos hasta D-1
        try:
            t0 = time.time()
            bundles = {}
            for h in (7, 30):
                target_col = f"ret_{h}d_fwd"
                if target_col not in df_train.columns:
                    continue
                bundles[h] = train_horizon(df_train, h, target_col)
            elapsed = time.time() - t0

            # Persistir bundles temporariamente para forecast_anchors
            tmp_dir = os.path.join(ROOT, "artifacts_eval", "wf_tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            import joblib
            for h, b in bundles.items():
                joblib.dump(b, os.path.join(tmp_dir, f"model_h{h}d.joblib"))

            # economic_utility usa features.iloc[-1] como "ahora"
            util = utility_wait_vs_sell(
                df_train, storage_cost_per_ton_month=STORAGE_USD_TON_MO,
                financing_rate_annual=FINANCING_RATE,
                horizon_days=HORIZON_DAYS, n_paths=2000,
                artifacts_dir=tmp_dir,
            )
            decision = util.get("decision", "INDIFFERENT")
            prob_wait = util.get("prob_wait_better_pct", 50)
            alpha_30d = bundles.get(30, {}).get("alpha")
        except Exception as e:
            print(f"  [WARN] D={D.date()} error: {e}")
            continue

        # Calcular precios USD/ton para cada estrategia
        to_ton = BU_PER_TON * CENTS_TO_USD
        sell_now_usd_ton  = price_D * to_ton
        # Si esperamos, descontamos costos
        months = HORIZON_DAYS / 30.0
        cost   = STORAGE_USD_TON_MO * months + (price_D * to_ton) * FINANCING_RATE * (HORIZON_DAYS / 365.0)
        wait_usd_ton      = price_D30 * to_ton - cost
        avg_usd_ton       = price_avg * to_ton - cost / 2  # costo prorrateado a mitad
        oracle_usd_ton    = price_max * to_ton - cost / 2

        # P&L per ton para cada estrategia
        pnl = {
            "MODEL":       wait_usd_ton if decision == "WAIT"
                           else 0.5 * sell_now_usd_ton + 0.5 * wait_usd_ton if decision == "INDIFFERENT"
                           else sell_now_usd_ton,
            "ALWAYS_SELL": sell_now_usd_ton,
            "ALWAYS_WAIT": wait_usd_ton,
            "AVERAGE":     avg_usd_ton,
            "ORACLE":      oracle_usd_ton,
        }

        rows.append({
            "decision_date": D, "exit_date": D30,
            "price_D":   price_D, "price_D30": price_D30,
            "price_avg": round(price_avg, 2), "price_max": round(price_max, 2),
            "decision":  decision, "prob_wait_pct": prob_wait, "alpha_30d": alpha_30d,
            "elapsed_s": round(elapsed, 1),
            **{f"pnl_{k}_usd_ton": round(v, 2) for k, v in pnl.items()},
        })
        print(f"  D={D.date()} → {decision:11s}  α30={alpha_30d}  "
              f"sell=${sell_now_usd_ton:6.2f}  wait=${wait_usd_ton:6.2f}  "
              f"model=${pnl['MODEL']:6.2f}  ({elapsed:.1f}s)")

    if not rows:
        print("\nSin decisiones suficientes.")
        return

    R = pd.DataFrame(rows)
    out_path = os.path.join(ROOT, "artifacts_eval", "producer_backtest.csv")
    R.to_csv(out_path, index=False)
    print(f"\n💾 {out_path}")

    # ── Resumen ────────────────────────────────────────────
    n = len(R)
    print(f"\n┌─────────────────────┬──────────┬──────────┬──────────┬───────────┐")
    print(f"│ Estrategia          │ P&L tot  │ P&L avg  │ vs MODEL │ vs ORACLE │")
    print(f"├─────────────────────┼──────────┼──────────┼──────────┼───────────┤")
    model_total = R["pnl_MODEL_usd_ton"].sum() * TONS_PER_DECISION
    oracle_total = R["pnl_ORACLE_usd_ton"].sum() * TONS_PER_DECISION
    for col, label in [
        ("pnl_MODEL_usd_ton",       "Modelo"),
        ("pnl_ALWAYS_SELL_usd_ton", "Always sell (D)"),
        ("pnl_ALWAYS_WAIT_usd_ton", "Always wait (D+30)"),
        ("pnl_AVERAGE_usd_ton",     "Average price"),
        ("pnl_ORACLE_usd_ton",      "Oracle (cota sup)"),
    ]:
        total = R[col].sum() * TONS_PER_DECISION
        avg   = R[col].mean()
        vs_model  = (total / model_total - 1) * 100 if model_total else 0
        vs_oracle = (total / oracle_total) * 100
        print(f"│ {label:<19} │ ${total:>7.0f} │ ${avg:>6.2f}  │ {vs_model:+6.1f} % │ {vs_oracle:>7.1f} % │")
    print(f"└─────────────────────┴──────────┴──────────┴──────────┴───────────┘")
    print(f"  ({n} decisiones × {TONS_PER_DECISION} ton = {n*TONS_PER_DECISION} ton totales)")

    # Adherencia/calidad de la decisión
    correct_wait = ((R["decision"] == "WAIT")     & (R["price_D30"] > R["price_D"])).sum()
    correct_sell = ((R["decision"] == "SELL_NOW") & (R["price_D30"] < R["price_D"])).sum()
    decisions_taken = (R["decision"] != "INDIFFERENT").sum()
    print(f"\n  Acierto direccional: WAIT correcto={correct_wait}, SELL correcto={correct_sell}, "
          f"total decisiones tomadas={decisions_taken}/{n} "
          f"({(correct_wait+correct_sell)/max(decisions_taken,1)*100:.0f}% si no contamos INDIFFERENT)")
    print(f"  Mix de decisiones: {dict(R['decision'].value_counts())}")


if __name__ == "__main__":
    run_backtest()
