"""
scripts/producer_strategy_backtest_v2.py
Backtest walk-forward unificado:
  1. Frecuencia de decisión 7 días (vs 21d original) → ~50 decisiones / año
  2. 3 perfiles distintos de productor (storage / financing diferentes)
  3. 2 horizontes (7d y 30d)

→ una sola pasada expensiva de walk-forward training, múltiples evaluaciones por
  fecha con economic_utility reusando los modelos entrenados.

Usa N_SEEDS=1 (single seed) para que termine en tiempo razonable.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Forzar single-seed para velocidad (el backtest no necesita multi-seed)
os.environ["HORIZONS_N_SEEDS"] = "1"

from src.model.train_horizons   import train_horizon
from src.model.economic_utility import utility_wait_vs_sell, BU_PER_TON, CENTS_TO_USD

FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")

DECISION_FREQ_DAYS = 7
LOOKBACK_DAYS      = 365
TRAIN_YEARS        = 5
TONS_PER_DECISION  = 100

# Escenarios de productor (storage USD/ton/mes, financing anual)
SCENARIOS = {
    "default":     {"storage": 6.0,  "financing": 0.08},
    "no_storage":  {"storage": 0.0,  "financing": 0.05},   # productor con storage propio + crédito barato
    "high_cost":   {"storage": 10.0, "financing": 0.15},   # productor con costos altos
}
HORIZONS_TO_TEST = [7, 30]


def main():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    end = df["Date"].max()
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)

    decision_dates = []
    d = start
    while d <= end - pd.Timedelta(days=max(HORIZONS_TO_TEST)):
        future = df[df["Date"] >= d]
        if future.empty:
            break
        decision_dates.append(future.iloc[0]["Date"])
        d = future.iloc[0]["Date"] + pd.Timedelta(days=DECISION_FREQ_DAYS)

    print(f"Backtest v2 — {len(decision_dates)} decisiones (freq={DECISION_FREQ_DAYS}d) "
          f"× {len(SCENARIOS)} escenarios × {len(HORIZONS_TO_TEST)} horizontes\n")

    rows = []   # un row por (date, scenario, horizon)
    tmp_dir = os.path.join(ROOT, "artifacts_eval", "wf_tmp_v2")
    os.makedirs(tmp_dir, exist_ok=True)
    import joblib

    for D in decision_dates:
        df_train = df[df["Date"] < D].copy()
        if len(df_train) < TRAIN_YEARS * 252:
            continue

        # Entrenar modelos 7d y 30d una sola vez
        try:
            t0 = time.time()
            for h in HORIZONS_TO_TEST:
                target_col = f"ret_{h}d_fwd"
                if target_col not in df_train.columns:
                    continue
                bundle = train_horizon(df_train, h, target_col)
                joblib.dump(bundle, os.path.join(tmp_dir, f"model_h{h}d.joblib"))
            elapsed_train = time.time() - t0
        except Exception as e:
            print(f"  [WARN] D={D.date()} train error: {e}")
            continue

        # Para cada escenario × horizonte, calcular utilidad y P&L real
        for scen_name, params in SCENARIOS.items():
            for H in HORIZONS_TO_TEST:
                # Precio de salida
                exit_rows = df[df["Date"] >= D + pd.Timedelta(days=H)]
                if exit_rows.empty:
                    continue
                D_exit = exit_rows.iloc[0]["Date"]
                price_D    = float(df.loc[df["Date"] == D, "Soybeans"].iloc[0])
                price_exit = float(df.loc[df["Date"] == D_exit, "Soybeans"].iloc[0])

                try:
                    util = utility_wait_vs_sell(
                        df_train,
                        storage_cost_per_ton_month=params["storage"],
                        financing_rate_annual=params["financing"],
                        horizon_days=H, n_paths=2000,
                        artifacts_dir=tmp_dir,
                    )
                    decision  = util.get("decision", "INDIFFERENT")
                    prob_wait = util.get("prob_wait_better_pct", 50)
                except Exception as e:
                    print(f"  [WARN] D={D.date()} {scen_name} {H}d util error: {e}")
                    continue

                # P&L real en USD/ton para cada estrategia
                to_ton = BU_PER_TON * CENTS_TO_USD
                months = H / 30.0
                cost   = params["storage"] * months + (price_D * to_ton) * params["financing"] * (H / 365.0)
                sell_now_v = price_D    * to_ton
                wait_v     = price_exit * to_ton - cost
                model_v    = (wait_v if decision == "WAIT"
                              else 0.5 * sell_now_v + 0.5 * wait_v if decision == "INDIFFERENT"
                              else sell_now_v)

                rows.append({
                    "date": D, "scenario": scen_name, "horizon": H,
                    "decision": decision, "prob_wait_pct": prob_wait,
                    "price_D": price_D, "price_exit": price_exit,
                    "pnl_MODEL":       round(model_v, 2),
                    "pnl_ALWAYS_SELL": round(sell_now_v, 2),
                    "pnl_ALWAYS_WAIT": round(wait_v, 2),
                })
        print(f"  D={D.date()}  ({elapsed_train:.1f}s train, "
              f"{len(SCENARIOS)*len(HORIZONS_TO_TEST)} evals)")

    if not rows:
        print("Sin decisiones evaluadas.")
        return

    R = pd.DataFrame(rows)
    out_path = os.path.join(ROOT, "artifacts_eval", "producer_backtest_v2.csv")
    R.to_csv(out_path, index=False)
    print(f"\n💾 {out_path}  ({len(R)} filas)")

    # ── Resumen por (scenario, horizon) ────────────────────────
    print(f"\n┌─────────────┬─────┬──────┬──────────┬──────────┬──────────┬──────────┬─────────┐")
    print(f"│ Scenario    │ H   │ N    │ Modelo   │ AlwaysSel│ AlwaysWai│ vs Sell  │ Mix     │")
    print(f"├─────────────┼─────┼──────┼──────────┼──────────┼──────────┼──────────┼─────────┤")
    for scen in SCENARIOS:
        for H in HORIZONS_TO_TEST:
            sub = R[(R["scenario"] == scen) & (R["horizon"] == H)]
            if sub.empty:
                continue
            n = len(sub)
            m_avg  = sub["pnl_MODEL"].mean()
            s_avg  = sub["pnl_ALWAYS_SELL"].mean()
            w_avg  = sub["pnl_ALWAYS_WAIT"].mean()
            lift_s = (m_avg - s_avg) / s_avg * 100
            decs   = dict(sub["decision"].value_counts())
            mix    = "/".join(f"{k}:{v}" for k, v in decs.items())[:25]
            print(f"│ {scen:<11} │ {H:>2}d │ {n:>4} │ ${m_avg:>6.2f} │ ${s_avg:>6.2f} │ ${w_avg:>6.2f} │ {lift_s:+6.2f}% │ {mix:<7} │")
    print(f"└─────────────┴─────┴──────┴──────────┴──────────┴──────────┴──────────┴─────────┘")

    # Significancia: tests de Wilcoxon paired (modelo vs always-sell)
    print(f"\n  Significancia estadística (Wilcoxon paired modelo vs always-sell):")
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    if wilcoxon is not None:
        for scen in SCENARIOS:
            for H in HORIZONS_TO_TEST:
                sub = R[(R["scenario"] == scen) & (R["horizon"] == H)]
                if len(sub) < 10:
                    continue
                diff = sub["pnl_MODEL"] - sub["pnl_ALWAYS_SELL"]
                if (diff != 0).sum() < 5:
                    print(f"    {scen:<11} {H:>2}d: modelo == always-sell en todas las decisiones (N/A)")
                    continue
                stat, pval = wilcoxon(sub["pnl_MODEL"], sub["pnl_ALWAYS_SELL"], zero_method="wilcox")
                sign = "✓ significativo" if pval < 0.05 else "× ruido"
                print(f"    {scen:<11} {H:>2}d:  Δ_avg=${diff.mean():+6.2f}/ton  p={pval:.3f}  {sign}")


if __name__ == "__main__":
    main()
