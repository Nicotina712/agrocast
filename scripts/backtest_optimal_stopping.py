"""
scripts/backtest_optimal_stopping.py
Walk-forward backtest: optimal stopping (DP) vs heurística (economic_utility) vs
naive (always-sell, always-wait).

En cada fecha D del último año, entrenamos modelo horizons y calculamos las
3 decisiones; luego comparamos P&L real con outcome 30d después.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Forzar single-seed y desactivar Granger durante backtest (queremos comparar puro)
os.environ["HORIZONS_N_SEEDS"]      = "1"
os.environ["USE_GRANGER_PRUNING"]   = "0"

from src.model.train_horizons   import train_horizon
from src.model.economic_utility import utility_wait_vs_sell, BU_PER_TON, CENTS_TO_USD

FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")
DECISION_FREQ_DAYS = 14
LOOKBACK_DAYS      = 365
TRAIN_YEARS        = 5
HORIZON_DAYS       = 30
N_PATHS            = 800
STORAGE_USD        = 6.0
FINANCING          = 0.08


def main():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    end = df["Date"].max()
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)

    decision_dates = []
    d = start
    while d <= end - pd.Timedelta(days=HORIZON_DAYS):
        future = df[df["Date"] >= d]
        if future.empty:
            break
        decision_dates.append(future.iloc[0]["Date"])
        d = future.iloc[0]["Date"] + pd.Timedelta(days=DECISION_FREQ_DAYS)

    print(f"Backtest optimal stopping vs heurística: {len(decision_dates)} decisiones\n")

    rows = []
    tmp_dir = os.path.join(ROOT, "artifacts_eval", "wf_opt_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    import joblib
    from src.model.optimal_stopping import optimal_stopping_decision

    for D in decision_dates:
        df_train = df[df["Date"] < D].copy()
        if len(df_train) < TRAIN_YEARS * 252:
            continue

        future = df[df["Date"] >= D + pd.Timedelta(days=HORIZON_DAYS)]
        if future.empty:
            continue
        D_exit = future.iloc[0]["Date"]
        price_D    = float(df.loc[df["Date"] == D, "Soybeans"].iloc[0])
        price_exit = float(df.loc[df["Date"] == D_exit, "Soybeans"].iloc[0])

        try:
            t0 = time.time()
            for h in (7, 30):
                target_col = f"ret_{h}d_fwd"
                if target_col not in df_train.columns:
                    continue
                bundle = train_horizon(df_train, h, target_col)
                joblib.dump(bundle, os.path.join(tmp_dir, f"model_h{h}d.joblib"))

            # Heurística (economic_utility)
            heur = utility_wait_vs_sell(df_train, storage_cost_per_ton_month=STORAGE_USD,
                                         financing_rate_annual=FINANCING,
                                         horizon_days=HORIZON_DAYS, n_paths=N_PATHS,
                                         artifacts_dir=tmp_dir)
            heur_dec = heur.get("decision", "INDIFFERENT")

            # Optimal stopping (DP)
            opt = optimal_stopping_decision(df_train, storage_cost_per_ton_month=STORAGE_USD,
                                              financing_rate_annual=FINANCING,
                                              horizon_days=HORIZON_DAYS, n_paths=N_PATHS,
                                              artifacts_dir=tmp_dir)
            opt_dec = opt.get("decision_now", "INDIFFERENT")
            opt_reservation = opt.get("reservation_today_usd_ton", 0)
            elapsed = time.time() - t0
        except Exception as e:
            print(f"  [WARN] D={D.date()} error: {e}")
            continue

        # P&L real
        to_ton = BU_PER_TON * CENTS_TO_USD
        months = HORIZON_DAYS / 30.0
        cost = STORAGE_USD * months + (price_D * to_ton) * FINANCING * (HORIZON_DAYS / 365.0)
        sell_v = price_D    * to_ton
        wait_v = price_exit * to_ton - cost

        # Mapeo decisión → P&L
        def pnl(dec):
            return (wait_v if dec == "WAIT"
                    else 0.5 * sell_v + 0.5 * wait_v if dec == "INDIFFERENT"
                    else sell_v)

        row = {
            "date":       D, "exit": D_exit,
            "price_D":    price_D, "price_exit": price_exit,
            "heur_dec":   heur_dec, "opt_dec": opt_dec,
            "opt_reservation": opt_reservation,
            "pnl_HEUR":         round(pnl(heur_dec), 2),
            "pnl_OPTSTOP":      round(pnl(opt_dec), 2),
            "pnl_ALWAYS_SELL":  round(sell_v, 2),
            "pnl_ALWAYS_WAIT":  round(wait_v, 2),
            "elapsed_s":        round(elapsed, 1),
        }
        rows.append(row)
        print(f"  D={D.date()}  HEUR={heur_dec[:9]:9s}  OPT={opt_dec[:9]:9s}  "
              f"reserv=${opt_reservation:.0f}  "
              f"sell=${sell_v:.1f} wait=${wait_v:.1f}  ({elapsed:.0f}s)")

    if not rows:
        print("\nSin decisiones evaluadas.")
        return

    R = pd.DataFrame(rows)
    out_path = os.path.join(ROOT, "artifacts_eval", "backtest_opt_stop.csv")
    R.to_csv(out_path, index=False)
    print(f"\n💾 {out_path}")

    # Resumen
    print(f"\n┌──────────────────────┬──────────┬──────────┬──────────┐")
    print(f"│ Estrategia           │ Total    │ Avg/ton  │ vs ALWS  │")
    print(f"├──────────────────────┼──────────┼──────────┼──────────┤")
    base = R["pnl_ALWAYS_SELL"].mean()
    for col, label in [
        ("pnl_HEUR",          "Heurística"),
        ("pnl_OPTSTOP",       "Optimal stopping"),
        ("pnl_ALWAYS_SELL",   "Always sell"),
        ("pnl_ALWAYS_WAIT",   "Always wait"),
    ]:
        avg = R[col].mean()
        tot = R[col].sum()
        lift = (avg / base - 1) * 100
        print(f"│ {label:<20} │ ${tot:>7.0f} │ ${avg:>6.2f}  │ {lift:+6.2f}% │")
    print(f"└──────────────────────┴──────────┴──────────┴──────────┘")
    print(f"  N={len(R)} decisiones")

    # Significancia
    try:
        from scipy.stats import wilcoxon
        for label, col in [("OPTSTOP vs HEUR",       "pnl_OPTSTOP"),
                           ("OPTSTOP vs ALWAYS_SELL", "pnl_OPTSTOP")]:
            other = "pnl_HEUR" if "HEUR" in label else "pnl_ALWAYS_SELL"
            d = R[col] - R[other]
            if (d != 0).sum() < 5:
                print(f"  {label}: empate exacto en >{len(R)-5}/{len(R)} casos (no test)")
                continue
            stat, pval = wilcoxon(R[col], R[other], zero_method="wilcox")
            sign = "✓ significativo" if pval < 0.05 else "× ruido"
            print(f"  Wilcoxon {label}: Δavg=${d.mean():+.2f}/ton  p={pval:.3f}  {sign}")
    except ImportError:
        pass

    # Mix de decisiones
    print(f"\n  Mix HEUR: {dict(R['heur_dec'].value_counts())}")
    print(f"  Mix OPT:  {dict(R['opt_dec'].value_counts())}")


if __name__ == "__main__":
    main()
