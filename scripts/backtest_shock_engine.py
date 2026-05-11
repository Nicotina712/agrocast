"""
scripts/backtest_shock_engine.py
Backtest walk-forward: ¿la recomendación condicional del Shock Engine
le gana al always-sell o al modelo regular DURANTE shocks pasados?

Diseño:
  1. Recorremos el catálogo histórico de shocks (top 15%)
  2. Para cada shock D, simulamos lo que el Shock Engine habría dicho
     (build_catalog hasta D-1, find_analogs, build_recommendation)
  3. Mapeamos action → P&L 30d real:
        WAIT_FOR_PEAK → vender al day 30 (suponiendo que vendemos el día 30)
                        ó al peak detectable (en backtest usamos day 30 sin look-ahead)
        SELL_NOW       → vender hoy
        AMBIGUOUS      → mitad y mitad
  4. Comparamos vs always-sell, always-wait
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model.shock_engine import (build_catalog, find_analogs,
                                      aggregate_outcomes, build_recommendation)
from src.model.economic_utility import BU_PER_TON, CENTS_TO_USD

FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")
HORIZON_DAYS = 30
STORAGE_USD  = 6.0
FINANCING    = 0.08
TONS         = 100


def main():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    # Catálogo completo histórico
    print("Construyendo catálogo histórico…")
    cat_full = build_catalog(df)
    print(f"  Total shocks en catálogo: {len(cat_full)}")
    print(f"  Strong (top 5%): {int(cat_full['is_strong'].sum())}")
    print(f"  Near (top 15%):  {int((~cat_full['is_strong']).sum())}")

    # Para cada shock, evaluar action recomendada y P&L
    rows = []
    print(f"\nEvaluando {len(cat_full)} shocks (puede tomar varios minutos)…")
    last_report = time.time()
    for i, sh in cat_full.iterrows():
        D = sh["Date"]
        # Sólo evaluamos shocks con outcome 30d completo
        future = df[df["Date"] >= D + pd.Timedelta(days=HORIZON_DAYS)]
        if future.empty:
            continue
        D_exit = future.iloc[0]["Date"]

        # Catalog histórico HASTA D-1 (sin look-ahead)
        df_train = df[df["Date"] < D].copy()
        if len(df_train) < 252:   # mínimo 1 año de history
            continue
        cat_hist = build_catalog(df_train)

        # Reconstruir current dict (catalog usa columnas *_at_shock)
        current = {
            "is_shock":        True,
            "is_strong_shock": bool(sh["is_strong"]),
            "is_near_shock":   not bool(sh["is_strong"]),
            "shock_type":      sh["shock_type"],
            "shock_direction": sh["shock_direction"],
            "window_active":   sh["window_active"],
            "ret_5d_pct":      float(sh["ret_5d_at_shock"]) * 100,
            "ret_10d_pct":     float(sh["ret_10d_at_shock"]) * 100,
            "oil_5d_pct":      float(sh["oil_5d_at_shock"]) * 100,
            "oil_10d_pct":     float(sh["oil_10d_at_shock"]) * 100,
            "ret_active_pct":  float(sh["ret_active_at_shock"]) * 100,
            "oil_active_pct":  float(sh["oil_active_at_shock"]) * 100,
        }

        analogs = find_analogs(cat_hist, current)
        stats   = aggregate_outcomes(analogs)
        rec     = build_recommendation(current, stats)

        # P&L
        p_D    = float(sh["Soybeans"])
        p_exit = float(future.iloc[0]["Soybeans"])
        to_ton = BU_PER_TON * CENTS_TO_USD
        months = HORIZON_DAYS / 30.0
        cost   = STORAGE_USD * months + (p_D * to_ton) * FINANCING * (HORIZON_DAYS / 365.0)
        sell_v = p_D    * to_ton
        wait_v = p_exit * to_ton - cost

        # Mapear action → P&L
        action = rec.get("action", "NO_SHOCK")
        if action in ("WAIT_FOR_PEAK", "WAIT_RECOVERY"):
            pnl = wait_v
        elif action == "SELL_NOW":
            pnl = sell_v
        else:    # AMBIGUOUS, INSUFFICIENT_ANALOGS, NO_SHOCK → mitad
            pnl = 0.5 * sell_v + 0.5 * wait_v

        rows.append({
            "date": D, "exit_date": D_exit,
            "shock_type":      sh["shock_type"],
            "shock_direction": sh["shock_direction"],
            "is_strong":       bool(sh["is_strong"]),
            "ret_active":      float(sh["ret_active_at_shock"]),
            "n_analogs":       stats.get("n", 0),
            "rec_action":      action,
            "rec_confidence":  rec.get("confidence", "low"),
            "pnl_SHOCK_ENG":   round(pnl, 2),
            "pnl_ALWAYS_SELL": round(sell_v, 2),
            "pnl_ALWAYS_WAIT": round(wait_v, 2),
            "p_D":             p_D,
            "p_exit":          p_exit,
        })

        if time.time() - last_report > 30:
            print(f"  Procesados {len(rows)}/{len(cat_full)}…")
            last_report = time.time()

    R = pd.DataFrame(rows)
    print(f"\nTotal evaluados: {len(R)}")
    out_path = os.path.join(ROOT, "artifacts_eval", "backtest_shock_engine.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    R.to_csv(out_path, index=False)
    print(f"💾 {out_path}")

    # ── Análisis global ─────────────────────────────────────
    print(f"\n══════════════════════════════════════════════════════════════")
    print(f"  Resumen global ({len(R)} shocks evaluados)")
    print(f"══════════════════════════════════════════════════════════════")
    base = R["pnl_ALWAYS_SELL"].mean()
    for col, label in [
        ("pnl_SHOCK_ENG",    "Shock Engine"),
        ("pnl_ALWAYS_SELL",  "Always Sell"),
        ("pnl_ALWAYS_WAIT",  "Always Wait"),
    ]:
        avg = R[col].mean()
        lift = (avg / base - 1) * 100
        print(f"  {label:<20}  avg=${avg:>7.2f}  lift_vs_sell={lift:+6.2f}%")

    from scipy.stats import wilcoxon
    diff = R["pnl_SHOCK_ENG"] - R["pnl_ALWAYS_SELL"]
    if (diff != 0).sum() >= 5:
        stat, pval = wilcoxon(R["pnl_SHOCK_ENG"], R["pnl_ALWAYS_SELL"], zero_method="wilcox")
        sign = "✓ significativo" if pval < 0.05 else "× ruido"
        print(f"  Wilcoxon SHOCK_ENG vs ALWAYS_SELL: Δ={diff.mean():+.2f}/ton  p={pval:.4f}  {sign}")

    # ── Por tipo de shock ──────────────────────────────────
    print(f"\n══════════════════════════════════════════════════════════════")
    print(f"  Por tipo de shock")
    print(f"══════════════════════════════════════════════════════════════")
    for stype in R["shock_type"].unique():
        sub = R[R["shock_type"] == stype]
        avg_se = sub["pnl_SHOCK_ENG"].mean()
        avg_as = sub["pnl_ALWAYS_SELL"].mean()
        lift = (avg_se / avg_as - 1) * 100 if avg_as else 0
        try:
            d = sub["pnl_SHOCK_ENG"] - sub["pnl_ALWAYS_SELL"]
            pv = wilcoxon(sub["pnl_SHOCK_ENG"], sub["pnl_ALWAYS_SELL"], zero_method="wilcox")[1] if (d != 0).sum() >= 5 else None
        except Exception:
            pv = None
        pv_s = f"p={pv:.3f}" if pv is not None else "p=n/a"
        print(f"  {stype:<20} N={len(sub):>3}  SE=${avg_se:.2f}  AS=${avg_as:.2f}  "
              f"lift={lift:+5.2f}%  {pv_s}")

    # ── Por strong vs near ────────────────────────────────
    print(f"\n══════════════════════════════════════════════════════════════")
    print(f"  Por fortaleza del shock")
    print(f"══════════════════════════════════════════════════════════════")
    for is_strong, label in [(True, "Strong (top 5%)"), (False, "Near (top 15%)")]:
        sub = R[R["is_strong"] == is_strong]
        if sub.empty:
            continue
        avg_se = sub["pnl_SHOCK_ENG"].mean()
        avg_as = sub["pnl_ALWAYS_SELL"].mean()
        lift = (avg_se / avg_as - 1) * 100 if avg_as else 0
        print(f"  {label:<20} N={len(sub):>3}  SE=${avg_se:.2f}  AS=${avg_as:.2f}  "
              f"lift={lift:+5.2f}%")

    # ── Por action recomendada ─────────────────────────────
    print(f"\n══════════════════════════════════════════════════════════════")
    print(f"  Por action recomendada")
    print(f"══════════════════════════════════════════════════════════════")
    for action in R["rec_action"].unique():
        sub = R[R["rec_action"] == action]
        if sub.empty:
            continue
        avg_se = sub["pnl_SHOCK_ENG"].mean()
        avg_as = sub["pnl_ALWAYS_SELL"].mean()
        avg_aw = sub["pnl_ALWAYS_WAIT"].mean()
        lift = (avg_se / avg_as - 1) * 100 if avg_as else 0
        print(f"  {action:<22} N={len(sub):>3}  SE=${avg_se:.2f}  AS=${avg_as:.2f}  "
              f"AW=${avg_aw:.2f}  lift={lift:+5.2f}%")


if __name__ == "__main__":
    main()
