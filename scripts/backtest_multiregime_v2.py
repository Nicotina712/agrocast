"""
scripts/backtest_multiregime.py
Backtest walk-forward del optimal stopping + heurística + always-sell sobre
ventanas históricas de 12 meses cada una, identificando regímenes distintos.

Ventanas (con TRAIN_YEARS=5 para todas, rolling):
  R1) 2021-04 → 2022-04: post-COVID rally + supply chain crisis
  R2) 2022-04 → 2023-04: Ucrania + récord prices + recovery
  R3) 2023-04 → 2024-04: post-war normalization
  R4) 2024-04 → 2025-04: lateral + early tariff fears
  R5) 2025-04 → 2026-05: tariff actual + Brasil récord (ya tested, control)

Para cada ventana:
  - Walk-forward train horizons (single-seed, sin Granger pruning para ser puro)
  - Decisiones de HEUR + OPT cada 14d
  - Comparación P&L vs always-sell + always-wait
  - p-values Wilcoxon paired

Output: tabla de lift por régimen + análisis cualitativo.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# V2: Granger pruning ON + features.csv con spike features (post 2026-05-03)
# Reproduce el modelo de PRODUCCIÓN actualizado para validar lift OOS por régimen.
os.environ["HORIZONS_N_SEEDS"]    = "1"
os.environ["USE_GRANGER_PRUNING"] = "1"

from src.model.train_horizons   import train_horizon
from src.model.economic_utility import utility_wait_vs_sell, BU_PER_TON, CENTS_TO_USD
from src.model.optimal_stopping import optimal_stopping_decision

FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")
TRAIN_YEARS  = 5
HORIZON_DAYS = 30
N_PATHS      = 600
DECISION_FREQ_DAYS = 14
STORAGE_USD  = 6.0
FINANCING    = 0.08

REGIMES = [
    {"name": "R1_post_covid_2021",   "start": "2021-04-01", "end": "2022-04-01",
     "label": "Post-COVID rally + supply chain"},
    {"name": "R2_ukraine_2022",      "start": "2022-04-01", "end": "2023-04-01",
     "label": "Ucrania + máximos históricos"},
    {"name": "R3_normalization_2023","start": "2023-04-01", "end": "2024-04-01",
     "label": "Normalización post-Ucrania"},
    {"name": "R4_lateral_2024",      "start": "2024-04-01", "end": "2025-04-01",
     "label": "Lateral + early tariff fears"},
    {"name": "R5_tariff_2025",       "start": "2025-04-01", "end": "2026-05-01",
     "label": "Tariff Trump 2.0 + Brasil récord (control)"},
]


def run_window(df: pd.DataFrame, start: str, end: str, label: str) -> dict:
    start_ts = pd.to_datetime(start)
    end_ts   = pd.to_datetime(end)
    window = df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)]
    if window.empty:
        return {"ok": False, "error": "ventana vacía"}

    # Generar fechas de decisión
    decision_dates = []
    d = start_ts
    while d <= end_ts - pd.Timedelta(days=HORIZON_DAYS):
        future = window[window["Date"] >= d]
        if future.empty:
            break
        decision_dates.append(future.iloc[0]["Date"])
        d = future.iloc[0]["Date"] + pd.Timedelta(days=DECISION_FREQ_DAYS)

    print(f"\n══════ {label} ══════ ({len(decision_dates)} decisiones)")

    rows = []
    tmp_dir = os.path.join(ROOT, "artifacts_eval", "wf_multi_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    import joblib

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
            for h in (7, 30):
                target_col = f"ret_{h}d_fwd"
                if target_col not in df_train.columns:
                    continue
                bundle = train_horizon(df_train, h, target_col)
                joblib.dump(bundle, os.path.join(tmp_dir, f"model_h{h}d.joblib"))

            heur = utility_wait_vs_sell(df_train, storage_cost_per_ton_month=STORAGE_USD,
                                          financing_rate_annual=FINANCING,
                                          horizon_days=HORIZON_DAYS, n_paths=N_PATHS,
                                          artifacts_dir=tmp_dir)
            opt = optimal_stopping_decision(df_train, storage_cost_per_ton_month=STORAGE_USD,
                                              financing_rate_annual=FINANCING,
                                              horizon_days=HORIZON_DAYS, n_paths=N_PATHS,
                                              artifacts_dir=tmp_dir)
        except Exception as e:
            print(f"  [WARN] D={D.date()} error: {e}")
            continue

        to_ton = BU_PER_TON * CENTS_TO_USD
        months = HORIZON_DAYS / 30.0
        cost = STORAGE_USD * months + (price_D * to_ton) * FINANCING * (HORIZON_DAYS / 365.0)
        sell_v = price_D    * to_ton
        wait_v = price_exit * to_ton - cost

        def pnl(dec):
            return (wait_v if dec == "WAIT"
                    else 0.5 * sell_v + 0.5 * wait_v if dec == "INDIFFERENT"
                    else sell_v)

        rows.append({
            "date": D, "label": label,
            "price_D": price_D, "price_exit": price_exit,
            "heur_dec":   heur.get("decision", "INDIFFERENT"),
            "opt_dec":    opt.get("decision_now", "INDIFFERENT"),
            "opt_reservation": opt.get("reservation_today_usd_ton", 0),
            "pnl_HEUR":         round(pnl(heur.get("decision", "INDIFFERENT")), 2),
            "pnl_OPTSTOP":      round(pnl(opt.get("decision_now", "INDIFFERENT")), 2),
            "pnl_ALWAYS_SELL":  round(sell_v, 2),
            "pnl_ALWAYS_WAIT":  round(wait_v, 2),
        })

    if not rows:
        return {"ok": False, "error": "ninguna decisión válida"}

    R = pd.DataFrame(rows)

    # Métricas por estrategia
    metrics = {}
    base = R["pnl_ALWAYS_SELL"].mean()
    for col, name in [("pnl_HEUR", "Heurística"),
                       ("pnl_OPTSTOP", "OptStop"),
                       ("pnl_ALWAYS_SELL", "AlwaysSell"),
                       ("pnl_ALWAYS_WAIT", "AlwaysWait")]:
        avg = R[col].mean()
        lift = (avg / base - 1) * 100
        metrics[name] = {"avg": float(avg), "lift_pct": float(lift)}

    # Wilcoxon paired
    from scipy.stats import wilcoxon
    p_values = {}
    for col, name in [("pnl_HEUR", "Heur_vs_Sell"), ("pnl_OPTSTOP", "Opt_vs_Sell")]:
        try:
            d = R[col] - R["pnl_ALWAYS_SELL"]
            if (d != 0).sum() < 5:
                p_values[name] = None
            else:
                _, pval = wilcoxon(R[col], R["pnl_ALWAYS_SELL"], zero_method="wilcox")
                p_values[name] = float(pval)
        except Exception:
            p_values[name] = None

    # Volatilidad y dirección del régimen (caracterización)
    log_ret = np.log(R["price_exit"] / R["price_D"]).dropna()
    regime_stats = {
        "n_decisions":     int(len(R)),
        "price_mean":      round(float(R["price_D"].mean()), 2),
        "price_std":       round(float(R["price_D"].std()), 2),
        "ret_30d_mean":    round(float(log_ret.mean()) * 100, 3),  # % drift mensual
        "ret_30d_std":     round(float(log_ret.std()) * 100, 3),   # % vol mensual
        "wait_wins":       int((R["pnl_ALWAYS_WAIT"] > R["pnl_ALWAYS_SELL"]).sum()),
        "wait_wins_pct":   round((R["pnl_ALWAYS_WAIT"] > R["pnl_ALWAYS_SELL"]).mean() * 100, 1),
    }

    return {"ok": True, "label": label,
            "metrics": metrics, "p_values": p_values,
            "regime_stats": regime_stats,
            "n": int(len(R)),
            "mix_heur": dict(R["heur_dec"].value_counts()),
            "mix_opt":  dict(R["opt_dec"].value_counts()),
            "rows": R.to_dict("records"),
            }


def main():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    print(f"Backtest multi-régimen: features {df['Date'].min().date()} → {df['Date'].max().date()}")

    results = []
    t_total = time.time()
    for r in REGIMES:
        t0 = time.time()
        out = run_window(df, r["start"], r["end"], r["label"])
        if out.get("ok"):
            out["regime_id"] = r["name"]
            out["start"]     = r["start"]
            out["end"]       = r["end"]
            results.append(out)

            # Imprimir resumen del régimen
            m = out["metrics"]; rs = out["regime_stats"]; pv = out["p_values"]
            print(f"  → drift={rs['ret_30d_mean']:+.2f}%/30d  vol={rs['ret_30d_std']:.2f}%/30d  "
                  f"wait_wins={rs['wait_wins']}/{rs['n_decisions']} ({rs['wait_wins_pct']}%)")
            print(f"  → AlwaysSell ${m['AlwaysSell']['avg']:.2f}  "
                  f"AlwaysWait ${m['AlwaysWait']['avg']:.2f} ({m['AlwaysWait']['lift_pct']:+.2f}%)")
            print(f"  → Heur ${m['Heurística']['avg']:.2f} ({m['Heurística']['lift_pct']:+.2f}%)  "
                  f"p_vs_sell={pv.get('Heur_vs_Sell')}")
            print(f"  → Opt  ${m['OptStop']['avg']:.2f} ({m['OptStop']['lift_pct']:+.2f}%)  "
                  f"p_vs_sell={pv.get('Opt_vs_Sell')}")
            print(f"  → mix HEUR: {out['mix_heur']}  |  mix OPT: {out['mix_opt']}")
            print(f"  ({time.time() - t0:.1f}s)")
        else:
            print(f"  [SKIP] {r['name']}: {out.get('error')}")

    # ── TABLA MAESTRA COMPARATIVA ─────────────────────────────────
    print(f"\n\n══════════════════════════════════════════════════════════════════")
    print(f"  TABLA MAESTRA — lift % vs Always-Sell por régimen")
    print(f"══════════════════════════════════════════════════════════════════")
    print(f"  {'Régimen':<35} {'N':>4}  {'Drift':>7}  {'Vol':>6}  "
          f"{'Heur':>8}  {'Opt':>8}  {'AlwaysWait':>10}  {'p_Heur':>7}  {'p_Opt':>7}")
    print(f"  {'-'*35} {'-'*4}  {'-'*7}  {'-'*6}  "
          f"{'-'*8}  {'-'*8}  {'-'*10}  {'-'*7}  {'-'*7}")
    for out in results:
        rs = out["regime_stats"]; m = out["metrics"]; pv = out["p_values"]
        ph = pv.get("Heur_vs_Sell")
        po = pv.get("Opt_vs_Sell")
        print(f"  {out['label'][:35]:<35} {rs['n_decisions']:>4}  "
              f"{rs['ret_30d_mean']:>+6.2f}%  {rs['ret_30d_std']:>5.2f}%  "
              f"{m['Heurística']['lift_pct']:>+7.2f}%  "
              f"{m['OptStop']['lift_pct']:>+7.2f}%  "
              f"{m['AlwaysWait']['lift_pct']:>+9.2f}%  "
              f"{ph:>7.3f}  {po:>7.3f}" if ph is not None and po is not None
              else f"  {out['label'][:35]:<35} {rs['n_decisions']:>4}  "
                   f"{rs['ret_30d_mean']:>+6.2f}%  {rs['ret_30d_std']:>5.2f}%  "
                   f"{m['Heurística']['lift_pct']:>+7.2f}%  "
                   f"{m['OptStop']['lift_pct']:>+7.2f}%  "
                   f"{m['AlwaysWait']['lift_pct']:>+9.2f}%      n/a      n/a")

    # Persistir
    import json as _json
    out_path = os.path.join(ROOT, "artifacts_eval", "backtest_multiregime_v2.json")
    with open(out_path, "w") as f:
        _json.dump([{k: v for k, v in r.items() if k != "rows"} for r in results],
                    f, indent=2, default=float)
    print(f"\n💾 {out_path}")

    rows_all = pd.concat([pd.DataFrame(r["rows"]) for r in results], ignore_index=True)
    rows_all.to_csv(os.path.join(ROOT, "artifacts_eval", "backtest_multiregime_v2.csv"), index=False)
    print(f"💾 backtest_multiregime_v2.csv ({len(rows_all)} filas)")
    print(f"\nTotal: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
