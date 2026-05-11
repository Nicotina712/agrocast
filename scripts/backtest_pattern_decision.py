"""
scripts/backtest_pattern_decision.py
Backtest formal del Pattern Decision Engine (Fase 2):
  - Multi-horizon classifier (Idea A, fp_weight=2.0) ya validado
  - Delta regressor (Fase 2.1)
  - PARTIAL_SELL combinado (Fase 2.2)

Compara contra:
  - Always-Sell (baseline empírico imbatible)
  - Always-Wait (devuelve si esperar pagó pasivamente)
  - Split 50/50 (estrategia naive)
  - Oracle (techo teórico, sabe el outcome)
  - Decision Classifier solo (sin partial, threshold 0.5) — para aislar el aporte del delta + partial

Por: profile × horizonte. (Régimen y año los dejamos para una iteración posterior
si Fase 2 muestra lift; ver PATTERN_DECISION_ROADMAP).

Output: artifacts_eval/backtest_pattern_decision.csv + tabla resumen.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model.decision_classifier import (
    _compute_cost_pct, _build_decision_features, train_decision_classifier,
    _combine_partial_sell, PRODUCER_PROFILES, FEATURE_COLS,
)

FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")
HORIZONS_BT  = [7, 15, 30]
TRAIN_YEARS  = 5
TEST_FREQ_DAYS = 7
REFIT_EVERY  = 90
DEFAULT_PRICE_TON = 400.0


def fraction_pnl(decision: str, ret_pct: float, cost_pct: float) -> float:
    """P&L hipotético en USD/ton normalizado (precio típico) para una decisión partial."""
    # decision_label define sell_pct
    label_to_sell_pct = {
        "SELL_100": 1.0,
        "SELL_70":  0.7,
        "SPLIT_50": 0.5,
        "HOLD_70":  0.3,
        "HOLD_100": 0.0,
    }
    sell_frac = label_to_sell_pct.get(decision, 1.0)
    hold_frac = 1.0 - sell_frac
    # PnL = sell_frac * 0 (precio actual referencia) + hold_frac * (ret - cost)
    # Normalizado: el lift sobre always-sell es hold_frac * (ret - cost)
    return hold_frac * (ret_pct - cost_pct)


def main():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    df = _build_decision_features(df)

    end = df["Date"].max()
    test_start = end - pd.DateOffset(years=5)
    test_idx = []
    d = test_start
    while d <= end - pd.Timedelta(days=max(HORIZONS_BT)):
        future = df[df["Date"] >= d]
        if future.empty: break
        idx = int(df.index[df["Date"] == future.iloc[0]["Date"]][0])
        test_idx.append(idx)
        d = future.iloc[0]["Date"] + pd.Timedelta(days=TEST_FREQ_DAYS)

    print(f"Backtest formal Pattern Decision Engine")
    print(f"  Test dates: {len(test_idx)}, freq={TEST_FREQ_DAYS}d, ventana=5 años")
    print(f"  Profiles: {list(PRODUCER_PROFILES.keys())}")
    print(f"  Horizons: {HORIZONS_BT}\n")

    # Para cada profile × horizonte, walk-forward
    all_rows = []
    for prof_name, prof in PRODUCER_PROFILES.items():
        for h in HORIZONS_BT:
            print(f"\n══ Profile: {prof_name:<14}  Horizonte: {h}d ══")
            t0 = time.time()
            ret_col = f"ret_{h}d_fwd"
            if ret_col not in df.columns:
                df[ret_col] = df["Soybeans"].pct_change(h).shift(-h)

            last_refit = None
            bundle = None
            rows = []

            for idx_t in test_idx:
                D = df.iloc[idx_t]["Date"]
                actual_ret = df.iloc[idx_t].get(ret_col)
                if pd.isna(actual_ret):
                    continue

                # Costo del horizonte para este profile (precio actual)
                p_now_cents = float(df.iloc[idx_t]["Soybeans"])
                p_now_ton = p_now_cents * 0.01 * 36.7437
                cost_pct = _compute_cost_pct(
                    prof["storage"], prof["financing"], p_now_ton,
                    prof.get("quality_risk_per_month", 0.0), horizon_days=h
                )

                # Refit walk-forward
                if (last_refit is None) or ((D - last_refit).days >= REFIT_EVERY):
                    train_df = df[df["Date"] < D].copy()
                    if len(train_df) < 200:
                        continue
                    bundle = train_decision_classifier(train_df, cost_pct=cost_pct, horizon_days=h)
                    if not bundle.get("ok"):
                        continue
                    last_refit = D

                # Predict
                X = df.iloc[[idx_t]][bundle["features"]].fillna(0).replace([np.inf, -np.inf], 0)
                try:
                    p_raw = float(bundle["model"].predict_proba(X)[0, 1])
                    p_cal = float(bundle["calibrator"].transform([p_raw])[0]) if bundle.get("calibrator") else p_raw
                    p_cal = float(np.clip(p_cal, 0.0, 1.0))
                except Exception:
                    continue

                # Delta predicted
                delta_pct = None
                if bundle.get("delta_model") is not None:
                    try:
                        delta_pct = float(bundle["delta_model"].predict(X)[0])
                    except Exception:
                        delta_pct = None

                # Decisiones a comparar
                # 1. Pattern Engine (clasificador + delta + partial)
                pe_label, pe_sell, _, _ = _combine_partial_sell(p_cal, delta_pct, cost_pct)
                # 2. Classifier-only binario (threshold 0.5, sin partial)
                clf_label = "SELL_100" if p_cal < 0.5 else "HOLD_100"
                # 3-5. Naive
                always_sell = "SELL_100"
                always_wait = "HOLD_100"
                split_50    = "SPLIT_50"
                # 6. Oracle (sabe el outcome)
                oracle = "HOLD_100" if (actual_ret - cost_pct) > 0 else "SELL_100"

                rows.append({
                    "date": D, "horizon": h, "profile": prof_name,
                    "actual_ret_pct":  float(actual_ret) * 100,
                    "cost_pct":        cost_pct * 100,
                    "p_cal":           round(p_cal, 3),
                    "delta_pred_pct":  round(delta_pct * 100, 3) if delta_pct is not None else None,
                    # Decisiones
                    "pe_label":        pe_label,
                    "clf_label":       clf_label,
                    # PnLs (lift sobre always-sell, en %)
                    "pnl_pe":          fraction_pnl(pe_label,    actual_ret, cost_pct) * 100,
                    "pnl_clf":         fraction_pnl(clf_label,   actual_ret, cost_pct) * 100,
                    "pnl_always_sell": fraction_pnl(always_sell, actual_ret, cost_pct) * 100,
                    "pnl_always_wait": fraction_pnl(always_wait, actual_ret, cost_pct) * 100,
                    "pnl_split_50":    fraction_pnl(split_50,    actual_ret, cost_pct) * 100,
                    "pnl_oracle":      fraction_pnl(oracle,      actual_ret, cost_pct) * 100,
                })

            elapsed = time.time() - t0
            if not rows:
                print(f"  Sin datos suficientes")
                continue
            R = pd.DataFrame(rows)
            n = len(R)
            print(f"  N={n}, time={elapsed:.1f}s")

            # Métricas agregadas
            metrics = {}
            for col, label in [
                ("pnl_pe",          "Pattern Engine"),
                ("pnl_clf",         "Classifier-only"),
                ("pnl_always_sell", "Always-Sell"),
                ("pnl_always_wait", "Always-Wait"),
                ("pnl_split_50",    "Split 50/50"),
                ("pnl_oracle",      "Oracle"),
            ]:
                avg = float(R[col].mean())
                metrics[label] = avg

            # Wilcoxon paired vs always-sell (que es 0)
            try:
                _, p_pe = wilcoxon(R["pnl_pe"], R["pnl_always_sell"], zero_method="zsplit")
                _, p_clf = wilcoxon(R["pnl_clf"], R["pnl_always_sell"], zero_method="zsplit")
            except Exception:
                p_pe = p_clf = None

            print(f"  {'Strategy':<18} {'Avg PnL%':>10}  {'p_vs_AS':>9}")
            print(f"  {'-'*18} {'-'*10}  {'-'*9}")
            for label, avg in metrics.items():
                star = "✓" if avg > 0 else " "
                if label == "Pattern Engine" and p_pe is not None:
                    print(f"  {label:<18} {avg:>+9.4f}%  {p_pe:>8.4f}{star}")
                elif label == "Classifier-only" and p_clf is not None:
                    print(f"  {label:<18} {avg:>+9.4f}%  {p_clf:>8.4f}{star}")
                else:
                    print(f"  {label:<18} {avg:>+9.4f}%  {'—':>9}{star}")

            # Mix de decisiones del Pattern Engine
            mix = R["pe_label"].value_counts().to_dict()
            print(f"  Mix PE: {mix}")

            all_rows.extend(rows)

    # Guardar todo
    if all_rows:
        R_all = pd.DataFrame(all_rows)
        out_path = os.path.join(ROOT, "artifacts_eval", "backtest_pattern_decision.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        R_all.to_csv(out_path, index=False)
        print(f"\n💾 {out_path}")

        # Tabla resumen final
        print(f"\n══════════════════════════════════════════════════════════════")
        print(f"  TABLA MAESTRA — Pattern Engine vs Always-Sell por (profile, h)")
        print(f"══════════════════════════════════════════════════════════════")
        print(f"  {'Profile':<14} {'H':>4}  {'PE':>9}  {'CLF':>9}  {'AW':>9}  {'Split':>9}  {'Oracle':>9}  {'p_PE':>7}  {'p_CLF':>7}")
        print(f"  {'-'*14} {'-'*4}  " + "  ".join(["-"*9]*5) + "  " + "-"*7 + "  " + "-"*7)
        summary_rows = []
        for prof_name in PRODUCER_PROFILES.keys():
            for h in HORIZONS_BT:
                sub = R_all[(R_all["profile"] == prof_name) & (R_all["horizon"] == h)]
                if sub.empty: continue
                pe   = float(sub["pnl_pe"].mean())
                clf  = float(sub["pnl_clf"].mean())
                aw   = float(sub["pnl_always_wait"].mean())
                spl  = float(sub["pnl_split_50"].mean())
                orc  = float(sub["pnl_oracle"].mean())
                try:
                    _, p_pe  = wilcoxon(sub["pnl_pe"], sub["pnl_always_sell"], zero_method="zsplit")
                    _, p_clf = wilcoxon(sub["pnl_clf"], sub["pnl_always_sell"], zero_method="zsplit")
                except Exception:
                    p_pe = p_clf = float("nan")
                star_pe = "✓" if pe > 0 and p_pe < 0.10 else (" " if pe <= 0 else ".")
                summary_rows.append({
                    "profile": prof_name, "horizon": h,
                    "pe": pe, "clf": clf, "aw": aw, "split": spl, "oracle": orc,
                    "p_pe": p_pe, "p_clf": p_clf,
                })
                print(f"  {prof_name:<14} {h:>3}d  {pe:>+8.4f}% {clf:>+8.4f}% {aw:>+8.4f}% {spl:>+8.4f}% {orc:>+8.4f}%  {p_pe:>7.4f}  {p_clf:>7.4f}{star_pe}")

        # Promedio global
        if summary_rows:
            S = pd.DataFrame(summary_rows)
            print(f"\n  GLOBAL avg:  PE={S['pe'].mean():+.4f}%  CLF={S['clf'].mean():+.4f}%  AW={S['aw'].mean():+.4f}%  Split={S['split'].mean():+.4f}%  Oracle={S['oracle'].mean():+.4f}%")
            print(f"  Configs con PE>0:  {(S['pe']>0).sum()}/{len(S)}")
            print(f"  Configs con PE>0 y p<0.10:  {((S['pe']>0) & (S['p_pe']<0.10)).sum()}/{len(S)}")

            S.to_csv(os.path.join(ROOT, "artifacts_eval", "backtest_pattern_decision_summary.csv"), index=False)


if __name__ == "__main__":
    main()
