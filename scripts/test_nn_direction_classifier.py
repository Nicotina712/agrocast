"""
scripts/test_nn_direction_classifier.py
Test focalizado: NN binary direction classifier a 7d.

Hipótesis: el espacio de estado predice SIGNO mejor que magnitud.
NN-analog regresor ponderado falla porque la magnitud predicha es muy
extrema (los vecinos son raros), pero la dirección puede ser correcta.

Evaluamos un clasificador NN puro:
  - Pred = +1 si la mayoría ponderada de vecinos tuvo ret_7d > 0
  - Pred = −1 si la mayoría ponderada tuvo ret_7d ≤ 0

Comparamos contra baselines simples:
  - Random (50%)
  - Always +1 (predice subir)
  - Sign(mom_5d) — predice continuación de momentum
  - Sign(mom_20d)

Métrica principal: DirAcc + binomial test + CI 95%.
N ampliado a ~250 con freq=3d, ventana 5 años.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")

CONFIGS = {
    "top5_granger":  ["price_vs_ma20", "price_vs_ma90", "mom_20d", "mom_5d", "rsi_14"],
    "top3_economic": ["Oil_chg7", "mom_20d", "news_sentiment"],
    "extended":      ["price_vs_ma20", "price_vs_ma90", "mom_20d", "mom_5d", "rsi_14",
                       "Oil_chg7", "news_sentiment", "vol_30d"],
}

HORIZONS       = [5, 7, 14]
K_NEIGHBORS    = 25
ZSCORE_WINDOW  = 252
MIN_GAP_DAYS   = 60
TEST_FREQ_DAYS = 3   # más denso → N mayor


def zscore_rolling(s: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    rm = s.rolling(window, min_periods=60).mean().shift(1)
    rs = s.rolling(window, min_periods=60).std().shift(1).replace(0, 1e-6)
    return (s - rm) / rs


def evaluate_direction(df, Z, test_idx, target_col):
    rows = []
    for idx_t in test_idx:
        D = df.iloc[idx_t]["Date"]
        target = df.iloc[idx_t].get(target_col)
        if pd.isna(target):
            continue
        actual_sign = 1 if target > 0 else -1
        v_t = Z.iloc[idx_t].values

        gap_date = D - pd.Timedelta(days=MIN_GAP_DAYS)
        hist_mask = (df["Date"] < gap_date) & df[target_col].notna()
        hist_idx = df.index[hist_mask]
        if len(hist_idx) < K_NEIGHBORS * 2:
            continue

        V_hist = Z.loc[hist_idx].values
        dists = np.sqrt(((V_hist - v_t) ** 2).sum(axis=1))
        top_k = np.argsort(dists)[:K_NEIGHBORS]
        neighbor_idx = hist_idx[top_k]
        neighbor_rets = df.loc[neighbor_idx, target_col].values

        # Voto ponderado por inverso de distancia
        weights = 1.0 / (dists[top_k] + 1e-6)
        weights /= weights.sum()
        prob_up = float(((neighbor_rets > 0).astype(float) * weights).sum())
        pred_sign = 1 if prob_up >= 0.5 else -1

        # Baselines de comparación
        mom5  = df.iloc[idx_t].get("mom_5d", 0)
        mom20 = df.iloc[idx_t].get("mom_20d", 0)

        rows.append({
            "date":   D,
            "actual_ret":  float(target),
            "actual_sign": actual_sign,
            "prob_up":     prob_up,
            "pred_nn_sign": pred_sign,
            "pred_always_up": 1,
            "pred_sign_mom5":  1 if mom5  > 0 else -1,
            "pred_sign_mom20": 1 if mom20 > 0 else -1,
        })
    return pd.DataFrame(rows)


def report(R: pd.DataFrame, label: str) -> dict:
    n = len(R)
    if n == 0:
        return {}
    metrics = {}
    for col in ["pred_nn_sign", "pred_always_up", "pred_sign_mom5", "pred_sign_mom20"]:
        correct = int((R[col] == R["actual_sign"]).sum())
        acc = correct / n
        # Binomial test: ¿acc significativamente > 0.5?
        bt = binomtest(correct, n, p=0.5, alternative="greater")
        ci = bt.proportion_ci(confidence_level=0.95, method="exact")
        metrics[col] = {
            "n": n, "correct": correct, "acc_pct": round(acc * 100, 2),
            "p_binomial_greater": round(bt.pvalue, 4),
            "ci95_low": round(ci.low * 100, 2),
            "ci95_high": round(ci.high * 100, 2),
        }
    return metrics


def main():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    for h in HORIZONS:
        col = f"ret_{h}d_fwd"
        if col not in df.columns:
            df[col] = df["Soybeans"].pct_change(h).shift(-h)

    end = df["Date"].max()
    test_start = end - pd.DateOffset(years=5)   # ventana ampliada
    test_idx = []
    d = test_start
    while d <= end - pd.Timedelta(days=max(HORIZONS)):
        future = df[df["Date"] >= d]
        if future.empty:
            break
        idx = int(df.index[df["Date"] == future.iloc[0]["Date"]][0])
        test_idx.append(idx)
        d = future.iloc[0]["Date"] + pd.Timedelta(days=TEST_FREQ_DAYS)
    print(f"Test dates ampliados: {len(test_idx)} (5 años, freq=3d)\n")

    # Pre-compute Z para cada config
    Z_cache = {}
    for cfg_name, feats in CONFIGS.items():
        avail = [f for f in feats if f in df.columns]
        Z_cache[cfg_name] = pd.DataFrame({f: zscore_rolling(df[f]) for f in avail}).fillna(0).replace([np.inf, -np.inf], 0)

    # Evaluación por config × horizonte
    print(f"{'Config':<16} {'H':>4}  {'N':>4}  {'NN_acc':>8} {'p_NN':>7} {'CI95':>15}  "
          f"{'AlwUp':>7} {'Mom5':>7} {'Mom20':>7}")
    print("-" * 100)
    rows = []
    for cfg_name, Z in Z_cache.items():
        for h in HORIZONS:
            target_col = f"ret_{h}d_fwd"
            R = evaluate_direction(df, Z, test_idx, target_col)
            m = report(R, f"{cfg_name}/{h}d")
            if not m:
                continue
            nn = m["pred_nn_sign"]
            au = m["pred_always_up"]["acc_pct"]
            m5 = m["pred_sign_mom5"]["acc_pct"]
            m20 = m["pred_sign_mom20"]["acc_pct"]
            stars = "✓✓" if nn["p_binomial_greater"] < 0.05 else ("✓" if nn["p_binomial_greater"] < 0.10 else "")
            print(f"{cfg_name:<16} {h:>3}d  {nn['n']:>4}  {nn['acc_pct']:>6.2f}% {nn['p_binomial_greater']:>7.4f}  "
                  f"[{nn['ci95_low']:.1f}-{nn['ci95_high']:.1f}]  "
                  f"{au:>6.2f}% {m5:>6.2f}% {m20:>6.2f}%  {stars}")
            rows.append({"config": cfg_name, "horizon": h, **nn,
                          "always_up_acc": au, "mom5_acc": m5, "mom20_acc": m20})

    R_all = pd.DataFrame(rows)
    R_all.to_csv(os.path.join(ROOT, "artifacts_eval", "test_nn_direction_classifier.csv"), index=False)

    # ── Decisión final ─────────────────────────────────────
    print(f"\n══════════════════════════════════════════════════════════════")
    print(f"  Configs con DirAcc significativamente > 50% (p<0.10)")
    print(f"══════════════════════════════════════════════════════════════")
    sig = R_all[R_all["p_binomial_greater"] < 0.10].sort_values("p_binomial_greater")
    if sig.empty:
        print(f"  ❌ Ninguna config × horizonte supera 50% con p<0.10")
    else:
        for _, row in sig.iterrows():
            print(f"  ✓ {row['config']:<18} h={row['horizon']:>2}d  "
                  f"acc={row['acc_pct']:.2f}%  p={row['p_binomial_greater']:.4f}  "
                  f"CI95=[{row['ci95_low']:.1f}-{row['ci95_high']:.1f}]")

    # ── Análisis adicional: ¿NN supera a baselines (always-up, mom)? ─
    print(f"\n══════════════════════════════════════════════════════════════")
    print(f"  ¿NN supera a baselines simples (always-up, mom5, mom20)?")
    print(f"══════════════════════════════════════════════════════════════")
    for _, row in R_all.iterrows():
        beats_au  = row["acc_pct"] > row["always_up_acc"]
        beats_m5  = row["acc_pct"] > row["mom5_acc"]
        beats_m20 = row["acc_pct"] > row["mom20_acc"]
        if beats_au and beats_m5 and beats_m20:
            print(f"  ✓ {row['config']:<18} h={row['horizon']:>2}d gana a TODOS los baselines  "
                  f"(NN={row['acc_pct']:.1f}% > AU={row['always_up_acc']:.1f}, "
                  f"M5={row['mom5_acc']:.1f}, M20={row['mom20_acc']:.1f})")

    print(f"\n💾 artifacts_eval/test_nn_direction_classifier.csv")


if __name__ == "__main__":
    main()
