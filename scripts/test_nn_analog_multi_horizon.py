"""
scripts/test_nn_analog_multi_horizon.py
Test NN-analog en MÚLTIPLES horizontes para descartar que la falla a 30d
sea generalizable. Quizás a horizontes más cortos (microestructura) sí hay
señal explotable.

Horizontes: 5d, 7d, 14d, 30d
Configs (las menos malas del test reducido): top-3 economic, top-5 Granger,
                                              solo mom_20d, solo macro
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")

CONFIGS = {
    "top5_granger":   ["price_vs_ma20", "price_vs_ma90", "mom_20d", "mom_5d", "rsi_14"],
    "top3_economic":  ["Oil_chg7", "mom_20d", "news_sentiment"],
    "macro_exog":     ["Oil_chg7", "Dollar_chg7", "soy_corn_ratio", "enso_oni"],
    "only_mom20":     ["mom_20d"],
    "vol_complex":    ["vol_30d", "vol_60d", "rsi_14"],
}

HORIZONS = [5, 7, 14, 30]
K_NEIGHBORS    = 20
ZSCORE_WINDOW  = 252
MIN_GAP_DAYS   = 60
TEST_FREQ_DAYS = 7


def zscore_rolling(s: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    rm = s.rolling(window, min_periods=60).mean().shift(1)
    rs = s.rolling(window, min_periods=60).std().shift(1).replace(0, 1e-6)
    return (s - rm) / rs


def evaluate(df, Z, test_idx, target_col, label):
    """NN-analog evaluation para un target_col específico."""
    rows = []
    for idx_t in test_idx:
        D = df.iloc[idx_t]["Date"]
        target = df.iloc[idx_t].get(target_col)
        if pd.isna(target):
            continue
        v_t = Z.iloc[idx_t].values

        gap_date = D - pd.Timedelta(days=MIN_GAP_DAYS)
        hist_mask = (df["Date"] < gap_date) & df[target_col].notna()
        hist_idx = df.index[hist_mask]
        if len(hist_idx) < K_NEIGHBORS * 2:
            continue

        V_hist = Z.loc[hist_idx].values
        dists  = np.sqrt(((V_hist - v_t) ** 2).sum(axis=1))
        top_k = np.argsort(dists)[:K_NEIGHBORS]
        neighbor_idx = hist_idx[top_k]
        neighbor_rets = df.loc[neighbor_idx, target_col].values

        weights = 1.0 / (dists[top_k] + 1e-6)
        weights /= weights.sum()
        pred_nn = float((neighbor_rets * weights).sum())
        rows.append({"actual": float(target), "pred_nn": pred_nn})

    if not rows:
        return None
    R = pd.DataFrame(rows)
    err_nn = (R["pred_nn"] - R["actual"]).abs()
    err_zero = R["actual"].abs()
    mae_nn = float(err_nn.mean())
    mae_rw = float(err_zero.mean())
    lift = (mae_rw - mae_nn) / mae_rw * 100
    diracc = float((np.sign(R["pred_nn"]) == np.sign(R["actual"])).mean() * 100)
    try:
        _, pval = wilcoxon(err_nn, err_zero)
    except Exception:
        pval = float("nan")
    return {"label": label, "n": len(R), "mae_nn": mae_nn, "mae_rw": mae_rw,
            "lift_pct": lift, "diracc_pct": diracc, "pval": pval}


def main():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    # Targets forward por horizonte
    for h in HORIZONS:
        col = f"ret_{h}d_fwd"
        if col not in df.columns:
            df[col] = df["Soybeans"].pct_change(h).shift(-h)

    end = df["Date"].max()
    test_start = end - pd.DateOffset(years=2)
    test_idx = []
    d = test_start
    while d <= end - pd.Timedelta(days=max(HORIZONS)):
        future = df[df["Date"] >= d]
        if future.empty:
            break
        idx = int(df.index[df["Date"] == future.iloc[0]["Date"]][0])
        test_idx.append(idx)
        d = future.iloc[0]["Date"] + pd.Timedelta(days=TEST_FREQ_DAYS)
    print(f"Test dates: {len(test_idx)}\n")

    # Pre-compute Z para cada config (no depende del horizonte)
    Z_cache = {}
    for cfg_name, feats in CONFIGS.items():
        avail = [f for f in feats if f in df.columns]
        if not avail:
            continue
        Z_cache[cfg_name] = pd.DataFrame({f: zscore_rolling(df[f]) for f in avail}).fillna(0).replace([np.inf, -np.inf], 0)

    # Evaluación cruzada: configs × horizontes
    results = []
    for cfg_name, Z in Z_cache.items():
        for h in HORIZONS:
            target_col = f"ret_{h}d_fwd"
            res = evaluate(df, Z, test_idx, target_col, f"{cfg_name}/{h}d")
            if res:
                res["config"] = cfg_name
                res["horizon"] = h
                res["n_dims"] = Z.shape[1]
                results.append(res)

    R = pd.DataFrame(results)

    # ── Tabla comparativa por horizonte ─────────────────────────
    print(f"\n══════════════════════════════════════════════════════════════════════════")
    print(f"  Tabla comparativa — NN-analog por horizonte (lift % vs always-sell)")
    print(f"══════════════════════════════════════════════════════════════════════════")
    print(f"  {'Config':<18} {'Dims':>4}  ", end="")
    for h in HORIZONS:
        print(f"{h}d_lift  {h}d_DirAcc ", end="  ")
    print()
    print(f"  {'-'*18} {'-'*4}  ", end="")
    for h in HORIZONS:
        print(f"{'-'*7}  {'-'*9} ", end="  ")
    print()

    for cfg in CONFIGS.keys():
        sub = R[R["config"] == cfg].set_index("horizon")
        if sub.empty:
            continue
        dims = int(sub["n_dims"].iloc[0])
        print(f"  {cfg:<18} {dims:>4}  ", end="")
        for h in HORIZONS:
            if h in sub.index:
                lift = sub.loc[h, "lift_pct"]
                da   = sub.loc[h, "diracc_pct"]
                star = "✓" if sub.loc[h, "pval"] < 0.10 and lift > 0 else " "
                print(f"{lift:+6.2f}%{star}  {da:>6.1f}%   ", end="  ")
            else:
                print(f"{'n/a':>9}  {'n/a':>9}  ", end="  ")
        print()

    print(f"\n  ✓ = lift > 0 con p<0.10 (es decir, NN-analog GANA al always-sell)")

    # ── Mejor config por horizonte ──────────────────────────────
    print(f"\n══════════════════════════════════════════════════════════════════════════")
    print(f"  Mejor config por horizonte")
    print(f"══════════════════════════════════════════════════════════════════════════")
    for h in HORIZONS:
        sub = R[R["horizon"] == h].sort_values("lift_pct", ascending=False)
        if sub.empty:
            continue
        best = sub.iloc[0]
        print(f"  Horizonte {h}d  →  best: {best['config']:<18} "
              f"lift={best['lift_pct']:+6.2f}%  DirAcc={best['diracc_pct']:.1f}%  "
              f"p={best['pval']:.3f}  MAE_NN={best['mae_nn']*100:.3f}%")

    # ── Hay GANADORES? ──────────────────────────────────────────
    winners = R[(R["lift_pct"] > 0) & (R["pval"] < 0.10)]
    print(f"\n══════════════════════════════════════════════════════════════════════════")
    print(f"  Configs SIGNIFICATIVAS que ganan al always-sell")
    print(f"══════════════════════════════════════════════════════════════════════════")
    if winners.empty:
        print(f"  ❌ Ninguna config × horizonte gana significativamente al always-sell.")
    else:
        for _, row in winners.iterrows():
            print(f"  ✓ {row['config']:<18} h={row['horizon']:>2}d  "
                  f"lift={row['lift_pct']:+.2f}%  DirAcc={row['diracc_pct']:.1f}%  p={row['pval']:.3f}")

    # Persistir
    R.to_csv(os.path.join(ROOT, "artifacts_eval", "test_nn_analog_horizons.csv"), index=False)
    print(f"\n💾 artifacts_eval/test_nn_analog_horizons.csv")


if __name__ == "__main__":
    main()
