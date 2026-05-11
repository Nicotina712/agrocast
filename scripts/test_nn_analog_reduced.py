"""
scripts/test_nn_analog_reduced.py
Test de NN-analog con FEATURE REDUCTION para descartar curse of dimensionality
como causa de la falla.

Probamos múltiples subsets:
  A) Top-5 Granger causales (price_vs_ma20, price_vs_ma90, mom_20d, mom_5d, rsi_14)
  B) Top-3 por importancia económica (Oil, mom_20d, news_sentiment)
  C) Solo price-relative (price_vs_ma20, price_vs_ma90, ma20_vs_ma90, mom_20d)
  D) Solo macro/exógenas (Oil_chg7, Dollar_chg7, soy_corn_ratio, enso_oni)
  E) PCA(3) sobre las 24 originales
  F) PCA(5) sobre las 24 originales
  G) PCA(8) sobre las 24 originales

Si NINGUNO supera a always-sell con MAE → la falla NO es dimensionalidad.
Es que no hay señal explotable en estas features.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")

# Subsets a probar
SUBSETS = {
    "A_top5_granger":   ["price_vs_ma20", "price_vs_ma90", "mom_20d", "mom_5d", "rsi_14"],
    "B_top3_economic":  ["Oil_chg7", "mom_20d", "news_sentiment"],
    "C_price_relative": ["price_vs_ma20", "price_vs_ma90", "ma20_vs_ma90", "mom_20d"],
    "D_macro_exog":     ["Oil_chg7", "Dollar_chg7", "soy_corn_ratio", "enso_oni"],
    "E_just_2":         ["mom_20d", "rsi_14"],   # mínimo absoluto
    "F_just_1":         ["mom_20d"],              # 1D, debería ser robusto a curse
}

# Features para PCA
FULL_FEATS = [
    "price_vs_ma20", "price_vs_ma90", "ma20_vs_ma90", "ma90_slope",
    "mom_5d", "mom_20d", "mom_60d", "rsi_14", "vol_30d", "vol_60d",
    "soy_corn_ratio", "soy_corn_ratio_dev", "soy_oil_ratio", "crush_spread_dev",
    "Oil_chg7", "Dollar_chg7", "news_sentiment", "news_velocity_7d",
    "cot_noncomm_long_pct", "cot_index", "cot_contrarian_score",
    "enso_oni", "month_sin", "month_cos",
]

K_NEIGHBORS    = 20
HORIZON_DAYS   = 30
ZSCORE_WINDOW  = 252
MIN_GAP_DAYS   = 60
TEST_FREQ_DAYS = 7


def zscore_rolling(s: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    rm = s.rolling(window, min_periods=60).mean().shift(1)
    rs = s.rolling(window, min_periods=60).std().shift(1).replace(0, 1e-6)
    return (s - rm) / rs


def evaluate_subset(df: pd.DataFrame, Z: pd.DataFrame, test_idx: list, label: str) -> dict:
    """Para un subset Z (DataFrame de features ya z-scored), corre NN-analog y devuelve métricas."""
    rows = []
    for idx_t in test_idx:
        D = df.iloc[idx_t]["Date"]
        target = df.iloc[idx_t].get("ret_30d_fwd")
        if pd.isna(target):
            continue
        v_t = Z.iloc[idx_t].values

        gap_date = D - pd.Timedelta(days=MIN_GAP_DAYS)
        hist_mask = (df["Date"] < gap_date) & df["ret_30d_fwd"].notna()
        hist_idx = df.index[hist_mask]
        if len(hist_idx) < K_NEIGHBORS * 2:
            continue

        V_hist = Z.loc[hist_idx].values
        dists  = np.sqrt(((V_hist - v_t) ** 2).sum(axis=1))
        top_k = np.argsort(dists)[:K_NEIGHBORS]
        neighbor_idx = hist_idx[top_k]
        neighbor_rets = df.loc[neighbor_idx, "ret_30d_fwd"].values

        weights = 1.0 / (dists[top_k] + 1e-6)
        weights /= weights.sum()
        pred_nn = float((neighbor_rets * weights).sum())

        rows.append({"actual": float(target), "pred_nn": pred_nn,
                      "dist_med": float(np.median(dists[top_k]))})

    R = pd.DataFrame(rows)
    if R.empty:
        return {"label": label, "n": 0}

    err_nn   = (R["pred_nn"] - R["actual"]).abs()
    err_zero = R["actual"].abs()
    mae_nn   = float(err_nn.mean())
    mae_rw   = float(err_zero.mean())
    lift     = (mae_rw - mae_nn) / mae_rw * 100
    diracc   = float((np.sign(R["pred_nn"]) == np.sign(R["actual"])).mean() * 100)
    try:
        _, pval = wilcoxon(err_nn, err_zero)
    except Exception:
        pval = float("nan")

    return {
        "label": label, "n": len(R), "n_dims": Z.shape[1],
        "mae_nn":  mae_nn,  "mae_rw": mae_rw,
        "lift_pct": lift, "diracc_pct": diracc, "pval": pval,
    }


def main():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    df["ret_30d_fwd"] = df["Soybeans"].pct_change(30).shift(-30)

    end = df["Date"].max()
    test_start = end - pd.DateOffset(years=2)
    test_idx = []
    d = test_start
    while d <= end - pd.Timedelta(days=HORIZON_DAYS):
        future = df[df["Date"] >= d]
        if future.empty:
            break
        idx = int(df.index[df["Date"] == future.iloc[0]["Date"]][0])
        test_idx.append(idx)
        d = future.iloc[0]["Date"] + pd.Timedelta(days=TEST_FREQ_DAYS)
    print(f"Test dates: {len(test_idx)}\n")

    results = []

    # ── Subsets manuales ────────────────────────────────────────
    for key, feats in SUBSETS.items():
        avail = [f for f in feats if f in df.columns]
        if not avail:
            continue
        Z = pd.DataFrame({f: zscore_rolling(df[f]) for f in avail}).fillna(0).replace([np.inf, -np.inf], 0)
        res = evaluate_subset(df, Z, test_idx, key)
        results.append(res)
        print(f"  {key:<22} n_dims={res['n_dims']:>2}  N={res['n']:>3}  "
              f"MAE_NN={res['mae_nn']*100:.3f}%  MAE_RW={res['mae_rw']*100:.3f}%  "
              f"lift={res['lift_pct']:+6.2f}%  DirAcc={res['diracc_pct']:.1f}%  p={res['pval']:.3f}")

    # ── PCA ─────────────────────────────────────────────────────
    avail_full = [f for f in FULL_FEATS if f in df.columns]
    Z_full = pd.DataFrame({f: zscore_rolling(df[f]) for f in avail_full}).fillna(0).replace([np.inf, -np.inf], 0)

    try:
        from sklearn.decomposition import PCA
        # Para evitar look-ahead estricto, fitteamos PCA con datos hasta cada fecha test.
        # Aquí simplificamos: PCA sobre toda la serie. Si igual no funciona, no hay
        # razón para hacer la versión rigurosa.
        for n_comp in (3, 5, 8):
            pca = PCA(n_components=n_comp, random_state=42)
            Z_pca = pd.DataFrame(pca.fit_transform(Z_full),
                                  index=Z_full.index,
                                  columns=[f"pc{i}" for i in range(n_comp)])
            var_explained = pca.explained_variance_ratio_.sum() * 100
            res = evaluate_subset(df, Z_pca, test_idx, f"G_PCA({n_comp})")
            res["var_explained_pct"] = var_explained
            results.append(res)
            print(f"  G_PCA({n_comp:>2})              n_dims={n_comp:>2}  N={res['n']:>3}  "
                  f"MAE_NN={res['mae_nn']*100:.3f}%  MAE_RW={res['mae_rw']*100:.3f}%  "
                  f"lift={res['lift_pct']:+6.2f}%  DirAcc={res['diracc_pct']:.1f}%  p={res['pval']:.3f}  "
                  f"var={var_explained:.0f}%")
    except ImportError:
        print("  [SKIP] sklearn no disponible para PCA")

    # ── Resumen ─────────────────────────────────────────────────
    print(f"\n══════════════════════════════════════════════════════════════")
    print(f"  Resumen — ¿alguna config supera always-sell?")
    print(f"══════════════════════════════════════════════════════════════")
    R = pd.DataFrame(results).sort_values("lift_pct", ascending=False)
    n_winners = int((R["lift_pct"] > 0).sum())
    n_significant = int((R["pval"] < 0.10).sum())
    print(f"  Configs probadas: {len(R)}")
    print(f"  Configs con lift > 0:    {n_winners}/{len(R)}")
    print(f"  Configs con p<0.10:       {n_significant}/{len(R)}")
    if n_winners > 0:
        winners = R[R["lift_pct"] > 0]
        print(f"\n  GANADORES (lift > 0):")
        for _, row in winners.iterrows():
            print(f"    {row['label']:<22} dims={row['n_dims']}  lift={row['lift_pct']:+.2f}%  p={row['pval']:.3f}")
    else:
        print(f"\n  ❌ Ninguna config supera al always-sell.")
        print(f"  Conclusión: el problema NO es curse of dimensionality.")
        print(f"  La señal direccional no está en estas features para este OOS.")


if __name__ == "__main__":
    main()
