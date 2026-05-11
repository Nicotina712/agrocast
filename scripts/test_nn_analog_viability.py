"""
scripts/test_nn_analog_viability.py
Test de viabilidad de Nearest-Neighbor Analog Forecasting (NNAF).

Pregunta: si tomamos un vector multidimensional del "estado del sistema" en
fecha D y buscamos los top-K vecinos más cercanos del pasado, ¿la mediana
de outcomes (ret_30d) de esos vecinos es mejor predictor que always-sell?

Diseño walk-forward (sin look-ahead):
  1. Para cada fecha D en el último año
  2. Construir vector de estado al día D (z-scored rolling 252d)
  3. Filtrar histórico: solo fechas D' < D (no contamos el futuro)
  4. Buscar top-K=20 vecinos por distancia euclidiana
  5. Mediana de ret_30d_fwd de esos vecinos = predicción
  6. Comparar MAE vs:
       - Always-sell (predice ret=0)
       - Random walk (predice ret=0, equivalente)
       - Mediana global de ret_30d (predicción ingenua histórica)

Si NN-analog gana al baseline con p<0.10 → idea viable, construir módulo.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")

# Features de estado: covers price relativo, vol, exógenas, COT, momentum, news, climate
STATE_FEATURES = [
    # Price-relative
    "price_vs_ma20", "price_vs_ma90", "ma20_vs_ma90", "ma90_slope",
    # Momentum
    "mom_5d", "mom_20d", "mom_60d", "rsi_14",
    # Volatility
    "vol_30d", "vol_60d",
    # Cross-commodity
    "soy_corn_ratio", "soy_corn_ratio_dev", "soy_oil_ratio", "crush_spread_dev",
    # Exógenas macro
    "Oil_chg7", "Dollar_chg7",
    # News
    "news_sentiment", "news_velocity_7d",
    # COT
    "cot_noncomm_long_pct", "cot_index", "cot_contrarian_score",
    # Climate
    "enso_oni",
    # Stagionality
    "month_sin", "month_cos",
]

K_NEIGHBORS    = 20
HORIZON_DAYS   = 30
ZSCORE_WINDOW  = 252   # 1 año
MIN_GAP_DAYS   = 60    # vecino debe estar al menos 60 días antes (no fechas adyacentes)
N_TEST_DATES   = 100
TEST_FREQ_DAYS = 7     # 1 fecha por semana


def main():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    df["ret_30d_fwd"] = df["Soybeans"].pct_change(30).shift(-30)

    # Filtrar features disponibles
    available = [f for f in STATE_FEATURES if f in df.columns]
    missing = [f for f in STATE_FEATURES if f not in df.columns]
    print(f"Features disponibles: {len(available)}/{len(STATE_FEATURES)}")
    if missing:
        print(f"  Faltantes (se omiten): {missing}")
    print(f"  Usadas: {available}\n")

    # ── Construir matriz de estado z-scored rolling ─────────────────
    # Esto evita look-ahead: el z-score de día D usa solo media/std de días < D
    print(f"Construyendo z-scores rolling {ZSCORE_WINDOW}d…")
    Z = pd.DataFrame(index=df.index)
    for f in available:
        s = df[f]
        rolling_mean = s.rolling(ZSCORE_WINDOW, min_periods=60).mean().shift(1)
        rolling_std  = s.rolling(ZSCORE_WINDOW, min_periods=60).std().shift(1).replace(0, 1e-6)
        Z[f] = (s - rolling_mean) / rolling_std
    Z = Z.fillna(0).replace([np.inf, -np.inf], 0)

    # ── Test dates: 1 por semana en el último año ──────────────────
    end = df["Date"].max()
    test_start = end - pd.DateOffset(years=2)   # ventana extendida
    test_idx = []
    d = test_start
    while d <= end - pd.Timedelta(days=HORIZON_DAYS):
        future = df[df["Date"] >= d]
        if future.empty:
            break
        idx = int(df.index[df["Date"] == future.iloc[0]["Date"]][0])
        test_idx.append(idx)
        d = future.iloc[0]["Date"] + pd.Timedelta(days=TEST_FREQ_DAYS)
    print(f"Test dates: {len(test_idx)}")

    # ── Para cada fecha test, hacer NN-analog ──────────────────────
    rows = []
    for idx_t in test_idx:
        D = df.iloc[idx_t]["Date"]
        target = df.iloc[idx_t].get("ret_30d_fwd")
        if pd.isna(target):
            continue
        v_t = Z.iloc[idx_t].values

        # Histórico válido: < D - MIN_GAP_DAYS, con ret_30d_fwd válido
        gap_date = D - pd.Timedelta(days=MIN_GAP_DAYS)
        hist_mask = (df["Date"] < gap_date) & df["ret_30d_fwd"].notna()
        hist_idx = df.index[hist_mask]
        if len(hist_idx) < K_NEIGHBORS * 2:
            continue

        # Distancia euclidiana
        V_hist = Z.loc[hist_idx].values
        dists  = np.sqrt(((V_hist - v_t) ** 2).sum(axis=1))
        # Top-K vecinos
        top_k = np.argsort(dists)[:K_NEIGHBORS]
        neighbor_idx = hist_idx[top_k]
        neighbor_rets = df.loc[neighbor_idx, "ret_30d_fwd"].values

        # Predicciones
        pred_nn_median = float(np.median(neighbor_rets))
        pred_nn_mean   = float(np.mean(neighbor_rets))
        # Predicción ponderada por inverso de distancia
        weights = 1.0 / (dists[top_k] + 1e-6)
        weights /= weights.sum()
        pred_nn_weighted = float((neighbor_rets * weights).sum())
        # Baseline 1: mediana global histórica
        pred_global_med = float(df.loc[hist_idx, "ret_30d_fwd"].median())
        # Baseline 2: always-sell (= 0 ret esperado)
        pred_zero = 0.0

        rows.append({
            "date": D, "actual_ret": float(target),
            "pred_nn_median":   pred_nn_median,
            "pred_nn_mean":     pred_nn_mean,
            "pred_nn_weighted": pred_nn_weighted,
            "pred_global_med":  pred_global_med,
            "pred_zero":        pred_zero,
            "neighbors_dist_med": float(np.median(dists[top_k])),
            "neighbors_dist_max": float(np.max(dists[top_k])),
        })

    R = pd.DataFrame(rows)
    print(f"N evaluados: {len(R)}")

    # ── Comparativa de errores ─────────────────────────────────────
    print(f"\n══════════════════════════════════════════════════════════════")
    print(f"  Comparativa de predictores (MAE en ret_30d)")
    print(f"══════════════════════════════════════════════════════════════")

    def mae(col):  return float((R[col] - R["actual_ret"]).abs().mean())
    def mape(col): return float(((R[col] - R["actual_ret"]).abs() / R["actual_ret"].abs().clip(lower=1e-3)).mean())
    def diracc(col):
        """Direction accuracy: signo correcto."""
        return float(((np.sign(R[col]) == np.sign(R["actual_ret"]))).mean() * 100)

    base_mae = mae("pred_zero")
    print(f"  {'Predictor':<25}  {'MAE':>9}  {'Lift_vs_RW':>11}  {'DirAcc%':>9}")
    print(f"  {'-'*25}  {'-'*9}  {'-'*11}  {'-'*9}")
    for col, label in [
        ("pred_zero",        "Always-sell (RW)"),
        ("pred_global_med",  "Mediana global"),
        ("pred_nn_median",   "NN-analog (mediana)"),
        ("pred_nn_mean",     "NN-analog (mean)"),
        ("pred_nn_weighted", "NN-analog (weighted)"),
    ]:
        m = mae(col)
        lift = (base_mae - m) / base_mae * 100
        diracc_v = diracc(col) if col != "pred_zero" else 0
        print(f"  {label:<25}  {m*100:>7.3f}%  {lift:>+9.2f}%  {diracc_v:>7.1f}%")

    # Wilcoxon paired NN vs zero
    from scipy.stats import wilcoxon
    err_nn   = (R["pred_nn_weighted"] - R["actual_ret"]).abs()
    err_zero = (R["pred_zero"]        - R["actual_ret"]).abs()
    try:
        stat, pval = wilcoxon(err_nn, err_zero)
        sign = "✓ significativo" if pval < 0.10 else "× ruido"
        print(f"\n  Wilcoxon NN-weighted vs always-sell: Δ_avg={(err_nn-err_zero).mean()*100:+.4f}%  "
              f"p={pval:.4f}  {sign}")
    except Exception as e:
        print(f"\n  Wilcoxon falló: {e}")

    # ── Análisis condicional: ¿gana en fechas con vecinos cercanos? ─
    print(f"\n══════════════════════════════════════════════════════════════")
    print(f"  Por proximidad de vecinos (top quartil de distancia)")
    print(f"══════════════════════════════════════════════════════════════")
    median_dist = R["neighbors_dist_med"].median()
    near = R[R["neighbors_dist_med"] <= R["neighbors_dist_med"].quantile(0.50)]
    far  = R[R["neighbors_dist_med"] >  R["neighbors_dist_med"].quantile(0.50)]
    for label, sub in [("Vecinos cercanos (mitad)", near), ("Vecinos lejanos (mitad)", far)]:
        if sub.empty: continue
        m_nn   = float((sub["pred_nn_weighted"] - sub["actual_ret"]).abs().mean())
        m_zero = float((sub["pred_zero"]        - sub["actual_ret"]).abs().mean())
        lift = (m_zero - m_nn) / m_zero * 100
        diracc_v = float((np.sign(sub["pred_nn_weighted"]) == np.sign(sub["actual_ret"])).mean() * 100)
        print(f"  {label:<27} N={len(sub):>3}  MAE_NN={m_nn*100:.3f}%  "
              f"MAE_RW={m_zero*100:.3f}%  lift={lift:+.2f}%  DirAcc={diracc_v:.1f}%")

    # Persistir
    R.to_csv(os.path.join(ROOT, "artifacts_eval", "test_nn_analog.csv"), index=False)
    print(f"\n💾 artifacts_eval/test_nn_analog.csv")


if __name__ == "__main__":
    main()
