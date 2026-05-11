"""
scripts/experiments_h1_h2_h6.py
Tres experimentos en una pasada:

H1 — Predecir volatilidad realizada 30d (en lugar del precio)
     Hipótesis: vol clustering hace que σ_t+30 sea más predecible que p_t+30.
     Baselines: naive (vol_t), EWMA (lambda=0.94), random walk en vol.

H2 — Predecir spreads (soja/maíz y basis Uruguay)
     Hipótesis: los spreads tienen mean reversion explotable; el nivel del precio no.
     Baselines: RW en spread, MA reversion (precio ↔ media).

H6 — Causal feature pruning (Granger)
     Hipótesis: features con p>0.10 en Granger no aportan; pruning reduce ruido.
     Procedimiento: Granger lags 1-5, mantener p<0.10, retrain horizons, comparar.

Salida unificada: tabla con MAE/lift por experimento + estadística (Wilcoxon).
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")
OOS_YEARS    = 1
TRAIN_YEARS  = 5
SAMPLE_STEP  = 5

# ────────────────────────────────────────────────────────────────────
# Carga
# ────────────────────────────────────────────────────────────────────
def load():
    return pd.read_csv(FEATURES_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════
# H1 — Volatilidad
# ════════════════════════════════════════════════════════════════════
def add_realized_vol(df: pd.DataFrame, window_fwd: int = 30) -> pd.DataFrame:
    """Computa realized vol forward: std de log returns en [t, t+window]."""
    d = df.copy()
    d["log_ret"] = np.log(d["Soybeans"] / d["Soybeans"].shift(1))
    # Rolling forward std (vol realizada futura)
    d["rv_fwd_30d"] = (
        d["log_ret"].rolling(window=window_fwd).std()
        .shift(-window_fwd)
        .mul(np.sqrt(252))   # anualizada
    )
    # Vol realizada pasada (para naive baselines)
    d["rv_past_30d"] = d["log_ret"].rolling(window=window_fwd).std().mul(np.sqrt(252))
    d["rv_past_60d"] = d["log_ret"].rolling(window=60).std().mul(np.sqrt(252))
    return d


def ewma_vol(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics-style EWMA volatility, anualizada."""
    var = returns.pow(2)
    ewma = var.ewm(alpha=1 - lam, adjust=False).mean()
    return ewma.pow(0.5).mul(np.sqrt(252))


def h1_volatility(df: pd.DataFrame) -> dict:
    print("\n══════ H1 — Volatilidad 30d ══════")
    d = add_realized_vol(df, window_fwd=30)
    d["rv_ewma"] = ewma_vol(d["log_ret"])
    d = d.dropna(subset=["rv_fwd_30d"]).reset_index(drop=True)

    end       = d["Date"].max()
    oos_start = end - pd.DateOffset(years=OOS_YEARS)

    train = d[d["Date"] < oos_start].copy()
    oos   = d[d["Date"] >= oos_start].copy().iloc[::SAMPLE_STEP]
    if len(oos) < 10:
        return {"ok": False, "error": "OOS insuficiente"}

    # Features: usar todo el set numérico salvo target y leakage
    EXCLUDE = {"Date", "Soybeans", "Soybeans_log", "log_ret", "rv_fwd_30d",
               "ret_1d_fwd", "ret_7d_fwd", "ret_14d_fwd", "ret_30d_fwd",
               "Soybeans_High", "Soybeans_Low", "Soybeans_Open"}
    feats = [c for c in d.select_dtypes(include=[np.number]).columns if c not in EXCLUDE]

    # Train XGB sobre rv_fwd_30d (target)
    from xgboost import XGBRegressor
    Xtr = train[feats].fillna(0).replace([np.inf, -np.inf], 0)
    ytr = train["rv_fwd_30d"]
    print(f"  train: {len(train)} filas, {len(feats)} features")
    t0 = time.time()
    model = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.8, random_state=42,
                         eval_metric="mae")
    model.fit(Xtr, ytr, verbose=False)
    print(f"  trained in {time.time()-t0:.1f}s")

    # Predict OOS
    Xo = oos[feats].fillna(0).replace([np.inf, -np.inf], 0)
    pred_xgb   = model.predict(Xo)
    actual     = oos["rv_fwd_30d"].values

    # Baselines en OOS
    pred_naive_30   = oos["rv_past_30d"].values
    pred_naive_60   = oos["rv_past_60d"].values
    pred_ewma       = oos["rv_ewma"].values

    def mae(p, y): return float(np.mean(np.abs(p - y)))
    def mape(p, y): return float(np.mean(np.abs((p - y) / y))) * 100

    rw_mae = mae(pred_naive_30, actual)

    rows = [
        ("Naive 30d (RV pasado 30d)", pred_naive_30),
        ("Naive 60d (RV pasado 60d)", pred_naive_60),
        ("EWMA RiskMetrics λ=0.94",   pred_ewma),
        ("XGB sobre features",        pred_xgb),
    ]
    print(f"\n  ┌─────────────────────────────┬─────────┬─────────┬───────────┐")
    print(  f"  │ Modelo                      │ MAE     │ MAPE %  │ Lift vs RW│")
    print(  f"  ├─────────────────────────────┼─────────┼─────────┼───────────┤")
    summary = []
    for name, p in rows:
        m = mae(p, actual); mp = mape(p, actual)
        lift = (rw_mae - m) / rw_mae * 100
        print(f"  │ {name:<27} │ {m:7.4f} │ {mp:6.2f}  │ {lift:+8.1f}% │")
        summary.append({"model": name, "mae": m, "mape": mp, "lift_vs_rw": lift})
    print(  f"  └─────────────────────────────┴─────────┴─────────┴───────────┘")
    print(f"  N obs OOS = {len(oos)}")

    # Significancia: paired test XGB vs Naive RW
    from scipy.stats import wilcoxon
    err_xgb   = np.abs(pred_xgb   - actual)
    err_naive = np.abs(pred_naive_30 - actual)
    try:
        stat, pval = wilcoxon(err_xgb, err_naive)
        print(f"  Wilcoxon paired (XGB vs Naive): p={pval:.4f}  "
              f"{'✓ significativo' if pval < 0.05 else '× ruido'}")
    except Exception:
        pval = None

    return {"ok": True, "n_obs": int(len(oos)), "summary": summary, "p_xgb_vs_rw": pval}


# ════════════════════════════════════════════════════════════════════
# H2 — Spreads
# ════════════════════════════════════════════════════════════════════
def h2_spreads(df: pd.DataFrame) -> dict:
    print("\n══════ H2 — Spreads (mean reversion) ══════")
    d = df.copy()
    if "soy_corn_ratio" not in d.columns:
        return {"ok": False, "error": "soy_corn_ratio missing"}

    # Targets forward 30d
    d["ratio_fwd_30d"] = d["soy_corn_ratio"].shift(-30)
    if "basis_uy_usd_ton" in d.columns:
        d["basis_fwd_30d"] = d["basis_uy_usd_ton"].shift(-30)
    if "crush_spread" in d.columns:
        d["crush_fwd_30d"] = d["crush_spread"].shift(-30)

    end       = d["Date"].max()
    oos_start = end - pd.DateOffset(years=OOS_YEARS)
    train = d[d["Date"] < oos_start].copy()
    oos   = d[d["Date"] >= oos_start].copy().iloc[::SAMPLE_STEP]
    if len(oos) < 10:
        return {"ok": False, "error": "OOS insuficiente"}

    out = {"ok": True, "spreads": []}

    for spread_col, target_col in [
        ("soy_corn_ratio",   "ratio_fwd_30d"),
        ("crush_spread",     "crush_fwd_30d"),
        ("basis_uy_usd_ton", "basis_fwd_30d"),
    ]:
        if spread_col not in d.columns or target_col not in d.columns:
            continue
        tr = train.dropna(subset=[target_col]).copy()
        oo = oos.dropna(subset=[target_col]).copy()
        if len(oo) < 10:
            continue

        # Features: el nivel actual + momentum del spread + macro/exógenos
        EXCLUDE = {"Date", "Soybeans", "Soybeans_log", target_col,
                   "ret_1d_fwd", "ret_7d_fwd", "ret_14d_fwd", "ret_30d_fwd",
                   "ratio_fwd_30d", "crush_fwd_30d", "basis_fwd_30d",
                   "Soybeans_High", "Soybeans_Low", "Soybeans_Open"}
        feats = [c for c in d.select_dtypes(include=[np.number]).columns if c not in EXCLUDE]

        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             eval_metric="mae")
        Xtr = tr[feats].fillna(0).replace([np.inf, -np.inf], 0)
        ytr = tr[target_col]
        model.fit(Xtr, ytr, verbose=False)

        Xo = oo[feats].fillna(0).replace([np.inf, -np.inf], 0)
        pred_xgb  = model.predict(Xo)
        pred_rw   = oo[spread_col].values        # Random walk del spread
        # Mean reversion: si tenemos MA del spread, predecir hacia él
        ma_col = f"{spread_col}_ma" if f"{spread_col}_ma" in d.columns else None
        if ma_col is not None:
            pred_mr = oo[ma_col].values
        else:
            ma_train = tr[spread_col].rolling(60, min_periods=30).mean()
            ma_train = ma_train.iloc[-1] if not ma_train.empty else float(tr[spread_col].mean())
            pred_mr  = np.full(len(oo), ma_train)

        actual = oo[target_col].values
        def mae(p, y): return float(np.mean(np.abs(p - y)))
        rw_mae = mae(pred_rw, actual)

        rows = [
            ("RW (spread_t)",         pred_rw),
            ("Mean reversion (MA)",   pred_mr),
            ("XGB sobre features",    pred_xgb),
        ]
        print(f"\n  ── Target: {spread_col} → t+30  (n={len(oo)}) ──")
        for name, p in rows:
            m = mae(p, actual)
            lift = (rw_mae - m) / rw_mae * 100
            print(f"    {name:<28} MAE={m:.4f}  lift_vs_rw={lift:+6.1f}%")
            out["spreads"].append({"spread": spread_col, "model": name,
                                    "mae": m, "lift_vs_rw": lift, "n": int(len(oo))})

        # Direction accuracy: ¿el spread sube o baja?
        from scipy.stats import wilcoxon
        try:
            err_xgb = np.abs(pred_xgb - actual)
            err_rw  = np.abs(pred_rw  - actual)
            _, pval = wilcoxon(err_xgb, err_rw)
            print(f"    Wilcoxon XGB vs RW: p={pval:.4f}  "
                  f"{'✓' if pval < 0.05 else '×'}")
        except Exception:
            pass

    return out


# ════════════════════════════════════════════════════════════════════
# H6 — Granger pruning
# ════════════════════════════════════════════════════════════════════
def h6_granger(df: pd.DataFrame, max_lag: int = 5, p_threshold: float = 0.10) -> dict:
    print(f"\n══════ H6 — Granger pruning (lag≤{max_lag}, p<{p_threshold}) ══════")
    from statsmodels.tsa.stattools import grangercausalitytests
    import warnings
    warnings.filterwarnings("ignore")

    if "ret_30d_fwd" not in df.columns:
        return {"ok": False, "error": "ret_30d_fwd missing"}

    EXCLUDE = {"Date", "Soybeans", "Soybeans_log", "target_log",
               "ret_1d_fwd", "ret_7d_fwd", "ret_14d_fwd", "ret_30d_fwd",
               "Soybeans_High", "Soybeans_Low", "Soybeans_Open"}
    feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in EXCLUDE]

    target = df["ret_30d_fwd"].dropna()
    causal = []
    skipped = 0

    print(f"  Testeando {len(feats)} features...")
    for i, f in enumerate(feats):
        s = df[[f]].iloc[target.index].copy()
        s["target"] = target.values
        s = s.dropna()
        if len(s) < 100 or s[f].std() < 1e-8:
            skipped += 1
            continue
        try:
            res = grangercausalitytests(s[["target", f]], maxlag=max_lag, verbose=False)
            # Min p-value entre los lags
            p_min = min(res[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1))
            if p_min < p_threshold:
                causal.append((f, p_min))
        except Exception:
            skipped += 1

    causal.sort(key=lambda x: x[1])
    print(f"  Causales (p<{p_threshold}): {len(causal)} / {len(feats)} (skipped={skipped})")
    print(f"  Top 10 causales:")
    for f, p in causal[:10]:
        print(f"    {f:<35}  p={p:.4f}")

    # Re-train horizons con feature set podado y comparar MAE
    pruned = [c for c, _ in causal]
    if len(pruned) < 5:
        return {"ok": True, "causal": causal, "note": "too few causal features for retraining"}

    print(f"\n  Re-entrenando modelo 30d con {len(pruned)} features causales (Granger)...")
    end       = df["Date"].max()
    oos_start = end - pd.DateOffset(years=OOS_YEARS)
    train = df[df["Date"] < oos_start].dropna(subset=["ret_30d_fwd"]).copy()
    oos   = df[(df["Date"] >= oos_start)].copy()

    # Sample step para tener pares (t, t+30)
    pairs = []
    eligible = oos.iloc[::SAMPLE_STEP]
    for _, row in eligible.iterrows():
        t30 = row["Date"] + pd.Timedelta(days=30)
        future = df[df["Date"] >= t30]
        if future.empty:
            continue
        pairs.append({"date": row["Date"], "p_t": float(row["Soybeans"]),
                      "p_t30": float(future.iloc[0]["Soybeans"]),
                      "idx": int(df.index[df["Date"] == row["Date"]][0])})

    from xgboost import XGBRegressor

    def fit_and_eval(features):
        m = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.8, random_state=42,
                         eval_metric="mae")
        Xtr = train[features].fillna(0).replace([np.inf, -np.inf], 0)
        ytr = train["ret_30d_fwd"]
        m.fit(Xtr, ytr, verbose=False)
        Xo = df.iloc[[p["idx"] for p in pairs]][features].fillna(0).replace([np.inf, -np.inf], 0)
        pred_ret = m.predict(Xo)
        preds = np.array([p["p_t"] for p in pairs]) * (1 + pred_ret)
        actuals = np.array([p["p_t30"] for p in pairs])
        return preds, actuals

    p_full, actuals = fit_and_eval(feats)
    p_prun, _       = fit_and_eval(pruned)
    rw  = np.array([p["p_t"] for p in pairs])
    rw_mae = float(np.mean(np.abs(rw - actuals)))
    mae_full = float(np.mean(np.abs(p_full - actuals)))
    mae_prun = float(np.mean(np.abs(p_prun - actuals)))

    print(f"\n  ┌─────────────────────────────┬─────────┬───────────┐")
    print(  f"  │ Feature set                 │ MAE     │ Lift vs RW│")
    print(  f"  ├─────────────────────────────┼─────────┼───────────┤")
    print(  f"  │ Random Walk                 │ {rw_mae:7.2f} │      0.0% │")
    print(  f"  │ Full ({len(feats)} features)         │ {mae_full:7.2f} │ {(rw_mae-mae_full)/rw_mae*100:+6.1f}% │")
    print(  f"  │ Pruned Granger ({len(pruned)})          │ {mae_prun:7.2f} │ {(rw_mae-mae_prun)/rw_mae*100:+6.1f}% │")
    print(  f"  └─────────────────────────────┴─────────┴───────────┘")
    print(f"  N obs = {len(pairs)}")

    return {"ok": True, "n_causal": len(causal), "n_total": len(feats),
            "top_causal": [{"feature": f, "p": p} for f, p in causal[:20]],
            "mae_full": mae_full, "mae_pruned": mae_prun, "rw_mae": rw_mae}


# ════════════════════════════════════════════════════════════════════
def main():
    df = load()
    print(f"Features cargadas: {df.shape} | rango: {df['Date'].min().date()} → {df['Date'].max().date()}")

    results = {}
    results["H1"] = h1_volatility(df)
    results["H2"] = h2_spreads(df)
    results["H6"] = h6_granger(df)

    # Persistir resumen
    out_path = os.path.join(ROOT, "artifacts_eval", "experiments_h1_h2_h6.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n💾 {out_path}")


if __name__ == "__main__":
    main()
