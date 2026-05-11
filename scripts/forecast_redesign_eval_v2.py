"""
scripts/forecast_redesign_eval_v2.py
Segunda ronda: pasos 1, 2 y 3 sobre el ganador de la ronda anterior (#3 retornos+WF).

Paso 1 — Conformal calibration de bandas Q10/Q90 (cobertura objetivo 80%).
Paso 2 — Ensemble #3 + Random Walk con α optimizado sobre OOS (split val/test).
Paso 3 — Refit cada 30d en walk-forward (vs 60d de la ronda anterior).

Para evitar leakage, dividimos el OOS año en:
   - Validación: primeros ~6 meses (calibrar α y conformal cuantile).
   - Test: últimos ~6 meses (métricas reportadas).
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")
EVAL_DIR     = os.path.join(ROOT, "artifacts_eval")
os.makedirs(EVAL_DIR, exist_ok=True)

HORIZON      = 30
OOS_YEARS    = 1
TRAIN_YEARS  = 4
MAX_FEATURES = 25
SAMPLE_STEP  = 5

EXCLUDE = {
    "Date", "Soybeans", "Soybeans_log", "target_log",
    "ret_1d_fwd", "ret_7d_fwd", "ret_14d_fwd", "ret_30d_fwd",
    "Soybeans_High", "Soybeans_Low", "Soybeans_Open",
    "Maize_High", "Maize_Low", "Maize_Open",
    "SoybeanMeal_High", "SoybeanMeal_Low", "SoybeanMeal_Open",
    "SoybeanOil_High",  "SoybeanOil_Low",  "SoybeanOil_Open",
    "released_at", "as_of_date", "wasde_date", "week_ending",
}


def load():
    return pd.read_csv(FEATURES_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)


def select_features(df, target_col, max_n=MAX_FEATURES):
    num = df.select_dtypes(include=[np.number]).copy()
    num[target_col] = df[target_col]
    corr = num.corr()[target_col].abs().sort_values(ascending=False)
    return [c for c in corr.index if c not in EXCLUDE and c != target_col][:max_n]


def fit_xgb(X, y, **kw):
    m = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="mae", **kw,
    )
    m.fit(X, y, verbose=False)
    return m


def fit_xgb_quantile(X, y, alpha):
    m = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        objective="reg:quantileerror", quantile_alpha=alpha,
    )
    m.fit(X, y, verbose=False)
    return m


def make_eval_pairs(df, oos_start, oos_end, step=SAMPLE_STEP):
    pairs = []
    eligible = df[(df["Date"] >= oos_start) & (df["Date"] <= oos_end)].iloc[::step]
    for _, row in eligible.iterrows():
        t30 = row["Date"] + pd.Timedelta(days=HORIZON)
        future = df[df["Date"] >= t30]
        if future.empty:
            continue
        pairs.append({
            "idx":      int(df.index[df["Date"] == row["Date"]][0]),
            "date":     row["Date"],
            "price_t":  float(row["Soybeans"]),
            "price_t30": float(future.iloc[0]["Soybeans"]),
            "month":    int(row["Date"].month),
        })
    return pairs


def walk_forward_returns(df, pairs, refit_every_days):
    """Retorna preds, q10, q90 + lista de modelos para inspección."""
    preds, q10s, q90s = [], [], []
    last_refit = None
    cur_model = cur_q10 = cur_q90 = cur_feats = None
    refit_count = 0
    t0 = time.time()
    for p in pairs:
        if (last_refit is None) or ((p["date"] - last_refit).days >= refit_every_days):
            train_start = p["date"] - pd.DateOffset(years=TRAIN_YEARS)
            tr = df[(df["Date"] >= train_start) & (df["Date"] < p["date"])].copy()
            tr = tr.dropna(subset=["ret_30d_fwd"])
            cur_feats = select_features(tr, "ret_30d_fwd")
            Xtr = tr[cur_feats].fillna(0).replace([np.inf,-np.inf],0)
            ytr = tr["ret_30d_fwd"]
            cur_model = fit_xgb(Xtr, ytr)
            cur_q10   = fit_xgb_quantile(Xtr, ytr, 0.10)
            cur_q90   = fit_xgb_quantile(Xtr, ytr, 0.90)
            last_refit = p["date"]
            refit_count += 1
        Xrow = df.iloc[[p["idx"]]][cur_feats].fillna(0).replace([np.inf,-np.inf],0)
        rp  = float(cur_model.predict(Xrow)[0])
        r10 = float(cur_q10.predict(Xrow)[0])
        r90 = float(cur_q90.predict(Xrow)[0])
        preds.append(p["price_t"] * (1 + rp))
        q10s.append(p["price_t"]  * (1 + min(r10, rp)))
        q90s.append(p["price_t"]  * (1 + max(r90, rp)))
    elapsed = time.time() - t0
    return (np.array(preds), np.array(q10s), np.array(q90s),
            {"refits": refit_count, "elapsed_s": elapsed,
             "last_model": cur_model, "last_q10": cur_q10, "last_q90": cur_q90,
             "last_feats": cur_feats})


def metrics(preds, actuals, rw_mae=None, ps=None):
    mae  = float(np.mean(np.abs(preds - actuals)))
    mape = float(np.mean(np.abs((preds - actuals) / actuals))) * 100
    out  = {"mae": mae, "mape": mape}
    if rw_mae is not None:
        out["lift_pct"] = (rw_mae - mae) / rw_mae * 100
    if ps is not None:
        dp = np.sign(preds - ps); dr = np.sign(actuals - ps)
        valid = dr != 0
        out["dir_acc"] = float(np.mean(dp[valid] == dr[valid]) * 100) if valid.any() else float("nan")
    return out


# ─────────────────────────────────────────────────────────────────
# Conformal calibration: delta tal que cobertura ≈ 1-alpha
# ─────────────────────────────────────────────────────────────────
def conformal_delta(actuals_val, q10_val, q90_val, target=0.80):
    """
    Calcula expansión simétrica δ tal que P(Q10-δ ≤ y ≤ Q90+δ) ≥ target.
    Toma como score la "distancia fuera de la banda" (signed) de cada obs val.
    """
    above = actuals_val - q90_val      # positivo si y supera Q90
    below = q10_val - actuals_val      # positivo si y queda debajo de Q10
    score = np.maximum(above, below)
    score = np.maximum(score, 0)       # 0 si está dentro
    # delta = quantile(target) sobre los scores → expansión necesaria para cubrir target%
    delta = float(np.quantile(score, target))
    return max(delta, 0.0)


def main():
    df = load()
    oos_end   = df["Date"].max()
    oos_start = oos_end - pd.DateOffset(years=OOS_YEARS)
    val_end   = oos_start + (oos_end - oos_start) / 2
    test_start = val_end + pd.Timedelta(days=1)
    print(f"OOS total: {oos_start.date()} → {oos_end.date()}")
    print(f"  Val  : {oos_start.date()} → {val_end.date()}  (calibra α y conformal δ)")
    print(f"  Test : {test_start.date()} → {oos_end.date()}  (métricas reportadas)\n")

    pairs_all  = make_eval_pairs(df, oos_start, oos_end)
    pairs_val  = [p for p in pairs_all if p["date"] <= val_end]
    pairs_test = [p for p in pairs_all if p["date"] >= test_start]
    print(f"Obs: total={len(pairs_all)}, val={len(pairs_val)}, test={len(pairs_test)}")

    # ── Walk-forward 30d y 60d ─────────────────────────────────
    print("\n[WF refit cada 60d]")
    p60, q10_60, q90_60, info60 = walk_forward_returns(df, pairs_all, 60)
    print(f"   {info60['refits']} refits, {info60['elapsed_s']:.1f}s")

    print("\n[WF refit cada 30d]")
    p30, q10_30, q90_30, info30 = walk_forward_returns(df, pairs_all, 30)
    print(f"   {info30['refits']} refits, {info30['elapsed_s']:.1f}s")

    actuals = np.array([p["price_t30"] for p in pairs_all])
    pt      = np.array([p["price_t"]   for p in pairs_all])

    # Indices val/test
    val_mask  = np.array([p["date"] <= val_end       for p in pairs_all])
    test_mask = np.array([p["date"] >= test_start    for p in pairs_all])

    # ── Baselines en TEST ──────────────────────────────────────
    rw_test    = pt[test_mask]
    actuals_te = actuals[test_mask]
    pt_te      = pt[test_mask]
    rw_mae_te  = float(np.mean(np.abs(rw_test - actuals_te)))

    # Seasonal naive
    train_df = df[df["Date"] < oos_start].copy()
    train_df["ret30"] = train_df["Soybeans"].pct_change(30).shift(-30)
    seas_map = train_df.groupby(train_df["Date"].dt.month)["ret30"].mean().to_dict()
    seas_test = np.array([p["price_t"] * (1 + (seas_map.get(p["month"],0.0) or 0.0))
                          for p, m in zip(pairs_all, test_mask) if m])

    # ── Paso 2 — Ensemble α óptimo en VAL ──────────────────────
    def best_alpha(preds_val, actuals_val, pt_val, grid=np.linspace(0, 1, 21)):
        best_a, best_mae = 0.0, float("inf")
        for a in grid:
            blend = a * preds_val + (1 - a) * pt_val
            m = float(np.mean(np.abs(blend - actuals_val)))
            if m < best_mae:
                best_a, best_mae = float(a), m
        return best_a, best_mae

    a60, mae_v60 = best_alpha(p60[val_mask],  actuals[val_mask], pt[val_mask])
    a30, mae_v30 = best_alpha(p30[val_mask],  actuals[val_mask], pt[val_mask])
    print(f"\n[Ensemble α óptimo en val]")
    print(f"   refit 60d → α* = {a60:.2f}  (MAE val ensemble = {mae_v60:.2f})")
    print(f"   refit 30d → α* = {a30:.2f}  (MAE val ensemble = {mae_v30:.2f})")

    # Aplicar α óptimo al test
    ens60_te = a60 * p60[test_mask] + (1 - a60) * pt[test_mask]
    ens30_te = a30 * p30[test_mask] + (1 - a30) * pt[test_mask]

    # ── Paso 1 — Conformal δ en VAL aplicado a TEST ────────────
    delta60 = conformal_delta(actuals[val_mask], q10_60[val_mask], q90_60[val_mask], 0.80)
    delta30 = conformal_delta(actuals[val_mask], q10_30[val_mask], q90_30[val_mask], 0.80)
    q10_60_cal_te = q10_60[test_mask] - delta60
    q90_60_cal_te = q90_60[test_mask] + delta60
    q10_30_cal_te = q10_30[test_mask] - delta30
    q90_30_cal_te = q90_30[test_mask] + delta30
    print(f"\n[Conformal δ calibrada en val]")
    print(f"   refit 60d → δ = ${delta60:.2f}")
    print(f"   refit 30d → δ = ${delta30:.2f}")

    def coverage(q10, q90, y):
        return float(np.mean((y >= q10) & (y <= q90)) * 100)

    def band_w(q10, q90):
        return float(np.mean(q90 - q10))

    # ── Tabla final TEST ───────────────────────────────────────
    rows = [
        ("Random Walk",                         rw_test,        None, None),
        ("Seasonal naive",                      seas_test,      None, None),
        ("#3 WF refit 60d",                     p60[test_mask], q10_60[test_mask], q90_60[test_mask]),
        ("#3 WF refit 60d + ensemble α=%.2f" % a60, ens60_te,   None, None),
        ("#3 WF refit 60d + conformal",         p60[test_mask], q10_60_cal_te, q90_60_cal_te),
        ("#3 WF refit 30d",                     p30[test_mask], q10_30[test_mask], q90_30[test_mask]),
        ("#3 WF refit 30d + ensemble α=%.2f" % a30, ens30_te,   None, None),
        ("#3 WF refit 30d + conformal",         p30[test_mask], q10_30_cal_te, q90_30_cal_te),
        ("#3 WF refit 30d + ens + conformal",   ens30_te,       q10_30_cal_te, q90_30_cal_te),
    ]

    print("\n┌─────────────────────────────────────────────┬─────────┬─────────┬───────────┬──────────┬──────────┬───────────┐")
    print("│ Modelo                                      │ MAE     │ MAPE %  │ Lift vs RW│ Dir.acc% │ Cov Q80% │ Banda $   │")
    print("├─────────────────────────────────────────────┼─────────┼─────────┼───────────┼──────────┼──────────┼───────────┤")
    summary = []
    for name, preds, q10, q90 in rows:
        m = metrics(preds, actuals_te, rw_mae_te, pt_te)
        cov_s = "—"; bw_s = "—"
        if q10 is not None and q90 is not None:
            cov = coverage(q10, q90, actuals_te); bw = band_w(q10, q90)
            cov_s = f"{cov:5.1f}"; bw_s = f"{bw:6.1f}"
        lift_s = f"{m['lift_pct']:+8.1f}%" if "lift_pct" in m else "    —   "
        diracc = m.get("dir_acc")
        dir_s = f"{diracc:6.1f}" if diracc is not None and not np.isnan(diracc) else "  —   "
        print(f"│ {name:<43} │ {m['mae']:7.2f} │ {m['mape']:6.2f}  │ {lift_s} │ {dir_s}   │ {cov_s}    │ {bw_s}    │")
        summary.append({"name": name, **m,
                        "coverage": coverage(q10, q90, actuals_te) if q10 is not None else None,
                        "band_width": band_w(q10, q90) if q10 is not None else None})
    print("└─────────────────────────────────────────────┴─────────┴─────────┴───────────┴──────────┴──────────┴───────────┘")

    out = {
        "oos_start":   str(oos_start.date()),
        "val_end":     str(val_end.date()),
        "test_start":  str(test_start.date()),
        "oos_end":     str(oos_end.date()),
        "n_val":       int(val_mask.sum()),
        "n_test":      int(test_mask.sum()),
        "alpha_60d":   a60,
        "alpha_30d":   a30,
        "delta_60d":   delta60,
        "delta_30d":   delta30,
        "results":     summary,
    }
    with open(os.path.join(EVAL_DIR, "eval_summary_v2.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    # Guardar el modelo refit-30d con calibración
    if info30.get("last_model") is not None:
        joblib.dump({
            "model": info30["last_model"],
            "q10":   info30["last_q10"],
            "q90":   info30["last_q90"],
            "features": info30["last_feats"],
            "alpha":  a30,
            "delta":  delta30,
        }, os.path.join(EVAL_DIR, "model_ret30_wf30_calibrated.joblib"))

    print("\n💾 artifacts_eval/eval_summary_v2.json + model_ret30_wf30_calibrated.joblib")


if __name__ == "__main__":
    main()
