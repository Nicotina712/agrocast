"""
Evaluación de la configuración ganadora (TY=5, WF refit 60d, α óptimo, conformal δ)
en horizontes 7d, 14d, 30d. Si los 3 superan al RW del horizonte respectivo,
podemos construir una curva diaria por interpolación lineal entre p_t, p_7d, p_14d, p_30d.

Targets disponibles en features.csv: ret_7d_fwd, ret_14d_fwd, ret_30d_fwd.
"""

import os, sys, time
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")
EVAL_DIR     = os.path.join(ROOT, "artifacts_eval")
os.makedirs(EVAL_DIR, exist_ok=True)

OOS_YEARS    = 1
TRAIN_YEARS  = 5
MAX_FEATURES = 25
SAMPLE_STEP  = 5
REFIT_DAYS   = 60
HORIZONS     = [7, 14, 30]

EXCLUDE = {
    "Date", "Soybeans", "Soybeans_log", "target_log",
    "ret_1d_fwd", "ret_7d_fwd", "ret_14d_fwd", "ret_30d_fwd",
    "Soybeans_High", "Soybeans_Low", "Soybeans_Open",
    "Maize_High", "Maize_Low", "Maize_Open",
    "SoybeanMeal_High", "SoybeanMeal_Low", "SoybeanMeal_Open",
    "SoybeanOil_High", "SoybeanOil_Low", "SoybeanOil_Open",
    "released_at", "as_of_date", "wasde_date", "week_ending",
}


def select_features(df, target_col):
    num = df.select_dtypes(include=[np.number]).copy()
    num[target_col] = df[target_col]
    corr = num.corr()[target_col].abs().sort_values(ascending=False)
    return [c for c in corr.index if c not in EXCLUDE and c != target_col][:MAX_FEATURES]


def fit_xgb(X, y, **kw):
    m = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.8, random_state=42,
                     eval_metric="mae", **kw)
    m.fit(X, y, verbose=False)
    return m


def fit_xgb_q(X, y, alpha):
    m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, random_state=42,
                     objective="reg:quantileerror", quantile_alpha=alpha)
    m.fit(X, y, verbose=False)
    return m


def make_pairs(df, oos_start, oos_end, horizon):
    pairs = []
    eligible = df[(df["Date"] >= oos_start) & (df["Date"] <= oos_end)].iloc[::SAMPLE_STEP]
    for _, row in eligible.iterrows():
        tH = row["Date"] + pd.Timedelta(days=horizon)
        future = df[df["Date"] >= tH]
        if future.empty:
            continue
        pairs.append({
            "idx": int(df.index[df["Date"] == row["Date"]][0]),
            "date": row["Date"], "price_t": float(row["Soybeans"]),
            "price_tH": float(future.iloc[0]["Soybeans"]),
        })
    return pairs


def walk_forward(df, pairs, target_col):
    preds, q10s, q90s = [], [], []
    last_refit = None
    cur_m = cur_q10 = cur_q90 = cur_f = None
    for p in pairs:
        if (last_refit is None) or ((p["date"] - last_refit).days >= REFIT_DAYS):
            tr = df[(df["Date"] >= p["date"] - pd.DateOffset(years=TRAIN_YEARS)) &
                    (df["Date"] < p["date"])].dropna(subset=[target_col])
            cur_f = select_features(tr, target_col)
            Xtr = tr[cur_f].fillna(0).replace([np.inf, -np.inf], 0)
            ytr = tr[target_col]
            cur_m   = fit_xgb(Xtr, ytr)
            cur_q10 = fit_xgb_q(Xtr, ytr, 0.10)
            cur_q90 = fit_xgb_q(Xtr, ytr, 0.90)
            last_refit = p["date"]
        Xrow = df.iloc[[p["idx"]]][cur_f].fillna(0).replace([np.inf, -np.inf], 0)
        rp  = float(cur_m.predict(Xrow)[0])
        r10 = float(cur_q10.predict(Xrow)[0])
        r90 = float(cur_q90.predict(Xrow)[0])
        preds.append(p["price_t"] * (1 + rp))
        q10s.append(p["price_t"]  * (1 + min(r10, rp)))
        q90s.append(p["price_t"]  * (1 + max(r90, rp)))
    return (np.array(preds), np.array(q10s), np.array(q90s),
            {"last_model": cur_m, "last_q10": cur_q10, "last_q90": cur_q90, "last_feats": cur_f})


def conformal_delta(y, q10, q90, target=0.80):
    score = np.maximum(np.maximum(y - q90, q10 - y), 0)
    return float(np.quantile(score, target))


def evaluate(df, horizon, target_col):
    oos_end    = df["Date"].max()
    oos_start  = oos_end - pd.DateOffset(years=OOS_YEARS)
    val_end    = oos_start + (oos_end - oos_start) / 2
    test_start = val_end + pd.Timedelta(days=1)

    pairs = make_pairs(df, oos_start, oos_end, horizon)
    val_mask  = np.array([p["date"] <= val_end    for p in pairs])
    test_mask = np.array([p["date"] >= test_start for p in pairs])
    actuals = np.array([p["price_tH"] for p in pairs])
    pt      = np.array([p["price_t"]  for p in pairs])

    preds, q10, q90, meta = walk_forward(df, pairs, target_col)

    grid = np.linspace(0, 1, 21)
    best_a, best_mae = 0.0, float("inf")
    for a in grid:
        blend = a * preds[val_mask] + (1 - a) * pt[val_mask]
        mae   = float(np.mean(np.abs(blend - actuals[val_mask])))
        if mae < best_mae:
            best_a, best_mae = float(a), mae

    delta = conformal_delta(actuals[val_mask], q10[val_mask], q90[val_mask], 0.80)

    rw_te = pt[test_mask]; a_te = actuals[test_mask]
    rw_mae   = float(np.mean(np.abs(rw_te - a_te)))
    base_mae = float(np.mean(np.abs(preds[test_mask] - a_te)))
    ens_te   = best_a * preds[test_mask] + (1 - best_a) * rw_te
    ens_mae  = float(np.mean(np.abs(ens_te - a_te)))

    q10_cal_te = q10[test_mask] - delta
    q90_cal_te = q90[test_mask] + delta
    cov   = float(np.mean((a_te >= q10_cal_te) & (a_te <= q90_cal_te))) * 100
    bw    = float(np.mean(q90_cal_te - q10_cal_te))

    dp = np.sign(ens_te - rw_te); dr = np.sign(a_te - rw_te)
    valid = dr != 0
    diracc = float(np.mean(dp[valid] == dr[valid])) * 100

    # Persistir el último modelo de cada horizonte
    joblib.dump({
        "horizon": horizon, "target": target_col,
        "model": meta["last_model"], "q10": meta["last_q10"], "q90": meta["last_q90"],
        "features": meta["last_feats"], "alpha": best_a, "delta": delta,
        "train_years": TRAIN_YEARS, "refit_days": REFIT_DAYS,
    }, os.path.join(EVAL_DIR, f"model_ret{horizon}d_wf60.joblib"))

    return {
        "horizon": horizon, "n_test": int(test_mask.sum()),
        "alpha": best_a, "delta": delta,
        "rw_mae": rw_mae, "base_mae": base_mae, "ens_mae": ens_mae,
        "lift_base": (rw_mae - base_mae) / rw_mae * 100,
        "lift_ens":  (rw_mae - ens_mae)  / rw_mae * 100,
        "diracc": diracc, "cov_q80": cov, "band_w": bw,
    }


def main():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    print(f"Multi-horizon eval — TY={TRAIN_YEARS}, refit={REFIT_DAYS}d\n")

    print("┌──────────┬───────┬──────┬───────┬─────────┬─────────┬─────────┬──────────┬──────────┬──────────┬──────────┐")
    print("│ Horizonte│ n_test│ α*   │ δ ($) │ MAE_RW  │ MAE_base│ MAE_ens │ Lift base│ Lift ens │ Dir acc% │ Cov Q80% │")
    print("├──────────┼───────┼──────┼───────┼─────────┼─────────┼─────────┼──────────┼──────────┼──────────┼──────────┤")
    for h in HORIZONS:
        target = f"ret_{h}d_fwd"
        r = evaluate(df, h, target)
        print(f"│ {h:>5}d    │ {r['n_test']:>5} │ {r['alpha']:.2f} │ {r['delta']:5.1f} │ "
              f"{r['rw_mae']:7.2f} │ {r['base_mae']:7.2f} │ {r['ens_mae']:7.2f} │ "
              f"{r['lift_base']:+7.1f}% │ {r['lift_ens']:+7.1f}% │ {r['diracc']:7.1f}  │ {r['cov_q80']:7.1f}  │")
    print("└──────────┴───────┴──────┴───────┴─────────┴─────────┴─────────┴──────────┴──────────┴──────────┴──────────┘")
    print(f"\n💾 Modelos guardados: artifacts_eval/model_ret{{7,14,30}}d_wf60.joblib")


if __name__ == "__main__":
    main()
