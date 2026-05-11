"""
scripts/forecast_redesign_eval.py
Evaluación de tres rediseños del forecast 30d, sin tocar producción.

#1  Modelo actual (target=log-precio) re-entrenado con rolling window 4 años
#2  Nuevo modelo (target=ret_30d_fwd directo) con ventana 4 años
#3  Walk-forward formal del #2 + quantile regression Q10/Q50/Q90

Todos comparados contra:
    - Random Walk (p_hat = p_t)
    - Seasonal naive (retorno medio del mes)
    - Modelo de producción tal-cual

Salida:
    - Tabla comparativa MAE / MAPE / Lift vs RW / cobertura Q10-Q90
    - artifacts_eval/{model_lvl_rolling, model_ret30, model_ret30_wf}.joblib
    - artifacts_eval/eval_summary.json
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")
PROD_MODEL   = os.path.join(ROOT, "artifacts", "model.joblib")
EVAL_DIR     = os.path.join(ROOT, "artifacts_eval")
os.makedirs(EVAL_DIR, exist_ok=True)

HORIZON      = 30
OOS_YEARS    = 1     # último año reservado para evaluación
TRAIN_YEARS  = 4     # ventana de entrenamiento rolling
MAX_FEATURES = 25

EXCLUDE = {
    "Date", "Soybeans", "Soybeans_log", "target_log",
    "ret_1d_fwd", "ret_7d_fwd", "ret_14d_fwd", "ret_30d_fwd",
    "Soybeans_High", "Soybeans_Low", "Soybeans_Open",
    "Maize_High", "Maize_Low", "Maize_Open",
    "SoybeanMeal_High", "SoybeanMeal_Low", "SoybeanMeal_Open",
    "SoybeanOil_High",  "SoybeanOil_Low",  "SoybeanOil_Open",
    "released_at", "as_of_date", "wasde_date", "week_ending",
}


# ─────────────────────────────────────────────────────────────────
def load():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def select_features(df, target_col, max_n=MAX_FEATURES):
    num = df.select_dtypes(include=[np.number]).copy()
    num[target_col] = df[target_col]
    corr = num.corr()[target_col].abs().sort_values(ascending=False)
    feats = [c for c in corr.index if c not in EXCLUDE and c != target_col][:max_n]
    return feats


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


# ─────────────────────────────────────────────────────────────────
# Construcción de pares (t, t+30) para evaluación 30d
# ─────────────────────────────────────────────────────────────────
def make_eval_pairs(df, oos_start, oos_end, step=5):
    """
    Devuelve lista de tuplas (idx_t, date_t, price_t, price_t30, month_t).
    Solo incluye dates donde t+30 calendario tiene match exacto o más cercano posterior.
    """
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


# ─────────────────────────────────────────────────────────────────
# Approach #1: log-price target, rolling 4y window
# ─────────────────────────────────────────────────────────────────
def simulate_30d_levels(start_idx, df, model, feats):
    """Simula 30 pasos con clip ±1% (réplica predict.py)."""
    base = float(df.iloc[start_idx]["Soybeans"])
    buf  = df["Soybeans"].iloc[max(0, start_idx-30):start_idx+1].tolist()
    cur  = df.iloc[start_idx].copy()
    for _ in range(HORIZON):
        cur["Soybeans_lag1"]  = buf[-1]
        if len(buf) >= 7:  cur["Soybeans_lag7"]  = buf[-7]
        if len(buf) >= 30: cur["Soybeans_lag30"] = buf[-30]
        X = pd.DataFrame([cur]).reindex(columns=feats, fill_value=0).fillna(0).replace([np.inf,-np.inf],0)
        pl = float(model.predict(X)[0])
        pl = max(min(pl, 10), -10)
        p  = float(np.expm1(pl))
        p  = 0.88*p + 0.12*base
        prev = buf[-1]
        p = float(np.clip(p, prev*0.99, prev*1.01))
        buf.append(p)
        cur["Soybeans"] = p
    return buf[-1]


def approach_1_levels_rolling(df, oos_start, oos_end):
    print("\n#1 ─ Modelo niveles (log) + rolling window 4 años")
    train_start = oos_start - pd.DateOffset(years=TRAIN_YEARS)
    train = df[(df["Date"] >= train_start) & (df["Date"] < oos_start)].copy()
    train["target_log"] = np.log1p(train["Soybeans"])
    feats = select_features(train, "target_log")
    X = train[feats].fillna(0).replace([np.inf,-np.inf],0)
    y = train["target_log"]
    t0 = time.time()
    model = fit_xgb(X, y)
    print(f"   train: {len(train)} filas, {len(feats)} feats, {time.time()-t0:.1f}s")
    joblib.dump({"model": model, "features": feats}, os.path.join(EVAL_DIR, "model_lvl_rolling.joblib"))

    # Evaluación 30d sobre OOS
    pairs = make_eval_pairs(df, oos_start, oos_end)
    preds = [simulate_30d_levels(p["idx"], df, model, feats) for p in pairs]
    actuals = [p["price_t30"] for p in pairs]
    return np.array(preds), np.array(actuals), pairs


# ─────────────────────────────────────────────────────────────────
# Approach #2: target = ret_30d_fwd, rolling 4y window, predicción directa
# ─────────────────────────────────────────────────────────────────
def approach_2_returns(df, oos_start, oos_end):
    print("\n#2 ─ Modelo retornos 30d directo + rolling 4 años")
    train_start = oos_start - pd.DateOffset(years=TRAIN_YEARS)
    train = df[(df["Date"] >= train_start) & (df["Date"] < oos_start)].copy()
    train = train.dropna(subset=["ret_30d_fwd"])
    feats = select_features(train, "ret_30d_fwd")
    X = train[feats].fillna(0).replace([np.inf,-np.inf],0)
    y = train["ret_30d_fwd"]
    t0 = time.time()
    model = fit_xgb(X, y)
    print(f"   train: {len(train)} filas, {len(feats)} feats, {time.time()-t0:.1f}s")
    joblib.dump({"model": model, "features": feats}, os.path.join(EVAL_DIR, "model_ret30.joblib"))

    pairs = make_eval_pairs(df, oos_start, oos_end)
    Xoos = df.iloc[[p["idx"] for p in pairs]][feats].fillna(0).replace([np.inf,-np.inf],0)
    ret_pred = model.predict(Xoos)
    preds   = np.array([p["price_t"] for p in pairs]) * (1 + ret_pred)
    actuals = np.array([p["price_t30"] for p in pairs])
    return preds, actuals, pairs, model, feats


# ─────────────────────────────────────────────────────────────────
# Approach #3: walk-forward del #2 + quantile regression
# ─────────────────────────────────────────────────────────────────
def approach_3_walkforward(df, oos_start, oos_end, refit_every_days=60):
    print(f"\n#3 ─ Walk-forward retornos 30d (refit cada {refit_every_days}d) + quantiles Q10/Q50/Q90")
    pairs = make_eval_pairs(df, oos_start, oos_end)
    if not pairs:
        return None

    preds, q10s, q50s, q90s, actuals = [], [], [], [], []
    last_refit_date = None
    cur_model = cur_q10 = cur_q90 = cur_feats = None
    refit_count = 0
    t0 = time.time()

    for p in pairs:
        # Decidir si refit
        need_refit = (last_refit_date is None) or \
                     ((p["date"] - last_refit_date).days >= refit_every_days)
        if need_refit:
            train_start = p["date"] - pd.DateOffset(years=TRAIN_YEARS)
            tr = df[(df["Date"] >= train_start) & (df["Date"] < p["date"])].copy()
            tr = tr.dropna(subset=["ret_30d_fwd"])
            cur_feats = select_features(tr, "ret_30d_fwd")
            Xtr = tr[cur_feats].fillna(0).replace([np.inf,-np.inf],0)
            ytr = tr["ret_30d_fwd"]
            cur_model = fit_xgb(Xtr, ytr)
            cur_q10   = fit_xgb_quantile(Xtr, ytr, alpha=0.10)
            cur_q90   = fit_xgb_quantile(Xtr, ytr, alpha=0.90)
            last_refit_date = p["date"]
            refit_count += 1

        Xrow = df.iloc[[p["idx"]]][cur_feats].fillna(0).replace([np.inf,-np.inf],0)
        rp = float(cur_model.predict(Xrow)[0])
        r10 = float(cur_q10.predict(Xrow)[0])
        r90 = float(cur_q90.predict(Xrow)[0])
        preds.append(p["price_t"] * (1 + rp))
        q10s.append(p["price_t"] * (1 + min(r10, rp)))
        q90s.append(p["price_t"] * (1 + max(r90, rp)))
        actuals.append(p["price_t30"])

    print(f"   {refit_count} refits, {len(pairs)} obs evaluadas, {time.time()-t0:.1f}s")
    # Persistir el último modelo
    joblib.dump({"model": cur_model, "q10": cur_q10, "q90": cur_q90, "features": cur_feats},
                os.path.join(EVAL_DIR, "model_ret30_wf.joblib"))
    return (np.array(preds), np.array(q10s), np.array(q90s),
            np.array(actuals), pairs)


# ─────────────────────────────────────────────────────────────────
# Production model bench (tal-cual está, sin retraining)
# ─────────────────────────────────────────────────────────────────
def bench_production(df, oos_start, oos_end):
    print("\n[bench] Modelo producción actual (sin retraining)")
    art = joblib.load(PROD_MODEL)
    model, feats = art["model"], art["features"]
    pairs = make_eval_pairs(df, oos_start, oos_end)
    preds = [simulate_30d_levels(p["idx"], df, model, feats) for p in pairs]
    actuals = [p["price_t30"] for p in pairs]
    return np.array(preds), np.array(actuals), pairs


def baselines(pairs, train_df):
    actuals = np.array([p["price_t30"] for p in pairs])
    rw = np.array([p["price_t"] for p in pairs])
    train = train_df.copy()
    train["ret30"] = train["Soybeans"].pct_change(30).shift(-30)
    seas_map = train.groupby(train["Date"].dt.month)["ret30"].mean().to_dict()
    seas = np.array([p["price_t"] * (1 + (seas_map.get(p["month"], 0.0) or 0.0)) for p in pairs])
    return rw, seas, actuals


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def metrics(name, preds, actuals, rw_mae):
    mae  = float(np.mean(np.abs(preds - actuals)))
    mape = float(np.mean(np.abs((preds - actuals) / actuals))) * 100
    lift = (rw_mae - mae) / rw_mae * 100
    return {"name": name, "mae": mae, "mape": mape, "lift_pct": lift}


def main():
    df = load()
    oos_end   = df["Date"].max()
    oos_start = oos_end - pd.DateOffset(years=OOS_YEARS)
    train_df  = df[df["Date"] < oos_start]
    print(f"OOS: {oos_start.date()} → {oos_end.date()}  ({len(df[df['Date']>=oos_start])} filas)")

    # Baselines
    pairs0 = make_eval_pairs(df, oos_start, oos_end)
    rw, seas, actuals = baselines(pairs0, train_df)
    rw_mae = float(np.mean(np.abs(rw - actuals)))

    rows = []
    rows.append(metrics("Random Walk",    rw,   actuals, rw_mae))
    rows.append(metrics("Seasonal naive", seas, actuals, rw_mae))

    # Bench producción
    p_prod, _, _ = bench_production(df, oos_start, oos_end)
    rows.append(metrics("Producción (actual)", p_prod, actuals, rw_mae))

    # #1
    p1, _, _ = approach_1_levels_rolling(df, oos_start, oos_end)
    rows.append(metrics("#1 niveles + rolling 4a", p1, actuals, rw_mae))

    # #2
    p2, _, _, _, _ = approach_2_returns(df, oos_start, oos_end)
    rows.append(metrics("#2 retornos 30d + rolling 4a", p2, actuals, rw_mae))

    # #3
    p3, q10, q90, a3, pairs3 = approach_3_walkforward(df, oos_start, oos_end)
    rows.append(metrics("#3 retornos 30d + walk-forward", p3, a3, rw_mae))

    # Cobertura Q10-Q90
    coverage = float(np.mean((a3 >= q10) & (a3 <= q90))) * 100
    band_w   = float(np.mean(q90 - q10))
    print(f"\n#3 cobertura empírica Q10-Q90: {coverage:.1f}% (objetivo 80%)  •  ancho medio: ${band_w:.1f}")

    # Direction accuracy (sube/baja vs precio_t)
    print("\n┌─────────────────────────────────────┬─────────┬─────────┬──────────────┬─────────────┐")
    print("│ Modelo                              │ MAE USD │ MAPE %  │ Lift vs RW   │ Dir.acc %   │")
    print("├─────────────────────────────────────┼─────────┼─────────┼──────────────┼─────────────┤")
    for r, preds in zip(rows, [rw, seas, p_prod, p1, p2, p3]):
        # direction_accuracy: signo(pred - p_t) vs signo(actual - p_t)
        ps = np.array([pp["price_t"] for pp in pairs0]) if r["name"] != "#3 retornos 30d + walk-forward" \
             else np.array([pp["price_t"] for pp in pairs3])
        ac = actuals if r["name"] != "#3 retornos 30d + walk-forward" else a3
        if len(preds) == len(ps) == len(ac):
            dir_pred  = np.sign(preds - ps)
            dir_real  = np.sign(ac - ps)
            valid = dir_real != 0
            diracc = float(np.mean(dir_pred[valid] == dir_real[valid])) * 100 if valid.any() else float("nan")
        else:
            diracc = float("nan")
        print(f"│ {r['name']:<35} │ {r['mae']:7.2f} │ {r['mape']:6.2f}  │ {r['lift_pct']:+10.1f} % │ {diracc:9.1f}   │")
    print("└─────────────────────────────────────┴─────────┴─────────┴──────────────┴─────────────┘")

    summary = {
        "oos_start": str(oos_start.date()),
        "oos_end":   str(oos_end.date()),
        "n_obs":     int(len(actuals)),
        "horizon":   HORIZON,
        "models":    rows,
        "q_coverage_pct": coverage,
        "q_band_width":   band_w,
    }
    with open(os.path.join(EVAL_DIR, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n💾 artifacts_eval/eval_summary.json  +  3 modelos guardados.")


if __name__ == "__main__":
    main()
