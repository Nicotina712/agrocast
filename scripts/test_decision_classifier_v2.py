"""
scripts/test_decision_classifier_v2.py
Análisis profundo del decision classifier con:
  1. Threshold variable (0.3 → 0.7)
  2. Costo del productor variable (low/med/high)
  3. Tipo de productor: high storage cost vs low storage cost
  4. P&L acumulado por configuración

Objetivo: encontrar la zona donde el clasificador SÍ aporta P&L vs always-sell.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")

HORIZON_DAYS    = 30
TRAIN_YEARS     = 5
TEST_FREQ_DAYS  = 7
REFIT_EVERY     = 90

# Tipos de productor (costo mensual como % del precio)
PRODUCER_PROFILES = {
    "low_cost":  0.010,   # 1.0%/mes (silo propio + cash) → más fácil que WAIT pague
    "medium":    0.0217,  # 2.17%/mes (default original)
    "high_cost": 0.030,   # 3.0%/mes (storage rentado + crédito caro)
}

# Thresholds a probar (probabilidad mínima para recomendar WAIT)
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def build_features(df: pd.DataFrame):
    d = df.copy()
    p_min_12m = d["Soybeans"].rolling(252, min_periods=60).min().shift(1)
    p_max_12m = d["Soybeans"].rolling(252, min_periods=60).max().shift(1)
    d["price_pct_in_12m_range"] = (d["Soybeans"] - p_min_12m) / (p_max_12m - p_min_12m + 1e-9)
    d["month"] = d["Date"].dt.month
    d["seasonal_ret_30d_med"] = d.groupby("month")["Soybeans"].transform(
        lambda s: s.pct_change(30).shift(-30).expanding(min_periods=12).median().shift(1)
    ).fillna(0)
    keep = ["news_sentiment", "news_velocity_7d", "mom_5d", "mom_20d", "rsi_14",
            "vol_30d", "vol_60d", "Oil_chg7", "Dollar_chg7",
            "soy_corn_ratio", "enso_oni", "month_sin", "month_cos",
            "cot_noncomm_long_pct", "wasde_bull_bias",
            "price_pct_in_12m_range", "seasonal_ret_30d_med"]
    avail = [c for c in keep if c in d.columns]
    return d, avail


def train_and_predict(df, feat_cols, cost_pct):
    """Entrena clasificador con target específico al cost_pct.
    Retorna DataFrame con prob_wait, actual_ret, label por fecha test."""
    df = df.copy()
    df["label_wait_paid"] = (df["ret_30d_fwd"] > cost_pct).astype(int)

    end = df["Date"].max()
    test_start = end - pd.DateOffset(years=5)
    test_idx = []
    d = test_start
    while d <= end - pd.Timedelta(days=HORIZON_DAYS):
        future = df[df["Date"] >= d]
        if future.empty: break
        idx = int(df.index[df["Date"] == future.iloc[0]["Date"]][0])
        test_idx.append(idx)
        d = future.iloc[0]["Date"] + pd.Timedelta(days=TEST_FREQ_DAYS)

    last_refit = None
    model = calibrator = None
    rows = []

    for idx_t in test_idx:
        D = df.iloc[idx_t]["Date"]
        target = df.iloc[idx_t].get("label_wait_paid")
        actual_ret = df.iloc[idx_t].get("ret_30d_fwd")
        if pd.isna(target) or pd.isna(actual_ret):
            continue

        if (last_refit is None) or ((D - last_refit).days >= REFIT_EVERY):
            train_start = D - pd.DateOffset(years=TRAIN_YEARS)
            tr = df[(df["Date"] >= train_start) & (df["Date"] < D)].dropna(subset=["label_wait_paid"]).copy()
            if len(tr) < 200:
                continue
            X_tr = tr[feat_cols].fillna(0).replace([np.inf, -np.inf], 0)
            y_tr = tr["label_wait_paid"]
            split_idx = int(len(tr) * 0.8)
            X_train, X_val = X_tr.iloc[:split_idx], X_tr.iloc[split_idx:]
            y_train, y_val = y_tr.iloc[:split_idx], y_tr.iloc[split_idx:]
            # Class weight para balancear (WAIT minoritario)
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            spw = max(n_neg / max(n_pos, 1), 1.0)
            model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                                    eval_metric="logloss", scale_pos_weight=spw)
            model.fit(X_train, y_train, verbose=False)
            try:
                p_val = model.predict_proba(X_val)[:, 1]
                calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                calibrator.fit(p_val, y_val.values.astype(float))
            except Exception:
                calibrator = None
            last_refit = D

        x_t = df.iloc[[idx_t]][feat_cols].fillna(0).replace([np.inf, -np.inf], 0)
        p_raw = float(model.predict_proba(x_t)[0, 1])
        p_cal = float(calibrator.transform([p_raw])[0]) if calibrator else p_raw
        rows.append({
            "date": D, "actual_label": int(target),
            "actual_ret": float(actual_ret),
            "p_calibrated": p_cal,
        })
    return pd.DataFrame(rows)


def evaluate_threshold(R: pd.DataFrame, threshold: float, cost_pct: float) -> dict:
    """Evalúa accuracy + P&L para un threshold dado."""
    R = R.copy()
    R["pred"] = (R["p_calibrated"] >= threshold).astype(int)
    R["pnl_sell"] = 0.0
    R["pnl_wait"] = R["actual_ret"] - cost_pct
    R["pnl_classifier"] = np.where(R["pred"] == 1, R["pnl_wait"], R["pnl_sell"])
    R["pnl_oracle"]     = np.where(R["actual_label"] == 1, R["pnl_wait"], R["pnl_sell"])

    acc = (R["pred"] == R["actual_label"]).mean()
    pnl_avg = R["pnl_classifier"].mean()
    pnl_sell = 0.0  # base
    pnl_oracle = R["pnl_oracle"].mean()
    n_wait = int(R["pred"].sum())

    # Precision / recall sobre la clase WAIT
    tp = int(((R["pred"] == 1) & (R["actual_label"] == 1)).sum())
    fp = int(((R["pred"] == 1) & (R["actual_label"] == 0)).sum())
    fn = int(((R["pred"] == 0) & (R["actual_label"] == 1)).sum())
    tn = int(((R["pred"] == 0) & (R["actual_label"] == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)

    return {"threshold": threshold, "n_wait_recommended": n_wait,
             "acc": acc, "precision_wait": prec, "recall_wait": rec,
             "pnl_avg_pct": pnl_avg, "pnl_oracle_pct": pnl_oracle,
             "lift_vs_sell_pct": (pnl_avg - pnl_sell) * 100,
             "n_total": len(R)}


def main():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    df["ret_30d_fwd"] = df["Soybeans"].pct_change(30).shift(-30)
    df, feat_cols = build_features(df)

    print(f"Features: {len(feat_cols)}")

    for prof_name, cost_pct in PRODUCER_PROFILES.items():
        print(f"\n══════════════════════════════════════════════════════════════")
        print(f"  PROFILE: {prof_name}  (costo={cost_pct*100:.2f}%/mes)")
        print(f"══════════════════════════════════════════════════════════════")
        # Distribución del label
        df["lbl"] = (df["ret_30d_fwd"] > cost_pct).astype(int)
        print(f"  Base rate WAIT en histórico: {df['lbl'].mean()*100:.1f}%")

        R = train_and_predict(df, feat_cols, cost_pct)
        if R.empty:
            print("  Sin datos.")
            continue
        print(f"  N evaluados: {len(R)}")

        # Tabla por threshold
        print(f"\n  {'Thr':>5} {'#WAIT':>6} {'Acc%':>6} {'PrecWAIT':>9} {'RecWAIT':>9} "
              f"{'PnL_avg%':>9} {'Lift_pp%':>9} {'Oracle%':>9}")
        rows = []
        for thr in THRESHOLDS:
            m = evaluate_threshold(R, thr, cost_pct)
            rows.append(m)
            star = "✓" if m["lift_vs_sell_pct"] > 0 else " "
            print(f"  {thr:>5.2f} {m['n_wait_recommended']:>6}  "
                  f"{m['acc']*100:>5.1f}%  {m['precision_wait']*100:>7.1f}%  "
                  f"{m['recall_wait']*100:>7.1f}%   "
                  f"{m['pnl_avg_pct']*100:>+6.3f}% {m['lift_vs_sell_pct']:>+6.3f}pp{star}  "
                  f"{m['pnl_oracle_pct']*100:>+6.3f}%")

        # Mejor threshold por P&L
        best = max(rows, key=lambda x: x["lift_vs_sell_pct"])
        print(f"\n  ⭐ Mejor threshold: {best['threshold']}  "
              f"PnL={best['pnl_avg_pct']*100:+.3f}%  "
              f"({'GANA' if best['lift_vs_sell_pct'] > 0 else 'pierde'} vs always-sell)")


if __name__ == "__main__":
    main()
