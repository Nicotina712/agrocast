"""
scripts/test_decision_classifier_v3.py
Test final del decision classifier con 3 mejoras:

A) Asymmetric loss: penalizar falsos positivos (recomendar WAIT cuando era
   SELL) más que falsos negativos. Esto sesga el modelo hacia conservador
   y reduce el costo del error promedio.

B) Stacking con always-sell baseline: el modelo solo OVERRIDE el always-sell
   cuando p_WAIT > τ_alto AND condiciones de confianza altas. Default = SELL.

C) Risk-aversion via expected utility: en lugar de threshold simétrico,
   usar criterio bayesiano: WAIT solo si E[wait | p] > E[sell] + risk_premium.

Combinamos las 3 en un mismo backtest sobre el mejor profile (low_cost).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")

HORIZON_DAYS    = 30
TRAIN_YEARS     = 5
TEST_FREQ_DAYS  = 7
REFIT_EVERY     = 90
COST_PCT        = 0.010   # low_cost (1.0 %/mes) — perfil más favorable


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


def train_classifier(tr, feat_cols, asymmetric_fp_weight: float = 1.0):
    """Entrena clasificador. asymmetric_fp_weight > 1 → penaliza más los
    falsos positivos (decir WAIT cuando era SELL).

    Usa sample_weight: muestras de clase negativa (label=0) reciben peso
    asymmetric_fp_weight. Si modelo predice 1 sobre estas, costo es mayor.
    """
    X_tr = tr[feat_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_tr = tr["label_wait_paid"]
    split_idx = int(len(tr) * 0.8)
    X_train, X_val = X_tr.iloc[:split_idx], X_tr.iloc[split_idx:]
    y_train, y_val = y_tr.iloc[:split_idx], y_tr.iloc[split_idx:]

    # Sample weights para asymmetric loss:
    # FP = decir WAIT (pred=1) cuando era SELL (label=0) → peso a label=0
    sample_w = np.where(y_train == 0, asymmetric_fp_weight, 1.0)

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw = max(n_neg / max(n_pos, 1), 1.0)
    model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=42,
                            eval_metric="logloss", scale_pos_weight=spw / asymmetric_fp_weight)
    model.fit(X_train, y_train, sample_weight=sample_w, verbose=False)
    try:
        p_val = model.predict_proba(X_val)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(p_val, y_val.values.astype(float))
    except Exception:
        calibrator = None
    return model, calibrator


def walk_forward_predict(df, feat_cols, asymmetric_fp_weight: float = 1.0):
    """Walk-forward predict con asymmetric loss configurable."""
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
            model, calibrator = train_classifier(tr, feat_cols, asymmetric_fp_weight)
            last_refit = D
        x_t = df.iloc[[idx_t]][feat_cols].fillna(0).replace([np.inf, -np.inf], 0)
        p_raw = float(model.predict_proba(x_t)[0, 1])
        p_cal = float(calibrator.transform([p_raw])[0]) if calibrator else p_raw
        rows.append({"date": D, "actual_label": int(target),
                      "actual_ret": float(actual_ret), "p_calibrated": p_cal})
    return pd.DataFrame(rows)


def evaluate_decision(R, decision_col, cost_pct):
    """Evalúa un set de decisiones contra always-sell."""
    R = R.copy()
    R["pnl_sell"] = 0.0
    R["pnl_wait"] = R["actual_ret"] - cost_pct
    R["pnl"] = np.where(R[decision_col] == 1, R["pnl_wait"], R["pnl_sell"])
    return {
        "n":               len(R),
        "n_wait":          int(R[decision_col].sum()),
        "acc":             float((R[decision_col] == R["actual_label"]).mean()),
        "pnl_avg":         float(R["pnl"].mean()),
        "lift_vs_sell_pp": float(R["pnl"].mean() * 100),  # vs sell=0
        "pnl_oracle":      float(np.where(R["actual_label"] == 1, R["pnl_wait"], R["pnl_sell"]).mean()),
    }


def main():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    df["ret_30d_fwd"] = df["Soybeans"].pct_change(30).shift(-30)
    df["label_wait_paid"] = (df["ret_30d_fwd"] > COST_PCT).astype(int)
    df, feat_cols = build_features(df)

    print(f"Profile: low_cost (costo {COST_PCT*100:.1f}%/mes), 17 features\n")

    # ── Baseline original (no asymmetric, threshold 0.5) ───────
    print(f"══════════════════════════════════════════════════════════════")
    print(f"  BASELINE — sin asymmetric loss, threshold 0.5")
    print(f"══════════════════════════════════════════════════════════════")
    R_base = walk_forward_predict(df, feat_cols, asymmetric_fp_weight=1.0)
    R_base["pred_05"] = (R_base["p_calibrated"] >= 0.50).astype(int)
    base_metrics = evaluate_decision(R_base, "pred_05", COST_PCT)
    print(f"  N={base_metrics['n']}, n_WAIT={base_metrics['n_wait']}, acc={base_metrics['acc']*100:.1f}%, "
          f"PnL={base_metrics['pnl_avg']*100:+.3f}%  (oracle={base_metrics['pnl_oracle']*100:+.3f}%)")

    # ── A) Asymmetric loss: distintos pesos a falsos positivos ─
    print(f"\n══════════════════════════════════════════════════════════════")
    print(f"  IDEA A — Asymmetric loss (peso falsos positivos)")
    print(f"══════════════════════════════════════════════════════════════")
    print(f"  {'fp_weight':>10} {'n_WAIT':>7} {'acc%':>6} {'PnL%':>8} {'lift_vs_baseline':>18}")
    a_results = []
    for w in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        R = walk_forward_predict(df, feat_cols, asymmetric_fp_weight=w)
        R["pred_05"] = (R["p_calibrated"] >= 0.50).astype(int)
        m = evaluate_decision(R, "pred_05", COST_PCT)
        delta = (m["pnl_avg"] - base_metrics["pnl_avg"]) * 100
        a_results.append({"fp_weight": w, **m, "delta_baseline": delta})
        star = "✓" if m["pnl_avg"] > 0 else " "
        print(f"  {w:>10.2f} {m['n_wait']:>7} {m['acc']*100:>5.1f}% "
              f"{m['pnl_avg']*100:>+6.3f}% {delta:>+15.3f}pp{star}")

    # ── B) Stacking con always-sell baseline + threshold alto ──
    print(f"\n══════════════════════════════════════════════════════════════")
    print(f"  IDEA B — Stacking: solo override SELL si confianza muy alta")
    print(f"══════════════════════════════════════════════════════════════")
    print(f"  {'thr_override':>13} {'n_WAIT':>7} {'acc%':>6} {'PnL%':>8} {'lift_vs_baseline':>18}")
    b_results = []
    R_b = R_base   # usamos las predicciones de baseline (no asymmetric)
    for thr in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        R_b["pred_stack"] = (R_b["p_calibrated"] >= thr).astype(int)
        m = evaluate_decision(R_b, "pred_stack", COST_PCT)
        delta = (m["pnl_avg"] - base_metrics["pnl_avg"]) * 100
        b_results.append({"thr": thr, **m, "delta_baseline": delta})
        star = "✓" if m["pnl_avg"] > 0 else " "
        print(f"  {thr:>13.2f} {m['n_wait']:>7} {m['acc']*100:>5.1f}% "
              f"{m['pnl_avg']*100:>+6.3f}% {delta:>+15.3f}pp{star}")

    # ── C) Expected utility con risk premium ─────────────────
    print(f"\n══════════════════════════════════════════════════════════════")
    print(f"  IDEA C — Expected utility: WAIT si p·E[ret_wait] > E[sell] + risk_premium")
    print(f"══════════════════════════════════════════════════════════════")
    # E[ret_wait] aproximado de la distribución histórica de ret_30d_fwd
    avg_wait_when_wait_pays = df.loc[df["label_wait_paid"] == 1, "ret_30d_fwd"].mean() - COST_PCT
    avg_wait_when_sell_pays = df.loc[df["label_wait_paid"] == 0, "ret_30d_fwd"].mean() - COST_PCT
    print(f"  E[ret_wait | wait paga]: {avg_wait_when_wait_pays*100:+.3f}%")
    print(f"  E[ret_wait | sell paga]: {avg_wait_when_sell_pays*100:+.3f}%")
    print(f"  Threshold racional sin risk premium:")
    # E[wait] = p * avg_when_paid + (1-p) * avg_when_not_paid
    # WAIT si E[wait] > 0
    # p * avg_paid + (1-p) * avg_not_paid > 0
    # p * (avg_paid - avg_not_paid) > -avg_not_paid
    # p > -avg_not_paid / (avg_paid - avg_not_paid)
    rational_p = -avg_wait_when_sell_pays / (avg_wait_when_wait_pays - avg_wait_when_sell_pays)
    print(f"  → p > {rational_p:.3f}")

    print(f"\n  {'risk_prem':>10} {'p_thr':>7} {'n_WAIT':>7} {'acc%':>6} {'PnL%':>8} {'lift_vs_baseline':>18}")
    c_results = []
    for rp in [0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030]:
        # WAIT si p * avg_paid + (1-p) * avg_not_paid > rp
        p_thr = (rp - avg_wait_when_sell_pays) / (avg_wait_when_wait_pays - avg_wait_when_sell_pays)
        p_thr = max(0.0, min(0.99, p_thr))
        R_c = R_base.copy()
        R_c["pred_eu"] = (R_c["p_calibrated"] >= p_thr).astype(int)
        m = evaluate_decision(R_c, "pred_eu", COST_PCT)
        delta = (m["pnl_avg"] - base_metrics["pnl_avg"]) * 100
        c_results.append({"risk_premium": rp, "p_thr": p_thr, **m, "delta_baseline": delta})
        star = "✓" if m["pnl_avg"] > 0 else " "
        print(f"  {rp*100:>+8.2f}% {p_thr:>7.3f} {m['n_wait']:>7} {m['acc']*100:>5.1f}% "
              f"{m['pnl_avg']*100:>+6.3f}% {delta:>+15.3f}pp{star}")

    # ── Ganador absoluto ──────────────────────────────────────
    print(f"\n══════════════════════════════════════════════════════════════")
    print(f"  GANADOR — mejor configuración")
    print(f"══════════════════════════════════════════════════════════════")
    all_configs = (
        [("A_asym_w" + str(r["fp_weight"]), r) for r in a_results] +
        [("B_stack_thr" + str(r["thr"]), r) for r in b_results] +
        [("C_eu_rp" + str(r["risk_premium"]), r) for r in c_results]
    )
    all_configs.append(("BASELINE_thr05", base_metrics))
    best = max(all_configs, key=lambda x: x[1]["pnl_avg"])
    print(f"  ⭐ {best[0]}")
    print(f"  PnL: {best[1]['pnl_avg']*100:+.3f}%  (vs always-sell: {best[1]['pnl_avg']*100:+.3f}pp)")
    print(f"  acc: {best[1]['acc']*100:.1f}%  n_WAIT: {best[1]['n_wait']}/{best[1]['n']}")
    if best[1]["pnl_avg"] > 0:
        print(f"  ✓✓ GANA al always-sell — primera config rentable encontrada")
    else:
        print(f"  × Sigue perdiendo, pero gap reducido ({(best[1]['pnl_avg'] - base_metrics['pnl_avg'])*100:+.3f}pp vs baseline)")


if __name__ == "__main__":
    main()
