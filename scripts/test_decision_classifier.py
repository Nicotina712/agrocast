"""
scripts/test_decision_classifier.py
Test de viabilidad: clasificador binario de DECISIÓN cost-aware.

Re-formulación radical: en lugar de predecir el ret_30d_fwd y dejar al
productor traducirlo en decisión, entrenamos directo sobre la decisión
correcta histórica.

TARGET (binario):
  Para cada día histórico D, calcular:
    util_sell = price_D
    util_wait = price_D+30 - storage_cost - financing_cost
    label = 1 si util_wait > util_sell, 0 si no

Es decir, label = "el oráculo retrospectivo dice WAIT".

FEATURES (compuestas, decision-relevant — alineadas con la lógica del usuario):
  - precio_vs_percentil_12m: ¿estamos en percentil bajo del año? (favorece WAIT)
  - estacionalidad: ¿este mes históricamente sube en próximos 30d?
  - sentimiento_news: composite
  - momentum/RSI: continuación o reversión
  - macro forward: oil, USD, ENSO
  - costos efectivos del productor (storage, financing por bushel/ton)

Por qué esto puede funcionar (y los regresores fallan):
  - El target ya integra los costos en la decisión correcta
  - No requiere predicción exacta del precio, solo identificar régimenes
    donde el "wait" tiene retorno esperado > costos
  - Usamos calibración isotonic para que la probabilidad sea confiable

Métricas:
  - Accuracy del clasificador
  - Brier score (calibración)
  - Utilidad económica: ¿seguir las predicciones gana al always-sell?
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")

HORIZON_DAYS    = 30
STORAGE_USD_TON = 6.0
FINANCING       = 0.08
TRAIN_YEARS     = 5
TEST_FREQ_DAYS  = 7

# Costo per bushel (target estaba en cents/bu, BU_PER_TON=36.74)
# Costo por bu = (storage + financing*price) * months / BU_PER_TON
# Pero como target es ret_30d_fwd que ya es relativo, traducimos costos a %
# costo_pct ≈ (storage_per_ton/30d + financing/12) / price_per_ton
# storage 6 USD/ton/mes, financing 8% anual = 0.667% mensual
# Para precio típico $400/ton: storage = 6/400 = 1.5%, financing = 0.667%
# Total costo mensual ~ 2.17% del precio
COST_PCT_MONTH  = 0.0217   # ~2.17% del precio en 30d


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye features decision-aware alineadas con la lógica del usuario."""
    d = df.copy()
    # 1. Precio relativo a percentil 12 meses
    p_min_12m = d["Soybeans"].rolling(252, min_periods=60).min().shift(1)
    p_max_12m = d["Soybeans"].rolling(252, min_periods=60).max().shift(1)
    d["price_pct_in_12m_range"] = (d["Soybeans"] - p_min_12m) / (p_max_12m - p_min_12m + 1e-9)

    # 2. Estacionalidad: retorno histórico mediano de este mes a 30d
    d["month"] = d["Date"].dt.month
    monthly_ret = d.groupby("month")["Soybeans"].apply(
        lambda s: s.pct_change(30).shift(-30).expanding(min_periods=12).median()
    ).reset_index(level=0, drop=True)
    # Aproximación más simple: rolling histórico por mes
    d["seasonal_ret_30d_med"] = d.groupby("month")["Soybeans"].transform(
        lambda s: s.pct_change(30).shift(-30).expanding(min_periods=12).median().shift(1)
    ).fillna(0)

    # 3. Indicadores existentes que el usuario mencionó
    keep = ["news_sentiment", "news_velocity_7d", "mom_5d", "mom_20d", "rsi_14",
            "vol_30d", "vol_60d", "Oil_chg7", "Dollar_chg7",
            "soy_corn_ratio", "enso_oni", "month_sin", "month_cos",
            "cot_noncomm_long_pct", "wasde_bull_bias",
            "price_pct_in_12m_range", "seasonal_ret_30d_med"]
    avail = [c for c in keep if c in d.columns]
    return d, avail


def main():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    df["ret_30d_fwd"] = df["Soybeans"].pct_change(30).shift(-30)

    # ── Construir target binario "WAIT pagó" ────────────────────
    # ret_30d_fwd > costo_30d → wait fue mejor que sell
    df["label_wait_paid"] = (df["ret_30d_fwd"] > COST_PCT_MONTH).astype(int)

    df, feat_cols = build_features(df)
    print(f"Features ({len(feat_cols)}): {feat_cols}")
    print(f"\nDistribución del label binario (WAIT pagó):")
    counts = df["label_wait_paid"].value_counts().sort_index()
    print(f"  0 (SELL fue mejor): {counts.get(0, 0)} ({counts.get(0,0)/len(df)*100:.1f}%)")
    print(f"  1 (WAIT pagó):       {counts.get(1, 0)} ({counts.get(1,0)/len(df)*100:.1f}%)")

    # ── Walk-forward eval ──────────────────────────────────────
    end = df["Date"].max()
    test_start = end - pd.DateOffset(years=5)   # 5 años test
    test_idx = []
    d = test_start
    while d <= end - pd.Timedelta(days=HORIZON_DAYS):
        future = df[df["Date"] >= d]
        if future.empty: break
        idx = int(df.index[df["Date"] == future.iloc[0]["Date"]][0])
        test_idx.append(idx)
        d = future.iloc[0]["Date"] + pd.Timedelta(days=TEST_FREQ_DAYS)

    print(f"\nTest dates: {len(test_idx)}, freq=7d, ventana=5 años")

    # Para velocidad: re-train cada 90 días en lugar de cada decisión
    REFIT_EVERY = 90
    last_refit = None
    model = calibrator = None
    rows = []

    for idx_t in test_idx:
        D = df.iloc[idx_t]["Date"]
        target = df.iloc[idx_t].get("label_wait_paid")
        actual_ret = df.iloc[idx_t].get("ret_30d_fwd")
        if pd.isna(target) or pd.isna(actual_ret):
            continue

        # Refit si toca
        need_refit = (last_refit is None) or ((D - last_refit).days >= REFIT_EVERY)
        if need_refit:
            train_start = D - pd.DateOffset(years=TRAIN_YEARS)
            tr = df[(df["Date"] >= train_start) & (df["Date"] < D)].dropna(subset=["label_wait_paid"]).copy()
            if len(tr) < 200:
                continue
            X_tr = tr[feat_cols].fillna(0).replace([np.inf, -np.inf], 0)
            y_tr = tr["label_wait_paid"]
            # Split val (últimos 20%) para isotonic calibration
            split_idx = int(len(tr) * 0.8)
            X_train, X_val = X_tr.iloc[:split_idx], X_tr.iloc[split_idx:]
            y_train, y_val = y_tr.iloc[:split_idx], y_tr.iloc[split_idx:]

            model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                                    eval_metric="logloss")
            model.fit(X_train, y_train, verbose=False)
            # Isotonic calibration sobre val
            try:
                p_val = model.predict_proba(X_val)[:, 1]
                calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                calibrator.fit(p_val, y_val.values.astype(float))
            except Exception:
                calibrator = None
            last_refit = D

        # Predict para D
        x_t = df.iloc[[idx_t]][feat_cols].fillna(0).replace([np.inf, -np.inf], 0)
        p_raw = float(model.predict_proba(x_t)[0, 1])
        p_cal = float(calibrator.transform([p_raw])[0]) if calibrator else p_raw
        pred = 1 if p_cal >= 0.5 else 0

        rows.append({
            "date": D, "actual_label": int(target),
            "actual_ret": float(actual_ret),
            "p_raw": p_raw, "p_calibrated": p_cal,
            "pred": pred,
        })

    R = pd.DataFrame(rows)
    print(f"\nN evaluados: {len(R)}")

    # ── Métricas básicas ──────────────────────────────────────
    correct = int((R["pred"] == R["actual_label"]).sum())
    n = len(R)
    acc = correct / n
    bt = binomtest(correct, n, p=0.5, alternative="greater")
    print(f"\n══════════════════════════════════════════════════════════════")
    print(f"  Accuracy del clasificador binario")
    print(f"══════════════════════════════════════════════════════════════")
    print(f"  Accuracy: {acc*100:.2f}% ({correct}/{n})")
    print(f"  Binomial test p (> 0.5): {bt.pvalue:.4f}")
    print(f"  Brier score (calibrated): {((R['p_calibrated'] - R['actual_label'])**2).mean():.4f}")
    print(f"  Brier score (raw):        {((R['p_raw']        - R['actual_label'])**2).mean():.4f}")

    # Baselines simples
    p_base_wait = R["actual_label"].mean()   # base rate WAIT
    p_base_sell = 1 - p_base_wait
    print(f"\n  Base rate WAIT: {p_base_wait*100:.1f}%")
    print(f"  Always-WAIT acc: {p_base_wait*100:.2f}%")
    print(f"  Always-SELL acc: {p_base_sell*100:.2f}%")
    print(f"  Random (mejor base): {max(p_base_wait, p_base_sell)*100:.2f}%")

    # ── Utilidad económica del clasificador ─────────────────
    print(f"\n══════════════════════════════════════════════════════════════")
    print(f"  Utilidad económica (P&L hipotético)")
    print(f"══════════════════════════════════════════════════════════════")
    # Convertir actual_ret a USD/ton: precio típico $400/ton, ret en %
    R["pnl_sell"] = 0.0   # base normalizada
    R["pnl_wait"] = R["actual_ret"] - COST_PCT_MONTH   # ganancia neta de esperar
    # P&L del clasificador: si pred=1 (WAIT), gano pnl_wait; si pred=0 (SELL), gano 0
    R["pnl_classifier"] = np.where(R["pred"] == 1, R["pnl_wait"], R["pnl_sell"])
    R["pnl_always_sell"] = R["pnl_sell"]
    R["pnl_always_wait"] = R["pnl_wait"]
    R["pnl_oracle"] = np.where(R["actual_label"] == 1, R["pnl_wait"], R["pnl_sell"])

    for col, label in [
        ("pnl_classifier",   "Clasificador"),
        ("pnl_always_sell",  "Always-Sell"),
        ("pnl_always_wait",  "Always-Wait"),
        ("pnl_oracle",       "Oracle (techo teórico)"),
    ]:
        avg = R[col].mean()
        win_pct = (R[col] > R["pnl_always_sell"]).mean() * 100
        print(f"  {label:<22} avg_pnl={avg*100:+6.3f}%  win_vs_sell={win_pct:.1f}%")

    # P&L ponderado por probabilidad (acción graduada)
    # Si p>0.5 → fracción WAIT proporcional a confianza
    R["pnl_graded"] = R["p_calibrated"] * R["pnl_wait"] + (1 - R["p_calibrated"]) * R["pnl_sell"]
    print(f"  Graded (prob-weighted)  avg_pnl={R['pnl_graded'].mean()*100:+6.3f}%")

    # ── Por confianza (¿alta confianza = más acierto?) ─────
    print(f"\n══════════════════════════════════════════════════════════════")
    print(f"  ¿Alta confianza ↔ más acierto?")
    print(f"══════════════════════════════════════════════════════════════")
    R["confidence"] = (R["p_calibrated"] - 0.5).abs()
    for q_label, q_low, q_high in [("Conf alta (top 25%)", 0.75, 1.0),
                                     ("Conf media (50-75%)", 0.50, 0.75),
                                     ("Conf baja (bottom 50%)", 0.0, 0.50)]:
        mask = R["confidence"].between(R["confidence"].quantile(q_low),
                                         R["confidence"].quantile(q_high))
        sub = R[mask]
        if sub.empty: continue
        sub_acc = (sub["pred"] == sub["actual_label"]).mean() * 100
        sub_pnl = sub["pnl_classifier"].mean() * 100
        print(f"  {q_label:<25} N={len(sub):>3}  acc={sub_acc:.1f}%  avg_pnl={sub_pnl:+.3f}%")

    R.to_csv(os.path.join(ROOT, "artifacts_eval", "test_decision_classifier.csv"), index=False)
    print(f"\n💾 artifacts_eval/test_decision_classifier.csv")


if __name__ == "__main__":
    main()
