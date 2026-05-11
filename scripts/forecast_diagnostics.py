"""
scripts/forecast_diagnostics.py
Diagnóstico del forecast 30d: tareas A y B (no modifica pipeline productivo).

A) Compara MAE 30d del modelo actual vs baselines (random walk, seasonal naive)
   sobre el holdout OOS (último 20% de features.csv).

B) Re-genera el forecast 30d para la fecha más reciente, recalculando
   features estructurales (mom_*, rsi_14, vol_*, ma*, ratios, crush_spread)
   en cada paso a partir del price_buffer. Compara contra forecast.csv actual.
"""

import os, sys
import numpy as np
import pandas as pd
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")
MODEL_PATH   = os.path.join(ROOT, "artifacts", "model.joblib")
FORECAST_CSV = os.path.join(ROOT, "artifacts", "forecast.csv")

HORIZON   = 30
LAST_YEARS = 1   # ventana de evaluación OOS para baselines


# ─────────────────────────────────────────────────────────────────
# Carga
# ─────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    art = joblib.load(MODEL_PATH)
    model    = art["model"]
    features = art["features"]
    return df, model, features


# ─────────────────────────────────────────────────────────────────
# Helpers de feature recalculo (espejo de build_features.py)
# ─────────────────────────────────────────────────────────────────
def _safe_pct(arr, periods):
    if len(arr) <= periods or arr[-periods - 1] in (0, None) or np.isnan(arr[-periods - 1]):
        return 0.0
    return float((arr[-1] - arr[-periods - 1]) / arr[-periods - 1])


def _rsi(arr, periods=14):
    if len(arr) < periods + 1:
        return 50.0
    s = pd.Series(arr[-(periods + 1):])
    delta = s.diff().dropna()
    gain = delta.clip(lower=0).mean()
    loss = (-delta.clip(upper=0)).mean()
    if loss == 0:
        return 100.0 if gain > 0 else 50.0
    rs = gain / loss
    return float(100 - 100 / (1 + rs))


def _vol(arr, window):
    if len(arr) < window + 1:
        return 0.0
    s = pd.Series(arr[-(window + 1):]).pct_change().dropna()
    return float(s.std() if len(s) else 0.0)


def _ma(arr, window):
    if len(arr) < window:
        return float(np.mean(arr))
    return float(np.mean(arr[-window:]))


def recompute_structural(row, price_buffer, maize_last, oil_last, meal_last, soyoil_last):
    """
    Actualiza in-place las features estructurales de `row` (Series) usando
    el price_buffer (lista de precios de soja, último = paso anterior).
    Las exógenas (Maize, Oil, etc.) se mantienen fijas en el último valor real.
    """
    p = price_buffer[-1]
    # Momentum
    row["mom_5d"]  = _safe_pct(price_buffer, 5)
    row["mom_20d"] = _safe_pct(price_buffer, 20)
    row["mom_60d"] = _safe_pct(price_buffer, 60)
    # RSI / vol
    row["rsi_14"]  = _rsi(price_buffer, 14)
    row["vol_10d"] = _vol(price_buffer, 10)
    row["vol_30d"] = _vol(price_buffer, 30)
    # Medias móviles vs precio
    ma20 = _ma(price_buffer, 20)
    ma90 = _ma(price_buffer, 90)
    row["price_vs_ma20"] = (p - ma20) / ma20 if ma20 else 0.0
    row["price_vs_ma90"] = (p - ma90) / ma90 if ma90 else 0.0
    row["ma20_vs_ma90"]  = (ma20 - ma90) / ma90 if ma90 else 0.0
    # Pendiente MA90 sobre últimos 5 días
    if len(price_buffer) >= 95:
        ma90_5d_ago = float(np.mean(price_buffer[-95:-5]))
        row["ma90_slope"] = (ma90 - ma90_5d_ago) / ma90 if ma90 else 0.0
    # Ratios y spreads (las exógenas no se reproyectan → son constantes)
    if maize_last:
        row["soy_corn_ratio"] = p / maize_last
    if soyoil_last is not None and meal_last is not None:
        meal_cb = (meal_last / 2204.6) * 100 * 48
        oil_cb  = soyoil_last * 11
        row["crush_spread"] = meal_cb + oil_cb - p
    # Lags
    if len(price_buffer) >= 1:  row["Soybeans_lag1"]  = price_buffer[-1]
    if len(price_buffer) >= 7:  row["Soybeans_lag7"]  = price_buffer[-7]
    if len(price_buffer) >= 30: row["Soybeans_lag30"] = price_buffer[-30]
    return row


# ─────────────────────────────────────────────────────────────────
# A) Baselines vs modelo (30d, OOS)
# ─────────────────────────────────────────────────────────────────
def task_A_baselines(df, model, features, eval_years=LAST_YEARS, recompute=False):
    """
    Para cada t en la ventana OOS donde existe t+30:
      - actual = price_{t+30}
      - rw     = price_t
      - seasonal = price_t * (1 + avg_ret30_for_month_in_train)
      - model_static = simulación 30 pasos (predict.py original — features fijas)
      - model_dyn    = simulación 30 pasos con features estructurales recalculadas
    """
    df = df.copy()
    cutoff = df["Date"].max() - pd.DateOffset(years=eval_years)
    train_df = df[df["Date"] < cutoff].copy()
    oos_df   = df[df["Date"] >= cutoff].copy()

    # Curva estacional ret_30d por mes (calibrada solo en train)
    train_df["ret30"] = train_df["Soybeans"].pct_change(30).shift(-30)
    seasonal = train_df.groupby(train_df["Date"].dt.month)["ret30"].mean().to_dict()

    # Iteramos sobre filas OOS donde t+HORIZON existe en df global
    df_idx = df.set_index("Date")
    results = []

    # Sub-muestreo cada 5 días para que la simulación dinámica sea viable
    sample_step = 5
    eligible = oos_df.iloc[::sample_step]

    print(f"\n📊 [A] Evaluando {len(eligible)} fechas OOS (cada {sample_step}d, ventana={eval_years}a)...")

    for _, row in eligible.iterrows():
        t = row["Date"]
        t30 = t + pd.Timedelta(days=HORIZON)
        # Buscamos la fila más cercana >= t30
        future_rows = df[df["Date"] >= t30]
        if future_rows.empty:
            continue
        actual = float(future_rows.iloc[0]["Soybeans"])
        p_t = float(row["Soybeans"])

        # RW
        rw = p_t
        # Seasonal
        seas_ret = seasonal.get(t.month, 0.0) or 0.0
        seas = p_t * (1 + seas_ret)
        # Modelo — simulación 30 pasos
        m_static = simulate_30d(df, row, model, features, recompute=False)
        m_dyn    = simulate_30d(df, row, model, features, recompute=True)

        results.append({
            "date":   t,
            "actual": actual,
            "rw":     rw,
            "seas":   seas,
            "m_stat": m_static,
            "m_dyn":  m_dyn,
        })

    R = pd.DataFrame(results)
    if R.empty:
        print("   ⚠️ Sin datos suficientes para evaluar.")
        return

    def mae(col):  return float(np.mean(np.abs(R[col] - R["actual"])))
    def mape(col): return float(np.mean(np.abs((R[col] - R["actual"]) / R["actual"]))) * 100
    def diracc(col):
        return float(np.mean(np.sign(R[col] - R["actual"].shift(1).fillna(R["actual"])) ==
                              np.sign(R["actual"] - R["actual"].shift(1).fillna(R["actual"])))) * 100

    print("\n┌──────────────────────────────┬─────────┬─────────┬──────────────┐")
    print("│ Modelo                       │ MAE USD │ MAPE %  │ Lift vs RW   │")
    print("├──────────────────────────────┼─────────┼─────────┼──────────────┤")
    rw_mae = mae("rw")
    for label, col in [
        ("Random Walk (p_t)",          "rw"),
        ("Seasonal naive (mes)",       "seas"),
        ("Modelo actual (estático)",   "m_stat"),
        ("Modelo + features dinámicas","m_dyn"),
    ]:
        m = mae(col)
        mp = mape(col)
        lift = (rw_mae - m) / rw_mae * 100
        print(f"│ {label:<28} │ {m:7.2f} │ {mp:6.2f}  │ {lift:+10.1f} % │")
    print("└──────────────────────────────┴─────────┴─────────┴──────────────┘")
    print(f"   n obs = {len(R)}  •  período = {R['date'].min().date()} → {R['date'].max().date()}")

    return R


# ─────────────────────────────────────────────────────────────────
# Simulación 30d (con o sin recálculo de features estructurales)
# ─────────────────────────────────────────────────────────────────
def simulate_30d(df_full, start_row, model, feature_cols, recompute=False, steps=30):
    """
    Simula el forecast loop a partir de start_row.
    El precio inicial es start_row["Soybeans"]; price_buffer se inicializa
    con los 90 días previos a start_row para soportar MA90.
    """
    start_idx = df_full.index[df_full["Date"] == start_row["Date"]]
    if len(start_idx) == 0:
        return float("nan")
    si = int(start_idx[0])
    hist_window = df_full["Soybeans"].iloc[max(0, si - 90):si + 1].tolist()
    base_price = float(start_row["Soybeans"])
    price_buffer = list(hist_window)

    # Exógenas constantes (último valor real)
    maize_last  = float(start_row.get("Maize", 0)) or None
    oil_last    = float(start_row.get("Oil", 0)) or None
    meal_last   = float(start_row.get("SoybeanMeal", 0)) or None
    soyoil_last = float(start_row.get("SoybeanOil", 0)) or None

    current = start_row.copy()
    for i in range(steps):
        if recompute:
            current = recompute_structural(current, price_buffer,
                                           maize_last, oil_last, meal_last, soyoil_last)
        else:
            # solo lags (réplica de predict.py productivo)
            if len(price_buffer) >= 1:  current["Soybeans_lag1"]  = price_buffer[-1]
            if len(price_buffer) >= 7:  current["Soybeans_lag7"]  = price_buffer[-7]
            if len(price_buffer) >= 30: current["Soybeans_lag30"] = price_buffer[-30]

        X = pd.DataFrame([current]).reindex(columns=feature_cols, fill_value=0)
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        pred_log = float(model.predict(X)[0])
        pred_log = max(min(pred_log, 10), -10)
        pred = float(np.expm1(pred_log))
        # Ancla 88/12
        pred = 0.88 * pred + 0.12 * base_price
        # Clip ±1% diario (idéntico a predict.py)
        prev = price_buffer[-1]
        pred = float(np.clip(pred, prev * 0.99, prev * 1.01))

        price_buffer.append(pred)
        current["Soybeans"] = pred

    return price_buffer[-1]


# ─────────────────────────────────────────────────────────────────
# B) Forecast actual con recompute=True para hoy
# ─────────────────────────────────────────────────────────────────
def task_B_forecast_dynamic(df, model, features):
    print("\n📊 [B] Forecast 30d para la última fecha disponible — features dinámicas\n")

    last_row = df.iloc[-1]
    base_price = float(last_row["Soybeans"])
    print(f"   Anclado en {last_row['Date'].date()} → ${base_price:.2f}\n")

    # Estático (réplica del actual)
    path_static = simulate_30d_path(df, last_row, model, features, recompute=False)
    path_dyn    = simulate_30d_path(df, last_row, model, features, recompute=True)

    # Cargar forecast.csv productivo para comparar
    if os.path.exists(FORECAST_CSV):
        prod = pd.read_csv(FORECAST_CSV)
        print(f"   forecast.csv productivo: {prod.iloc[0]['Soybeans']:.2f} → {prod.iloc[-1]['Soybeans']:.2f}")

    print("\n   ── Trayectoria comparada ─────────────────────────────────────")
    print("   day │ static (current) │ dinámico (B)")
    for i in (0, 4, 9, 14, 19, 24, 29):
        s = path_static[i] if i < len(path_static) else None
        d = path_dyn[i]    if i < len(path_dyn)    else None
        print(f"   {i+1:>3}  │  {s:>10.2f}      │  {d:>10.2f}")

    # Métricas de forma
    def returns(p): return [(p[i] - p[i-1]) / p[i-1] for i in range(1, len(p))]
    rs_stat = returns(path_static)
    rs_dyn  = returns(path_dyn)

    print(f"\n   ── Forma de la curva ─────────────────────────────────────────")
    print(f"   Static  → cambio total = {(path_static[-1]/base_price-1)*100:+.2f}%, "
          f"std(rets) = {np.std(rs_stat)*100:.3f}%, "
          f"% pasos clip+ = {sum(1 for r in rs_stat if r>=0.0099)/len(rs_stat)*100:.0f}%")
    print(f"   Dyn     → cambio total = {(path_dyn[-1]/base_price-1)*100:+.2f}%, "
          f"std(rets) = {np.std(rs_dyn)*100:.3f}%, "
          f"% pasos clip+ = {sum(1 for r in rs_dyn if r>=0.0099)/len(rs_dyn)*100:.0f}%")
    print(f"\n   Si el % de pasos saturando el clip baja sustancialmente, B confirma")
    print(f"   que la rampa es artefacto de features congeladas, no del modelo.")


def simulate_30d_path(df_full, start_row, model, feature_cols, recompute, steps=30):
    si = int(df_full.index[df_full["Date"] == start_row["Date"]][0])
    hist_window = df_full["Soybeans"].iloc[max(0, si - 90):si + 1].tolist()
    base_price = float(start_row["Soybeans"])
    price_buffer = list(hist_window)
    maize_last  = float(start_row.get("Maize", 0)) or None
    oil_last    = float(start_row.get("Oil", 0)) or None
    meal_last   = float(start_row.get("SoybeanMeal", 0)) or None
    soyoil_last = float(start_row.get("SoybeanOil", 0)) or None

    current = start_row.copy()
    out = []
    for i in range(steps):
        if recompute:
            current = recompute_structural(current, price_buffer,
                                           maize_last, oil_last, meal_last, soyoil_last)
        else:
            if len(price_buffer) >= 1:  current["Soybeans_lag1"]  = price_buffer[-1]
            if len(price_buffer) >= 7:  current["Soybeans_lag7"]  = price_buffer[-7]
            if len(price_buffer) >= 30: current["Soybeans_lag30"] = price_buffer[-30]

        X = pd.DataFrame([current]).reindex(columns=feature_cols, fill_value=0)
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        pred_log = float(model.predict(X)[0])
        pred_log = max(min(pred_log, 10), -10)
        pred = float(np.expm1(pred_log))
        pred = 0.88 * pred + 0.12 * base_price
        prev = price_buffer[-1]
        pred = float(np.clip(pred, prev * 0.99, prev * 1.01))
        price_buffer.append(pred)
        current["Soybeans"] = pred
        out.append(pred)
    return out


if __name__ == "__main__":
    df, model, features = load_data()
    print(f"Features cargadas: {len(features)}  •  Filas: {len(df)}  •  "
          f"Rango: {df['Date'].min().date()} → {df['Date'].max().date()}")
    task_A_baselines(df, model, features)
    task_B_forecast_dynamic(df, model, features)
