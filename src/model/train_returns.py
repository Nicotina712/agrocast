"""
src/model/train_returns.py
Entrena un XGBClassifier para predecir la DIRECCIÓN del precio a 7 días.

Cambio clave vs versión anterior (XGBRegressor):
  - Predecir el valor exacto del retorno es casi imposible (r²≈0 en finanzas).
  - Predecir la DIRECCIÓN (sube/baja) es más factible y directamente útil.
  - El clasificador devuelve una probabilidad P(sube) en [0,1]:
      P > 0.58 → BUY, P < 0.42 → SELL, 0.42–0.58 → HOLD
  - Esto elimina el problema de predicciones degeneradas (todo≈0) de raíz.
"""

import os

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

NON_FEATURE_COLS = {"Date", "Soybeans",
                     "ret_1d_fwd", "ret_7d_fwd", "ret_14d_fwd", "ret_30d_fwd",
                     "ret_7d_net", "ret_14d_net", "direction",
                     "realized_vol_14d",
                     # OHLC del mismo día = look-ahead implícito
                     "Soybeans_High", "Soybeans_Low", "Soybeans_Open",
                     "Maize_High", "Maize_Low", "Maize_Open",
                     "SoybeanMeal_High", "SoybeanMeal_Low", "SoybeanMeal_Open",
                     "SoybeanOil_High",  "SoybeanOil_Low",  "SoybeanOil_Open",
                     # auxiliares point-in-time
                     "released_at", "as_of_date", "wasde_date", "week_ending"}
TARGET_REG = "ret_14d_fwd"  # horizonte 14d (más señal, menos ruido que 7d)
VOL_TARGET = "realized_vol_14d"  # head adicional: vol esperada (sizing/conviction)
EMBARGO_DAYS = 18                # = horizonte (14) + buffer evento (4)


def train_returns_model(features_path: str, artifacts_dir: str) -> XGBClassifier:
    """
    Entrena XGBClassifier para predecir si el precio sube en 7 días.

    Parámetros
    ----------
    features_path : path completo al CSV de features
    artifacts_dir : directorio donde guardar returns_model.joblib
    """
    print(f"📂 Cargando features desde: {features_path}")

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"No se encontró features CSV en: {features_path}")

    df = pd.read_csv(features_path)
    df = df.dropna(subset=[TARGET_REG])
    df = df.replace([float("inf"), float("-inf")], 0)

    # ── Target NETO de costos ──────────────────────────────────────
    # Restamos el costo round-trip estimado del retorno antes de binarizar.
    # Si el día tiene event_cost_mult > 1, el threshold sube en consecuencia.
    try:
        from src.trader.costs import estimate_round_trip_cost_pct
        if "event_cost_mult" in df.columns:
            cost_pct = df["event_cost_mult"].apply(
                lambda m: estimate_round_trip_cost_pct(float(m))
            )
        else:
            base_cost = estimate_round_trip_cost_pct(1.0)
            cost_pct = pd.Series(base_cost, index=df.index)
    except Exception:
        cost_pct = pd.Series(0.0005, index=df.index)  # 5 bps fallback

    df["ret_14d_net"] = df[TARGET_REG] - cost_pct
    df["direction"]   = (df["ret_14d_net"] > 0).astype(int)

    # ── Volatilidad realizada forward-looking (target del head de vol) ──
    # Std de retornos diarios en la ventana D+1..D+14, anualizada.
    if VOL_TARGET not in df.columns and "Soybeans" in df.columns:
        daily_ret = df["Soybeans"].pct_change()
        df[VOL_TARGET] = (
            daily_ret.shift(-1).rolling(14, min_periods=7).std() * np.sqrt(252)
        ).bfill().fillna(0)
    target_dist = df["direction"].value_counts()
    print(f"📊 Target: {target_dist[1]} subidas ({target_dist[1]/len(df)*100:.1f}%) "
          f"| {target_dist[0]} bajadas ({target_dist[0]/len(df)*100:.1f}%)")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS | {"direction"}]
    print(f"📊 Features para clasificador: {len(feature_cols)}")

    X     = df[feature_cols].fillna(0)
    y     = df["direction"]
    y_vol = df[VOL_TARGET].fillna(0) if VOL_TARGET in df.columns else None

    # ── Split temporal 80/20 con embargo ──────────────────────────
    # Embargo entre train y val para evitar contaminación por la ventana
    # forward del target (ret_14d_fwd usa D+1..D+14) y eventos macro programados.
    split = int(len(df) * 0.8)
    if "Date" in df.columns:
        df_dates    = pd.to_datetime(df["Date"])
        cut_date    = df_dates.iloc[split - 1]
        embargo_end = cut_date + pd.Timedelta(days=EMBARGO_DAYS)
        val_mask    = df_dates > embargo_end
        train_mask  = df_dates <= cut_date
        X_train, y_train = X[train_mask], y[train_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]
        if y_vol is not None:
            y_vol_train = y_vol[train_mask]
            y_vol_val   = y_vol[val_mask]
        print(f"   [Embargo] {EMBARGO_DAYS}d entre {cut_date.date()} y {embargo_end.date()} "
              f"({split - len(X_train) + (len(df) - split - len(X_val))} filas dropeadas)")
    else:
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]
        if y_vol is not None:
            y_vol_train = y_vol.iloc[:split]
            y_vol_val   = y_vol.iloc[split:]

    # scale_pos_weight balancea clases si hay desbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw   = float(n_neg / n_pos) if n_pos > 0 else 1.0

    # ── Clasificador ──────────────────────────────────────────────
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.05,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # ── Métricas ──────────────────────────────────────────────────
    probs  = model.predict_proba(X_val)[:, 1]
    preds  = (probs > 0.5).astype(int)
    acc    = accuracy_score(y_val, preds)
    try:
        auc = roc_auc_score(y_val, probs)
    except Exception:
        auc = float("nan")

    print(f"📊 Clasificador — Accuracy: {acc*100:.1f}% | ROC-AUC: {auc:.3f}")
    print(f"   P(sube) — mean: {probs.mean():.3f} | std: {probs.std():.3f} "
          f"| min: {probs.min():.3f} | max: {probs.max():.3f}")

    # BUY/SELL/HOLD distribution con thresholds por defecto
    buy_pct  = (probs > 0.58).mean() * 100
    sell_pct = (probs < 0.42).mean() * 100
    hold_pct = 100 - buy_pct - sell_pct
    print(f"   Señales (val) — BUY: {buy_pct:.0f}% | SELL: {sell_pct:.0f}% | HOLD: {hold_pct:.0f}%")

    # ── Head de volatilidad esperada (multi-output) ───────────────
    # Predice realized_vol_14d para sizing/conviction. No reemplaza al
    # clasificador — se entrena en paralelo con las mismas features.
    vol_model = None
    vol_mae   = None
    if y_vol is not None and y_vol_train.std() > 1e-6:
        try:
            vol_model = XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, gamma=0.05,
                random_state=42, eval_metric="mae",
            )
            vol_model.fit(X_train, y_vol_train,
                          eval_set=[(X_val, y_vol_val)], verbose=False)
            vol_pred = vol_model.predict(X_val)
            vol_mae  = float(mean_absolute_error(y_vol_val, vol_pred))
            print(f"📊 Vol head — MAE={vol_mae:.4f} | "
                  f"pred mean={vol_pred.mean():.3f} target mean={y_vol_val.mean():.3f}")
        except Exception as _e:
            print(f"   [WARN] vol head falló: {_e}")
            vol_model = None

    # ── Guardar ───────────────────────────────────────────────────
    os.makedirs(artifacts_dir, exist_ok=True)
    out_path = os.path.join(artifacts_dir, "returns_model.joblib")

    joblib.dump({
        "model":       model,
        "vol_model":   vol_model,
        "feature_cols": feature_cols,
        "model_type":  "classifier",    # señal para predict_returns
        "buy_thresh":  0.58,
        "sell_thresh": 0.42,
        "val_acc":     round(acc, 4),
        "val_auc":     round(auc, 4) if not np.isnan(auc) else None,
        "vol_mae":     round(vol_mae, 4) if vol_mae is not None else None,
        "embargo_days": EMBARGO_DAYS,
    }, out_path)

    print(f"✅ Clasificador guardado en {out_path}")
    return model
