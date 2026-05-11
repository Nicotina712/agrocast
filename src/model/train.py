"""
src/model/train.py
Entrena el modelo Ridge de predicción de precios de soja.

✅ FIX CRÍTICO — Data Leakage:
   La selección de features por correlación con target_log incluía
   ret_7d_fwd, ret_1d_fwd, ret_30d_fwd (retornos del FUTURO).
   El modelo veía el futuro durante el entrenamiento → métricas
   infladas pero predicciones inútiles en producción.
   → Ahora esas columnas están explícitamente excluidas.

✅ FIX: el path de artifacts se resuelve desde PROJECT_ROOT pasado
   como parámetro, en lugar de depender de __file__.
"""

import os

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Columnas que NUNCA deben ser features (leakage o irrelevantes)
EXCLUDE_FROM_FEATURES = {
    "target_log",
    "Soybeans",
    "Soybeans_log",
    # ✅ retornos forward = data leakage
    "ret_1d_fwd",
    "ret_7d_fwd",
    "ret_14d_fwd",
    "ret_30d_fwd",
    # ✅ OHLC intradía del MISMO día = look-ahead implícito
    # High/Low/Open del día t están disponibles después del cierre que estamos
    # tratando de predecir → dominan 94% del feature importance pero el modelo
    # explica el día actual en lugar de predecir el siguiente.
    "Soybeans_High", "Soybeans_Low", "Soybeans_Open",
    "Maize_High", "Maize_Low", "Maize_Open",
    "SoybeanMeal_High", "SoybeanMeal_Low", "SoybeanMeal_Open",
    "SoybeanOil_High",  "SoybeanOil_Low",  "SoybeanOil_Open",
    # columnas de fecha
    "Date",
    # auxiliares point-in-time
    "released_at", "as_of_date", "wasde_date", "week_ending",
}

MAX_FEATURES = 25


def train_model(
    df: pd.DataFrame,
    target: str = "Soybeans",
    artifacts_dir: str | None = None,
    rolling_window_years: int | None = None,
) -> None:
    """
    Entrena un modelo Ridge para predecir el precio de soja.

    Parámetros
    ----------
    df                   : DataFrame con features (salida de make_features)
    target               : columna objetivo
    artifacts_dir        : directorio donde guardar model.joblib
    rolling_window_years : si se pasa, limita el entrenamiento a los últimos N años
                           (ventana deslizante en lugar de datos acumulativos)
    """
    print("\n🤖 Entrenando modelo de precios…")

    df = df.copy()

    # ── Ventana deslizante (rolling retrain) ──────────────────────
    if rolling_window_years is not None and "Date" in df.columns:
        cutoff = pd.Timestamp.today() - pd.DateOffset(years=rolling_window_years)
        df_full_len = len(df)
        df = df[pd.to_datetime(df["Date"]) >= cutoff]
        print(f"   📅 Rolling window {rolling_window_years}a: {df_full_len} → {len(df)} filas "
              f"(desde {cutoff.date()})")

    # ── Limpieza ──────────────────────────────────────────────────
    df = df.dropna(subset=[target])
    df = df.ffill().fillna(0)
    df = df.replace([np.inf, -np.inf], 0)

    # ── Target en log ─────────────────────────────────────────────
    df["target_log"] = np.log1p(df[target])

    # ── Selección de features ─────────────────────────────────────
    numeric_df = df.select_dtypes(include=[np.number])
    corr       = numeric_df.corr()["target_log"].abs().sort_values(ascending=False)

    selected = [
        c for c in corr.index
        if c not in EXCLUDE_FROM_FEATURES
    ][:MAX_FEATURES]

    print(f"\n📊 Features seleccionadas ({len(selected)}):")
    print(selected)

    # ── Matrices ──────────────────────────────────────────────────
    X = df[selected]
    y = df["target_log"]

    # ── Split temporal (80/20) ────────────────────────────────────
    split   = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ── Entrenamiento ─────────────────────────────────────────────
    model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=30,
        eval_metric="mae",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Métricas ──────────────────────────────────────────────────
    preds  = np.expm1(model.predict(X_test))
    y_real = np.expm1(y_test)

    mae  = mean_absolute_error(y_real, preds)
    rmse = np.sqrt(mean_squared_error(y_real, preds))

    print(f"\n📊 Métricas (test set):")
    print(f"   MAE  : {mae:.2f}")
    print(f"   RMSE : {rmse:.2f}")

    # Feature importance top-5
    imp = sorted(zip(selected, model.feature_importances_), key=lambda x: -x[1])
    print("   Top features: " + ", ".join(f"{n}({v:.3f})" for n, v in imp[:5]))

    # ── Modelos de cuantiles Q10 / Q90 para bandas de confianza ──
    # XGBoost soporta reg:quantileerror desde v1.7. Si la versión es
    # antigua o falla, se omiten sin romper el pipeline.
    q10_model = None
    q90_model = None
    try:
        _q_params = dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:quantileerror",
        )
        q10_model = XGBRegressor(**_q_params, quantile_alpha=0.10)
        q10_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        q90_model = XGBRegressor(**_q_params, quantile_alpha=0.90)
        q90_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Cobertura empírica: cuántos puntos del test caen entre Q10 y Q90
        q10_preds = np.expm1(q10_model.predict(X_test))
        q90_preds = np.expm1(q90_model.predict(X_test))
        coverage = float(((y_real >= q10_preds) & (y_real <= q90_preds)).mean())
        print(f"   📐 Cuantiles Q10/Q90 — cobertura test: {coverage*100:.1f}% (objetivo 80%)")
    except Exception as _qe:
        print(f"   [INFO] Quantile models omitidos (XGBoost <1.7 o error): {_qe}")
        q10_model = None
        q90_model = None

    # ── Guardar ───────────────────────────────────────────────────
    if artifacts_dir is None:
        # Fallback: artifacts/ en la raíz del proyecto
        artifacts_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "artifacts",
        )

    os.makedirs(artifacts_dir, exist_ok=True)

    out_path = os.path.join(artifacts_dir, "model.joblib")
    joblib.dump({
        "model":      model,
        "features":   selected,
        "q10_model":  q10_model,
        "q90_model":  q90_model,
    }, out_path)

    print(f"💾 Modelo guardado en {out_path}")
