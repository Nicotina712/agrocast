"""
src/model/predict.py
Genera el forecast de precios de soja a 30 días.

✅ FIX: en el loop de forecast se actualizaba solo lag1 pero no lag7
   ni lag30. El modelo "olvidaba" la historia después del primer paso.
   → Ahora se mantiene un buffer histórico y se actualizan todos los lags.

✅ FIX: la carga del modelo verificaba la existencia del archivo pero
   no producía un mensaje de error claro si faltaba.
"""

import os

import joblib
import numpy as np
import pandas as pd


def forecast_30d(
    df: pd.DataFrame,
    target: str,
    date_col: str,
    artifacts_dir: str,
    real_last_date=None,
    steps: int = 30,
) -> pd.DataFrame:
    """
    Genera un forecast de `steps` días para el precio del target.

    Parámetros
    ----------
    df             : DataFrame con features históricas
    target         : columna objetivo ("Soybeans")
    date_col       : columna de fecha ("Date")
    artifacts_dir  : directorio con model.joblib
    real_last_date : última fecha real conocida (para anclar el horizonte)
    steps          : días a proyectar (default 30)

    Retorna
    -------
    DataFrame con columnas Date y Soybeans.
    """

    # ── Fecha de inicio del forecast ──────────────────────────────
    if real_last_date is not None:
        last_date = pd.to_datetime(real_last_date)
    else:
        last_date = pd.to_datetime(df[date_col]).sort_values().iloc[-1]

    today = pd.Timestamp.today().normalize()
    if last_date < today - pd.Timedelta(days=30):
        print("⚠️  Fecha del modelo desactualizada, usando fecha actual")
        last_date = today

    # ── Cargar modelo ─────────────────────────────────────────────
    model_path = os.path.join(artifacts_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modelo no encontrado en {model_path}. Ejecutar pipeline primero."
        )

    model_data = joblib.load(model_path)
    model      = model_data["model"]
    features   = model_data["features"]
    q10_model  = model_data.get("q10_model")
    q90_model  = model_data.get("q90_model")

    # ── Precio base para ancla ────────────────────────────────────
    base_price  = float(df[target].iloc[-1])
    daily_vol   = float(df[target].pct_change().tail(60).std())  # volatilidad diaria histórica

    # ✅ Buffer de precios para actualizar lags correctamente
    # Guardamos los últimos 30 precios reales + predichos
    price_buffer = list(df[target].values[-30:])
    use_quantile_bands = (q10_model is not None) and (q90_model is not None)

    # Capturar features de noticias del último dato real para
    # proyectarlas de forma atenuada a lo largo del forecast
    NEWS_COLS = ["news_sentiment", "news_volume", "news_impact",
                 "china_score", "weather_score", "macro_score", "news_shock"]

    current = df.iloc[-1:].copy()
    forecasts = []
    forecasts_q10 = []
    forecasts_q90 = []

    for i in range(steps):

        # ── Atenuar features de noticias con el tiempo ────────────
        # Las noticias tienen efecto decreciente: en el día 1 pesan 100%,
        # al día 15 ya pesan 50%, al día 30 pesan ~0%.
        decay = max(0.0, 1.0 - i / steps)
        for col in NEWS_COLS:
            if col in current.columns:
                current[col] = df.iloc[-1][col] * decay if col in df.columns else 0.0

        # ── Actualizar lags desde el buffer ───────────────────────
        if len(price_buffer) >= 1:
            current[f"{target}_lag1"] = price_buffer[-1]
        if len(price_buffer) >= 7:
            current[f"{target}_lag7"] = price_buffer[-7]
        if len(price_buffer) >= 30:
            current[f"{target}_lag30"] = price_buffer[-30]

        # ── Predicción ────────────────────────────────────────────
        X = current.reindex(columns=features, fill_value=0)
        X = X.replace([np.inf, -np.inf], 0)

        pred_log = float(model.predict(X)[0])
        pred_log = max(min(pred_log, 10), -10)
        pred     = np.expm1(pred_log)

        # ── Ancla al precio base (12%) — reduce deriva agresiva ───
        pred = (0.88 * pred) + (0.12 * base_price)

        # ── Limitar cambio diario máximo (±1.0%) ──────────────────
        # Volatilidad histórica soja CBOT: ~0.8% diario. ±1% da margen
        # sin permitir forecasts de +15% en 30 días.
        prev_price = price_buffer[-1]
        pred = float(np.clip(
            pred,
            prev_price * 0.990,
            prev_price * 1.010,
        ))

        forecasts.append(pred)
        price_buffer.append(pred)

        # ── Bandas cuantílicas (Q10/Q90) ──────────────────────────
        if use_quantile_bands:
            q10_log = float(q10_model.predict(X)[0])
            q90_log = float(q90_model.predict(X)[0])
            q10_log = max(min(q10_log, 10), -10)
            q90_log = max(min(q90_log, 10), -10)
            q10_raw = np.expm1(q10_log)
            q90_raw = np.expm1(q90_log)
            q10_anc = (0.88 * q10_raw) + (0.12 * base_price)
            q90_anc = (0.88 * q90_raw) + (0.12 * base_price)
            # Asegurar que Q10 ≤ pred ≤ Q90
            forecasts_q10.append(min(q10_anc, pred))
            forecasts_q90.append(max(q90_anc, pred))
        else:
            forecasts_q10.append(None)
            forecasts_q90.append(None)

        current[target] = pred

    # ── Armar DataFrame de salida ─────────────────────────────────
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=steps,
        freq="D",
    )

    # ── Bandas de confianza ───────────────────────────────────────
    # Prioridad: modelos Q10/Q90 del pipeline (más honestos).
    # Fallback: fórmula basada en OOS MAE si los modelos cuantílicos no están.
    if use_quantile_bands:
        upper = [q if q is not None else f for q, f in zip(forecasts_q90, forecasts)]
        lower = [max(q, 0) if q is not None else 0 for q in forecasts_q10]
        print("   📐 Bandas de confianza: modelos Q10/Q90")
    else:
        oos_mae_usc = 24.5
        residual_vol = oos_mae_usc / base_price if base_price > 0 else daily_vol
        band_vol = max(residual_vol / np.sqrt(30), daily_vol * 0.5)
        upper = [f * (1 + band_vol * np.sqrt(i + 1)) for i, f in enumerate(forecasts)]
        lower = [max(f * (1 - band_vol * np.sqrt(i + 1)), 0) for i, f in enumerate(forecasts)]
        print("   📐 Bandas de confianza: fórmula OOS MAE (sin modelos cuantílicos)")

    return pd.DataFrame({
        "Date":     future_dates,
        "Soybeans": forecasts,
        "upper":    upper,
        "lower":    lower,
    })


def save_forecast_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    print(f"💾 Forecast guardado en {path}")
