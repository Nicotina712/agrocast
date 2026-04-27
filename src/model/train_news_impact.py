"""
src/model/train_news_impact.py
Módulo experimental: entrena un XGBoost incorporando el impacto histórico
de noticias sobre retornos de soja.

Fuentes de datos de noticias:
  - GDELT tone histórico (2018-actualidad)
  - WASDE proximity (regla: segundo martes de cada mes)
  - Señales externas históricas: ext_brazil_signal, ext_china_signal
  - Sentimiento RSS/NewsAPI acumulado (data/news_sentiment_history.csv)

Los últimos dos empiezan con pocos días de datos y mejoran con el tiempo.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

_SRC          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _PROJECT_ROOT)

from src.data.fetch_gdelt        import fetch_gdelt_tone
from src.data.wasde_dates        import add_wasde_features
from src.features.news_history   import load_news_sentiment_history

NON_FEATURE_COLS = {"Date", "Soybeans", "ret_1d_fwd", "ret_7d_fwd", "ret_30d_fwd"}
TARGET           = "ret_7d_fwd"

# Todas las columnas consideradas "de noticias" para medir su impacto
NEWS_COLS = [
    "gdelt_tone",
    "wasde_days_ahead", "wasde_days_behind", "wasde_window",
    "ext_brazil_signal", "ext_china_signal",   # señales históricas CSV
    "rss_sentiment", "rss_volume",             # acumulado RSS/NewsAPI
    "rss_china_score", "rss_weather_score",
]

ARTIFACTS_DIR = os.path.join(_PROJECT_ROOT, "artifacts", "news_impact")


def _train_xgb(X_train, y_train, X_val, y_val) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def train_news_impact_model(features_path: str) -> dict:
    """
    Entrena el modelo news-impact y retorna un dict con métricas y datos
    para visualización.
    """
    print("📂 Cargando features base...")
    df = pd.read_csv(features_path, parse_dates=["Date"])
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], 0)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    # ── GDELT ────────────────────────────────────────────────────
    gdelt = fetch_gdelt_tone(start_year=int(df["Date"].dt.year.min()))
    if not gdelt.empty:
        gdelt["Date"] = pd.to_datetime(gdelt["Date"]).dt.normalize()
        df = df.merge(gdelt, on="Date", how="left")
        df["gdelt_tone"] = df["gdelt_tone"].ffill().fillna(0)
        coverage = round(df["gdelt_tone"].astype(bool).mean() * 100, 1)
        print(f"✅ GDELT: {coverage}% de fechas con datos")
    else:
        df["gdelt_tone"] = 0.0
        coverage = 0.0
        print("⚠️  GDELT sin datos, usando 0")

    # ── WASDE ────────────────────────────────────────────────────
    print("📅 Añadiendo features WASDE...")
    df = add_wasde_features(df, "Date")

    # ── Sentimiento RSS/NewsAPI acumulado ─────────────────────────
    rss_hist = load_news_sentiment_history()
    if not rss_hist.empty:
        rss_hist["Date"] = pd.to_datetime(rss_hist["Date"]).dt.normalize()
        rss_rename = {
            "news_sentiment": "rss_sentiment",
            "news_volume":    "rss_volume",
            "china_score":    "rss_china_score",
            "weather_score":  "rss_weather_score",
            "macro_score":    "rss_macro_score",
        }
        rss_hist = rss_hist.rename(columns=rss_rename)
        rss_cols = [c for c in rss_rename.values() if c in rss_hist.columns]
        df = df.merge(rss_hist[["Date"] + rss_cols], on="Date", how="left")
        for c in rss_cols:
            if c in df.columns:
                df[c] = df[c].ffill().fillna(0)
        n_rss = rss_hist["Date"].nunique()
        print(f"✅ RSS/NewsAPI history: {n_rss} días acumulados")
    else:
        for c in ["rss_sentiment", "rss_volume", "rss_china_score", "rss_weather_score"]:
            df[c] = 0.0
        print("ℹ️  Sin historial RSS/NewsAPI aún (se acumula con cada actualización)")

    # ── Split features ────────────────────────────────────────────
    all_feature_cols  = [c for c in df.columns if c not in NON_FEATURE_COLS]
    news_feature_cols = [c for c in NEWS_COLS if c in all_feature_cols]
    base_feature_cols = [c for c in all_feature_cols if c not in news_feature_cols]

    print(f"📊 Features base: {len(base_feature_cols)} | Features noticias: {len(news_feature_cols)}")
    print(f"   News features activas: {news_feature_cols}")

    split = int(len(df) * 0.8)
    y     = df[TARGET]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    # ── Modelo CON noticias ───────────────────────────────────────
    X_full       = df[all_feature_cols].fillna(0)
    X_full_train = X_full.iloc[:split]
    X_full_val   = X_full.iloc[split:]

    print("🤖 Entrenando modelo news-impact...")
    model_news = _train_xgb(X_full_train, y_train, X_full_val, y_val)
    preds_news = model_news.predict(X_full_val).astype(float)
    mae_news   = float(mean_absolute_error(y_val, preds_news))

    # ── Modelo SIN noticias (baseline) ───────────────────────────
    X_base       = df[base_feature_cols].fillna(0)
    X_base_train = X_base.iloc[:split]
    X_base_val   = X_base.iloc[split:]

    print("🤖 Entrenando modelo base (sin noticias)...")
    model_base = _train_xgb(X_base_train, y_train, X_base_val, y_val)
    preds_base = model_base.predict(X_base_val).astype(float)
    mae_base   = float(mean_absolute_error(y_val, preds_base))

    improvement = (mae_base - mae_news) / mae_base * 100 if mae_base > 0 else 0.0
    print(f"📊 MAE base: {mae_base:.5f} | MAE news: {mae_news:.5f} | Mejora: {improvement:.1f}%")

    # ── Importancia de features de noticias ──────────────────────
    importance_dict = dict(zip(all_feature_cols, model_news.feature_importances_))
    news_importance = {
        k: round(float(v), 5)
        for k, v in importance_dict.items()
        if k in news_feature_cols
    }

    # ── Histórico GDELT + precio ──────────────────────────────────
    gdelt_history = []
    sample = df[["Date", "gdelt_tone", "Soybeans"]].tail(365).copy()
    for _, row in sample.iterrows():
        if pd.notna(row["Soybeans"]):
            gdelt_history.append({
                "Date":  str(row["Date"])[:10],
                "tone":  round(float(row["gdelt_tone"]), 3),
                "price": round(float(row["Soybeans"]),   2),
            })

    # ── Fechas WASDE próximas (futuras desde hoy) ─────────────────
    from src.data.wasde_dates import _second_tuesday
    from datetime import date as _date, timedelta as _td
    _today = _date.today()
    _yr, _mo = _today.year, _today.month
    wasde_recent = []
    while len(wasde_recent) < 4:
        _dt = _second_tuesday(_yr, _mo)
        if _dt >= _today:
            wasde_recent.append(str(_dt))
        _mo += 1
        if _mo > 12:
            _mo, _yr = 1, _yr + 1

    # ── Señal actual del modelo news-impact ──────────────────────
    last_row     = df[all_feature_cols].iloc[[-1]].fillna(0)
    current_pred = float(model_news.predict(last_row)[0])
    preds_series = pd.Series(preds_news)
    low_t  = float(preds_series.quantile(0.33))
    high_t = float(preds_series.quantile(0.67))
    # Doble guardia contra señales ruidosas:
    # 1. Spread mínimo entre percentiles (distribución demasiado estrecha → HOLD)
    # 2. El retorno predicho debe superar un umbral absoluto mínimo de ±0.5%
    #    para considerar la señal accionable (evita SELL con +0.04%)
    MIN_ABS_RETURN = 0.005   # 0.5% retorno semanal mínimo para señal activa
    if abs(high_t - low_t) < 1e-6:
        ni_signal = "HOLD"
    elif current_pred > high_t and current_pred > MIN_ABS_RETURN:
        ni_signal = "BUY"
    elif current_pred < low_t and current_pred < -MIN_ABS_RETURN:
        ni_signal = "SELL"
    else:
        ni_signal = "HOLD"

    # ── Info de fuentes activas ───────────────────────────────────
    rss_days = rss_hist["Date"].nunique() if not rss_hist.empty else 0
    sources_active = {
        "gdelt":        coverage > 0,
        "wasde":        True,
        "ext_signals":  "ext_brazil_signal" in df.columns,
        "rss_history":  rss_days > 0,
        "rss_days":     rss_days,
    }

    # ── Guardar ───────────────────────────────────────────────────
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(
        {"model": model_news, "feature_cols": all_feature_cols},
        os.path.join(ARTIFACTS_DIR, "model.joblib"),
    )

    result = {
        "mae_news_model":          round(mae_news,    5),
        "mae_base_model":          round(mae_base,    5),
        "improvement_pct":         round(improvement, 1),
        "news_feature_importance": news_importance,
        "gdelt_coverage_pct":      coverage,
        "gdelt_history":           gdelt_history,
        "wasde_recent":            wasde_recent,
        "current_signal":          ni_signal,
        "current_pred_return":     round(current_pred * 100, 3),
        "n_train":                 split,
        "n_val":                   len(df) - split,
        "trained_at":              pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "sources_active":          sources_active,
    }

    with open(os.path.join(ARTIFACTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ Artefactos guardados en {ARTIFACTS_DIR}")
    return result
