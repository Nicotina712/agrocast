"""
src/features/news_features.py
Convierte artículos de noticias en features diarias para el modelo.

✅ FIX CRÍTICO: cuando articles=[] (lista vacía), la función antes
   retornaba un DataFrame completamente vacío, lo que hacía que el merge
   en pipeline.py fallara silenciosamente y las features de noticias
   quedaran siempre en 0.
   → Ahora retorna un DataFrame con las columnas correctas y 0 filas,
     que el merge maneja con how="left" sin perder datos.

✅ FIX: cuando el pipeline no tiene artículos nuevos (modo offline),
   se retornan columnas con valores neutros en lugar de nada.
"""

import pandas as pd

# Columnas de salida que siempre deben existir
FEATURE_COLS = [
    "Date",
    "news_sentiment",
    "news_impact",
    "news_volume",
    "china_score",
    "weather_score",
    "macro_score",
]

TOPICS = ("china", "weather", "macro")


def _empty_features() -> pd.DataFrame:
    """DataFrame vacío con las columnas correctas para merge seguro."""
    return pd.DataFrame(columns=FEATURE_COLS)


def build_news_features(articles: list) -> pd.DataFrame:
    """
    Agrega artículos por día y construye features de noticias.

    Parámetros
    ----------
    articles : lista de dicts con claves:
               publishedAt, sentiment, impact, topic, title

    Retorna
    -------
    DataFrame con columnas: Date, news_sentiment, news_impact,
    news_volume, china_score, weather_score, macro_score.
    Retorna DataFrame vacío (con columnas) si no hay artículos.
    """
    if not articles:
        # ✅ Retornar estructura vacía con columnas definidas
        return _empty_features()

    df = pd.DataFrame(articles)

    # ── Validar columnas mínimas ──────────────────────────────────
    required = {"publishedAt", "sentiment", "impact", "title"}
    missing  = required - set(df.columns)
    if missing:
        print(f"⚠️  news_features: columnas faltantes: {missing}")
        return _empty_features()

    # ── Fecha ─────────────────────────────────────────────────────
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    df = df.dropna(subset=["publishedAt"])

    if df.empty:
        return _empty_features()

    df["Date"] = df["publishedAt"].dt.floor("D")

    # ── Agregación base ───────────────────────────────────────────
    agg = (
        df.groupby("Date")
        .agg(
            news_sentiment=("sentiment", "mean"),
            news_impact=("impact",    "mean"),
            news_volume=("title",     "count"),
        )
    )

    # ── Topics ────────────────────────────────────────────────────
    if "topic" in df.columns:
        for topic in TOPICS:
            counts = (
                df[df["topic"] == topic]
                .groupby("Date")
                .size()
                .rename(f"{topic}_score")
            )
            agg = agg.join(counts, how="left")
    else:
        for topic in TOPICS:
            agg[f"{topic}_score"] = 0

    # ── Limpieza ──────────────────────────────────────────────────
    agg = agg.fillna(0).reset_index()

    # ── Normalización suave ───────────────────────────────────────
    agg["news_sentiment"] = agg["news_sentiment"] * 1.5
    agg["news_impact"]    = agg["news_impact"]    * 2.0
    agg["news_volume"]    = agg["news_volume"]    / 20.0

    for topic in TOPICS:
        agg[f"{topic}_score"] = agg[f"{topic}_score"] / 10.0

    return agg[FEATURE_COLS]
