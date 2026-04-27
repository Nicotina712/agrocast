"""
src/features/news_history.py
Acumula el sentimiento diario de noticias (RSS + NewsAPI) en un CSV persistente.

Cada vez que el pipeline procesa noticias, guarda el score del día.
Con el tiempo, esto construye un historial que el modelo news-impact puede usar.
"""

import os
from datetime import date

import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HISTORY_PATH = os.path.join(_PROJECT_ROOT, "data", "news_sentiment_history.csv")

_COLS = ["Date", "news_sentiment", "news_volume",
         "china_score", "weather_score", "macro_score"]


def save_daily_sentiment(
    sentiment: float,
    volume: int,
    china_score:   float = 0.0,
    weather_score: float = 0.0,
    macro_score:   float = 0.0,
) -> None:
    """Guarda o actualiza el sentimiento de noticias de hoy en el CSV histórico."""
    today = str(date.today())
    new_row = {
        "Date":           today,
        "news_sentiment": round(float(sentiment),    4),
        "news_volume":    int(volume),
        "china_score":    round(float(china_score),  4),
        "weather_score":  round(float(weather_score),4),
        "macro_score":    round(float(macro_score),  4),
    }

    if os.path.exists(_HISTORY_PATH):
        df = pd.read_csv(_HISTORY_PATH)
        df = df[df["Date"] != today]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        os.makedirs(os.path.dirname(_HISTORY_PATH), exist_ok=True)
        df = pd.DataFrame([new_row])

    df = df.sort_values("Date").reset_index(drop=True)
    df.to_csv(_HISTORY_PATH, index=False)
    print(f"📝 Sentimiento guardado: {today} | sent={sentiment:.3f} | vol={volume}")


def load_news_sentiment_history() -> pd.DataFrame:
    """
    Carga el historial de sentimiento de noticias diario.
    Retorna DataFrame con columnas: Date, news_sentiment, news_volume, etc.
    """
    if not os.path.exists(_HISTORY_PATH):
        return pd.DataFrame(columns=_COLS)
    try:
        df = pd.read_csv(_HISTORY_PATH, parse_dates=["Date"])
        return df.sort_values("Date").reset_index(drop=True)
    except Exception as e:
        print(f"⚠️  Error cargando news_sentiment_history: {e}")
        return pd.DataFrame(columns=_COLS)
