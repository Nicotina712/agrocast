"""
src/data/fetch_trends.py
Reemplaza GDELT con Google Trends vía pytrends (libre, sin API key).

Google Trends mide el interés de búsqueda relativo (0-100) para palabras
clave a lo largo del tiempo. Para "soybean" + "soja", picos de búsqueda
correlacionan con eventos de mercado (reportes USDA, eventos climáticos,
noticias China). Es un proxy de atención/sentimiento del mercado.

Fuente: Google Trends (datos semanales, hasta 5 años de historia)
Cache: data/trends_soybean.csv (TTL 3 días)
"""

import os
import time
from datetime import datetime, timedelta

import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_PATH   = os.path.join(_PROJECT_ROOT, "data", "trends_soybean.csv")
_TTL_DAYS     = 3


def _cache_fresh() -> bool:
    if not os.path.exists(_CACHE_PATH):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(_CACHE_PATH))
    return age < timedelta(days=_TTL_DAYS)


def fetch_google_trends(start_year: int = 2021) -> pd.DataFrame:
    """
    Descarga interés de búsqueda para "soybean" desde Google Trends.
    Retorna DataFrame con columnas: Date, trends_interest, trends_norm.

    trends_interest: valor 0-100 de Google Trends (semanal, interpolado a diario)
    trends_norm:     normalizado a [-1, 1] para usar como feature del modelo
    """
    if _cache_fresh():
        print(f"📂 Google Trends desde caché")
        return pd.read_csv(_CACHE_PATH, parse_dates=["Date"])

    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("  [WARN] pytrends no instalado — ejecuta: pip install pytrends")
        return pd.DataFrame(columns=["Date", "trends_interest", "trends_norm"])

    print("📡 Descargando Google Trends 'soybean'...")
    try:
        pt = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        timeframe = f"{start_year}-01-01 {datetime.now().strftime('%Y-%m-%d')}"
        pt.build_payload(
            kw_list=["soybean"],
            cat=0,
            timeframe=timeframe,
            geo="",
        )
        data = pt.interest_over_time()

        if data.empty:
            print("  [WARN] Google Trends devolvió datos vacíos")
            return pd.DataFrame(columns=["Date", "trends_interest", "trends_norm"])

        data = data.reset_index()[["date", "soybean"]].rename(
            columns={"date": "Date", "soybean": "trends_interest"}
        )
        data["Date"] = pd.to_datetime(data["Date"])

        # Interpolación diaria (datos originales son semanales)
        daily_idx = pd.date_range(data["Date"].min(), data["Date"].max(), freq="D")
        data = data.set_index("Date").reindex(daily_idx).interpolate("linear").reset_index()
        data.columns = ["Date", "trends_interest"]

        # Normalizar a [-1, 1]: 50 = neutral → 0, 100 = máximo → 1, 0 = mínimo → -1
        data["trends_norm"] = (data["trends_interest"] - 50) / 50.0

        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        data.to_csv(_CACHE_PATH, index=False)
        print(f"  ✅ Google Trends: {len(data)} días guardados")
        return data

    except Exception as e:
        print(f"  [WARN] Google Trends error: {e}")
        return pd.DataFrame(columns=["Date", "trends_interest", "trends_norm"])
