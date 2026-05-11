"""
src/data/load_external.py
Carga fuentes de datos externas adicionales para enriquecer el modelo:

  1. CSVs de señales de noticias históricas (data/external/)
     - brazil_conab : sentimiento noticias cosecha Brasil (2018-2025)
     - china_demand : sentimiento demanda China (2018-2025)
     - logistics_bdi: sentimiento flete/logística Baltic Dry Index (2008-2025)

  2. FRED (Federal Reserve Economic Data) — opcional, requiere FRED_API_KEY
     - DEXBZUS : Real brasileño / USD (diario) — competitividad exportaciones
     - DEXCHUS : Yuan chino / USD (diario) — poder adquisitivo importaciones

Todos los datos se interpolan/forward-fill a frecuencia diaria para
poder fusionarlos con el DataFrame de precios del modelo.
"""

import os
import sys
from datetime import date, timedelta

import pandas as pd

_SCRIPT_DIR   = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_EXTERNAL_DIR = os.path.join(_PROJECT_ROOT, "data", "external")

# Columnas útiles del wide CSV (solo las que tienen valores no-cero con cobertura real)
# logistics_bdi_NewsIndex EXCLUIDA: todos sus valores son 0.0 → sin señal útil
_USEFUL_COLS = [
    "brazil_conab_NewsIndex",
    "china_demand_NewsIndex",
]


# ── 1. CSVs de noticias históricas ──────────────────────────────────

def load_news_signals() -> pd.DataFrame:
    """
    Carga data/external/news_signals_wide.csv y lo convierte a frecuencia
    diaria con forward-fill. Retorna DataFrame con columnas:
      Date, brazil_conab_signal, china_demand_signal, logistics_bdi_signal
    """
    path = os.path.join(_EXTERNAL_DIR, "news_signals_wide.csv")
    if not os.path.exists(path):
        print("⚠️  news_signals_wide.csv no encontrado — se omiten señales históricas")
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Conservar solo columnas con buena cobertura histórica
    available = [c for c in _USEFUL_COLS if c in df.columns]
    if not available:
        return pd.DataFrame()

    df = df[["Date"] + available].copy()

    # Renombrar para claridad en el modelo
    rename = {
        "brazil_conab_NewsIndex":  "ext_brazil_signal",
        "china_demand_NewsIndex":  "ext_china_signal",
        "logistics_bdi_NewsIndex": "ext_bdi_signal",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Convertir a diario con forward-fill (los datos son mensuales)
    df = df.set_index("Date").resample("D").ffill().reset_index()

    print(f"✅ Señales históricas externas: {df.shape[0]} días "
          f"| cols: {[c for c in df.columns if c != 'Date']}")
    return df


# ── 2. FRED (macro: BRL/USD y CNY/USD) ──────────────────────────────

_FRED_CACHE_PATH = os.path.join(_PROJECT_ROOT, "data", "fred_cache.csv")


def _load_fred_cache() -> pd.DataFrame:
    """Caché de la última respuesta exitosa de FRED. Crítico: cuando FRED
    devuelve 502 transient, sin caché el feature set cambia (faltan brl_usd
    y cny_usd) y los modelos del A/B retoman α distinto entre runs."""
    if not os.path.exists(_FRED_CACHE_PATH):
        return pd.DataFrame()
    try:
        df = pd.read_csv(_FRED_CACHE_PATH, parse_dates=["Date"])
        return df
    except Exception:
        return pd.DataFrame()


def load_fred_data(start_date: date | None = None) -> pd.DataFrame:
    """
    Descarga BRL/USD y CNY/USD desde FRED si FRED_API_KEY está configurada.
    Cuando la API falla (502 transient u otra), cae al caché en disco
    `data/fred_cache.csv` para mantener el feature set estable entre runs.

    Para obtener una clave gratuita: https://fred.stlouisfed.org/docs/api/api_key.html
    """
    fred_key = os.getenv("FRED_API_KEY", "").strip()
    if not fred_key:
        print("ℹ️  FRED_API_KEY no configurada — se omiten datos macro FRED")
        cached = _load_fred_cache()
        if not cached.empty:
            print(f"   [FRED] usando cache en disco ({len(cached)} filas)")
        return cached

    if start_date is None:
        start_date = date(2015, 1, 1)
    end_date = date.today()

    try:
        sys.path.insert(0, _PROJECT_ROOT)
        from src.connectors.fred import FredConnector

        fred = FredConnector(api_key=fred_key)

        # BRL/USD y CNY/USD — daily frequency
        df = fred.fetch(
            start=start_date,
            end=end_date,
            series=["DEXBZUS", "DEXCHUS"],
            frequency="d",
        )
        df = df.rename(columns={
            "DEXBZUS": "brl_usd",  # Reales por dólar — sube = Brasil exporta más barato
            "DEXCHUS": "cny_usd",  # Yuanes por dólar — sube = China importa más caro
        })

        # Forward-fill fines de semana y feriados
        df = df.set_index("Date").resample("D").ffill().reset_index()

        print(f"✅ FRED: {df.shape[0]} días | BRL/USD y CNY/USD desde {start_date}")

        # Persistir caché para tolerar fallas transientes en runs siguientes
        try:
            df.to_csv(_FRED_CACHE_PATH, index=False)
        except Exception as _e:
            print(f"   [FRED] no pude persistir cache: {_e}")
        return df

    except Exception as e:
        print(f"⚠️  Error cargando FRED: {e}")
        cached = _load_fred_cache()
        if not cached.empty:
            print(f"   [FRED] fallback a cache ({len(cached)} filas, último: {cached['Date'].max().date()})")
        return cached


# ── 3. USDA Export Inspections semanales ────────────────────────────

def load_usda_inspections_external(start_date: date | None = None) -> pd.DataFrame:
    """
    Carga Export Inspections semanales de soja (USDA-FGIS).
    Expande a frecuencia diaria con forward-fill para el merge.
    """
    try:
        from src.data.usda_inspections import load_usda_inspections
        start_year = start_date.year if start_date else None
        df = load_usda_inspections(start_year=start_year)
        if df.empty:
            return pd.DataFrame()

        cols = ["Date", "insp_soy_bu", "insp_soy_4wk_avg", "insp_soy_yoy", "insp_soy_zscore"]
        df = df[[c for c in cols if c in df.columns]].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").resample("D").ffill().reset_index()
        return df
    except Exception as e:
        print(f"   [WARN] USDA inspections: {e}")
        return pd.DataFrame()


# ── 4. COT (Commitment of Traders — CFTC) ───────────────────────────

def load_cot_external() -> pd.DataFrame:
    """
    Carga datos COT semanales de soybeans (CFTC Disaggregated Futures Only).
    Features: cot_commercial_net, cot_noncomm_net, cot_noncomm_long_pct, cot_index.
    """
    try:
        from src.data.fetch_cot import fetch_cot_soybeans
        df = fetch_cot_soybeans()
        if df.empty:
            return pd.DataFrame()
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception as e:
        print(f"   [WARN] COT: {e}")
        return pd.DataFrame()


# ── 5. Google Trends (reemplaza GDELT) ──────────────────────────────

def load_trends_external() -> pd.DataFrame:
    """
    Carga interés de búsqueda de Google Trends para 'soybean'.
    Reemplaza GDELT que era inestable (rate limits, JSON errors).
    """
    try:
        from src.data.fetch_trends import fetch_google_trends
        df = fetch_google_trends()
        if df.empty:
            return pd.DataFrame()
        df["Date"] = pd.to_datetime(df["Date"])
        return df[["Date", "trends_interest", "trends_norm"]]
    except Exception as e:
        print(f"   [WARN] Google Trends: {e}")
        return pd.DataFrame()


# ── 6. Función unificada ─────────────────────────────────────────────

def load_all_external(start_date: date | None = None) -> pd.DataFrame:
    """
    Carga y fusiona todas las fuentes externas.
    Retorna un único DataFrame con Date + todas las columnas externas.
    Si alguna fuente falla, se ignora sin romper el pipeline.
    """
    frames = []

    news = load_news_signals()
    if not news.empty:
        frames.append(news)

    fred = load_fred_data(start_date)
    if not fred.empty:
        frames.append(fred)

    usda = load_usda_inspections_external(start_date)
    if not usda.empty:
        frames.append(usda)

    cot = load_cot_external()
    if not cot.empty:
        frames.append(cot)

    trends = load_trends_external()
    if not trends.empty:
        frames.append(trends)

    if not frames:
        return pd.DataFrame()

    # Merge secuencial por Date
    result = frames[0]
    for df in frames[1:]:
        result = result.merge(df, on="Date", how="outer")

    result = result.sort_values("Date").reset_index(drop=True)
    print(f"📦 Total datos externos: {result.shape[1]-1} columnas | {result.shape[0]} días")
    return result
