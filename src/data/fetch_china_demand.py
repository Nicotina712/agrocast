"""
src/data/fetch_china_demand.py
China Demand Intelligence Module.

China compra ~60-65% de la soja exportada globalmente.
Cuando la demanda china sube/baja, arrastra el precio de Chicago.

Componentes:
  1. Importaciones anuales — USDA FAS bulk CSV (ZIP descargable, sin bloqueo)
  2. Crush margin implícito (CBOT ZM=F / ZS=F como proxy)
  3. CNY/USD → interpretación de poder adquisitivo
  4. Comparación YoY + Z-score vs. histórico

Cache del ZIP USDA: data/psd_oilseeds.csv (TTL 7 días — datos mensuales)
Cache principal: data/china_demand.json (TTL 24h)
"""

import io
import json
import os
import zipfile
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd

_PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_PATH    = os.path.join(_PROJECT_ROOT, "data", "china_demand.json")
_PSD_CSV_PATH  = os.path.join(_PROJECT_ROOT, "data", "psd_oilseeds.csv")
_TTL_HOURS     = 24
_PSD_TTL_DAYS  = 7

_PSD_ZIP_URL   = "https://apps.fas.usda.gov/psdonline/downloads/psd_oilseeds_csv.zip"
_MEAL_TICKER   = "ZM=F"
_SOY_TICKER    = "ZS=F"
SOYBEAN_MEAL_YIELD = 0.80


def _cache_valid() -> bool:
    if not os.path.exists(_CACHE_PATH):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(_CACHE_PATH))
    return age < timedelta(hours=_TTL_HOURS)


def _psd_csv_fresh() -> bool:
    if not os.path.exists(_PSD_CSV_PATH):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(_PSD_CSV_PATH))
    return age < timedelta(days=_PSD_TTL_DAYS)


def _download_psd_csv() -> pd.DataFrame | None:
    """Descarga y cachea el CSV PSD de oleaginosas de USDA FAS."""
    try:
        import requests
        print("   [China] Descargando PSD Oilseeds CSV de USDA FAS (~3.7MB)...")
        r = requests.get(_PSD_ZIP_URL, timeout=60)
        if r.status_code != 200:
            return None
        z   = zipfile.ZipFile(io.BytesIO(r.content))
        df  = pd.read_csv(z.open("psd_oilseeds.csv"))
        df.to_csv(_PSD_CSV_PATH, index=False)
        print(f"   [China] PSD CSV guardado: {len(df):,} filas")
        return df
    except Exception as e:
        print(f"   [China] Error descargando PSD CSV: {e}")
        return None


def _load_psd_csv() -> pd.DataFrame | None:
    """Carga PSD CSV desde caché o lo descarga si está vencido."""
    if _psd_csv_fresh() and os.path.exists(_PSD_CSV_PATH):
        try:
            return pd.read_csv(_PSD_CSV_PATH)
        except Exception:
            pass
    return _download_psd_csv()


def _fetch_china_imports_psd(year: int) -> dict:
    """
    Importaciones y crush anuales de China desde USDA FAS PSD bulk CSV.
    China code = 'CH', Soybean commodity code = 2222000.
    Valores en 1000 MT → convertir a MMT dividiendo por 1000.
    """
    try:
        df = _load_psd_csv()
        if df is None or df.empty:
            return {}

        china_soy = df[
            (df["Country_Code"] == "CH") &
            (df["Commodity_Code"] == 2222000)
        ]

        result = {}
        for attr, key in [("Imports", "imports_mmt"), ("Crush", "crush_mmt"),
                           ("Domestic Consumption", "consumption_mmt")]:
            row = china_soy[
                (china_soy["Attribute_Description"] == attr) &
                (china_soy["Market_Year"] == year)
            ]
            if not row.empty:
                val = row["Value"].iloc[0]
                if pd.notna(val):
                    result[key] = round(float(val) / 1000, 2)  # 1000 MT → MMT

        # Historial de importaciones para YoY y Z-score
        hist = china_soy[china_soy["Attribute_Description"] == "Imports"].copy()
        hist = hist.dropna(subset=["Value"]).sort_values("Market_Year")
        hist["imports_mmt"] = hist["Value"] / 1000
        result["history"] = hist[["Market_Year", "imports_mmt"]].tail(10).to_dict("records")

        return result
    except Exception as e:
        print(f"   [China] PSD CSV parse error: {e}")
        return {}


def _compute_crush_margin() -> dict:
    """
    Calcula el crush margin implícito usando futuros CBOT como proxy.
    Crush margin = (Soybean Meal precio × rendimiento) - Soybean precio
    En USD/ton.
    """
    try:
        import yfinance as yf

        soy  = yf.Ticker(_SOY_TICKER).history(period="5d")
        meal = yf.Ticker(_MEAL_TICKER).history(period="5d")

        if soy.empty or meal.empty:
            return {"margin_usd_ton": None, "signal": None}

        soy_price_usc_bu  = float(soy["Close"].iloc[-1])
        meal_price_usd_ton = float(meal["Close"].iloc[-1])  # USD/short ton

        # Convertir soja a USD/ton
        soy_usd_ton = (soy_price_usc_bu / 100) * 36.744

        # Gross Processing Margin (GPM)
        # 1 ton soja → 0.80 ton harina + 0.185 ton aceite (descartamos aceite aquí)
        meal_revenue = meal_price_usd_ton * SOYBEAN_MEAL_YIELD * (1000 / 907.185)  # short ton → metric ton
        gpm = meal_revenue - soy_usd_ton

        if gpm > 20:
            signal = "POSITIVO"   # crushers incentivados → demanda soja alta
            note   = f"Margen positivo ({gpm:+.1f} USD/ton) — crushers chinos incentivados a importar soja."
        elif gpm > -10:
            signal = "NEUTRAL"
            note   = f"Margen moderado ({gpm:+.1f} USD/ton) — demanda dentro de parámetros normales."
        else:
            signal = "NEGATIVO"   # márgenes negativos → demanda reducida
            note   = f"Margen negativo ({gpm:+.1f} USD/ton) — presión sobre demanda de soja para crush."

        return {
            "margin_usd_ton":      round(gpm, 2),
            "soy_price_usd_ton":   round(soy_usd_ton, 2),
            "meal_price_usd_ton":  round(meal_price_usd_ton, 2),
            "signal":              signal,
            "note":                note,
        }
    except Exception as e:
        print(f"   [China] Crush margin error: {e}")
        return {"margin_usd_ton": None, "signal": None, "note": f"Error: {e}"}


def _get_cny_usd() -> float | None:
    """Lee CNY/USD desde el archivo FRED o usa fallback."""
    try:
        raw_path = os.path.join(_PROJECT_ROOT, "data", "raw_market.csv")
        import pandas as pd
        df = pd.read_csv(raw_path, parse_dates=["Date"])
        if "cny_usd" in df.columns:
            return float(df["cny_usd"].dropna().iloc[-1])
    except Exception:
        pass
    return None


def get_china_demand() -> dict:
    """
    Retorna módulo de demanda china: importaciones + crush margin + CNY signal.

    Retorna dict con:
      imports_mmt_current   : importaciones anuales actuales (MMT)
      imports_mmt_prev      : importaciones año anterior (MMT)
      imports_yoy_pct       : variación interanual
      crush_margin          : dict con margin_usd_ton, signal
      cny_usd               : tipo de cambio
      cny_signal            : FORTALECIMIENTO / DEBILITAMIENTO / ESTABLE
      demand_score          : 0-100 (100=demanda muy fuerte)
      interpretation        : texto
      as_of                 : fecha
    """
    if _cache_valid():
        try:
            with open(_CACHE_PATH) as f:
                data = json.load(f)
            print(f"   [China] Caché válido ({data.get('as_of', '?')})")
            return data
        except Exception:
            pass

    print("   [China] Calculando módulo de demanda China…")
    year = date.today().year

    # ── Importaciones desde USDA FAS PSD bulk CSV ──────────────────────
    psd_curr = _fetch_china_imports_psd(year)
    psd_prev = _fetch_china_imports_psd(year - 1)
    crush    = _compute_crush_margin()
    cny_usd  = _get_cny_usd()

    curr_mmt = psd_curr.get("imports_mmt")
    prev_mmt = psd_prev.get("imports_mmt")

    # Fallback: si el año actual no está en el CSV, usar el más reciente disponible
    if curr_mmt is None and psd_curr.get("history"):
        last = psd_curr["history"][-1]
        curr_mmt = last.get("imports_mmt")
        print(f"   [China] Usando último dato disponible: MY {last.get('Market_Year')} = {curr_mmt} MMT")

    yoy_pct = round((curr_mmt - prev_mmt) / prev_mmt * 100, 1) if (curr_mmt and prev_mmt and prev_mmt > 0) else None

    # Z-score histórico
    history = psd_curr.get("history", [])
    zscore  = None
    if history and curr_mmt:
        vals   = [h["imports_mmt"] for h in history if h.get("imports_mmt")]
        mean_h = float(np.mean(vals))
        std_h  = float(np.std(vals)) if len(vals) > 2 else 1.0
        zscore = round((curr_mmt - mean_h) / (std_h + 1e-8), 2)

    # CNY signal: CNY más alto (más yuanes/dólar) = yuan más débil = soja más cara para China
    cny_signal = None
    cny_note   = None
    if cny_usd:
        # Históricamente: CNY/USD entre 6.5-7.3
        if cny_usd > 7.2:
            cny_signal = "DEBILITAMIENTO"
            cny_note   = f"Yuan débil ({cny_usd:.2f} CNY/USD) — soja en USD más cara para importadores chinos → señal bajista."
        elif cny_usd < 6.8:
            cny_signal = "FORTALECIMIENTO"
            cny_note   = f"Yuan fuerte ({cny_usd:.2f} CNY/USD) — soja más asequible para China → señal alcista."
        else:
            cny_signal = "ESTABLE"
            cny_note   = f"Yuan en rango normal ({cny_usd:.2f} CNY/USD) — impacto FX neutro."

    # Demand score compuesto
    score = 50
    crush_signal = crush.get("signal")
    if crush_signal == "POSITIVO":
        score += 20
    elif crush_signal == "NEGATIVO":
        score -= 20

    if yoy_pct is not None:
        if yoy_pct > 5:
            score += 15
        elif yoy_pct < -5:
            score -= 15

    if cny_signal == "FORTALECIMIENTO":
        score += 10
    elif cny_signal == "DEBILITAMIENTO":
        score -= 10

    score = max(0, min(100, round(score)))

    if score >= 65:
        interp = "Demanda china fuerte — importaciones en alza y márgenes de crush positivos."
        demand_signal = "FUERTE"
    elif score >= 40:
        interp = "Demanda china en niveles normales — sin señales extremas."
        demand_signal = "NORMAL"
    else:
        interp = "Señales de demanda china débil — márgenes de crush negativos o caída de importaciones."
        demand_signal = "DEBIL"

    result = {
        "imports_mmt_current":  curr_mmt,
        "imports_mmt_prev":     prev_mmt,
        "imports_yoy_pct":      yoy_pct,
        "imports_zscore":       zscore,
        "crush_margin":         crush,
        "cny_usd":              cny_usd,
        "cny_signal":           cny_signal,
        "cny_note":             cny_note,
        "demand_score":         score,
        "demand_signal":        demand_signal,
        "interpretation":       interp,
        "data_source":          "usda_fas_psd_csv",
        "history":              psd_curr.get("history", []),
        "as_of":                date.today().isoformat(),
    }

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"   [China] Score: {score} | Crush: {crush_signal} | YoY: {yoy_pct}% | CNY: {cny_signal}")
    return result
