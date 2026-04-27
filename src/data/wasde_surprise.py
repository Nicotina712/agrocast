"""
src/data/wasde_surprise.py
WASDE Surprise Detection — dos capas:

  1. Proxy automático: mide la reacción del precio de soja en los 3 días
     posteriores a cada reporte WASDE como estimador del "surprise".
     Un movimiento fuerte = el mercado se sorprendió (bullish o bearish).

  2. Opcional: si existe data/wasde_actuals.csv (formato: Date,soy_ending_stocks_mmt)
     cargado manualmente desde USDA PSD, calcula el MoM change real como surprise.

Salida (features diarias, forward-fill):
  wasde_surprise_proxy  : retorno acumulado del precio en los 3d post-WASDE
                          (positivo = WASDE bullish, negativo = bearish)
  wasde_surprise_ma3    : media móvil de 3 reportes (tendencia de sesgo WASDE)
  wasde_bull_bias       : fracción de últimos 6 reportes con surprise positivo
  wasde_days_to_next    : días hasta el próximo WASDE (0–35)
"""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.data.wasde_dates import get_wasde_dates

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_ACTUALS_PATH = os.path.join(_PROJECT_ROOT, "data", "wasde_actuals.csv")


def _wasde_surprise_from_price(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estima el surprise de cada WASDE como el retorno acumulado del precio
    en los 3 días hábiles posteriores al reporte.

    Parámetros
    ----------
    price_df : DataFrame con columnas ['Date', 'Soybeans']
    """
    df = price_df[["Date", "Soybeans"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    wasde_ts = get_wasde_dates(start_year=int(df.index.year.min()))

    records = []
    for wd in wasde_ts:
        # Precio el día del reporte (o el más cercano anterior)
        before = df[df.index <= wd]
        if before.empty:
            continue
        p_before = float(before["Soybeans"].iloc[-1])

        # Precio 3 días hábiles después
        after = df[(df.index > wd) & (df.index <= wd + pd.Timedelta(days=7))]
        if after.empty or len(after) < 1:
            continue
        p_after = float(after["Soybeans"].iloc[min(2, len(after) - 1)])

        surprise = (p_after - p_before) / p_before if p_before > 0 else 0.0
        # Point-in-time: el surprise proxy usa precios de D+1..D+3 post-WASDE,
        # por lo que sólo es observable ~3 días hábiles después del reporte.
        # released_at = wasde_date + 5 días calendario (~3 hábiles).
        released_at = wd + pd.Timedelta(days=5)
        records.append({
            "Date": released_at,
            "wasde_date": wd,
            "released_at": released_at,
            "wasde_surprise_proxy": surprise,
        })

    if not records:
        return pd.DataFrame(columns=["Date", "wasde_surprise_proxy",
                                     "wasde_surprise_ma3", "wasde_bull_bias"])

    surprises = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
    # Drop wasde_date helper column for downstream merge cleanliness
    if "wasde_date" in surprises.columns:
        surprises = surprises.drop(columns=["wasde_date"])
    surprises["wasde_surprise_ma3"]  = surprises["wasde_surprise_proxy"].rolling(3, min_periods=1).mean()
    surprises["wasde_bull_bias"]     = (
        surprises["wasde_surprise_proxy"]
        .rolling(6, min_periods=1)
        .apply(lambda x: (x > 0).mean())
    )
    return surprises


def _wasde_surprise_from_actuals() -> pd.DataFrame:
    """
    Carga data/wasde_actuals.csv si existe.
    Formato esperado: Date,soy_ending_stocks_mmt
    Calcula MoM change como surprise normalizado.
    """
    if not os.path.exists(_ACTUALS_PATH):
        return pd.DataFrame()

    try:
        df = pd.read_csv(_ACTUALS_PATH, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        if "soy_ending_stocks_mmt" not in df.columns:
            return pd.DataFrame()

        df["wasde_stocks_mom"] = df["soy_ending_stocks_mmt"].diff()
        std = df["wasde_stocks_mom"].std()
        df["wasde_stocks_surprise"] = (df["wasde_stocks_mom"] / std).fillna(0).clip(-3, 3)

        print(f"   [WASDE] Actuals cargados: {len(df)} reportes | "
              f"último: {df['Date'].max().date()}")
        return df[["Date", "soy_ending_stocks_mmt", "wasde_stocks_surprise"]]
    except Exception as e:
        print(f"   [WARN] wasde_actuals.csv error: {e}")
        return pd.DataFrame()


def build_wasde_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye features WASDE diarias para merge con features principales.

    Parámetros
    ----------
    price_df : DataFrame con al menos ['Date', 'Soybeans']

    Retorna DataFrame diario con columnas WASDE listo para merge.
    """
    print("   [WASDE] Calculando surprise proxy desde precio…")
    surprises = _wasde_surprise_from_price(price_df)

    actuals = _wasde_surprise_from_actuals()
    if not actuals.empty:
        surprises = surprises.merge(actuals, on="Date", how="left")

    if surprises.empty:
        return pd.DataFrame(columns=["Date"])

    # Expandir a frecuencia diaria con forward-fill (el surprise persiste hasta el siguiente WASDE)
    price_dates = pd.to_datetime(price_df["Date"].unique())
    date_range  = pd.DataFrame({"Date": pd.date_range(price_dates.min(), price_dates.max(), freq="D")})

    surprises["Date"] = pd.to_datetime(surprises["Date"])
    daily = date_range.merge(surprises, on="Date", how="left").sort_values("Date")

    # Forward-fill: el último surprise conocido es la información disponible
    # Excluimos columnas de timestamp auxiliares (wasde_date, released_at)
    drop_aux = [c for c in ("wasde_date", "released_at") if c in daily.columns]
    if drop_aux:
        daily = daily.drop(columns=drop_aux)
    feat_cols = [c for c in daily.columns if c != "Date"]
    daily[feat_cols] = daily[feat_cols].ffill().fillna(0)

    n_reports = surprises["wasde_surprise_proxy"].notna().sum()
    bull_pct   = (surprises["wasde_surprise_proxy"] > 0).mean() * 100
    print(f"   [WASDE] {n_reports} reportes | bullish: {bull_pct:.0f}% | "
          f"last surprise: {surprises['wasde_surprise_proxy'].iloc[-1]*100:.1f}%")

    return daily
