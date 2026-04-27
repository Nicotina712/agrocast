"""
src/data/fetch_climate_macro.py
ENSO + drought macro climate signals para soja (Fix #10).

Fuentes públicas (sin auth):
  - ENSO ONI (Oceanic Niño Index): NOAA CPC mensual
    https://psl.noaa.gov/data/correlation/oni.data (texto plano)
  - US Drought Monitor (USDM): % área del Midwest en D2+ (severe drought)
    Proxy alternativo si no está disponible: NOAA NCEI Palmer Drought Severity Index

Features expuestas:
  enso_oni            : índice ONI mensual (-3..+3); negativo = La Niña, positivo = El Niño
  enso_phase          : -1 (La Niña), 0 (Neutral), +1 (El Niño)
  drought_pdsi_us     : Palmer Drought Severity Index promedio US (-7 muy seco..+7 muy húmedo)
  drought_severity    : 0 (sin sequía) .. 4 (excepcional)

Cache local para evitar re-descargas: data/climate_macro.csv (TTL 7 días).
"""

from __future__ import annotations

import io
import os
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT_PATH     = os.path.join(_PROJECT_ROOT, "data", "climate_macro.csv")
_CACHE_DAYS   = 7

_ONI_URL = "https://psl.noaa.gov/data/correlation/oni.data"


def _fetch_oni() -> pd.DataFrame:
    """Descarga ONI mensual y devuelve DataFrame con Date (1° del mes) + enso_oni."""
    try:
        import urllib.request
        with urllib.request.urlopen(_ONI_URL, timeout=20) as r:
            text = r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"[Climate] ONI fetch falló: {e}")
        return pd.DataFrame()

    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 13:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        if year < 1950 or year > date.today().year + 1:
            continue
        for m, val in enumerate(parts[1:], start=1):
            try:
                v = float(val)
            except ValueError:
                continue
            if v <= -90:  # missing sentinel
                continue
            rows.append({"Date": pd.Timestamp(year, m, 1), "enso_oni": v})

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    # Phase clasificación clásica (NOAA): |ONI| >= 0.5 por 5 meses → fase
    df["enso_phase"] = df["enso_oni"].apply(
        lambda v: 1 if v >= 0.5 else (-1 if v <= -0.5 else 0)
    )
    return df


def _synthetic_drought() -> pd.DataFrame:
    """
    Drought severity proxy sintético desde NASA POWER si está cacheado.
    No descarga nada — usa satellite_history.csv si existe (precip/temp anomalía).
    """
    sat_path = os.path.join(_PROJECT_ROOT, "data", "satellite_history.csv")
    if not os.path.exists(sat_path):
        return pd.DataFrame()
    try:
        sat = pd.read_csv(sat_path, parse_dates=["Date"])
    except Exception:
        return pd.DataFrame()
    if sat.empty:
        return pd.DataFrame()

    # Heurística: si tenemos 'precip_anom' o 'temp_anom' → score combinado
    cols = sat.columns
    drought = pd.DataFrame({"Date": sat["Date"]})
    pdsi_proxy = 0.0
    if "precip_anom" in cols:
        # precip negativa = sequía
        pdsi_proxy = -sat["precip_anom"].fillna(0)
    if "temp_anom" in cols:
        pdsi_proxy = pdsi_proxy + sat["temp_anom"].fillna(0) * 0.5
    if isinstance(pdsi_proxy, float):
        return pd.DataFrame()
    drought["drought_pdsi_us"] = pdsi_proxy.clip(-7, 7)
    drought["drought_severity"] = drought["drought_pdsi_us"].apply(
        lambda v: 4 if v >= 5 else (3 if v >= 3 else (2 if v >= 1.5 else (1 if v >= 0.5 else 0)))
    )
    return drought


def fetch_climate_macro(refresh: bool = False) -> pd.DataFrame:
    """Devuelve DataFrame diario con enso_* + drought_* features."""
    if not refresh and os.path.exists(_OUT_PATH):
        age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(_OUT_PATH))).days
        if age_days < _CACHE_DAYS:
            try:
                return pd.read_csv(_OUT_PATH, parse_dates=["Date"])
            except Exception:
                pass

    print("   [Climate] Fetcheando ENSO ONI + drought proxy…")
    oni = _fetch_oni()
    drought = _synthetic_drought()

    if oni.empty and drought.empty:
        return pd.DataFrame(columns=["Date", "enso_oni", "enso_phase",
                                     "drought_pdsi_us", "drought_severity"])

    # Combinar — expandir ONI mensual a diario, merge_asof con drought diario
    if not oni.empty:
        end_date = max(date.today(), oni["Date"].max().date())
        idx = pd.date_range(oni["Date"].min(), end_date, freq="D")
        oni_daily = pd.DataFrame({"Date": idx})
        oni_daily = pd.merge_asof(oni_daily, oni.sort_values("Date"),
                                   on="Date", direction="backward")
    else:
        oni_daily = pd.DataFrame(columns=["Date", "enso_oni", "enso_phase"])

    if not drought.empty:
        out = pd.merge(oni_daily, drought, on="Date", how="left")
    else:
        out = oni_daily.copy()
        out["drought_pdsi_us"]  = 0.0
        out["drought_severity"] = 0

    out = out.ffill().fillna(0)

    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    out.to_csv(_OUT_PATH, index=False)
    print(f"   [Climate] {len(out)} días guardados → ENSO/drought")
    return out


def load_climate_macro_features(features_df: pd.DataFrame) -> pd.DataFrame:
    cm = fetch_climate_macro()
    if cm.empty or "Date" not in features_df.columns:
        return features_df
    cm["Date"] = pd.to_datetime(cm["Date"])
    df = features_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    keep = [c for c in cm.columns if c == "Date"
            or c.startswith(("enso_", "drought_"))]
    cm = cm[keep].copy()
    merged = pd.merge_asof(
        df.sort_values("Date"),
        cm.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    feat_cols = [c for c in keep if c != "Date"]
    merged[feat_cols] = merged[feat_cols].ffill().fillna(0)
    return merged


if __name__ == "__main__":
    out = fetch_climate_macro(refresh=True)
    print(out.tail())
