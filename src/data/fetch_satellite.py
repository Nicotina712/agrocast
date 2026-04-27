"""
src/data/fetch_satellite.py
Datos satelitales/climáticos para zonas sojeras clave (NASA POWER API).

NASA POWER es gratis, sin API key, y entrega:
  - Precipitación (PRECTOTCORR, mm/día)
  - Temperatura mínima/máxima/promedio (T2M_MIN, T2M_MAX, T2M)
  - Radiación solar (ALLSKY_SFC_SW_DWN)
  - Humedad relativa (RH2M)

Zonas (centroides aproximados):
  - US Iowa (corn belt soja)        : 42.0, -93.5
  - Argentina (Pampa Húmeda)         : -33.5, -61.5
  - Brasil (Mato Grosso)             : -12.5, -55.5
  - Uruguay (litoral oeste)          : -32.5, -58.0

Como proxy de NDVI/condición de cultivo construimos un *crop_stress_score*
combinando déficit de lluvia (vs media histórica) y desvío térmico.
Un score positivo = stress (sequía o calor extremo) → bullish para el precio.

Salida: data/satellite_history.csv con columnas Date, region, precip_mm,
        tmean_c, tmax_c, rh_pct, srad, crop_stress_score.

NOTA: NDVI puro requiere Sentinel/MODIS (auth + tiles); NASA POWER es la mejor
opción “gratis y rápida” para una señal climática diaria que correlaciona con
condición de cultivo. Si en el futuro se quiere NDVI verdadero, integrar
Sentinel Hub o NASA AppEEARS — la firma de este módulo permanece igual.
"""

from __future__ import annotations

import io
import os
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT_PATH     = os.path.join(_PROJECT_ROOT, "data", "satellite_history.csv")

_REGIONS = {
    "us_iowa":   (42.0, -93.5),
    "ar_pampa":  (-33.5, -61.5),
    "br_mt":     (-12.5, -55.5),
    "uy_oeste":  (-32.5, -58.0),
}

_PARAMS = "PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,RH2M,ALLSKY_SFC_SW_DWN"
_BASE   = "https://power.larc.nasa.gov/api/temporal/daily/point"


def _fetch_region(lat: float, lon: float, start: date, end: date,
                  timeout: int = 30) -> Optional[pd.DataFrame]:
    params = {
        "parameters": _PARAMS,
        "community":  "AG",
        "longitude":  lon,
        "latitude":   lat,
        "start":      start.strftime("%Y%m%d"),
        "end":        end.strftime("%Y%m%d"),
        "format":     "JSON",
    }
    try:
        r = requests.get(_BASE, params=params, timeout=timeout)
        r.raise_for_status()
        js = r.json()
    except Exception as e:
        print(f"[NASA POWER] fallo lat={lat} lon={lon}: {e}")
        return None

    p = js.get("properties", {}).get("parameter", {})
    if not p:
        return None
    df = pd.DataFrame(p)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "Date"
    df = df.replace(-999, np.nan).rename(columns={
        "PRECTOTCORR":         "precip_mm",
        "T2M":                 "tmean_c",
        "T2M_MAX":             "tmax_c",
        "T2M_MIN":             "tmin_c",
        "RH2M":                "rh_pct",
        "ALLSKY_SFC_SW_DWN":   "srad",
    })
    return df.reset_index()


def _crop_stress_score(df: pd.DataFrame) -> pd.Series:
    """
    Score simple en [-3, 3]: combina déficit pluvial 30d y desvío T máx.
    + = sequía / calor (bullish soja);  − = exceso de agua / frío.
    """
    precip_30d = df["precip_mm"].rolling(30, min_periods=10).sum()
    p_mean     = precip_30d.expanding(min_periods=60).mean()
    p_std      = precip_30d.expanding(min_periods=60).std()
    p_z        = -((precip_30d - p_mean) / (p_std + 1e-9))  # negativo precip → score+

    t_mean     = df["tmax_c"].expanding(min_periods=60).mean()
    t_std      = df["tmax_c"].expanding(min_periods=60).std()
    t_z        = (df["tmax_c"] - t_mean) / (t_std + 1e-9)

    score = (p_z.fillna(0) + t_z.fillna(0)).clip(-3, 3)
    return score


def fetch_satellite_window(days_back: int = 365) -> Optional[pd.DataFrame]:
    """Descarga ventana reciente para todas las regiones y persiste."""
    end   = date.today() - timedelta(days=2)  # POWER tiene 1-2 días de lag
    start = end - timedelta(days=days_back)

    frames = []
    for name, (lat, lon) in _REGIONS.items():
        df = _fetch_region(lat, lon, start, end)
        if df is None or df.empty:
            continue
        df["region"] = name
        df["crop_stress_score"] = _crop_stress_score(df)
        frames.append(df)

    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["region", "Date"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    out.to_csv(_OUT_PATH, index=False)
    print(f"[NASA POWER] {len(out)} filas, {out['region'].nunique()} regiones -> {_OUT_PATH}")
    return out


def load_satellite_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mergea agregados climáticos por fecha (promedio entre regiones productoras
    pesado por importancia: US 0.35, BR 0.35, AR 0.20, UY 0.10).
    """
    if not os.path.exists(_OUT_PATH):
        return features_df

    try:
        sat = pd.read_csv(_OUT_PATH, parse_dates=["Date"])
    except Exception:
        return features_df

    if sat.empty:
        return features_df

    weights = {"us_iowa": 0.35, "br_mt": 0.35, "ar_pampa": 0.20, "uy_oeste": 0.10}
    sat["w"] = sat["region"].map(weights).fillna(0)

    grouped = sat.groupby("Date")
    agg_rows = []
    for d, g in grouped:
        wsum = g["w"].sum()
        if wsum <= 0:
            continue
        agg_rows.append({
            "Date":              d,
            "sat_precip_mm":     float((g["precip_mm"]    * g["w"]).sum() / wsum),
            "sat_tmax_c":        float((g["tmax_c"]       * g["w"]).sum() / wsum),
            "sat_tmean_c":       float((g["tmean_c"]      * g["w"]).sum() / wsum),
            "sat_rh_pct":        float((g["rh_pct"]       * g["w"]).sum() / wsum),
            "sat_srad":          float((g["srad"]         * g["w"]).sum() / wsum),
            "sat_crop_stress":   float((g["crop_stress_score"] * g["w"]).sum() / wsum),
        })
    if not agg_rows:
        return features_df

    agg = pd.DataFrame(agg_rows).sort_values("Date").reset_index(drop=True)
    agg["sat_precip_30d"]      = agg["sat_precip_mm"].rolling(30, min_periods=5).sum()
    agg["sat_stress_30d_avg"]  = agg["sat_crop_stress"].rolling(30, min_periods=5).mean()

    df = features_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    merged = pd.merge_asof(
        df.sort_values("Date"),
        agg.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    sat_cols = [c for c in agg.columns if c != "Date"]
    merged[sat_cols] = merged[sat_cols].ffill().fillna(0)
    return merged


if __name__ == "__main__":
    out = fetch_satellite_window(365)
    if out is not None:
        print(out.tail())
