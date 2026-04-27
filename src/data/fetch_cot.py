"""
src/data/fetch_cot.py
Descarga el reporte COT (Commitment of Traders) de la CFTC.

Fuente: CFTC Disaggregated Futures-Only Report (históricos 2021-2024)
URL: https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip

Features generados (semanales, interpolados a diario):
  - cot_commercial_net   : posición neta comerciales (hedgers) — contrarian signal
  - cot_noncomm_net      : posición neta money managers (especuladores)
  - cot_noncomm_long_pct : % long de money managers sobre open interest
  - cot_index            : COT Index (0-100): percentil 52 semanas de noncomm_net

El COT Index cerca de 100 = especuladores muy largos → señal bajista contrarian.
El COT Index cerca de 0   = especuladores muy cortos → señal alcista contrarian.
"""

import os
import io
import zipfile
import numpy as np
import pandas as pd
import requests
from datetime import date

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_PATH   = os.path.join(_PROJECT_ROOT, "data", "cot_soybeans.csv")
_CACHE_DAYS   = 7

_SOY_MARKET = "SOYBEANS - CHICAGO BOARD OF TRADE"

_HIST_YEARS = [2021, 2022, 2023, 2024, 2025]
_URL_TEMPLATE = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"


def _download_year(year: int) -> pd.DataFrame:
    url = _URL_TEMPLATE.format(year=year)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            txt = [n for n in z.namelist() if n.endswith(".txt")][0]
            with z.open(txt) as f:
                df = pd.read_csv(f, low_memory=False)
        soy = df[df["Market_and_Exchange_Names"] == _SOY_MARKET].copy()
        print(f"[COT] {year}: {len(soy)} filas de soybeans")
        return soy
    except Exception as e:
        print(f"[COT] Error {year}: {e}")
        return pd.DataFrame()


def _parse_cot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    col_map = {
        "Report_Date_as_YYYY-MM-DD":   "Date",
        "Prod_Merc_Positions_Long_All":  "comm_long",
        "Prod_Merc_Positions_Short_All": "comm_short",
        "M_Money_Positions_Long_All":    "mm_long",
        "M_Money_Positions_Short_All":   "mm_short",
        "Open_Interest_All":             "open_interest",
    }
    present = {k: v for k, v in col_map.items() if k in df.columns}
    df = df[list(present.keys())].rename(columns=present).copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["comm_long", "comm_short", "mm_long", "mm_short", "open_interest"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    df["cot_commercial_net"]   = df["comm_long"] - df["comm_short"]
    df["cot_noncomm_net"]      = df["mm_long"] - df["mm_short"]
    df["cot_noncomm_long_pct"] = (
        df["mm_long"] / df["open_interest"].replace(0, np.nan) * 100
    )

    # COT Index: percentil rodante 52 semanas
    df["cot_index"] = (
        df["cot_noncomm_net"]
        .rolling(52, min_periods=10)
        .apply(
            lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-9) * 100,
            raw=True,
        )
    )

    return df[["Date", "cot_commercial_net", "cot_noncomm_net",
               "cot_noncomm_long_pct", "cot_index"]].copy()


def fetch_cot_soybeans(force_refresh: bool = False) -> pd.DataFrame:
    """
    Retorna DataFrame diario (interpolado) con features COT de soybeans.
    Usa caché local; refresca si tiene más de _CACHE_DAYS días.
    """
    if os.path.exists(_CACHE_PATH) and not force_refresh:
        mtime = date.fromtimestamp(os.path.getmtime(_CACHE_PATH))
        if (date.today() - mtime).days < _CACHE_DAYS:
            df = pd.read_csv(_CACHE_PATH, parse_dates=["Date"])
            print(f"[COT] Cache vigente: {len(df)} filas hasta {df['Date'].max().date()}")
            return df

    print("[COT] Descargando datos CFTC...")
    frames = []
    for year in _HIST_YEARS:
        raw = _download_year(year)
        if not raw.empty:
            frames.append(raw)

    if not frames:
        print("[COT] Sin datos disponibles")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    weekly   = _parse_cot(combined)

    if weekly.empty:
        return pd.DataFrame()

    weekly = weekly.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)

    # ── Point-in-time: COT data es Tuesday-as-of, publicado el viernes 15:30 ET.
    # Shift Date → released_at (= Tuesday + 3 días = viernes) para que el merge
    # con features sólo aporte info verdaderamente disponible ese día.
    weekly["as_of_date"] = weekly["Date"]
    weekly["Date"]       = weekly["Date"] + pd.Timedelta(days=3)
    weekly["released_at"] = weekly["Date"]

    # Interpolar a frecuencia diaria (a partir del primer release real)
    date_range = pd.date_range(weekly["Date"].min(), date.today(), freq="D")
    daily = weekly.drop(columns=["as_of_date", "released_at"]).set_index("Date").reindex(date_range).interpolate("linear")
    daily.index.name = "Date"
    daily = daily.reset_index()

    cot_cols = ["cot_commercial_net", "cot_noncomm_net", "cot_noncomm_long_pct", "cot_index"]
    daily[cot_cols] = daily[cot_cols].ffill().bfill().fillna(0)

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    daily.to_csv(_CACHE_PATH, index=False)
    print(f"[COT] {len(daily)} filas — {daily['Date'].min().date()} → {daily['Date'].max().date()}")

    return daily
