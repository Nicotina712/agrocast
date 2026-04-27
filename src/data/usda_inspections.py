"""
src/data/usda_inspections.py
Descarga y agrega el reporte semanal de Export Inspections del USDA-FGIS
para SOYBEANS. Los datos se cachen en data/usda_inspections.csv.

Fuente pública: https://fgisonline.ams.usda.gov/ExportGrainReport/CYxxxx.csv
- Una fila por certificado de embarque
- Columna 'Cert Date': YYYYMMDD (int)
- Columna 'Grain': 'SOYBEANS', 'CORN', etc.
- Columna 'Pounds': libras inspeccionadas

Salida: DataFrame con columnas:
  Date (datetime), insp_soy_lbs (float), insp_soy_bu (float),
  insp_soy_4wk_avg (float), insp_soy_yoy (float)
"""

import io
import os
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

CACHE_PATH  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "usda_inspections.csv",
)
BASE_URL    = "https://fgisonline.ams.usda.gov/ExportGrainReport/CY{year}.csv"
POUNDS_PER_BUSHEL = 60.0  # soja: 60 lb/bu (estándar USDA)


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features derivadas sobre un DataFrame con columnas Date + insp_soy_lbs."""
    df = df.copy()
    df["insp_soy_bu"]      = df["insp_soy_lbs"] / POUNDS_PER_BUSHEL
    df["insp_soy_4wk_avg"] = df["insp_soy_bu"].rolling(4, min_periods=1).mean()
    df["insp_soy_yoy"] = (
        df["insp_soy_bu"] / df["insp_soy_bu"].shift(52).replace(0, np.nan) - 1
    ).fillna(0).clip(-1, 3)
    roll_mean = df["insp_soy_bu"].rolling(52, min_periods=10).mean()
    roll_std  = df["insp_soy_bu"].rolling(52, min_periods=10).std().replace(0, 1)
    df["insp_soy_zscore"] = ((df["insp_soy_bu"] - roll_mean) / roll_std).fillna(0).clip(-3, 3)
    return df


def _fetch_year(year: int, retries: int = 2) -> pd.DataFrame:
    """Descarga export inspections de un año. Retry en IncompleteRead/timeout."""
    url = BASE_URL.format(year=year)
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.urlopen(url, timeout=30)
            df = pd.read_csv(io.BytesIO(req.read()), low_memory=False)
            break
        except Exception as e:
            last_err = e
            if attempt < retries:
                continue
            print(f"   [WARN] USDA inspections {year} (tras {retries+1} intentos): {e}")
            return pd.DataFrame()

    if "Grain" not in df.columns or "Pounds" not in df.columns:
        return pd.DataFrame()

    soy = df[df["Grain"].astype(str).str.upper() == "SOYBEANS"].copy()
    if soy.empty:
        return pd.DataFrame()

    soy["Cert Date"] = pd.to_numeric(soy["Cert Date"], errors="coerce")
    soy = soy.dropna(subset=["Cert Date"])
    soy["Date"] = pd.to_datetime(soy["Cert Date"].astype(int).astype(str), format="%Y%m%d", errors="coerce")
    soy = soy.dropna(subset=["Date"])

    soy["Pounds"] = pd.to_numeric(soy["Pounds"], errors="coerce").fillna(0)

    weekly = (
        soy.groupby(pd.Grouper(key="Date", freq="W-THU"))["Pounds"]
        .sum()
        .reset_index()
        .rename(columns={"Pounds": "insp_soy_lbs"})
    )
    weekly = weekly[weekly["insp_soy_lbs"] > 0]
    return weekly


def load_usda_inspections(
    start_year: int | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Devuelve DataFrame semanal de Export Inspections de soja con features derivadas.
    Descarga incremental: si existe caché con historial completo, solo actualiza
    el año actual en lugar de descargar todo desde start_year.

    Parámetros
    ----------
    start_year : primer año a descargar en descarga completa (por defecto: hace 10 años)
    refresh    : forzar descarga completa aunque exista caché
    """
    if start_year is None:
        start_year = datetime.now().year - 10

    current_year = datetime.now().year

    # ── Intentar actualización incremental (solo año actual) ─────────────
    if not refresh and os.path.exists(CACHE_PATH):
        cached = pd.read_csv(CACHE_PATH, parse_dates=["Date"])
        if not cached.empty:
            last = cached["Date"].max()
            days_old = (datetime.now() - last.to_pydatetime().replace(tzinfo=None)).days
            # Caché reciente (< 7 días) → sin actualizar
            if days_old < 7:
                print(f"   [USDA] Caché válido — último: {last.date()} ({days_old}d)")
                return _compute_features(cached)
            # Caché con historial completo → solo refrescar año actual
            cache_start_year = cached["Date"].min().year
            if cache_start_year <= start_year + 1:
                print(f"   [USDA] Actualizando solo {current_year} (caché desde {cache_start_year})…")
                df_curr = _fetch_year(current_year)
                if not df_curr.empty:
                    print(f"      {current_year}: {len(df_curr)} semanas OK")
                    merged = pd.concat([
                        cached[cached["Date"].dt.year < current_year][["Date", "insp_soy_lbs"]],
                        df_curr,
                    ]).sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
                    df = _compute_features(merged)
                    df.to_csv(CACHE_PATH, index=False)
                    print(f"   [USDA] {len(df)} semanas guardadas → {CACHE_PATH}")
                    return df
                # Si el incremental fallo (red caida o IncompleteRead), NO bajamos
                # 11 años — devolvemos el cache que ya tenemos. El proximo refresh
                # (6h despues) intentara de nuevo.
                print(f"   [USDA] Incremental fallo — usando caché existente ({len(cached)} semanas)")
                return _compute_features(cached)

    # ── Descarga completa (primera vez o caché incompleto) ───────────────
    print(f"   [USDA] Descargando Export Inspections {start_year}–{current_year}…")
    frames = []
    for yr in range(start_year, current_year + 1):
        df_yr = _fetch_year(yr)
        if not df_yr.empty:
            frames.append(df_yr)
            print(f"      {yr}: {len(df_yr)} semanas OK")

    if not frames:
        print("   [USDA] Sin datos descargados")
        return pd.DataFrame(columns=["Date", "insp_soy_lbs", "insp_soy_bu",
                                     "insp_soy_4wk_avg", "insp_soy_yoy"])

    df = pd.concat(frames).sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    df = _compute_features(df)

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    print(f"   [USDA] {len(df)} semanas guardadas → {CACHE_PATH}")
    return df
