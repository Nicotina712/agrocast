"""
src/data/fetch_crop_progress.py
USDA NASS Crop Progress weekly para soja (US).

Reporta cada lunes (durante temporada) desde NASS Quick Stats API:
  - % planted
  - % emerged
  - % blooming
  - % setting pods
  - % dropping leaves
  - % harvested
  - % condition (excellent/good/fair/poor/very poor) → good_excellent_pct

API pública: https://quickstats.nass.usda.gov/api
Requiere API key (gratis): https://quickstats.nass.usda.gov/api/api_key

Si la env var USDA_API_KEY no está, devuelve None silenciosamente —
el resto del pipeline sigue funcionando.

Salida: data/crop_progress.csv (fecha semanal del reporte + métricas).
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Optional

import pandas as pd
import requests

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT_PATH     = os.path.join(_PROJECT_ROOT, "data", "crop_progress.csv")

_API_BASE = "https://quickstats.nass.usda.gov/api/api_GET/"

_STATISTICCATS = {
    "PCT PLANTED":         "pct_planted",
    "PCT EMERGED":         "pct_emerged",
    "PCT BLOOMING":        "pct_blooming",
    "PCT SETTING PODS":    "pct_setting_pods",
    "PCT DROPPING LEAVES": "pct_dropping_leaves",
    "PCT HARVESTED":       "pct_harvested",
}

_CONDITION_CATS = ["EXCELLENT", "GOOD", "FAIR", "POOR", "VERY POOR"]


def _api_key() -> Optional[str]:
    return os.environ.get("USDA_API_KEY")


def _fetch_param(api_key: str, year: int, statisticcat: str,
                 short_desc_filter: str = "SOYBEANS") -> pd.DataFrame:
    params = {
        "key":             api_key,
        "commodity_desc":  "SOYBEANS",
        "statisticcat_desc": statisticcat,
        "agg_level_desc":  "NATIONAL",
        "year":            year,
        "format":          "JSON",
    }
    try:
        # Timeout corto y separado (connect, read) para que no bloquee el pipeline
        r = requests.get(_API_BASE, params=params, timeout=(8, 12))
        r.raise_for_status()
        js = r.json()
    except Exception as e:
        print(f"[USDA Crop Progress] {statisticcat} {year} fallo: {e}")
        return pd.DataFrame()

    rows = js.get("data", [])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def fetch_crop_progress(years_back: int = 3) -> Optional[pd.DataFrame]:
    key = _api_key()
    if not key:
        print("[USDA Crop Progress] USDA_API_KEY no seteada — skip.")
        return None

    cur_year = date.today().year
    years    = list(range(cur_year - years_back + 1, cur_year + 1))

    progress_frames = []
    for y in years:
        # PROGRESS devuelve TODAS las categorías en una sola llamada — no loopear
        df = _fetch_param(key, y, "PROGRESS")
        if df.empty:
            continue
        for cat, col in _STATISTICCATS.items():
            short_match = df["short_desc"].str.contains(
                cat.replace("PCT ", ""), case=False, na=False
            )
            sub = df[short_match].copy()
            if sub.empty:
                continue
            sub["Date"]  = pd.to_datetime(sub.get("week_ending"), errors="coerce")
            sub["value"] = pd.to_numeric(
                sub["Value"].astype(str).str.replace(",", ""), errors="coerce"
            )
            progress_frames.append(sub[["Date", "value"]].assign(metric=col))

    # Condition (good/excellent/etc.)
    # NASS devuelve 400 para CONDITION en años sin temporada activa todavía
    # (ej. enero-abril del año en curso, antes que el cultivo emerja).
    # En ese caso, caemos al año previo para no quedarnos sin la métrica.
    cond_frames = []
    cond_years_to_try = list(years)
    if cur_year in cond_years_to_try:
        # Si current year falla, agregamos cur_year-1 al fallback (sin duplicar)
        prev = cur_year - 1
        if prev not in cond_years_to_try:
            cond_years_to_try.append(prev)

    cond_year_succeeded = False
    for y in cond_years_to_try:
        df = _fetch_param(key, y, "CONDITION")
        if df.empty:
            if y == cur_year:
                print(f"[USDA Crop Progress] CONDITION {y} sin datos aun -> probando {cur_year-1}")
            continue
        cond_year_succeeded = True
        df["Date"]  = pd.to_datetime(df.get("week_ending"), errors="coerce")
        df["value"] = pd.to_numeric(
            df["Value"].astype(str).str.replace(",", ""), errors="coerce"
        )

        for cat in _CONDITION_CATS:
            mask = df["short_desc"].str.contains(cat, case=False, na=False)
            sub  = df[mask][["Date", "value"]].copy()
            if sub.empty:
                continue
            sub["metric"] = "cond_" + cat.lower().replace(" ", "_")
            cond_frames.append(sub)

    all_frames = progress_frames + cond_frames
    if not all_frames:
        return None

    long_df = pd.concat(all_frames, ignore_index=True).dropna(subset=["Date", "value"])
    wide    = long_df.pivot_table(index="Date", columns="metric",
                                   values="value", aggfunc="last").reset_index()

    # ── Point-in-time: Crop Progress refiere a la semana terminando domingo
    # pero se publica el lunes 16:00 ET. released_at = Date (week_ending) + 1 día.
    wide["week_ending"] = wide["Date"]
    wide["Date"]        = wide["Date"] + pd.Timedelta(days=1)
    wide["released_at"] = wide["Date"]

    if "cond_good" in wide.columns and "cond_excellent" in wide.columns:
        wide["good_excellent_pct"] = wide["cond_good"].fillna(0) + wide["cond_excellent"].fillna(0)
    if "cond_poor" in wide.columns and "cond_very_poor" in wide.columns:
        wide["poor_very_poor_pct"] = wide["cond_poor"].fillna(0) + wide["cond_very_poor"].fillna(0)

    wide = wide.sort_values("Date").reset_index(drop=True)

    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    wide.to_csv(_OUT_PATH, index=False)
    print(f"[USDA Crop Progress] {len(wide)} semanas -> {_OUT_PATH}")
    return wide


def load_crop_progress_features(features_df: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(_OUT_PATH):
        return features_df

    try:
        cp = pd.read_csv(_OUT_PATH, parse_dates=["Date"])
    except Exception:
        return features_df

    if cp.empty:
        return features_df

    keep_cols = [c for c in cp.columns if c == "Date" or c.startswith(("pct_", "good_", "poor_", "cond_"))]
    cp = cp[keep_cols].copy()
    # Si el CSV es viejo y no tiene released_at, asumimos que Date ya fue shifted en el último fetch.

    rename = {c: ("cp_" + c) for c in cp.columns if c != "Date" and not c.startswith("cp_")}
    cp = cp.rename(columns=rename)

    if "cp_good_excellent_pct" in cp.columns:
        cp["cp_condition_change_4w"] = cp["cp_good_excellent_pct"].diff(4)

    df = features_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    merged = pd.merge_asof(
        df.sort_values("Date"),
        cp.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    cp_cols = [c for c in cp.columns if c != "Date"]
    merged[cp_cols] = merged[cp_cols].ffill().fillna(0)
    return merged


if __name__ == "__main__":
    out = fetch_crop_progress(2)
    if out is not None:
        print(out.tail())
