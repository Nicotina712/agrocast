"""
src/infra/etl_manifest.py
Bronze/Silver/Gold ETL manifest (Fix #11).

No refactor completo del pipeline — capa de metadatos sobre los datasets
existentes para hacer explícita la calidad/contrato de cada layer.

Layers:
  bronze : raw ingest as-is del proveedor (CSV/JSON crudo)
           ej: data/raw_market.csv, data/cot_soybeans.csv (post-shift PIT)
  silver : limpio, tipado, joinable, point-in-time correcto
           ej: data/features.csv (después de build_features)
  gold   : feature-engineered, listo para training/serving
           ej: data/features.csv después de todos los enrichments + signals.csv

El manifest enumera para cada dataset:
  - layer (bronze/silver/gold)
  - path
  - source / fetcher
  - update_frequency
  - point_in_time (bool — si tiene released_at o shift correcto)
  - schema_cols (lista de columnas críticas)
  - last_updated (mtime)
  - row_count (samplea sin cargar todo)

Uso: from src.infra.etl_manifest import write_manifest, validate_manifest
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MANIFEST_PATH = os.path.join(_PROJECT_ROOT, "data", "etl_manifest.json")


# ── Spec del manifest (single source of truth) ─────────────────────
MANIFEST_SPEC = [
    # Bronze: raw inputs
    {"layer": "bronze", "name": "raw_market",
     "path": "data/raw_market.csv", "source": "yfinance + Investing",
     "frequency": "daily",  "point_in_time": True,
     "schema_cols": ["Date", "Soybeans", "Maize", "Oil"]},
    {"layer": "bronze", "name": "cot_soybeans",
     "path": "data/cot_soybeans.csv", "source": "CFTC weekly",
     "frequency": "weekly (Fri release)", "point_in_time": True,
     "schema_cols": ["Date", "cot_commercial_net", "cot_index"]},
    {"layer": "bronze", "name": "usda_inspections",
     "path": "data/usda_inspections.csv", "source": "USDA-FGIS",
     "frequency": "weekly (Thu)", "point_in_time": True,
     "schema_cols": ["Date", "insp_soy_lbs", "insp_soy_yoy"]},
    {"layer": "bronze", "name": "crop_progress",
     "path": "data/crop_progress.csv", "source": "USDA NASS",
     "frequency": "weekly (Mon release)", "point_in_time": True,
     "schema_cols": ["Date", "good_excellent_pct"]},
    {"layer": "bronze", "name": "wasde_official",
     "path": "data/wasde_official.json", "source": "USDA FAS PSD / static fallback",
     "frequency": "monthly", "point_in_time": False,
     "schema_cols": ["world", "argentina", "brazil", "usa"]},
    {"layer": "bronze", "name": "cme_history",
     "path": "data/cme_history.csv", "source": "CME public + Yahoo fallback",
     "frequency": "daily", "point_in_time": True,
     "schema_cols": ["Date", "front_settle", "total_oi", "front_volume"]},
    {"layer": "bronze", "name": "cvol_history",
     "path": "data/cvol_history.csv", "source": "Yahoo options chain",
     "frequency": "daily", "point_in_time": True,
     "schema_cols": ["Date", "iv_atm", "iv_call", "iv_put"]},
    {"layer": "bronze", "name": "satellite_history",
     "path": "data/satellite_history.csv", "source": "NASA POWER",
     "frequency": "5-7 days", "point_in_time": True,
     "schema_cols": ["Date", "precip_anom", "temp_anom"]},
    {"layer": "bronze", "name": "climate_macro",
     "path": "data/climate_macro.csv", "source": "NOAA ONI + drought proxy",
     "frequency": "monthly", "point_in_time": True,
     "schema_cols": ["Date", "enso_oni", "enso_phase"]},
    {"layer": "bronze", "name": "news_sentiment_history",
     "path": "data/news_sentiment_history.csv", "source": "RSS + LLM",
     "frequency": "intraday", "point_in_time": True,
     "schema_cols": ["Date", "news_sentiment", "news_volume"]},

    # Silver/Gold: built artifacts
    {"layer": "gold", "name": "features",
     "path": "data/features.csv", "source": "src/pipeline.py",
     "frequency": "on pipeline run", "point_in_time": True,
     "schema_cols": ["Date", "Soybeans", "ret_14d_fwd", "rsi_14",
                     "cot_index", "days_to_wasde", "event_intensity",
                     "cvol_iv_atm", "enso_oni"]},
    {"layer": "gold", "name": "signals",
     "path": "artifacts/signals.csv", "source": "src/model/predict_returns.py",
     "frequency": "on pipeline run", "point_in_time": True,
     "schema_cols": ["Date", "expected_return", "signal", "confidence",
                     "expected_vol"]},
]


def _stat_dataset(path: str) -> dict:
    full = os.path.join(_PROJECT_ROOT, path)
    if not os.path.exists(full):
        return {"exists": False, "row_count": None, "last_updated": None,
                "size_kb": None}
    size = os.path.getsize(full)
    mtime = datetime.fromtimestamp(os.path.getmtime(full)).isoformat(timespec="seconds")
    row_count = None
    try:
        if path.endswith(".csv"):
            # Conteo eficiente sin pandas
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                row_count = sum(1 for _ in f) - 1
        elif path.endswith(".json"):
            row_count = 1
    except Exception:
        pass
    return {
        "exists":       True,
        "row_count":    row_count,
        "last_updated": mtime,
        "size_kb":      round(size / 1024, 1),
    }


def write_manifest() -> dict:
    """Genera/actualiza data/etl_manifest.json con stats actuales de cada dataset."""
    entries = []
    for spec in MANIFEST_SPEC:
        stat = _stat_dataset(spec["path"])
        entries.append({**spec, **stat})

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_datasets":   len(entries),
        "n_missing":    sum(1 for e in entries if not e.get("exists")),
        "by_layer": {
            "bronze": [e for e in entries if e["layer"] == "bronze"],
            "silver": [e for e in entries if e["layer"] == "silver"],
            "gold":   [e for e in entries if e["layer"] == "gold"],
        },
    }
    os.makedirs(os.path.dirname(_MANIFEST_PATH), exist_ok=True)
    with open(_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"   [ETL] Manifest → {_MANIFEST_PATH} | "
          f"{manifest['n_datasets']} datasets, {manifest['n_missing']} faltantes")
    return manifest


def validate_manifest() -> list:
    """Devuelve lista de warnings sobre datasets stale o faltantes."""
    warnings = []
    now = datetime.now()
    for spec in MANIFEST_SPEC:
        stat = _stat_dataset(spec["path"])
        if not stat["exists"]:
            warnings.append(f"[MISSING] {spec['name']} ({spec['path']})")
            continue
        try:
            age_days = (now - datetime.fromisoformat(stat["last_updated"])).days
        except Exception:
            continue
        # Heurística de stale por frecuencia
        freq = spec.get("frequency", "")
        max_age = 60
        if "daily" in freq:    max_age = 5
        elif "weekly" in freq: max_age = 14
        elif "monthly" in freq:max_age = 45
        if age_days > max_age:
            warnings.append(f"[STALE] {spec['name']}: {age_days}d sin actualizar "
                            f"(frecuencia esperada: {freq})")
    return warnings


if __name__ == "__main__":
    write_manifest()
    for w in validate_manifest():
        print(w)
