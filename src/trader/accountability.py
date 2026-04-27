"""
src/trader/accountability.py
Compara forecasts pasados vs precios reales para medir accuracy del modelo.
Guarda snapshots en data/forecast_snapshots.json.
"""

import json
import os
from datetime import date, timedelta

import pandas as pd

_PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SNAPSHOTS_PATH = os.path.join(_PROJECT_ROOT, "data", "forecast_snapshots.json")
_HORIZONS       = [7, 14, 30]


def save_forecast_snapshot(
    price_now: float,
    signal: str,
    forecast_df: pd.DataFrame,
    score: int = None,
) -> None:
    """
    Guarda un snapshot del forecast actual. Se llama desde pipeline.py.
    forecast_df debe tener columnas Date y Soybeans.
    """
    snapshots = _load_snapshots()
    today_str = date.today().isoformat()

    if any(s["snapshot_date"] == today_str for s in snapshots):
        return

    snap = {
        "snapshot_date":    today_str,
        "price_at_snapshot": round(price_now, 2),
        "signal":           signal,
        "score":            score,
        "horizons":         {},
    }

    fc_dates  = pd.to_datetime(forecast_df["Date"])
    fc_prices = forecast_df["Soybeans"].values
    today_ts  = pd.Timestamp(today_str)

    for h in _HORIZONS:
        target_ts = today_ts + pd.Timedelta(days=h)
        diffs = (fc_dates - target_ts).abs()
        if len(diffs) == 0:
            continue
        idx = int(diffs.argmin())
        snap["horizons"][str(h)] = {
            "target_date":    fc_dates.iloc[idx].date().isoformat(),
            "forecast_price": round(float(fc_prices[idx]), 2),
        }

    snapshots.append(snap)
    _save_snapshots(snapshots[-365:])
    print(f"   [Accountability] Snapshot guardado: {today_str} | {signal} | ${price_now:.0f}")


def get_accountability_records() -> dict:
    """
    Retorna snapshots con precios reales y métricas de error donde disponibles.
    """
    snapshots = _load_snapshots()
    if not snapshots:
        return {"ok": True, "records": [], "summary": {}, "message": "Sin snapshots aún — se acumulan con cada ejecución del pipeline."}

    actual_prices = _load_actual_prices()
    records = []

    for snap in reversed(snapshots[-90:]):
        rec = {
            "snapshot_date":    snap["snapshot_date"],
            "price_at_snapshot": snap.get("price_at_snapshot"),
            "signal":           snap.get("signal", "—"),
            "score":            snap.get("score"),
            "horizons":         {},
        }

        for h_str, h_data in snap.get("horizons", {}).items():
            target_date    = h_data["target_date"]
            forecast_price = h_data["forecast_price"]
            actual         = actual_prices.get(target_date)

            h_rec = {
                "target_date":    target_date,
                "forecast_price": forecast_price,
                "actual_price":   actual,
                "available":      actual is not None,
            }

            if actual and snap.get("price_at_snapshot"):
                base = snap["price_at_snapshot"]
                h_rec["error_usc"]        = round(actual - forecast_price, 1)
                h_rec["error_pct"]        = round((actual - forecast_price) / forecast_price * 100, 2)
                h_rec["abs_error_pct"]    = round(abs(h_rec["error_pct"]), 2)
                fc_dir                    = forecast_price > base
                ac_dir                    = actual > base
                h_rec["direction_correct"] = fc_dir == ac_dir

            rec["horizons"][h_str] = h_rec

        records.append(rec)

    return {
        "ok":      True,
        "records": records,
        "summary": _compute_summary(records),
        "total_snapshots": len(snapshots),
    }


def _compute_summary(records: list) -> dict:
    summary = {}
    for h in _HORIZONS:
        h_str  = str(h)
        errors = []
        dirs   = []
        for r in records:
            hd = r["horizons"].get(h_str, {})
            if hd.get("available"):
                errors.append(hd["abs_error_pct"])
                if "direction_correct" in hd:
                    dirs.append(hd["direction_correct"])

        summary[f"mae_pct_{h}d"]       = round(sum(errors) / len(errors), 2) if errors else None
        summary[f"dir_accuracy_{h}d"]  = round(sum(dirs) / len(dirs) * 100, 1) if dirs else None
        summary[f"n_evaluated_{h}d"]   = len(errors)

    return summary


def _load_actual_prices() -> dict:
    raw_path = os.path.join(_PROJECT_ROOT, "data", "raw_market.csv")
    if not os.path.exists(raw_path):
        return {}
    try:
        df = pd.read_csv(raw_path, parse_dates=["Date"]).dropna(subset=["Soybeans"])
        return {row["Date"].date().isoformat(): round(float(row["Soybeans"]), 2) for _, row in df.iterrows()}
    except Exception:
        return {}


def _load_snapshots() -> list:
    if not os.path.exists(_SNAPSHOTS_PATH):
        return []
    try:
        with open(_SNAPSHOTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_snapshots(snapshots: list) -> None:
    os.makedirs(os.path.dirname(_SNAPSHOTS_PATH), exist_ok=True)
    with open(_SNAPSHOTS_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshots, f, ensure_ascii=False, indent=2)
