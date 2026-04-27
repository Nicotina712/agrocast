"""
src/trader/curve_features.py
Features de curva de futuros (term structure) para el modelo.

Cada vez que se llama a snapshot_curve(), persiste un row en
data/curve_history.csv con:
  - front_price_usc, next_price_usc
  - front_next_spread       (next - front, USc/bu)
  - front_next_spread_pct   (spread / front)
  - roll_yield_annualized   ((front-next)/front × 365/days)
  - structure               CONTANGO/BACKWARDATION/FLAT
  - days_between            días entre vencimientos

load_curve_features(features_df) hace ffill al merge para que el modelo
vea el último estado conocido de la curva en cada fila.
"""

import os
from datetime import date, datetime
from typing import Optional

import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CURVE_PATH   = os.path.join(_PROJECT_ROOT, "data", "curve_history.csv")


def _structure_label(spread_pct: float) -> str:
    if spread_pct > 0.005:
        return "CONTANGO"
    if spread_pct < -0.005:
        return "BACKWARDATION"
    return "FLAT"


def snapshot_curve() -> Optional[dict]:
    """Toma snapshot del front-month vs next-month y persiste en curve_history.csv."""
    try:
        from src.trader.term_structure import fetch_term_structure
    except Exception:
        return None

    ts = fetch_term_structure()
    if not ts or not isinstance(ts, dict) or not ts.get("contracts"):
        return None

    contracts = ts["contracts"]
    if len(contracts) < 2:
        return None

    front, nxt = contracts[0], contracts[1]
    front_price = float(front["price_usd_bu"]) * 100  # USD/bu → USc/bu
    next_price  = float(nxt["price_usd_bu"]) * 100
    days_between = max(1, int(nxt["days_to_expiry"]) - int(front["days_to_expiry"]))

    spread     = next_price - front_price
    spread_pct = spread / front_price if front_price > 0 else 0.0
    # roll yield anualizado: cuánto rinde "rolar" el contrato (signo: +backwardation gana)
    roll_yield_annual = (-spread / front_price) * (365.0 / days_between) if front_price > 0 else 0.0
    structure = _structure_label(spread_pct)

    row = {
        "Date":                  date.today().isoformat(),
        "front_price_usc":       round(front_price, 2),
        "next_price_usc":        round(next_price, 2),
        "front_next_spread":     round(spread, 2),
        "front_next_spread_pct": round(spread_pct, 5),
        "roll_yield_annual":     round(roll_yield_annual, 5),
        "days_between":          days_between,
        "structure":             structure,
    }

    # Append idempotente por fecha
    os.makedirs(os.path.dirname(_CURVE_PATH), exist_ok=True)
    if os.path.exists(_CURVE_PATH):
        try:
            hist = pd.read_csv(_CURVE_PATH)
            hist = hist[hist["Date"] != row["Date"]]
            hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
        except Exception:
            hist = pd.DataFrame([row])
    else:
        hist = pd.DataFrame([row])
    hist = hist.sort_values("Date").tail(1500)
    hist.to_csv(_CURVE_PATH, index=False)
    return row


def load_curve_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Mergea curve_history al features_df con ffill. Si no hay historial, agrega ceros."""
    cols = ["front_next_spread", "front_next_spread_pct",
            "roll_yield_annual", "days_between"]
    if not os.path.exists(_CURVE_PATH):
        for c in cols:
            features_df[c] = 0.0
        features_df["curve_contango"]     = 0
        features_df["curve_backwardation"] = 0
        return features_df

    try:
        hist = pd.read_csv(_CURVE_PATH, parse_dates=["Date"])
    except Exception:
        for c in cols:
            features_df[c] = 0.0
        features_df["curve_contango"]     = 0
        features_df["curve_backwardation"] = 0
        return features_df

    features_df["Date"] = pd.to_datetime(features_df["Date"])
    merged = features_df.merge(hist[["Date"] + cols + ["structure"]], on="Date", how="left")
    for c in cols:
        merged[c] = merged[c].ffill().fillna(0.0)
    # astype(object) explícito evita el FutureWarning de downcasting en pandas 2.2+
    merged["structure"] = merged["structure"].astype(object).ffill().fillna("FLAT")
    merged["curve_contango"]      = (merged["structure"] == "CONTANGO").astype(int)
    merged["curve_backwardation"] = (merged["structure"] == "BACKWARDATION").astype(int)
    merged = merged.drop(columns=["structure"])
    return merged


if __name__ == "__main__":
    print(snapshot_curve())
