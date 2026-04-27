"""
src/intel/llm_accountability.py
Snapshot diario del stance del Brief LLM y verificación a 7 días.

Cada llamada a `record_today()`:
  - Lee data/market_synthesis.json
  - Toma precio actual de data/raw_market.csv
  - Append a data/llm_snapshots.json (1 por día, dedupe por fecha)

`evaluate_due()` revisa snapshots con >=7 días y los marca como hit/miss
comparando stance con el retorno realizado a 7 días.

`get_summary()` devuelve estadísticas agregadas para mostrar en dashboard.
"""

import json
import os
from datetime import date, datetime, timedelta

import pandas as pd

_PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SNAP_PATH     = os.path.join(_PROJECT_ROOT, "data", "llm_snapshots.json")
_SYN_PATH      = os.path.join(_PROJECT_ROOT, "data", "market_synthesis.json")
_RAW_PATH      = os.path.join(_PROJECT_ROOT, "data", "raw_market.csv")
_HORIZON_DAYS  = 14
_NEUTRAL_BAND  = 0.015  # ±1.5% en 14d → "sin movimiento" (NEUTRAL acierta si cae acá)


def _load() -> list:
    if not os.path.exists(_SNAP_PATH):
        return []
    try:
        with open(_SNAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save(snaps: list) -> None:
    os.makedirs(os.path.dirname(_SNAP_PATH), exist_ok=True)
    with open(_SNAP_PATH, "w", encoding="utf-8") as f:
        json.dump(snaps[-365:], f, ensure_ascii=False, indent=2)


def _current_price() -> float | None:
    if not os.path.exists(_RAW_PATH):
        return None
    try:
        df = pd.read_csv(_RAW_PATH, parse_dates=["Date"]).sort_values("Date")
        return float(df["Soybeans"].iloc[-1])
    except Exception:
        return None


def record_today() -> dict | None:
    """Guarda snapshot del Brief LLM de hoy (idempotente por día)."""
    if not os.path.exists(_SYN_PATH):
        return None
    try:
        with open(_SYN_PATH, "r", encoding="utf-8") as f:
            syn = json.load(f)
    except Exception:
        return None

    stance     = syn.get("stance")
    conviction = syn.get("conviction")
    headline   = syn.get("headline")
    if not stance:
        return None

    price = _current_price()
    if price is None:
        return None

    today = date.today().isoformat()
    snaps = _load()
    if any(s["date"] == today for s in snaps):
        return next(s for s in snaps if s["date"] == today)

    snap = {
        "date":           today,
        "stance":         stance,
        "conviction":     conviction,
        "headline":       headline,
        "price_at_snapshot": round(price, 2),
        "verified":       False,
        "hit":            None,
        "ret_pct":        None,
        "verified_at":    None,
    }
    snaps.append(snap)
    _save(snaps)
    return snap


def evaluate_due() -> int:
    """Verifica snapshots con >=7 días contra retorno realizado. Devuelve cuántos verificó."""
    snaps = _load()
    if not snaps:
        return 0
    if not os.path.exists(_RAW_PATH):
        return 0

    df = pd.read_csv(_RAW_PATH, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    today = pd.Timestamp(date.today())
    n_done = 0

    for s in snaps:
        if s.get("verified"):
            continue
        snap_date = pd.Timestamp(s["date"])
        if (today - snap_date).days < _HORIZON_DAYS:
            continue

        target = snap_date + pd.Timedelta(days=_HORIZON_DAYS)
        future = df[df["Date"] >= target]
        if future.empty:
            continue
        price_then = float(future["Soybeans"].iloc[0])
        ret = (price_then / s["price_at_snapshot"]) - 1.0

        stance = s["stance"]
        if stance == "ALCISTA":
            hit = ret > _NEUTRAL_BAND
        elif stance == "BAJISTA":
            hit = ret < -_NEUTRAL_BAND
        else:  # NEUTRAL
            hit = abs(ret) <= _NEUTRAL_BAND

        s["verified"]    = True
        s["hit"]         = bool(hit)
        s["ret_pct"]     = round(ret * 100, 2)
        s["price_at_horizon"] = round(price_then, 2)
        s["verified_at"] = datetime.now().isoformat()
        n_done += 1

    if n_done:
        _save(snaps)
    return n_done


def get_summary() -> dict:
    """Resumen para dashboard."""
    evaluate_due()  # auto-verificar lo pendiente
    snaps = _load()
    verified = [s for s in snaps if s.get("verified")]
    hits     = sum(1 for s in verified if s.get("hit"))
    n_v      = len(verified)
    return {
        "ok":           True,
        "total":        len(snaps),
        "verified":     n_v,
        "pending":      len(snaps) - n_v,
        "hits":         hits,
        "hit_rate":     round(hits / n_v * 100, 1) if n_v else None,
        "recent":       snaps[-10:][::-1],
    }


if __name__ == "__main__":
    print("record_today:", record_today())
    print("evaluate_due:", evaluate_due())
    print(json.dumps(get_summary(), indent=2, ensure_ascii=False))
