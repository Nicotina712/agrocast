"""
src/intel/debate_repository.py
Persistent repository of multi-agent debates with ex-post evaluation.

Each debate is saved as a JSONL record in data/debate_history.jsonl.
After 7d, 15d, and 30d, the actual price is compared against the
predicted range and direction, building a calibrated track record.

This answers: "When the Fund Manager said BUY at $1208, was he right?"
"""

import json
import os
from datetime import date, datetime, timedelta

import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HISTORY_PATH = os.path.join(_PROJECT_ROOT, "data", "debate_history.jsonl")
_RAW_PATH     = os.path.join(_PROJECT_ROOT, "data", "raw_market.csv")

_HORIZONS = [
    {"key": "7d",  "days": 7},
    {"key": "15d", "days": 15},
    {"key": "30d", "days": 30},
]

_VERDICT_DIRECTION = {
    "STRONG_BUY":  1,
    "BUY":         1,
    "HOLD":        0,
    "SELL":       -1,
    "STRONG_SELL": -1,
}

_NEUTRAL_BAND = 0.015  # ±1.5% → HOLD verdict is correct inside this band


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_all() -> list[dict]:
    if not os.path.exists(_HISTORY_PATH):
        return []
    records = []
    try:
        with open(_HISTORY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception:
        pass
    return records


def _append_record(record: dict) -> None:
    os.makedirs(os.path.dirname(_HISTORY_PATH), exist_ok=True)
    with open(_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _rewrite_all(records: list[dict]) -> None:
    os.makedirs(os.path.dirname(_HISTORY_PATH), exist_ok=True)
    with open(_HISTORY_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_debate(debate_result: dict) -> dict:
    """
    Extract and persist a compact record from a full debate result.
    Returns the saved record. Idempotent: skips if same timestamp already exists.
    """
    ts = debate_result.get("timestamp", datetime.now().isoformat())

    existing = _load_all()
    if any(r.get("timestamp") == ts for r in existing):
        return next(r for r in existing if r.get("timestamp") == ts)

    verdict   = debate_result.get("verdict", {})
    agents    = debate_result.get("agents", {})
    bull      = agents.get("bull", {})
    bear      = agents.get("bear", {})
    risk      = agents.get("risk", {})
    tech      = agents.get("technical", {})
    bb        = verdict.get("bull_bear_balance", {})

    record = {
        "timestamp":          ts,
        "price_at_debate":    debate_result.get("current_price", 0),
        # Verdict
        "verdict":            verdict.get("verdict", "HOLD"),
        "confidence":         verdict.get("confidence", 0),
        "reasoning":          verdict.get("reasoning", ""),
        "price_range_7d":     verdict.get("price_range_7d", {}),
        "price_range_30d":    verdict.get("price_range_30d", {}),
        "position_sizing":    verdict.get("position_sizing", ""),
        "invalidation_conditions": verdict.get("invalidation_conditions", []),
        "key_watchlist":      verdict.get("key_watchlist", []),
        # Agent stances
        "bull_conviction":    bull.get("conviction", ""),
        "bull_thesis":        bull.get("thesis", ""),
        "bear_conviction":    bear.get("conviction", ""),
        "bear_thesis":        bear.get("thesis", ""),
        "bull_weight":        bb.get("bull_weight", 0.5),
        "bear_weight":        bb.get("bear_weight", 0.5),
        # Risk & technical
        "regime":             risk.get("regime", ""),
        "volatility":         risk.get("volatility_assessment", ""),
        "event_risk_7d":      risk.get("event_risk_next_7d", ""),
        "tech_trend":         tech.get("primary_trend", "") if tech else "",
        "tech_confidence":    tech.get("confidence", None) if tech else None,
        # Ex-post fields — filled by evaluate_due()
        "verified_7d":        False,
        "verified_15d":       False,
        "verified_30d":       False,
        "price_7d":           None,
        "price_15d":          None,
        "price_30d":          None,
        "ret_7d":             None,
        "ret_15d":            None,
        "ret_30d":            None,
        "in_range_7d":        None,
        "in_range_30d":       None,
        "direction_hit_7d":   None,
        "direction_hit_15d":  None,
        "direction_hit_30d":  None,
    }

    _append_record(record)
    return record


def evaluate_due() -> int:
    """
    Check all unverified debates and fill in ex-post results when enough
    time has passed. Returns the number of horizon evaluations performed.
    """
    records = _load_all()
    if not records:
        return 0
    if not os.path.exists(_RAW_PATH):
        return 0

    df = pd.read_csv(_RAW_PATH, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    today   = pd.Timestamp(date.today())
    n_done  = 0
    changed = False

    for r in records:
        ts        = pd.Timestamp(r["timestamp"])
        price_at  = r.get("price_at_debate", 0)
        if not price_at:
            continue

        for h in _HORIZONS:
            key          = h["key"]
            days         = h["days"]
            verified_key = f"verified_{key}"

            if r.get(verified_key):
                continue
            if (today - ts).days < days:
                continue

            target_date = ts + pd.Timedelta(days=days)
            future      = df[df["Date"] >= target_date]
            if future.empty:
                continue

            actual_price = float(future["Soybeans"].iloc[0])
            ret          = (actual_price / price_at) - 1.0

            r[f"price_{key}"] = round(actual_price, 2)
            r[f"ret_{key}"]   = round(ret * 100, 2)
            r[verified_key]   = True

            # Direction hit
            verdict_dir = _VERDICT_DIRECTION.get(r.get("verdict", "HOLD"), 0)
            if verdict_dir == 1:
                direction_hit = ret > _NEUTRAL_BAND
            elif verdict_dir == -1:
                direction_hit = ret < -_NEUTRAL_BAND
            else:
                direction_hit = abs(ret) <= _NEUTRAL_BAND
            r[f"direction_hit_{key}"] = bool(direction_hit)

            # Range hit (only for horizons where we have saved ranges)
            range_data = r.get(f"price_range_{key}", {})
            if range_data and key in ("7d", "30d"):
                low  = range_data.get("low", 0)
                high = range_data.get("high", 0)
                if low and high:
                    r[f"in_range_{key}"] = bool(low <= actual_price <= high)

            changed = True
            n_done += 1

    if changed:
        _rewrite_all(records)

    return n_done


def get_history(limit: int = 30) -> list[dict]:
    """Return the most recent `limit` debates, newest first."""
    evaluate_due()
    records = _load_all()
    return list(reversed(records[-limit:]))


def get_calibration_stats() -> dict:
    """
    Aggregate calibration statistics across all verified debates.
    Shows direction hit rates, range accuracy, and breakdown by verdict type.
    """
    evaluate_due()
    records = _load_all()

    horizon_stats = {}
    for h in _HORIZONS:
        key      = h["key"]
        verified = [r for r in records if r.get(f"verified_{key}")]
        hits     = [r for r in verified if r.get(f"direction_hit_{key}")]
        in_range_verified = [r for r in verified if r.get(f"in_range_{key}") is not None]
        in_range_hits     = [r for r in in_range_verified if r.get(f"in_range_{key}") is True]

        rets = [r[f"ret_{key}"] for r in verified if r.get(f"ret_{key}") is not None]

        horizon_stats[key] = {
            "total_debates":    len(records),
            "verified":         len(verified),
            "pending":          len(records) - len(verified),
            "direction_hits":   len(hits),
            "direction_hit_rate": round(len(hits) / len(verified) * 100, 1) if verified else None,
            "in_range_hits":    len(in_range_hits),
            "in_range_rate":    round(len(in_range_hits) / len(in_range_verified) * 100, 1) if in_range_verified else None,
            "avg_ret_pct":      round(sum(rets) / len(rets), 2) if rets else None,
        }

    # Breakdown by verdict type
    by_verdict: dict[str, dict] = {}
    for r in records:
        v = r.get("verdict", "HOLD")
        if v not in by_verdict:
            by_verdict[v] = {"count": 0, "hits_7d": 0, "verified_7d": 0}
        by_verdict[v]["count"] += 1
        if r.get("verified_7d"):
            by_verdict[v]["verified_7d"] += 1
            if r.get("direction_hit_7d"):
                by_verdict[v]["hits_7d"] += 1
    for v in by_verdict:
        vd = by_verdict[v]
        vd["hit_rate_7d"] = (
            round(vd["hits_7d"] / vd["verified_7d"] * 100, 1)
            if vd["verified_7d"] else None
        )

    # Regime performance (7d direction hit rate by regime)
    by_regime: dict[str, dict] = {}
    for r in records:
        reg = r.get("regime", "unknown")
        if reg not in by_regime:
            by_regime[reg] = {"count": 0, "hits_7d": 0, "verified_7d": 0}
        by_regime[reg]["count"] += 1
        if r.get("verified_7d"):
            by_regime[reg]["verified_7d"] += 1
            if r.get("direction_hit_7d"):
                by_regime[reg]["hits_7d"] += 1
    for reg in by_regime:
        rd = by_regime[reg]
        rd["hit_rate_7d"] = (
            round(rd["hits_7d"] / rd["verified_7d"] * 100, 1)
            if rd["verified_7d"] else None
        )

    return {
        "ok":             True,
        "total_debates":  len(records),
        "horizons":       horizon_stats,
        "by_verdict":     by_verdict,
        "by_regime":      by_regime,
        "as_of":          date.today().isoformat(),
    }


if __name__ == "__main__":
    print("evaluate_due:", evaluate_due())
    print(json.dumps(get_calibration_stats(), indent=2, ensure_ascii=False))
