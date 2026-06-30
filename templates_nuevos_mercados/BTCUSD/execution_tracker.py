"""
BTCUSD Bitcoin Robot — Execution Tracker
Syncs MT5 fills and generates performance reports.
"""

import os
import sys
import json
import argparse
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in [_HERE, _MVP_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import SYMBOL, ARTIFACTS_DIR, EXEC_LOG_FILE, PAPER_LOG_FILE

OUR_MAGIC = 20260601   # BTC robot magic number (corregido 2026-06-26: estaba 20260602=XAU, swapeado con live_runner)


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except Exception:
                    pass
    return lines


def compute_performance(trades: list) -> dict:
    if not trades:
        return {"total_trades": 0}

    longs  = [t for t in trades if t.get("signal") == "LONG"]
    shorts = [t for t in trades if t.get("signal") == "SHORT"]

    rr_vals   = [float(t["rr"])       for t in trades if t.get("rr")]
    risk_vals = [float(t["risk_usd"]) for t in trades if t.get("risk_usd")]
    confs     = [t.get("confidence", "?") for t in trades]

    avg_rr   = round(sum(rr_vals)   / len(rr_vals),   2) if rr_vals   else None
    avg_risk = round(sum(risk_vals) / len(risk_vals), 2) if risk_vals else None

    conf_dist = {}
    for c in confs:
        conf_dist[c] = conf_dist.get(c, 0) + 1

    # BTC regime distribution
    regimes = {}
    for t in trades:
        r = t.get("btc_regime", "unknown")
        regimes[r] = regimes.get(r, 0) + 1

    return {
        "total_trades":   len(trades),
        "longs":          len(longs),
        "shorts":         len(shorts),
        "avg_rr":         avg_rr,
        "avg_risk_usd":   avg_risk,
        "confidence_dist": conf_dist,
        "btc_regime_dist": regimes,
    }


def sync_fills() -> list:
    """Check MT5 positions for BTCUSD with our magic number."""
    try:
        from mt5_bridge import initialize, get_positions, is_connected
        if not is_connected():
            initialize()
        positions = get_positions(SYMBOL)
        ours = [p for p in (positions or []) if p.get("magic") == OUR_MAGIC]
        return ours
    except Exception as e:
        print(f"MT5 sync error: {e}")
        return []


def print_report():
    paper = _read_jsonl(PAPER_LOG_FILE)
    exec_ = _read_jsonl(EXEC_LOG_FILE)

    trades = [t for t in paper if t.get("signal") not in (None, "FLAT")]
    flat   = [t for t in paper if t.get("signal") == "FLAT"]
    stats  = compute_performance(trades)

    print("\n" + "="*55)
    print("  BTCUSD Bitcoin Robot — Performance Report")
    print("="*55)
    print(f"  Paper trades total : {stats.get('total_trades', 0)}")
    print(f"    LONG             : {stats.get('longs', 0)}")
    print(f"    SHORT            : {stats.get('shorts', 0)}")
    print(f"    FLAT (skipped)   : {len(flat)}")
    print(f"  Avg R:R            : {stats.get('avg_rr', 'N/A')}")
    print(f"  Avg Risk/trade     : ${stats.get('avg_risk_usd', 'N/A')}")
    print(f"  Confidence dist    : {stats.get('confidence_dist', {})}")
    print(f"  BTC Regime dist    : {stats.get('btc_regime_dist', {})}")
    print(f"  MT5 executions     : {len(exec_)}")
    print("="*55)

    if trades:
        print("\nRecent paper trades (last 5):")
        for t in trades[-5:]:
            ts  = t.get("timestamp", "")[:16].replace("T", " ")
            sig = t.get("signal", "?")
            ent = t.get("entry", "?")
            sl  = t.get("sl",    "?")
            tp  = t.get("tp",    "?")
            rr  = t.get("rr",    "?")
            cf  = t.get("confidence", "?")
            reg = t.get("btc_regime", "?")
            print(f"  {ts} | {sig:5s} @ ${ent:>10,.2f} | SL: ${sl:>10,.2f} | TP: ${tp:>10,.2f} | R:R {rr} | {cf} | {reg}")


def main():
    parser = argparse.ArgumentParser(description="BTCUSD Execution Tracker")
    parser.add_argument("--report", action="store_true", help="Print performance report")
    parser.add_argument("--sync",   action="store_true", help="Sync fills from MT5")
    args = parser.parse_args()

    if args.report:
        print_report()
    elif args.sync:
        fills = sync_fills()
        print(f"Open BTCUSD positions (magic={OUR_MAGIC}): {len(fills)}")
        for p in fills:
            print(f"  {p}")
    else:
        print_report()


if __name__ == "__main__":
    main()
