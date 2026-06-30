"""
XAUUSD Gold — Execution Tracker
Syncs MT5 fills and builds performance metrics for Gold trades.

Usage:
  cd templates_nuevos_mercados/XAUUSD
  python execution_tracker.py --sync      # sync MT5 fills
  python execution_tracker.py --report    # performance report
  python execution_tracker.py --dashboard # dashboard JSON
"""

import os
import sys
import io
import json
import argparse
from datetime import datetime, timedelta, timezone

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in [_HERE, _MVP_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import numpy as np

from config import (
    SYMBOL, ARTIFACTS_DIR,
    EXEC_LOG_FILE, PERF_FILE, PAPER_LOG_FILE,
    CAPITAL_USD, MAX_RISK_USD,
)
from mt5_bridge import initialize as mt5_init, is_connected as mt5_connected, get_positions

OUR_MAGIC = 20260602  # Gold robot magic number (corregido 2026-06-26: estaba 20260601=BTC, swapeado con live_runner; soja era 20260522)


# ─── Paper trade loader ───────────────────────────────────────────────────────

def load_paper_trades() -> list[dict]:
    if not os.path.exists(PAPER_LOG_FILE):
        return []
    trades = []
    with open(PAPER_LOG_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trades.append(json.loads(line))
                except Exception:
                    pass
    return trades


# ─── Execution log ────────────────────────────────────────────────────────────

def load_execution_log() -> list[dict]:
    if not os.path.exists(EXEC_LOG_FILE):
        return []
    entries = []
    with open(EXEC_LOG_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    return entries


def log_execution(entry: dict):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    entry["timestamp"] = datetime.now().isoformat()
    entry["symbol"]    = SYMBOL
    with open(EXEC_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ─── Performance metrics ──────────────────────────────────────────────────────

def compute_performance(trades: list[dict]) -> dict:
    """Compute basic performance stats from paper trades."""
    if not trades:
        return {"total_trades": 0, "win_rate": None, "avg_rr": None}

    signals_only = [t for t in trades if t.get("signal") not in ("FLAT", None)]
    total = len(signals_only)
    if total == 0:
        return {"total_trades": 0, "win_rate": None}

    # Count by direction
    longs  = sum(1 for t in signals_only if t.get("signal") == "LONG")
    shorts = sum(1 for t in signals_only if t.get("signal") == "SHORT")

    # R:R distribution
    rr_values = []
    for t in signals_only:
        rr_str = t.get("rr", "")
        if rr_str and ":" in str(rr_str):
            try:
                rr_values.append(float(str(rr_str).split(":")[0]))
            except Exception:
                pass

    # Risk distribution
    risk_vals = [t.get("risk_usd") for t in signals_only if t.get("risk_usd")]
    risk_vals = [r for r in risk_vals if r is not None]

    return {
        "total_trades":    total,
        "longs":           longs,
        "shorts":          shorts,
        "avg_rr":          round(float(np.mean(rr_values)), 2) if rr_values else None,
        "avg_risk_usd":    round(float(np.mean(risk_vals)), 2) if risk_vals else None,
        "capital_at_risk": round(float(np.sum(risk_vals)), 2) if risk_vals else 0,
        "pct_of_capital":  round(float(np.sum(risk_vals)) / CAPITAL_USD * 100, 1) if risk_vals else 0,
        "confidence_dist": {
            "HIGH":   sum(1 for t in signals_only if t.get("confidence") == "HIGH"),
            "MEDIUM": sum(1 for t in signals_only if t.get("confidence") == "MEDIUM"),
            "LOW":    sum(1 for t in signals_only if t.get("confidence") == "LOW"),
        },
        "computed_at": datetime.now().isoformat(),
    }


def save_performance(perf: dict):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(PERF_FILE, "w", encoding="utf-8") as f:
        json.dump(perf, f, indent=2, default=str)


# ─── Sync MT5 fills ───────────────────────────────────────────────────────────

def sync_fills():
    """Check open positions in MT5 for Gold."""
    if not mt5_connected():
        mt5_init()

    positions = get_positions()
    gold_pos = [p for p in positions if SYMBOL in str(p.get("symbol", ""))]

    print(f"\n=== XAUUSD Positions in MT5 ===")
    if not gold_pos:
        print("No open Gold positions.")
    else:
        for p in gold_pos:
            print(f"  Ticket: {p.get('ticket')} | {p.get('type')} | {p.get('volume')} lots")
            print(f"  Entry: {p.get('price_open')} | Current: {p.get('price_current')}")
            print(f"  PnL: ${p.get('profit', 0):.2f} | SL: {p.get('sl')} | TP: {p.get('tp')}")

    log_execution({"event": "sync", "positions": gold_pos})


# ─── Report ───────────────────────────────────────────────────────────────────

def print_report():
    trades = load_paper_trades()
    perf   = compute_performance(trades)
    save_performance(perf)

    print(f"\n=== XAUUSD Gold — Performance Report ===")
    print(f"Total signals: {perf['total_trades']}")
    print(f"  LONG: {perf.get('longs', 0)} | SHORT: {perf.get('shorts', 0)}")
    print(f"Avg R:R: {perf.get('avg_rr')}")
    print(f"Avg risk per trade: ${perf.get('avg_risk_usd')}")
    print(f"Total capital at risk (paper): ${perf.get('capital_at_risk')} ({perf.get('pct_of_capital')}% of ${CAPITAL_USD:.0f})")
    conf = perf.get("confidence_dist", {})
    print(f"Confidence: HIGH={conf.get('HIGH', 0)} MEDIUM={conf.get('MEDIUM', 0)} LOW={conf.get('LOW', 0)}")
    print(f"\nLast 5 paper trades:")
    for t in trades[-5:]:
        print(f"  {t.get('timestamp', '')[:16]} | {t.get('signal')} @ ${t.get('entry')} | SL:{t.get('sl')} TP:{t.get('tp')} | {t.get('lots')}lots | {t.get('confidence')}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="XAUUSD Execution Tracker")
    parser.add_argument("--sync",      action="store_true")
    parser.add_argument("--report",    action="store_true")
    parser.add_argument("--dashboard", action="store_true")
    args = parser.parse_args()

    if args.sync:
        sync_fills()
    elif args.report:
        print_report()
    elif args.dashboard:
        trades = load_paper_trades()
        perf   = compute_performance(trades)
        dashboard = {"symbol": SYMBOL, "performance": perf, "paper_trades": trades[-20:]}
        print(json.dumps(dashboard, indent=2, default=str))
    else:
        print_report()


if __name__ == "__main__":
    main()
