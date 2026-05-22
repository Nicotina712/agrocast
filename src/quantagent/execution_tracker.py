"""
src/quantagent/execution_tracker.py
Execution Tracker — Monitors MT5 fills and builds PnL feedback loop.

Tracks:
  - Signal → Fill latency (how fast was the order executed)
  - Slippage (signal price vs actual fill price)
  - Real PnL vs paper PnL comparison
  - Position lifecycle (entry → SL/TP hit → exit)
  - Running performance metrics

Data flow:
  1. live_runner.py emits signal → execution_tracker logs it
  2. MT5 EA executes the order → tracker polls MT5 for fill confirmation
  3. Tracker compares paper trade vs real execution
  4. Weekly: feeds performance data back to retrainer

Usage:
  python -m src.quantagent.execution_tracker --sync      # sync MT5 fills
  python -m src.quantagent.execution_tracker --report     # performance report
  python -m src.quantagent.execution_tracker --dashboard  # dashboard JSON
"""

import os
import sys
import io
import json
import time
import argparse
from datetime import datetime, date, timedelta, timezone
from typing import Optional

# Fix Windows console encoding (only when running as main script)
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_QA_DIR = os.path.join(_ROOT, "artifacts", "quantagent")
_EXEC_LOG = os.path.join(_QA_DIR, "execution_log.jsonl")
_PERF_FILE = os.path.join(_QA_DIR, "performance.json")
_DASHBOARD_FILE = os.path.join(_QA_DIR, "live_dashboard.json")
_PAPER_LOG = os.path.join(_QA_DIR, "paper_trades.jsonl")

# MT5 magic number used by our EA
OUR_MAGIC = 20260522


# =========================================================================
#  SYNC MT5 FILLS
# =========================================================================

def sync_mt5_fills(days: int = 7) -> dict:
    """
    Poll MT5 for recent trade history and match fills to our signals.
    Returns summary of matched/unmatched fills.
    """
    try:
        from src.intraday.data.mt5_bridge import (
            initialize, get_trade_history, get_positions, get_account_info,
        )
    except ImportError:
        return {"error": "mt5_bridge not available"}

    if not initialize():
        return {"error": "MT5 not connected"}

    # Load our signals
    signals = _load_signals()

    # Get MT5 trade history
    history = get_trade_history(days=days)
    positions = get_positions()
    account = get_account_info()

    # Filter our trades (by magic number or comment)
    our_deals = [d for d in history if d.get("magic") == OUR_MAGIC or
                 "QA" in (d.get("comment") or "")]

    matched = []
    unmatched_deals = []

    for deal in our_deals:
        deal_time = deal.get("time")
        if deal_time is None:
            continue

        # Try to match with a signal (within 5 minutes)
        best_match = None
        best_delta = timedelta(minutes=5)

        for sig in signals:
            sig_time = pd.to_datetime(sig.get("timestamp"))
            if sig_time.tzinfo is None:
                sig_time = sig_time.tz_localize("UTC")

            delta = abs(deal_time - sig_time)
            if delta < best_delta:
                best_delta = delta
                best_match = sig

        if best_match:
            # Calculate slippage
            signal_price = best_match.get("entry") or best_match.get("price_at_signal", 0)
            fill_price = deal.get("price", 0)
            slippage = 0
            if signal_price and fill_price:
                if best_match.get("signal") == "LONG":
                    slippage = fill_price - signal_price  # positive = worse
                elif best_match.get("signal") == "SHORT":
                    slippage = signal_price - fill_price

            match_entry = {
                "timestamp": datetime.now().isoformat(),
                "signal_time": best_match.get("timestamp"),
                "fill_time": deal_time.isoformat() if hasattr(deal_time, 'isoformat') else str(deal_time),
                "signal": best_match.get("signal"),
                "signal_price": signal_price,
                "fill_price": fill_price,
                "slippage": round(slippage, 2),
                "latency_seconds": best_delta.total_seconds(),
                "volume": deal.get("volume"),
                "profit": deal.get("profit", 0),
                "commission": deal.get("commission", 0),
                "swap": deal.get("swap", 0),
                "deal_ticket": deal.get("ticket"),
                "entry_type": "in" if deal.get("entry") == 0 else "out",
            }
            matched.append(match_entry)

            # Log to execution log
            _append_exec_log(match_entry)
        else:
            unmatched_deals.append(deal)

    result = {
        "timestamp": datetime.now().isoformat(),
        "total_deals": len(our_deals),
        "matched": len(matched),
        "unmatched": len(unmatched_deals),
        "open_positions": len(positions),
        "account": account,
        "avg_slippage": round(np.mean([m["slippage"] for m in matched]), 2) if matched else 0,
        "avg_latency_sec": round(np.mean([m["latency_seconds"] for m in matched]), 1) if matched else 0,
    }

    print(f"[exec] Synced: {len(matched)} matched, {len(unmatched_deals)} unmatched "
          f"out of {len(our_deals)} deals")
    if matched:
        print(f"[exec] Avg slippage: {result['avg_slippage']} | Avg latency: {result['avg_latency_sec']}s")

    return result


def _load_signals() -> list:
    """Load paper trade signals."""
    if not os.path.exists(_PAPER_LOG):
        return []
    signals = []
    with open(_PAPER_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    signals.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return signals


def _append_exec_log(entry: dict):
    """Append to execution log."""
    os.makedirs(_QA_DIR, exist_ok=True)
    with open(_EXEC_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# =========================================================================
#  PERFORMANCE METRICS
# =========================================================================

def build_performance_report() -> dict:
    """
    Build comprehensive performance report from paper trades + real executions.
    """
    # Paper trades
    signals = _load_signals()
    evaluated = [s for s in signals if s.get("evaluated") and s.get("signal") != "FLAT"]
    active = [s for s in signals if s.get("signal") != "FLAT"]

    # Execution log
    exec_entries = []
    if os.path.exists(_EXEC_LOG):
        with open(_EXEC_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        exec_entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    # Paper performance
    paper_perf = _calc_paper_performance(evaluated)

    # Real execution performance
    real_perf = _calc_real_performance(exec_entries)

    # Comparison
    comparison = {}
    if paper_perf.get("total_trades") and real_perf.get("total_fills"):
        comparison = {
            "paper_vs_real_pnl": round(
                (paper_perf.get("total_pnl_cents", 0) or 0) -
                (real_perf.get("total_profit_usd", 0) or 0), 2),
            "avg_slippage": real_perf.get("avg_slippage", 0),
            "fill_rate": round(real_perf["total_fills"] / paper_perf["total_trades"] * 100, 1)
                if paper_perf["total_trades"] > 0 else 0,
        }

    report = {
        "timestamp": datetime.now().isoformat(),
        "paper_performance": paper_perf,
        "real_performance": real_perf,
        "comparison": comparison,
        "total_signals": len(signals),
        "total_active": len(active),
        "total_evaluated": len(evaluated),
        "total_executions": len(exec_entries),
    }

    # Save
    os.makedirs(_QA_DIR, exist_ok=True)
    with open(_PERF_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def _calc_paper_performance(evaluated: list) -> dict:
    """Calculate paper trade performance metrics."""
    if not evaluated:
        return {"total_trades": 0}

    wins = [t for t in evaluated if (t.get("pnl_4h") or 0) > 0]
    losses = [t for t in evaluated if (t.get("pnl_4h") or 0) < 0]
    pnls = [t.get("pnl_4h", 0) or 0 for t in evaluated]

    # By confidence
    by_conf = {}
    for conf in ("HIGH", "MEDIUM", "LOW"):
        conf_trades = [t for t in evaluated if t.get("confidence") == conf]
        if conf_trades:
            conf_wins = sum(1 for t in conf_trades if (t.get("pnl_4h") or 0) > 0)
            by_conf[conf] = {
                "trades": len(conf_trades),
                "wins": conf_wins,
                "win_rate": round(conf_wins / len(conf_trades) * 100, 1),
            }

    # By direction
    by_dir = {}
    for d in ("LONG", "SHORT"):
        dir_trades = [t for t in evaluated if t.get("signal") == d]
        if dir_trades:
            dir_wins = sum(1 for t in dir_trades if (t.get("pnl_4h") or 0) > 0)
            by_dir[d] = {
                "trades": len(dir_trades),
                "wins": dir_wins,
                "win_rate": round(dir_wins / len(dir_trades) * 100, 1),
            }

    # Profit factor
    gross_wins = sum(p for p in pnls if p > 0)
    gross_losses = abs(sum(p for p in pnls if p < 0))
    pf = round(gross_wins / gross_losses, 2) if gross_losses > 0 else float("inf")

    return {
        "total_trades": len(evaluated),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(evaluated) * 100, 1),
        "avg_pnl_4h": round(np.mean(pnls), 2),
        "total_pnl_cents": round(sum(pnls), 2),
        "profit_factor": pf,
        "max_win": round(max(pnls), 2) if pnls else 0,
        "max_loss": round(min(pnls), 2) if pnls else 0,
        "by_confidence": by_conf,
        "by_direction": by_dir,
        "sl_hit_rate": round(sum(1 for t in evaluated if t.get("would_hit_sl")) / len(evaluated) * 100, 1),
        "tp_hit_rate": round(sum(1 for t in evaluated if t.get("would_hit_tp")) / len(evaluated) * 100, 1),
    }


def _calc_real_performance(exec_entries: list) -> dict:
    """Calculate real execution performance from MT5 fills."""
    if not exec_entries:
        return {"total_fills": 0}

    fills_in = [e for e in exec_entries if e.get("entry_type") == "in"]
    fills_out = [e for e in exec_entries if e.get("entry_type") == "out"]

    slippages = [e.get("slippage", 0) for e in fills_in if e.get("slippage") is not None]
    latencies = [e.get("latency_seconds", 0) for e in fills_in if e.get("latency_seconds") is not None]
    profits = [e.get("profit", 0) for e in exec_entries if e.get("profit")]
    commissions = [e.get("commission", 0) for e in exec_entries if e.get("commission")]

    return {
        "total_fills": len(exec_entries),
        "entries": len(fills_in),
        "exits": len(fills_out),
        "avg_slippage": round(np.mean(slippages), 2) if slippages else 0,
        "max_slippage": round(max(slippages), 2) if slippages else 0,
        "avg_latency_sec": round(np.mean(latencies), 1) if latencies else 0,
        "total_profit_usd": round(sum(profits), 2),
        "total_commission_usd": round(sum(commissions), 2),
        "net_profit_usd": round(sum(profits) + sum(commissions), 2),
    }


# =========================================================================
#  LIVE DASHBOARD DATA
# =========================================================================

def build_dashboard_data() -> dict:
    """
    Build JSON for live dashboard display.
    Combines: current signal, positions, account, performance.
    """
    dashboard = {
        "timestamp": datetime.now().isoformat(),
        "status": "offline",
    }

    # Latest signal
    signal_file = os.path.join(_QA_DIR, "latest_signal.json")
    if os.path.exists(signal_file):
        try:
            with open(signal_file, encoding="utf-8") as f:
                sig = json.load(f)
            dashboard["latest_signal"] = {
                "time": sig.get("timestamp"),
                "signal": sig.get("signal", {}).get("signal") if isinstance(sig.get("signal"), dict) else sig.get("signal"),
                "confidence": sig.get("signal", {}).get("confidence") if isinstance(sig.get("signal"), dict) else None,
                "price": sig.get("current_price"),
                "entry": sig.get("signal", {}).get("entry") if isinstance(sig.get("signal"), dict) else None,
                "sl": sig.get("signal", {}).get("stop_loss") if isinstance(sig.get("signal"), dict) else None,
                "tp": sig.get("signal", {}).get("take_profit") if isinstance(sig.get("signal"), dict) else None,
                "mode": sig.get("mode", "batch"),
                "data_source": sig.get("data_source", "yfinance"),
            }
        except Exception:
            pass

    # Live state
    live_state_file = os.path.join(_QA_DIR, "live_state.json")
    if os.path.exists(live_state_file):
        try:
            with open(live_state_file, encoding="utf-8") as f:
                ls = json.load(f)
            dashboard["live_state"] = {
                "date": ls.get("date"),
                "llm_calls_today": ls.get("llm_calls", 0),
                "cycles_today": ls.get("cycles", 0),
                "last_signal_time": ls.get("last_signal_time"),
            }
            dashboard["status"] = "live" if ls.get("date") == date.today().isoformat() else "stale"
        except Exception:
            pass

    # MT5 connection
    try:
        from src.intraday.data.mt5_bridge import initialize, is_connected, get_account_info, get_positions
        if initialize() and is_connected():
            dashboard["mt5"] = {
                "connected": True,
                "account": get_account_info(),
                "positions": get_positions(),
            }
            dashboard["status"] = "live"
    except Exception:
        dashboard["mt5"] = {"connected": False}

    # Performance summary
    if os.path.exists(_PERF_FILE):
        try:
            with open(_PERF_FILE, encoding="utf-8") as f:
                perf = json.load(f)
            dashboard["performance"] = {
                "paper": perf.get("paper_performance", {}),
                "real": perf.get("real_performance", {}),
                "comparison": perf.get("comparison", {}),
            }
        except Exception:
            pass

    # Model info
    model_metrics = os.path.join(_ROOT, "artifacts", "intraday", "intraday_train_metrics.json")
    if os.path.exists(model_metrics):
        try:
            with open(model_metrics) as f:
                mm = json.load(f)
            dashboard["model"] = {
                "mean_auc": mm.get("mean_auc"),
                "retrain_timestamp": mm.get("retrain_timestamp"),
                "n_features": mm.get("n_features"),
            }
        except Exception:
            pass

    # Save
    os.makedirs(_QA_DIR, exist_ok=True)
    with open(_DASHBOARD_FILE, "w", encoding="utf-8") as f:
        json.dump(dashboard, f, indent=2, default=str)

    return dashboard


# =========================================================================
#  CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="QuantAgent Execution Tracker")
    parser.add_argument("--sync", action="store_true", help="Sync MT5 fills with signals")
    parser.add_argument("--report", action="store_true", help="Build performance report")
    parser.add_argument("--dashboard", action="store_true", help="Build dashboard data")
    parser.add_argument("--days", type=int, default=7, help="Days of history to sync")
    args = parser.parse_args()

    if args.sync:
        result = sync_mt5_fills(days=args.days)
        print(json.dumps(result, indent=2, default=str))

    if args.report:
        report = build_performance_report()
        paper = report.get("paper_performance", {})
        real = report.get("real_performance", {})
        print(f"\n=== Performance Report ===")
        print(f"Paper: {paper.get('total_trades', 0)} trades, "
              f"WR: {paper.get('win_rate', 0)}%, PF: {paper.get('profit_factor', 0)}")
        print(f"Real: {real.get('total_fills', 0)} fills, "
              f"Net PnL: ${real.get('net_profit_usd', 0)}")

    if args.dashboard:
        data = build_dashboard_data()
        print(json.dumps(data, indent=2, default=str))

    if not (args.sync or args.report or args.dashboard):
        # Default: do all
        print("=== Sync ===")
        sync_mt5_fills(days=args.days)
        print("\n=== Report ===")
        build_performance_report()
        print("\n=== Dashboard ===")
        data = build_dashboard_data()
        print(json.dumps(data, indent=2, default=str))


if __name__ == "__main__":
    main()
