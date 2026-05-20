"""
QuantAgent-lite Paper Trade Logger.

Tracks every signal for performance evaluation.
After N bars, evaluates if the trade would have been profitable.
Compares against the XGBoost intraday model baseline.
"""

import os
import json
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LOG_DIR = os.path.join(_ROOT, "artifacts", "quantagent")
_LOG_FILE = os.path.join(_LOG_DIR, "paper_trades.jsonl")
_SUMMARY_FILE = os.path.join(_LOG_DIR, "paper_summary.json")


def log_signal(result: dict):
    """Append a signal to the paper trade log."""
    os.makedirs(_LOG_DIR, exist_ok=True)

    signal = result.get("signal", {})
    entry = {
        "timestamp": result.get("timestamp"),
        "price_at_signal": result.get("current_price"),
        "signal": signal.get("signal", "FLAT"),
        "confidence": signal.get("confidence", "LOW"),
        "entry": signal.get("entry"),
        "stop_loss": signal.get("stop_loss"),
        "take_profit": signal.get("take_profit"),
        "risk_reward": signal.get("risk_reward"),
        "contracts": signal.get("contracts", 0),
        "volatility_regime": signal.get("volatility_regime"),
        "trend_summary": result.get("trend_agent", {}).get("trend"),
        "trend_momentum": result.get("trend_agent", {}).get("momentum"),
        "setup_type": result.get("trend_agent", {}).get("setup", {}).get("type"),
        # Evaluation fields (filled later)
        "evaluated": False,
        "price_after_4h": None,
        "price_after_1d": None,
        "pnl_4h": None,
        "pnl_1d": None,
        "would_hit_sl": None,
        "would_hit_tp": None,
    }

    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def load_log() -> dict:
    """Load all paper trades and compute summary stats."""
    if not os.path.exists(_LOG_FILE):
        return {"trades": [], "summary": {}}

    trades = []
    with open(_LOG_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trades.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Summary
    total = len(trades)
    active = [t for t in trades if t.get("signal") != "FLAT"]
    flat = total - len(active)
    evaluated = [t for t in active if t.get("evaluated")]

    wins = [t for t in evaluated if (t.get("pnl_4h") or 0) > 0]
    losses = [t for t in evaluated if (t.get("pnl_4h") or 0) < 0]

    summary = {
        "total_signals": total,
        "active_signals": len(active),
        "flat_signals": flat,
        "evaluated": len(evaluated),
        "pending_evaluation": len(active) - len(evaluated),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(evaluated) * 100, 1) if evaluated else None,
        "avg_pnl_4h": round(
            sum(t.get("pnl_4h", 0) for t in evaluated) / len(evaluated), 2
        ) if evaluated else None,
        "by_confidence": {},
        "by_setup_type": {},
        "signal_distribution": {},
    }

    # Breakdown by confidence
    for conf in ("HIGH", "MEDIUM", "LOW"):
        conf_trades = [t for t in active if t.get("confidence") == conf]
        summary["by_confidence"][conf] = len(conf_trades)

    # Breakdown by setup type
    for t in active:
        st = t.get("setup_type", "unknown")
        summary["by_setup_type"][st] = summary["by_setup_type"].get(st, 0) + 1

    # Signal distribution
    for t in trades:
        sig = t.get("signal", "FLAT")
        summary["signal_distribution"][sig] = summary["signal_distribution"].get(sig, 0) + 1

    return {"trades": trades, "summary": summary}


def evaluate_pending(bars_df=None):
    """
    Evaluate pending paper trades against actual price data.
    Called periodically (e.g., when new bars arrive).
    """
    if not os.path.exists(_LOG_FILE) or bars_df is None or bars_df.empty:
        return 0

    import pandas as pd

    trades = []
    with open(_LOG_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trades.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not trades:
        return 0

    bars_df = bars_df.copy()
    bars_df["datetime"] = pd.to_datetime(bars_df["datetime"], utc=True)

    n_evaluated = 0
    for trade in trades:
        if trade.get("evaluated") or trade.get("signal") == "FLAT":
            continue

        ts = pd.to_datetime(trade["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")

        # Find bars 4h and 1d after signal
        future_bars = bars_df[bars_df["datetime"] > ts]
        if future_bars.empty:
            continue

        # 4h = ~4 bars at 60m
        if len(future_bars) >= 4:
            bar_4h = future_bars.iloc[3]
            trade["price_after_4h"] = round(float(bar_4h["close"]), 2)

            entry_price = trade.get("entry") or trade.get("price_at_signal", 0)
            if entry_price:
                if trade["signal"] == "LONG":
                    trade["pnl_4h"] = round(float(bar_4h["close"]) - entry_price, 2)
                elif trade["signal"] == "SHORT":
                    trade["pnl_4h"] = round(entry_price - float(bar_4h["close"]), 2)

            # Check SL/TP hits in first 4 bars
            first_4 = future_bars.iloc[:4]
            sl = trade.get("stop_loss")
            tp = trade.get("take_profit")
            if sl:
                if trade["signal"] == "LONG":
                    trade["would_hit_sl"] = bool(first_4["low"].min() <= sl)
                elif trade["signal"] == "SHORT":
                    trade["would_hit_sl"] = bool(first_4["high"].max() >= sl)
            if tp:
                if trade["signal"] == "LONG":
                    trade["would_hit_tp"] = bool(first_4["high"].max() >= tp)
                elif trade["signal"] == "SHORT":
                    trade["would_hit_tp"] = bool(first_4["low"].min() <= tp)

            trade["evaluated"] = True
            n_evaluated += 1

        # 1d = ~5 RTH bars
        if len(future_bars) >= 5:
            bar_1d = future_bars.iloc[4]
            trade["price_after_1d"] = round(float(bar_1d["close"]), 2)
            entry_price = trade.get("entry") or trade.get("price_at_signal", 0)
            if entry_price:
                if trade["signal"] == "LONG":
                    trade["pnl_1d"] = round(float(bar_1d["close"]) - entry_price, 2)
                elif trade["signal"] == "SHORT":
                    trade["pnl_1d"] = round(entry_price - float(bar_1d["close"]), 2)

    # Rewrite log
    if n_evaluated > 0:
        with open(_LOG_FILE, "w", encoding="utf-8") as f:
            for t in trades:
                f.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")

    return n_evaluated
