"""
QuantAgent-lite Runner — Orchestrates Trend + Risk agents on 60m bars.

Pipeline:
  1. Fetch latest 60m bars via tick_feed (yfinance)
  2. Build microstructure features
  3. Summarize price context for LLM agents
  4. Call Trend Agent → interpret chart
  5. Call Risk Agent → evaluate + size trade
  6. Synthesize final signal (BUY/SELL/FLAT + parameters)
  7. Log to paper_log for performance tracking

Cost gate: max 3 runs/day (every ~4h during RTH).
"""

import os
import sys
import json
import time
from datetime import datetime, date

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.intraday.data.tick_feed import fetch_intraday_bars
from src.intraday.features.microstructure import build_intraday_features
from src.quantagent.agents import call_trend_agent, call_risk_agent
from src.quantagent.paper_log import log_signal, load_log

_OUT_DIR = os.path.join(_ROOT, "artifacts", "quantagent")
_SIGNAL_FILE = os.path.join(_OUT_DIR, "latest_signal.json")
_GATE_FILE = os.path.join(_OUT_DIR, "run_gate.json")
_MAX_RUNS_PER_DAY = 3


def _check_gate() -> tuple[bool, str]:
    """Cost gate: max N runs per day."""
    today = date.today().isoformat()
    if os.path.exists(_GATE_FILE):
        try:
            with open(_GATE_FILE) as f:
                gate = json.load(f)
            if gate.get("date") == today:
                if gate.get("runs", 0) >= _MAX_RUNS_PER_DAY:
                    return False, f"max {_MAX_RUNS_PER_DAY} runs/day reached"
        except Exception:
            pass
    return True, "ok"


def _mark_ran():
    """Increment today's run counter."""
    today = date.today().isoformat()
    gate = {"date": today, "runs": 0}
    if os.path.exists(_GATE_FILE):
        try:
            with open(_GATE_FILE) as f:
                gate = json.load(f)
            if gate.get("date") != today:
                gate = {"date": today, "runs": 0}
        except Exception:
            gate = {"date": today, "runs": 0}
    gate["runs"] = gate.get("runs", 0) + 1
    os.makedirs(os.path.dirname(_GATE_FILE), exist_ok=True)
    with open(_GATE_FILE, "w") as f:
        json.dump(gate, f)


def _build_bars_summary(feat: pd.DataFrame) -> dict:
    """Build a compact summary of recent bars for LLM agents."""
    if feat.empty:
        return {}

    # Filter RTH only for cleaner data
    rth = feat[feat["is_rth"] == 1].copy() if "is_rth" in feat.columns else feat.copy()
    if rth.empty:
        rth = feat.copy()

    last = rth.iloc[-1]
    last_12 = rth.tail(12)

    # Current state
    current_state = {
        "close": round(float(last["close"]), 2),
        "session_date": str(last.get("date_ct", "")),
        "hour_ct": int(last.get("hour_ct", 0)),
        "mins_to_close": float(last.get("mins_to_close", 0)) if pd.notna(last.get("mins_to_close")) else None,
        "is_rth": bool(last.get("is_rth", 0)),
    }

    # Indicators
    def _safe(val):
        if pd.isna(val):
            return None
        return round(float(val), 4)

    indicators = {
        "rsi_14": _safe(last.get("rsi_14")),
        "atr_14": _safe(last.get("atr_14")),
        "atr_mean_30": _safe(rth["atr_14"].tail(30).mean()) if "atr_14" in rth else None,
        "ema_fast": _safe(last.get("ema_fast")),
        "ema_slow": _safe(last.get("ema_slow")),
        "ema_cross": _safe(last.get("ema_cross")),
        "realized_vol_30": _safe(last.get("realized_vol_30")),
        "vol_zscore_30": _safe(last.get("vol_zscore_30")),
        "vwap_session": _safe(last.get("vwap_session")),
        "vwap_dist": _safe(last.get("vwap_dist")),
        "body_pct": _safe(last.get("body_pct")),
        "upper_wick": _safe(last.get("upper_wick")),
        "lower_wick": _safe(last.get("lower_wick")),
        "cum_delta_5": _safe(last.get("cum_delta_5")),
        "cum_delta_20": _safe(last.get("cum_delta_20")),
    }

    # Returns
    returns = {
        "ret_1": _safe(last.get("ret_1")),
        "ret_3": _safe(last.get("ret_3")),
        "ret_12": _safe(last.get("ret_12")),
    }
    # Session return
    today_bars = rth[rth["date_ct"] == last.get("date_ct")]
    if not today_bars.empty:
        sess_open = today_bars.iloc[0]["open"]
        sess_ret = (float(last["close"]) - float(sess_open)) / float(sess_open) * 100
        returns["session_return"] = round(sess_ret, 3)

    # Recent bars (last 12)
    recent_bars = []
    for _, row in last_12.iterrows():
        recent_bars.append({
            "hour_ct": int(row.get("hour_ct", 0)),
            "open": round(float(row["open"]), 2),
            "high": round(float(row["high"]), 2),
            "low": round(float(row["low"]), 2),
            "close": round(float(row["close"]), 2),
            "volume": int(row.get("volume", 0)),
            "body_pct": _safe(row.get("body_pct")),
        })
    recent_bars.reverse()  # Most recent first

    # Session range today
    session_range = {}
    if not today_bars.empty:
        session_range = {
            "open": round(float(today_bars.iloc[0]["open"]), 2),
            "high": round(float(today_bars["high"].max()), 2),
            "low": round(float(today_bars["low"].min()), 2),
            "range": round(float(today_bars["high"].max() - today_bars["low"].min()), 2),
            "range_pct": round(
                float((today_bars["high"].max() - today_bars["low"].min()) / last["close"] * 100), 3
            ),
        }

    # Multi-day context (last 5 days)
    multi_day = []
    if "date_ct" in rth.columns:
        daily = rth.groupby("date_ct").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        ).tail(5)
        for dt, row in daily.iterrows():
            dr = (float(row["close"]) - float(row["open"])) / float(row["open"]) * 100
            multi_day.append({
                "date": str(dt),
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "daily_return": round(dr, 2),
            })

    return {
        "current_state": current_state,
        "indicators": indicators,
        "returns": returns,
        "recent_bars": recent_bars,
        "session_range": session_range,
        "multi_day": multi_day,
    }


def _synthesize_signal(trend: dict, risk: dict, current_price: float) -> dict:
    """Combine Trend + Risk agent outputs into final signal."""
    setup = trend.get("setup", {})
    direction = setup.get("direction", "FLAT")
    setup_confidence = setup.get("confidence", "LOW")

    trade_viable = risk.get("trade_viable", False)
    veto = risk.get("veto_reason")

    # Final decision
    if direction == "FLAT" or not trade_viable:
        return {
            "signal": "FLAT",
            "reason": veto or setup.get("description", "No clear setup"),
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "contracts": 0,
            "confidence": "LOW",
        }

    sl_info = risk.get("stop_loss", {})
    tp_info = risk.get("take_profit", {})
    size_info = risk.get("position_size", {})
    entry_zone = setup.get("entry_zone", {})

    entry = (entry_zone.get("low", current_price) + entry_zone.get("high", current_price)) / 2
    sl = sl_info.get("price")
    tp = tp_info.get("price")
    contracts = size_info.get("contracts_mzs", 1)

    # Validate R:R
    if sl and tp and entry:
        risk_dist = abs(entry - sl)
        reward_dist = abs(tp - entry)
        rr = reward_dist / risk_dist if risk_dist > 0 else 0
        if rr < 1.3:  # Slightly below 1.5 threshold to allow agent discretion
            return {
                "signal": "FLAT",
                "reason": f"R:R insuficiente ({rr:.1f}:1, min 1.5:1)",
                "entry": entry,
                "stop_loss": sl,
                "take_profit": tp,
                "contracts": 0,
                "confidence": "LOW",
            }

    return {
        "signal": direction,
        "reason": setup.get("description", ""),
        "entry": round(entry, 2) if entry else None,
        "stop_loss": round(sl, 2) if sl else None,
        "take_profit": round(tp, 2) if tp else None,
        "contracts": contracts,
        "confidence": setup_confidence,
        "risk_reward": tp_info.get("risk_reward", "?"),
        "max_hold_bars": risk.get("max_hold_bars"),
        "volatility_regime": risk.get("volatility_regime", "?"),
    }


def run_quantagent(force: bool = False) -> dict:
    """
    Main entry point. Runs the 2-agent pipeline on latest 60m bars.

    Returns dict with:
      signal, trend_agent, risk_agent, bars_summary, timestamp, etc.
    """
    os.makedirs(_OUT_DIR, exist_ok=True)

    # Cost gate
    if not force:
        can_run, reason = _check_gate()
        if not can_run:
            # Return cached signal
            if os.path.exists(_SIGNAL_FILE):
                with open(_SIGNAL_FILE, encoding="utf-8") as f:
                    cached = json.load(f)
                cached["from_cache"] = True
                cached["gate_reason"] = reason
                return cached
            return {"error": reason, "from_cache": True}

    t0 = time.time()

    # 1. Fetch and build features
    print("[QA] Fetching 60m bars...")
    bars = fetch_intraday_bars(interval="60m", use_cache=True, cache_max_age_min=60)
    if bars.empty:
        return {"error": "No bars available"}

    print("[QA] Building features...")
    feat = build_intraday_features(bars, interval="60m")

    # 2. Summarize for LLM
    summary = _build_bars_summary(feat)
    current_price = summary.get("current_state", {}).get("close", 0)

    # 3. Call agents
    print("[QA] Calling Trend Agent...")
    trend = call_trend_agent(summary)
    print(f"[QA] Trend: {trend.get('trend', '?')} | Setup: {trend.get('setup', {}).get('direction', '?')}")

    print("[QA] Calling Risk Agent...")
    risk = call_risk_agent(summary, trend)
    print(f"[QA] Risk: viable={risk.get('trade_viable', '?')} | vol={risk.get('volatility_regime', '?')}")

    # 4. Synthesize
    signal = _synthesize_signal(trend, risk, current_price)
    elapsed = time.time() - t0

    result = {
        "timestamp": datetime.now().isoformat(),
        "current_price": current_price,
        "signal": signal,
        "trend_agent": trend,
        "risk_agent": risk,
        "bars_summary": {
            "current_state": summary.get("current_state"),
            "indicators": summary.get("indicators"),
            "returns": summary.get("returns"),
            "session_range": summary.get("session_range"),
            "multi_day": summary.get("multi_day"),
        },
        "execution_time_seconds": round(elapsed, 1),
        "from_cache": False,
    }

    # 5. Save
    try:
        with open(_SIGNAL_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"[QA] Save failed: {e}")

    # 6. Log to paper trades
    try:
        log_signal(result)
    except Exception as e:
        print(f"[QA] Paper log failed: {e}")

    # 7. Mark gate
    _mark_ran()

    print(f"[QA] Done in {elapsed:.1f}s — signal={signal['signal']} conf={signal.get('confidence','?')}")
    return result


def get_latest_signal() -> dict | None:
    """Load latest signal from cache without running agents."""
    if not os.path.exists(_SIGNAL_FILE):
        return None
    try:
        with open(_SIGNAL_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_status() -> dict:
    """Return current status: gate, last run, paper trade stats."""
    today = date.today().isoformat()
    gate = {"date": today, "runs": 0}
    if os.path.exists(_GATE_FILE):
        try:
            with open(_GATE_FILE) as f:
                gate = json.load(f)
        except Exception:
            pass

    latest = get_latest_signal()
    log = load_log()

    return {
        "gate": {
            "today": today,
            "runs_today": gate.get("runs", 0) if gate.get("date") == today else 0,
            "max_runs": _MAX_RUNS_PER_DAY,
            "can_run": gate.get("runs", 0) < _MAX_RUNS_PER_DAY if gate.get("date") == today else True,
        },
        "last_signal": {
            "timestamp": latest.get("timestamp") if latest else None,
            "signal": latest.get("signal", {}).get("signal") if latest else None,
            "price": latest.get("current_price") if latest else None,
        },
        "paper_trades": log.get("summary", {}),
    }


if __name__ == "__main__":
    result = run_quantagent(force=True)
    sig = result.get("signal", {})
    print(f"\n{'='*60}")
    print(f"SIGNAL: {sig.get('signal', '?')}")
    print(f"Confidence: {sig.get('confidence', '?')}")
    print(f"Entry: {sig.get('entry', '?')}")
    print(f"SL: {sig.get('stop_loss', '?')} | TP: {sig.get('take_profit', '?')}")
    print(f"R:R: {sig.get('risk_reward', '?')}")
    print(f"Contracts MZS: {sig.get('contracts', 0)}")
    print(f"Time: {result.get('execution_time_seconds', '?')}s")
