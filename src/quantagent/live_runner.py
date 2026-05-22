"""
src/quantagent/live_runner.py
Live QuantAgent Runner — Polls MT5 every cycle during RTH.

Architecture:
  - Runs locally on the trader's PC (where MT5 terminal is open)
  - Polls MT5 for live 60m bars every CYCLE_MINUTES
  - Builds microstructure features on real-time data
  - Calls Trend + Risk agents (Claude API)
  - Publishes signal to local file + optional POST to Render API
  - Evaluates pending paper trades against live bars

Schedule:
  - RTH: 08:30-13:20 CT (soybean futures CBOT)
  - Cycles: every 15 min by default (adjustable)
  - Max 6 LLM calls/day to control costs (~$0.12/run)
  - Runs outside RTH in monitoring-only mode (no new signals)

Usage:
  python -m src.quantagent.live_runner              # run one cycle
  python -m src.quantagent.live_runner --loop        # continuous loop
  python -m src.quantagent.live_runner --diagnose    # connection test
"""

import os
import sys
import io
import json
import time
import argparse
import traceback
from datetime import datetime, date, timedelta, timezone

# Fix Windows console encoding (only when running as main script)
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.intraday.data.mt5_bridge import (
    initialize as mt5_init,
    shutdown as mt5_shutdown,
    is_connected as mt5_connected,
    fetch_mt5_bars,
    get_live_tick,
    get_account_info,
    get_positions,
    place_order,
    diagnose as mt5_diagnose,
)
from src.intraday.features.microstructure import build_intraday_features
from src.quantagent.agents import call_trend_agent, call_risk_agent
from src.quantagent.paper_log import log_signal, load_log
from src.quantagent.runner import (
    _build_bars_summary,
    _build_track_record,
    _build_fundamental_context,
    _synthesize_signal,
    _OUT_DIR,
    _SIGNAL_FILE,
)


# =========================================================================
#  CONFIG
# =========================================================================

CYCLE_MINUTES = int(os.environ.get("QA_CYCLE_MIN", "15"))
MAX_LLM_CALLS_PER_DAY = int(os.environ.get("QA_MAX_CALLS", "6"))
RTH_START_CT = (8, 30)   # 08:30 CT
RTH_END_CT = (13, 20)    # 13:20 CT
CT_OFFSET_HOURS = -5     # CT = UTC-5 (CDT during summer)
MIN_BARS_REQUIRED = 30   # need at least 30 bars for features
POST_TO_API = os.environ.get("QA_POST_API", "")  # optional: Render URL
EXECUTE_TRADES = os.environ.get("QA_EXECUTE", "0") == "1"  # set via --execute flag
DEFAULT_VOLUME = float(os.environ.get("QA_VOLUME", "1"))  # lot size (Sbean_N6 min=1)

_LIVE_STATE_FILE = os.path.join(_OUT_DIR, "live_state.json")
_LIVE_LOG_FILE = os.path.join(_OUT_DIR, "live_log.jsonl")


# =========================================================================
#  TIME HELPERS
# =========================================================================

def _now_ct() -> datetime:
    """Current time in CT (Central Time)."""
    utc_now = datetime.now(timezone.utc)
    ct = utc_now + timedelta(hours=CT_OFFSET_HOURS)
    return ct


def _is_rth(ct: datetime = None) -> bool:
    """Check if current CT time is within RTH (08:30 - 13:20)."""
    if ct is None:
        ct = _now_ct()
    t = ct.hour * 60 + ct.minute
    start = RTH_START_CT[0] * 60 + RTH_START_CT[1]
    end = RTH_END_CT[0] * 60 + RTH_END_CT[1]
    return start <= t <= end


def _mins_to_close(ct: datetime = None) -> int:
    """Minutes remaining until RTH close."""
    if ct is None:
        ct = _now_ct()
    t = ct.hour * 60 + ct.minute
    end = RTH_END_CT[0] * 60 + RTH_END_CT[1]
    return max(0, end - t)


def _is_weekday() -> bool:
    """Check if today is a weekday (Mon-Fri)."""
    return _now_ct().weekday() < 5


# =========================================================================
#  LLM COST GATE
# =========================================================================

def _load_live_state() -> dict:
    """Load persistent live runner state."""
    if os.path.exists(_LIVE_STATE_FILE):
        try:
            with open(_LIVE_STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"date": "", "llm_calls": 0, "cycles": 0, "last_signal_time": None}


def _save_live_state(state: dict):
    """Save live runner state."""
    os.makedirs(_OUT_DIR, exist_ok=True)
    with open(_LIVE_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def _can_call_llm(state: dict) -> tuple:
    """Check if we can make another LLM call today."""
    today = date.today().isoformat()
    if state.get("date") != today:
        state["date"] = today
        state["llm_calls"] = 0
        state["cycles"] = 0

    if state["llm_calls"] >= MAX_LLM_CALLS_PER_DAY:
        return False, f"max {MAX_LLM_CALLS_PER_DAY} LLM calls/day reached ({state['llm_calls']})"
    return True, "ok"


# =========================================================================
#  LIVE LOG
# =========================================================================

def _log_live_event(event_type: str, data: dict):
    """Append event to live log (JSONL)."""
    os.makedirs(_OUT_DIR, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "ct_time": _now_ct().strftime("%H:%M"),
        "type": event_type,
        **data,
    }
    try:
        with open(_LIVE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


# =========================================================================
#  CORE: EVALUATE PENDING WITH LIVE BARS
# =========================================================================

def _evaluate_pending_live(bars_df: pd.DataFrame) -> int:
    """Evaluate pending paper trades using live MT5 bar data."""
    from src.quantagent.paper_log import _LOG_FILE
    if not os.path.exists(_LOG_FILE) or bars_df.empty:
        return 0

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

    # Use the index as datetime (MT5 bridge sets DatetimeIndex)
    bar_times = bars_df.index.to_series()

    n_evaluated = 0
    for trade in trades:
        if trade.get("evaluated") or trade.get("signal") == "FLAT":
            continue

        ts = pd.to_datetime(trade["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")

        future_bars = bars_df[bars_df.index > ts]
        if len(future_bars) < 4:
            continue

        bar_4h = future_bars.iloc[3]
        entry_price = trade.get("entry") or trade.get("price_at_signal", 0)

        trade["price_after_4h"] = round(float(bar_4h["close"]), 2)

        if entry_price:
            if trade["signal"] == "LONG":
                trade["pnl_4h"] = round(float(bar_4h["close"]) - entry_price, 2)
            elif trade["signal"] == "SHORT":
                trade["pnl_4h"] = round(entry_price - float(bar_4h["close"]), 2)

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

        if len(future_bars) >= 5:
            bar_1d = future_bars.iloc[4]
            trade["price_after_1d"] = round(float(bar_1d["close"]), 2)
            if entry_price:
                if trade["signal"] == "LONG":
                    trade["pnl_1d"] = round(float(bar_1d["close"]) - entry_price, 2)
                elif trade["signal"] == "SHORT":
                    trade["pnl_1d"] = round(entry_price - float(bar_1d["close"]), 2)

        trade["evaluated"] = True
        trade["eval_source"] = "mt5_live"
        n_evaluated += 1

    if n_evaluated > 0:
        with open(_LOG_FILE, "w", encoding="utf-8") as f:
            for t in trades:
                f.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")

    return n_evaluated


# =========================================================================
#  CORE: RUN ONE CYCLE
# =========================================================================

def run_live_cycle(force_llm: bool = False) -> dict:
    """
    Run one live cycle:
      1. Connect to MT5 (if not connected)
      2. Fetch live 60m bars
      3. Build features
      4. Evaluate pending trades
      5. If RTH + can_call_llm: run Trend + Risk agents
      6. Publish signal
      7. Return result

    Args:
        force_llm: bypass cost gate for LLM calls

    Returns:
        dict with cycle result
    """
    os.makedirs(_OUT_DIR, exist_ok=True)
    t0 = time.time()
    ct = _now_ct()
    rth = _is_rth(ct)
    mtc = _mins_to_close(ct)
    state = _load_live_state()

    # Update cycle counter
    today = date.today().isoformat()
    if state.get("date") != today:
        state = {"date": today, "llm_calls": 0, "cycles": 0, "last_signal_time": None}
    state["cycles"] += 1

    print(f"\n{'='*60}")
    print(f"[LIVE] Cycle #{state['cycles']} | CT: {ct.strftime('%H:%M')} | "
          f"RTH: {'YES' if rth else 'NO'} | Mins to close: {mtc}")
    print(f"[LIVE] LLM calls today: {state['llm_calls']}/{MAX_LLM_CALLS_PER_DAY}")

    # 1. Connect to MT5
    if not mt5_init():
        err = "MT5 connection failed"
        _log_live_event("error", {"message": err})
        _save_live_state(state)
        return {"error": err}

    # 2. Fetch live bars
    print("[LIVE] Fetching live 60m bars from MT5...")
    bars = fetch_mt5_bars(interval="60m", n_bars=500)
    if bars.empty:
        err = "No bars from MT5"
        _log_live_event("error", {"message": err})
        _save_live_state(state)
        return {"error": err}

    # Get current tick
    tick = get_live_tick()
    current_price = tick["last"] if tick else float(bars["close"].iloc[-1])
    print(f"[LIVE] Got {len(bars)} bars | Current price: {current_price:.2f}")

    # 3. Build features
    #    MT5 bridge returns DatetimeIndex; microstructure expects 'datetime' column
    print("[LIVE] Building microstructure features...")
    try:
        bars_for_feat = bars.copy()
        if "datetime" not in bars_for_feat.columns:
            bars_for_feat = bars_for_feat.reset_index()
            # The index from MT5 becomes a column — rename to 'datetime'
            idx_col = bars_for_feat.columns[0]
            if idx_col != "datetime":
                bars_for_feat = bars_for_feat.rename(columns={idx_col: "datetime"})
        feat = build_intraday_features(bars_for_feat, interval="60m")
        print(f"[LIVE] Features built: {feat.shape}")
    except Exception as e:
        err = f"Feature build failed: {e}"
        print(f"[LIVE] {err}")
        _log_live_event("error", {"message": err})
        _save_live_state(state)
        return {"error": err}

    # 4. Evaluate pending paper trades
    print("[LIVE] Evaluating pending paper trades...")
    try:
        n_eval = _evaluate_pending_live(bars)
        if n_eval:
            print(f"[LIVE] Evaluated {n_eval} pending trades with live data")
    except Exception as e:
        print(f"[LIVE] Eval failed (non-blocking): {e}")

    # 5. Should we call LLM agents?
    run_agents = False
    skip_reason = None

    if not rth and not force_llm:
        skip_reason = "outside RTH"
    elif mtc < 30 and not force_llm:
        skip_reason = f"too close to RTH close ({mtc} min remaining)"
    elif not _is_weekday() and not force_llm:
        skip_reason = "weekend"
    else:
        can, reason = _can_call_llm(state)
        if not can and not force_llm:
            skip_reason = reason
        else:
            run_agents = True

    if not run_agents:
        print(f"[LIVE] Skipping LLM agents: {skip_reason}")
        _log_live_event("monitor", {
            "price": current_price,
            "bars": len(bars),
            "rth": rth,
            "skip_reason": skip_reason,
        })

        # Return monitoring data without signal
        elapsed = time.time() - t0
        result = {
            "timestamp": datetime.now().isoformat(),
            "mode": "monitor",
            "current_price": current_price,
            "skip_reason": skip_reason,
            "rth": rth,
            "bars_count": len(bars),
            "execution_time_seconds": round(elapsed, 1),
        }

        # Still include positions + account info
        try:
            result["positions"] = get_positions()
            result["account"] = get_account_info()
        except Exception:
            pass

        _save_live_state(state)
        return result

    # 6. Run QuantAgent pipeline
    print("[LIVE] Running QuantAgent pipeline with live MT5 data...")

    # Build summary for LLM
    summary = _build_bars_summary(feat)
    summary["current_state"]["close"] = round(current_price, 2)  # update with live tick

    # Load track record + fundamental context
    track_record = _build_track_record()
    fundamental_context = ""
    try:
        fundamental_context = _build_fundamental_context()
        if fundamental_context:
            print(f"[LIVE] Fundamental context: {len(fundamental_context)} chars")
    except Exception as e:
        print(f"[LIVE] Fundamental context failed (non-blocking): {e}")

    extra_context = track_record + fundamental_context

    # Call agents
    print("[LIVE] Calling Trend Agent...")
    trend = call_trend_agent(summary, track_record=extra_context)
    print(f"[LIVE] Trend: {trend.get('trend', '?')} | "
          f"Setup: {trend.get('setup', {}).get('direction', '?')} | "
          f"Confidence: {trend.get('setup', {}).get('confidence', '?')}")

    print("[LIVE] Calling Risk Agent...")
    risk = call_risk_agent(summary, trend, track_record=extra_context)
    print(f"[LIVE] Risk: viable={risk.get('trade_viable', '?')} | "
          f"vol={risk.get('volatility_regime', '?')}")

    # Count LLM calls
    state["llm_calls"] += 2  # trend + risk = 2 calls

    # 7. Synthesize signal
    signal = _synthesize_signal(trend, risk, current_price)
    elapsed = time.time() - t0

    # Build result
    result = {
        "timestamp": datetime.now().isoformat(),
        "mode": "live",
        "data_source": "mt5",
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
        "has_fundamental_context": bool(fundamental_context),
        "execution_time_seconds": round(elapsed, 1),
        "from_cache": False,
        "live_stats": {
            "llm_calls_today": state["llm_calls"],
            "cycles_today": state["cycles"],
            "rth": rth,
            "mins_to_close": mtc,
        },
    }

    # Add positions + account
    try:
        result["positions"] = get_positions()
        result["account"] = get_account_info()
    except Exception:
        pass

    # Evaluation stats
    try:
        eval_log = load_log()
        result["evaluation_stats"] = eval_log.get("summary", {})
    except Exception:
        pass

    # 8. Save signal
    try:
        with open(_SIGNAL_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"[LIVE] Signal saved to {_SIGNAL_FILE}")
    except Exception as e:
        print(f"[LIVE] Save failed: {e}")

    # 9. Log to paper trades
    try:
        log_signal(result)
        print("[LIVE] Paper trade logged")
    except Exception as e:
        print(f"[LIVE] Paper log failed: {e}")

    # 10. Log event
    _log_live_event("signal", {
        "signal": signal.get("signal"),
        "confidence": signal.get("confidence"),
        "price": current_price,
        "entry": signal.get("entry"),
        "sl": signal.get("stop_loss"),
        "tp": signal.get("take_profit"),
        "elapsed": round(elapsed, 1),
    })

    # 10b. EXECUTE on MT5 (demo or live)
    if EXECUTE_TRADES and signal.get("signal") in ("LONG", "SHORT"):
        direction = "BUY" if signal["signal"] == "LONG" else "SELL"
        sl_price = signal.get("stop_loss")
        tp_price = signal.get("take_profit")
        volume = DEFAULT_VOLUME

        print(f"\n[EXEC] Placing {direction} order | Vol: {volume} | "
              f"SL: {sl_price} | TP: {tp_price}")

        try:
            order_result = place_order(
                direction=direction,
                volume=volume,
                sl=sl_price,
                tp=tp_price,
                comment=f"QA_{signal.get('confidence', 'X')[:1]}",
                dry_run=False,
            )
            result["execution"] = order_result

            if order_result.get("ok"):
                print(f"[EXEC] ORDER FILLED | Ticket: {order_result.get('order')} | "
                      f"Price: {order_result.get('price')} | Vol: {order_result.get('volume')}")
                _log_live_event("execution", {
                    "direction": direction,
                    "fill_price": order_result.get("price"),
                    "ticket": order_result.get("order"),
                    "volume": order_result.get("volume"),
                    "signal_price": signal.get("entry"),
                    "slippage": round((order_result.get("price", 0) - (signal.get("entry") or 0)), 2)
                        if signal.get("entry") else None,
                })
            else:
                print(f"[EXEC] ORDER FAILED | Code: {order_result.get('retcode')} | "
                      f"Reason: {order_result.get('comment', order_result.get('error'))}")
                _log_live_event("exec_failed", {
                    "direction": direction,
                    "retcode": order_result.get("retcode"),
                    "error": order_result.get("comment", order_result.get("error")),
                })
        except Exception as e:
            print(f"[EXEC] Order error: {e}")
            _log_live_event("exec_error", {"error": str(e)})
    elif EXECUTE_TRADES and signal.get("signal") == "FLAT":
        print("[EXEC] Signal is FLAT - no order placed")

    # 11. Optional: POST to Render API
    if POST_TO_API:
        try:
            import urllib.request
            payload = json.dumps({
                "signal": signal,
                "price": current_price,
                "timestamp": result["timestamp"],
            }, default=str).encode("utf-8")
            req = urllib.request.Request(
                POST_TO_API,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                print(f"[LIVE] Posted to API: {resp.status}")
        except Exception as e:
            print(f"[LIVE] API post failed (non-blocking): {e}")

    state["last_signal_time"] = datetime.now().isoformat()
    _save_live_state(state)

    print(f"\n[LIVE] SIGNAL: {signal['signal']} | Conf: {signal.get('confidence','?')} | "
          f"Entry: {signal.get('entry','?')} | SL: {signal.get('stop_loss','?')} | "
          f"TP: {signal.get('take_profit','?')}")
    print(f"[LIVE] Done in {elapsed:.1f}s")

    return result


# =========================================================================
#  CONTINUOUS LOOP
# =========================================================================

def run_loop(cycle_minutes: int = None):
    """
    Run continuous loop: cycle every N minutes.
    Runs during and outside RTH (monitoring mode outside RTH).
    Ctrl+C to stop.
    """
    cycle_min = cycle_minutes or CYCLE_MINUTES
    print(f"\n{'='*60}")
    print(f"[LIVE] Starting continuous loop")
    print(f"[LIVE] Cycle interval: {cycle_min} min")
    print(f"[LIVE] RTH: {RTH_START_CT[0]}:{RTH_START_CT[1]:02d} - "
          f"{RTH_END_CT[0]}:{RTH_END_CT[1]:02d} CT")
    print(f"[LIVE] Max LLM calls/day: {MAX_LLM_CALLS_PER_DAY}")
    print(f"[LIVE] Execute trades: {'YES' if EXECUTE_TRADES else 'NO (paper only)'}")
    if EXECUTE_TRADES:
        print(f"[LIVE] Volume: {DEFAULT_VOLUME} lots")
    print(f"[LIVE] Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    try:
        while True:
            try:
                result = run_live_cycle()

                # Show quick summary
                mode = result.get("mode", "?")
                price = result.get("current_price", "?")
                if mode == "live":
                    sig = result.get("signal", {})
                    print(f"\n>> [{mode.upper()}] Price: {price} | "
                          f"Signal: {sig.get('signal','?')} ({sig.get('confidence','?')})")
                else:
                    print(f"\n>> [{mode.upper()}] Price: {price} | "
                          f"Skip: {result.get('skip_reason','?')}")

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"\n[LIVE] Cycle error: {e}")
                traceback.print_exc()
                _log_live_event("crash", {"error": str(e), "traceback": traceback.format_exc()})

            # Sleep until next cycle
            # Outside RTH: sleep longer (30 min)
            if _is_rth():
                sleep_sec = cycle_min * 60
            else:
                sleep_sec = max(cycle_min * 60, 30 * 60)  # at least 30 min outside RTH

            next_ct = _now_ct() + timedelta(seconds=sleep_sec)
            print(f"\n[LIVE] Next cycle at {next_ct.strftime('%H:%M')} CT "
                  f"(sleeping {sleep_sec // 60} min)...")
            time.sleep(sleep_sec)

    except KeyboardInterrupt:
        print("\n\n[LIVE] Stopped by user")
    finally:
        print("[LIVE] Shutting down MT5...")
        mt5_shutdown()
        print("[LIVE] Done.")


# =========================================================================
#  CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="QuantAgent Live Runner (MT5)")
    parser.add_argument("--loop", action="store_true", help="Run continuous loop")
    parser.add_argument("--cycle", type=int, default=None, help="Cycle interval in minutes")
    parser.add_argument("--force", action="store_true", help="Force LLM call (bypass gate)")
    parser.add_argument("--execute", action="store_true", help="Execute trades on MT5 (demo/live)")
    parser.add_argument("--volume", type=float, default=None, help="Lot size (default 0.01)")
    parser.add_argument("--diagnose", action="store_true", help="Run MT5 diagnostic only")
    args = parser.parse_args()

    # Enable execution mode
    if args.execute:
        global EXECUTE_TRADES, DEFAULT_VOLUME
        EXECUTE_TRADES = True
        if args.volume:
            DEFAULT_VOLUME = args.volume
        acc = None
        try:
            mt5_init()
            acc = get_account_info()
        except Exception:
            pass
        mode = acc.get("trade_mode", "?") if acc else "?"
        balance = acc.get("balance", "?") if acc else "?"
        print(f"\n*** EXECUTION MODE ON ***")
        print(f"*** Account: {mode.upper()} | Balance: ${balance} | Volume: {DEFAULT_VOLUME} lots ***")
        print(f"*** Trades will be placed automatically on MT5 ***\n")

    if args.diagnose:
        print("=== MT5 Live Runner Diagnostic ===\n")
        diag = mt5_diagnose()
        print(json.dumps(diag, indent=2, default=str))
        mt5_shutdown()
        return

    if args.loop:
        run_loop(cycle_minutes=args.cycle)
    else:
        result = run_live_cycle(force_llm=args.force)
        sig = result.get("signal", {})
        if isinstance(sig, dict):
            print(f"\nResult: {sig.get('signal', '?')} ({sig.get('confidence', '?')})")
        mt5_shutdown()


if __name__ == "__main__":
    main()
