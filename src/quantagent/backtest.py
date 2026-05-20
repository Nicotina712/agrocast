"""
QuantAgent-lite Backtest — Replay historical 60m bars through LLM agents.

Approach:
  - Walk through last N RTH days of 60m bars
  - At each sample point (~11:00 CT mid-session), build a snapshot
  - Call Trend + Risk agents with only data available up to that point
  - Record signal, entry, SL, TP
  - Evaluate against ACTUAL subsequent bars (4-bar lookahead)
  - Track cumulative P&L, drawdown, win rate

Cost control:
  - 1 sample per day × 30 days = 60 API calls ≈ $3-5
  - ~15-20 min total execution time

Usage:
  python -m src.quantagent.backtest [--days 30] [--capital 1000] [--sample-hour 11]
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, date

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.intraday.data.tick_feed import fetch_intraday_bars
from src.intraday.features.microstructure import build_intraday_features
from src.quantagent.agents import call_trend_agent, call_risk_agent

_OUT_DIR = os.path.join(_ROOT, "artifacts", "quantagent")
_BT_FILE = os.path.join(_OUT_DIR, "backtest_results.json")


def _build_snapshot_summary(feat: pd.DataFrame, idx: int) -> dict:
    """
    Build bars_summary as if we only see data up to feat.iloc[idx].
    Replicates runner._build_bars_summary but on a historical slice.
    """
    # Only use data up to (and including) idx — no future leakage
    hist = feat.iloc[:idx + 1].copy()

    # Filter RTH
    rth = hist[hist["is_rth"] == 1] if "is_rth" in hist.columns else hist
    if rth.empty:
        rth = hist

    last = rth.iloc[-1]
    last_12 = rth.tail(12)

    def _safe(val):
        if pd.isna(val):
            return None
        return round(float(val), 4)

    current_state = {
        "close": round(float(last["close"]), 2),
        "session_date": str(last.get("date_ct", "")),
        "hour_ct": int(last.get("hour_ct", 0)),
        "mins_to_close": float(last.get("mins_to_close", 0)) if pd.notna(last.get("mins_to_close")) else None,
        "is_rth": bool(last.get("is_rth", 0)),
    }

    indicators = {
        "rsi_14": _safe(last.get("rsi_14")),
        "atr_14": _safe(last.get("atr_14")),
        "atr_mean_30": _safe(rth["atr_14"].tail(30).mean()) if "atr_14" in rth.columns else None,
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

    returns = {
        "ret_1": _safe(last.get("ret_1")),
        "ret_3": _safe(last.get("ret_3")),
        "ret_12": _safe(last.get("ret_12")),
    }

    # Session return
    today_date = last.get("date_ct")
    today_bars = rth[rth["date_ct"] == today_date]
    if not today_bars.empty:
        sess_open = float(today_bars.iloc[0]["open"])
        sess_ret = (float(last["close"]) - sess_open) / sess_open * 100
        returns["session_return"] = round(sess_ret, 3)

    # Recent bars
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
    recent_bars.reverse()

    # Session range
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

    # Multi-day (last 5 completed days before current)
    multi_day = []
    if "date_ct" in rth.columns:
        daily = rth.groupby("date_ct").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        )
        # Exclude current day, take last 5
        completed = daily[daily.index < today_date].tail(5)
        for dt, row in completed.iterrows():
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


def _evaluate_trade(signal: dict, future_bars: pd.DataFrame, capital: float) -> dict:
    """
    Evaluate a trade signal against actual subsequent price bars.
    Returns P&L, SL/TP hit status, max adverse excursion, etc.
    """
    direction = signal.get("signal", "FLAT")
    entry = signal.get("entry")
    sl = signal.get("stop_loss")
    tp = signal.get("take_profit")
    contracts = signal.get("contracts", 0)

    if direction == "FLAT" or not entry or contracts == 0:
        return {
            "traded": False,
            "reason": "FLAT signal or no entry",
            "pnl_usd": 0,
            "pnl_pct": 0,
        }

    if future_bars.empty:
        return {
            "traded": False,
            "reason": "No future bars available",
            "pnl_usd": 0,
            "pnl_pct": 0,
        }

    # MZS: 1 point = $10/contract (1000 bushels, price in cents/bushel, so 1 cent = $10)
    # Actually: MZS = 1000 bushels, quoted in cents. 1 point (1 cent) = $10
    # Tick = 0.125 cents = $1.25
    point_value = 10.0  # USD per point per MZS contract

    # Walk through future bars to check SL/TP hits
    hit_sl = False
    hit_tp = False
    exit_price = None
    exit_bar = None
    max_favorable = 0
    max_adverse = 0

    # Max hold: use 4 bars (4 hours) or agent's recommendation
    max_bars = min(len(future_bars), signal.get("max_hold_bars", 4) or 4)

    for i in range(max_bars):
        bar = future_bars.iloc[i]
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])

        if direction == "LONG":
            # Check SL hit (price goes below SL)
            if sl and bar_low <= sl and not hit_tp:
                hit_sl = True
                exit_price = sl
                exit_bar = i + 1
                break
            # Check TP hit
            if tp and bar_high >= tp and not hit_sl:
                hit_tp = True
                exit_price = tp
                exit_bar = i + 1
                break
            # Track excursions
            max_favorable = max(max_favorable, bar_high - entry)
            max_adverse = min(max_adverse, bar_low - entry)

        elif direction == "SHORT":
            # Check SL hit (price goes above SL)
            if sl and bar_high >= sl and not hit_tp:
                hit_sl = True
                exit_price = sl
                exit_bar = i + 1
                break
            # Check TP hit
            if tp and bar_low <= tp and not hit_sl:
                hit_tp = True
                exit_price = tp
                exit_bar = i + 1
                break
            max_favorable = max(max_favorable, entry - bar_low)
            max_adverse = min(max_adverse, entry - bar_high)

    # If neither SL nor TP hit, exit at close of last bar
    if exit_price is None:
        last_close = float(future_bars.iloc[max_bars - 1]["close"])
        exit_price = last_close
        exit_bar = max_bars

    # Calculate P&L
    if direction == "LONG":
        pnl_points = exit_price - entry
    else:  # SHORT
        pnl_points = entry - exit_price

    pnl_usd = pnl_points * point_value * contracts
    pnl_pct = (pnl_usd / capital) * 100

    return {
        "traded": True,
        "direction": direction,
        "entry": round(entry, 2),
        "exit_price": round(exit_price, 2),
        "exit_bar": exit_bar,
        "hit_sl": hit_sl,
        "hit_tp": hit_tp,
        "sl": round(sl, 2) if sl else None,
        "tp": round(tp, 2) if tp else None,
        "contracts": contracts,
        "pnl_points": round(pnl_points, 2),
        "pnl_usd": round(pnl_usd, 2),
        "pnl_pct": round(pnl_pct, 2),
        "max_favorable_excursion": round(max_favorable, 2),
        "max_adverse_excursion": round(max_adverse, 2),
    }


def run_backtest(
    days: int = 30,
    capital: float = 1000.0,
    sample_hour: int = 11,
    verbose: bool = True,
) -> dict:
    """
    Run historical backtest of QuantAgent-lite.

    Args:
        days: Number of RTH days to test
        capital: Starting capital in USD
        sample_hour: CT hour to sample each day (default 11 = mid-session)
        verbose: Print progress

    Returns:
        Full backtest results with trade log and statistics
    """
    os.makedirs(_OUT_DIR, exist_ok=True)

    if verbose:
        print(f"[BT] Starting backtest: {days} days, ${capital:,.0f} capital")
        print(f"[BT] Fetching 60m bars...")

    # 1. Fetch all available 60m bars
    bars = fetch_intraday_bars(interval="60m", use_cache=True, cache_max_age_min=120)
    if bars.empty:
        return {"error": "No bars available"}

    if verbose:
        print(f"[BT] Building features on {len(bars)} bars...")

    feat = build_intraday_features(bars, interval="60m")

    # Filter RTH bars only
    rth_feat = feat[feat["is_rth"] == 1].copy() if "is_rth" in feat.columns else feat.copy()
    if rth_feat.empty:
        return {"error": "No RTH bars found"}

    # 2. Find sample points: one per day at ~sample_hour CT
    if "date_ct" not in rth_feat.columns or "hour_ct" not in rth_feat.columns:
        return {"error": "Missing date_ct/hour_ct columns"}

    unique_days = sorted(rth_feat["date_ct"].unique())

    if verbose:
        print(f"[BT] Total RTH days available: {len(unique_days)}")
        print(f"[BT] date_ct type: {type(unique_days[0]) if unique_days else '?'}")

    # Take last N+5 days (extra buffer for evaluation of last trades)
    test_days = unique_days[-(days + 5):-5] if len(unique_days) > days + 5 else unique_days[:-5]

    if not test_days:
        return {"error": f"Not enough historical days. Have {len(unique_days)}, need {days + 5}"}

    if verbose:
        print(f"[BT] {len(test_days)} test days from {test_days[0]} to {test_days[-1]}")

    # 3. Run agents on each sample point
    trades = []
    running_capital = capital
    peak_capital = capital
    max_drawdown = 0
    total_api_time = 0

    for i, day in enumerate(test_days):
        day_bars = rth_feat[rth_feat["date_ct"] == day]
        # Find bar closest to sample_hour
        sample_bars = day_bars[day_bars["hour_ct"] == sample_hour]
        if sample_bars.empty:
            # Try adjacent hours
            for h in [sample_hour - 1, sample_hour + 1, sample_hour - 2]:
                sample_bars = day_bars[day_bars["hour_ct"] == h]
                if not sample_bars.empty:
                    break
        if sample_bars.empty:
            if verbose:
                print(f"  [{i+1}/{len(test_days)}] {day}: no bar at hour ~{sample_hour}, skipping")
            continue

        # Use the index position in the full feat DataFrame
        sample_idx = feat.index.get_loc(sample_bars.index[-1])

        if verbose:
            price = round(float(feat.iloc[sample_idx]["close"]), 2)
            print(f"  [{i+1}/{len(test_days)}] {day} h{int(feat.iloc[sample_idx]['hour_ct']):02d}: "
                  f"price={price}, capital=${running_capital:,.2f} ...", end="", flush=True)

        # Build snapshot (no future leakage)
        try:
            snapshot = _build_snapshot_summary(feat, sample_idx)
        except Exception as e:
            if verbose:
                print(f" snapshot error: {e}")
            continue

        # Update capital reference for Risk Agent
        # We adjust the system prompt context via the snapshot
        snapshot["_backtest_capital"] = running_capital

        # Call agents
        t0 = time.time()
        try:
            trend = call_trend_agent(snapshot)
            risk = call_risk_agent(snapshot, trend)
        except Exception as e:
            if verbose:
                print(f" agent error: {e}")
            continue
        api_time = time.time() - t0
        total_api_time += api_time

        # Synthesize signal (reuse runner logic)
        from src.quantagent.runner import _synthesize_signal
        current_price = snapshot["current_state"]["close"]
        signal = _synthesize_signal(trend, risk, current_price)

        # Scale contracts to our capital (agents assume $10k, we have $1k)
        if signal.get("contracts", 0) > 0:
            scale = running_capital / 10000.0
            signal["contracts"] = max(1, round(signal["contracts"] * scale))

        # Evaluate against actual future bars
        future_start = sample_idx + 1
        future_end = min(sample_idx + 20, len(feat))  # up to 20 bars ahead
        future_bars = feat.iloc[future_start:future_end]

        evaluation = _evaluate_trade(signal, future_bars, running_capital)

        # Update running capital
        if evaluation["traded"]:
            running_capital += evaluation["pnl_usd"]
            running_capital = round(running_capital, 2)
            peak_capital = max(peak_capital, running_capital)
            current_dd = (peak_capital - running_capital) / peak_capital * 100
            max_drawdown = max(max_drawdown, current_dd)

        trade_record = {
            "day": str(day),
            "hour_ct": int(feat.iloc[sample_idx]["hour_ct"]),
            "price": round(float(feat.iloc[sample_idx]["close"]), 2),
            "trend": trend.get("trend", "?"),
            "trend_strength": trend.get("trend_strength", "?"),
            "momentum": trend.get("momentum", "?"),
            "setup_type": trend.get("setup", {}).get("type", "?"),
            "signal": signal.get("signal", "FLAT"),
            "confidence": signal.get("confidence", "LOW"),
            "volatility_regime": risk.get("volatility_regime", "?"),
            "evaluation": evaluation,
            "capital_after": running_capital,
            "api_time_s": round(api_time, 1),
        }
        trades.append(trade_record)

        if verbose:
            sig = signal.get("signal", "FLAT")
            pnl = evaluation.get("pnl_usd", 0)
            status = f" {sig}"
            if evaluation["traded"]:
                status += f" -> {'WIN' if pnl > 0 else 'LOSS'} ${pnl:+.2f}"
            print(status + f" ({api_time:.0f}s)")

        # Small delay to avoid rate limits
        if api_time < 2:
            time.sleep(1)

    # 4. Compute statistics
    active_trades = [t for t in trades if t["evaluation"].get("traded")]
    flat_trades = [t for t in trades if not t["evaluation"].get("traded")]
    wins = [t for t in active_trades if t["evaluation"]["pnl_usd"] > 0]
    losses = [t for t in active_trades if t["evaluation"]["pnl_usd"] <= 0]

    total_pnl = sum(t["evaluation"]["pnl_usd"] for t in active_trades)
    total_pnl_pct = ((running_capital - capital) / capital) * 100

    avg_win = np.mean([t["evaluation"]["pnl_usd"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["evaluation"]["pnl_usd"] for t in losses]) if losses else 0
    profit_factor = abs(sum(t["evaluation"]["pnl_usd"] for t in wins) /
                        sum(t["evaluation"]["pnl_usd"] for t in losses)) if losses and sum(
                            t["evaluation"]["pnl_usd"] for t in losses) != 0 else float("inf")

    # Sharpe approximation (daily returns)
    daily_returns = [t["evaluation"]["pnl_pct"] / 100 for t in active_trades]
    sharpe = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)) if len(
        daily_returns) > 1 and np.std(daily_returns) > 0 else 0

    stats = {
        "capital_initial": capital,
        "capital_final": running_capital,
        "total_pnl_usd": round(total_pnl, 2),
        "total_return_pct": round(total_pnl_pct, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "total_samples": len(trades),
        "active_trades": len(active_trades),
        "flat_signals": len(flat_trades),
        "flat_rate_pct": round(len(flat_trades) / len(trades) * 100, 1) if trades else 0,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(active_trades) * 100, 1) if active_trades else 0,
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "sharpe_annualized": round(sharpe, 2),
        "total_api_time_s": round(total_api_time, 0),
        "avg_api_time_s": round(total_api_time / len(trades), 1) if trades else 0,
        "period": {
            "start": str(test_days[0]) if test_days else None,
            "end": str(test_days[-1]) if test_days else None,
            "days": len(test_days),
        },
    }

    # Build equity curve
    equity_curve = [{"day": "start", "capital": capital}]
    running = capital
    for t in trades:
        if t["evaluation"].get("traded"):
            running += t["evaluation"]["pnl_usd"]
        equity_curve.append({"day": t["day"], "capital": round(running, 2)})

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "days": days,
            "capital": capital,
            "sample_hour_ct": sample_hour,
        },
        "statistics": stats,
        "equity_curve": equity_curve,
        "trades": trades,
    }

    # Save
    try:
        with open(_BT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        if verbose:
            print(f"\n[BT] Results saved to {_BT_FILE}")
    except Exception as e:
        print(f"[BT] Save error: {e}")

    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS — {stats['period']['start']} to {stats['period']['end']}")
        print(f"{'='*60}")
        print(f"Capital:     ${capital:,.2f} -> ${running_capital:,.2f} ({total_pnl_pct:+.1f}%)")
        print(f"P&L:         ${total_pnl:+,.2f}")
        print(f"Max DD:      {max_drawdown:.1f}%")
        print(f"Trades:      {len(active_trades)} active / {len(flat_trades)} flat "
              f"({stats['flat_rate_pct']:.0f}% flat rate)")
        print(f"Win rate:    {stats['win_rate_pct']:.1f}% ({len(wins)}W / {len(losses)}L)")
        print(f"Avg win:     ${avg_win:+.2f} | Avg loss: ${avg_loss:+.2f}")
        print(f"Profit fct:  {stats['profit_factor']}")
        print(f"Sharpe:      {sharpe:.2f}")
        print(f"API time:    {total_api_time:.0f}s total ({stats['avg_api_time_s']:.0f}s avg)")
        print(f"{'='*60}")

        # Verdict
        print("\n--- VERDICT ---")
        if len(active_trades) < 5:
            print("INSUFICIENTE: Muy pocos trades activos para evaluar. "
                  "Los agentes son muy conservadores o el mercado estuvo sin setup.")
        elif stats["win_rate_pct"] >= 50 and total_pnl > 0 and max_drawdown < 15:
            print("PROMETEDOR: Win rate > 50%, P&L positivo, drawdown controlado.")
            print("-> RECOMENDACION: Probar en cuenta demo con sizing minimo (1 MZS).")
        elif total_pnl > 0 and max_drawdown < 20:
            print("CAUTELOSO: P&L positivo pero metricas mixtas.")
            print("-> RECOMENDACION: Extender paper trading 30 dias mas antes de demo.")
        elif total_pnl <= 0 or max_drawdown >= 20:
            print("NO LISTO: P&L negativo o drawdown excesivo.")
            print("-> RECOMENDACION: Revisar parametros de agentes, NO ir a cuenta demo.")
        else:
            print("MIXTO: Evaluar trade-by-trade para entender patrones.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantAgent-lite Backtest")
    parser.add_argument("--days", type=int, default=30, help="Number of RTH days to test")
    parser.add_argument("--capital", type=float, default=1000, help="Starting capital USD")
    parser.add_argument("--sample-hour", type=int, default=11, help="CT hour to sample")
    args = parser.parse_args()

    results = run_backtest(
        days=args.days,
        capital=args.capital,
        sample_hour=args.sample_hour,
    )
