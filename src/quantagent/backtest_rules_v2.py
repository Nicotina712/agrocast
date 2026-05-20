"""
QuantAgent-lite Rule-Based Backtest V2 — Improved rules.

Changes from V1:
  1. Breakout filter: require volume confirmation (vol_zscore > 0.8) +
     multi-bar momentum alignment (ret_3 in same direction)
  2. Tighter SL: 1.2x ATR instead of 1.5x (closer stops, less damage)
  3. Wider TP: 3.0x ATR (better R:R, let winners run)
  4. Max risk per trade: 1.5% instead of 2% (survival priority)
  5. Kill HIGH-confidence breakouts without 3-bar momentum confirmation
  6. Add mean-reversion setup: price >2 ATR from VWAP + opposing delta
  7. Reduce max_hold to 3 bars (force faster exits)
  8. Add trailing stop logic: if MFE > 1 ATR, tighten SL to entry (breakeven)

Usage:
  python -m src.quantagent.backtest_rules_v2 [--days 200] [--capital 1000]
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.intraday.data.tick_feed import fetch_intraday_bars
from src.intraday.features.microstructure import build_intraday_features

_OUT_DIR = os.path.join(_ROOT, "artifacts", "quantagent")
_BT_FILE = os.path.join(_OUT_DIR, "backtest_rules_v2_results.json")


# ─── V2 Trend Agent ─────────────────────────────────────────────────────

def _detect_trend_v2(row):
    """Trend detection — same as V1."""
    ema_cross = row.get("ema_cross", 0) or 0
    vwap_dist = row.get("vwap_dist", 0) or 0
    ret_12 = row.get("ret_12", 0) or 0
    rsi = row.get("rsi_14", 50) or 50

    score = 0
    score += np.sign(ema_cross) * min(abs(ema_cross) * 300, 2)
    score += np.sign(vwap_dist) * min(abs(vwap_dist) * 200, 1.5)
    score += np.sign(ret_12) * min(abs(ret_12) * 100, 1)
    if rsi > 60:
        score += 0.5
    elif rsi < 40:
        score -= 0.5

    if score > 1.0:
        direction = "UP"
    elif score < -1.0:
        direction = "DOWN"
    else:
        direction = "LATERAL"

    abs_score = abs(score)
    strength = "strong" if abs_score > 2.5 else ("moderate" if abs_score > 1.5 else "weak")

    return direction, strength, score


def _detect_setup_v2(row, session_bars, trend_dir, trend_score):
    """
    V2 setup detection — key changes:
      - Breakout requires vol_zscore > 0.8 AND ret_3 alignment
      - Added mean-reversion setup
      - Pullback conditions slightly relaxed
      - Reversal RSI thresholds relaxed (68/32 instead of 72/28)
    """
    rsi = row.get("rsi_14", 50) or 50
    body_pct = row.get("body_pct", 0) or 0
    atr = row.get("atr_14", 0) or 0
    close = float(row["close"])
    vol_zscore = row.get("vol_zscore_30", 0) or 0
    ema_cross = row.get("ema_cross", 0) or 0
    vwap_dist = row.get("vwap_dist", 0) or 0
    cum_delta_5 = row.get("cum_delta_5", 0) or 0
    ret_1 = row.get("ret_1", 0) or 0
    ret_3 = row.get("ret_3", 0) or 0

    if session_bars is not None and not session_bars.empty:
        sess_high = float(session_bars["high"].max())
        sess_low = float(session_bars["low"].min())
        sess_range = sess_high - sess_low
    else:
        sess_high = close + atr
        sess_low = close - atr
        sess_range = atr * 2

    setup_type = "none"
    direction = "FLAT"
    confidence = "LOW"
    description = ""

    # ── 1. BREAKOUT (V2: stricter filters) ──
    near_high = close >= sess_high - atr * 0.3
    near_low = close <= sess_low + atr * 0.3
    vol_confirmed = vol_zscore > 0.8  # V2: was 0.5
    momentum_3bar_aligned_up = ret_3 > 0.001  # V2: 3-bar return confirms direction
    momentum_3bar_aligned_down = ret_3 < -0.001

    if (near_high and trend_dir == "UP" and abs(trend_score) > 2.0  # V2: was 1.5
            and vol_confirmed and momentum_3bar_aligned_up):
        setup_type = "breakout"
        direction = "LONG"
        # V2: HIGH only if delta also confirms
        if abs(trend_score) > 2.5 and cum_delta_5 > 0 and rsi < 70:
            confidence = "HIGH"
        else:
            confidence = "MEDIUM"
        description = f"Breakout UP confirmado: near high {sess_high:.2f}, vol_z={vol_zscore:.1f}, ret3={ret_3:.4f}"

    elif (near_low and trend_dir == "DOWN" and abs(trend_score) > 2.0
          and vol_confirmed and momentum_3bar_aligned_down):
        setup_type = "breakout"
        direction = "SHORT"
        if abs(trend_score) > 2.5 and cum_delta_5 < 0 and rsi > 30:
            confidence = "HIGH"
        else:
            confidence = "MEDIUM"
        description = f"Breakout DOWN confirmado: near low {sess_low:.2f}, vol_z={vol_zscore:.1f}, ret3={ret_3:.4f}"

    # ── 2. MEAN REVERSION (V2: NEW setup) ──
    elif vwap_dist > 0.003 and rsi > 65 and body_pct < 0 and cum_delta_5 < 0:
        # Price extended above VWAP + exhaustion signs
        setup_type = "mean_reversion"
        direction = "SHORT"
        confidence = "MEDIUM" if rsi > 70 else "LOW"
        description = f"Mean reversion SHORT: {vwap_dist:.4f} above VWAP, RSI={rsi:.1f}, bear candle"

    elif vwap_dist < -0.003 and rsi < 35 and body_pct > 0 and cum_delta_5 > 0:
        setup_type = "mean_reversion"
        direction = "LONG"
        confidence = "MEDIUM" if rsi < 30 else "LOW"
        description = f"Mean reversion LONG: {vwap_dist:.4f} below VWAP, RSI={rsi:.1f}, bull candle"

    # ── 3. PULLBACK (V2: slightly relaxed) ──
    elif trend_dir == "UP" and 38 < rsi < 55 and ema_cross > 0.0003 and ret_3 > -0.003:
        setup_type = "pullback"
        direction = "LONG"
        confidence = "MEDIUM" if abs(trend_score) > 1.8 and cum_delta_5 > 0 else "LOW"
        description = f"Pullback alcista: RSI={rsi:.1f}, EMA cross +, trend score={trend_score:.1f}"

    elif trend_dir == "DOWN" and 45 < rsi < 62 and ema_cross < -0.0003 and ret_3 < 0.003:
        setup_type = "pullback"
        direction = "SHORT"
        confidence = "MEDIUM" if abs(trend_score) > 1.8 and cum_delta_5 < 0 else "LOW"
        description = f"Pullback bajista: RSI={rsi:.1f}, EMA cross -, trend score={trend_score:.1f}"

    # ── 4. REVERSAL (V2: relaxed thresholds) ──
    elif rsi > 68 and body_pct < -0.25 and cum_delta_5 < 0:
        setup_type = "reversal"
        direction = "SHORT"
        confidence = "MEDIUM" if rsi > 75 else "LOW"
        description = f"Reversal bajista: RSI={rsi:.1f} OB + bear body + neg delta"

    elif rsi < 32 and body_pct > 0.25 and cum_delta_5 > 0:
        setup_type = "reversal"
        direction = "LONG"
        confidence = "MEDIUM" if rsi < 25 else "LOW"
        description = f"Reversal alcista: RSI={rsi:.1f} OS + bull body + pos delta"

    # ── 5. COMPRESSION ──
    elif sess_range < atr * 1.0 and abs(ema_cross) < 0.0008:
        setup_type = "compression"
        direction = "FLAT"
        description = f"Compresion: rango {sess_range:.2f} < ATR {atr:.2f}"

    # ── 6. RANGE (default) ──
    else:
        setup_type = "range"
        direction = "FLAT"
        description = f"Sin setup: score={trend_score:.1f}, RSI={rsi:.1f}, vol_z={vol_zscore:.1f}"

    return {
        "type": setup_type,
        "direction": direction,
        "confidence": confidence,
        "description": description,
    }


# ─── V2 Risk Agent ──────────────────────────────────────────────────────

def rule_risk_agent_v2(row, trend_output, capital):
    """V2 Risk Agent — tighter SL, wider TP, lower max risk."""
    atr = float(row.get("atr_14", 0) or 0)
    vol_zscore = float(row.get("vol_zscore_30", 0) or 0)
    mins_to_close = float(row.get("mins_to_close", 999) or 999)
    close = float(row["close"])

    setup = trend_output.get("setup", {})
    direction = setup.get("direction", "FLAT")
    confidence = setup.get("confidence", "LOW")

    # Volatility regime
    if vol_zscore > 2.5:
        vol_regime = "extreme"
    elif vol_zscore > 1.5:
        vol_regime = "elevated"
    elif vol_zscore > 0.5:
        vol_regime = "normal"
    else:
        vol_regime = "low"

    trade_viable = True
    veto_reason = None

    if direction == "FLAT":
        trade_viable = False
        veto_reason = "FLAT — sin setup"
    elif mins_to_close < 45:  # V2: was 30
        trade_viable = False
        veto_reason = f"{mins_to_close:.0f} min to close — too late"
    elif vol_zscore > 2.0:  # V2: was 2.5
        trade_viable = False
        veto_reason = f"Vol extreme z={vol_zscore:.1f}"
    elif confidence == "LOW":  # V2: veto ALL low confidence
        trade_viable = False
        veto_reason = "Low confidence — skip"
    elif atr < 1.0:  # V2: was 0.5
        trade_viable = False
        veto_reason = f"ATR too low ({atr:.2f})"

    sl_price = None
    tp_price = None
    sl_risk_usd = None
    rr_ratio = None
    contracts = 0

    if trade_viable and atr > 0:
        # V2: Tighter SL (1.2x ATR), wider TP (3.0x ATR)
        sl_distance = atr * 1.2
        tp_distance = atr * 3.0
        entry = close

        if direction == "LONG":
            sl_price = round(entry - sl_distance, 2)
            tp_price = round(entry + tp_distance, 2)
        elif direction == "SHORT":
            sl_price = round(entry + sl_distance, 2)
            tp_price = round(entry - tp_distance, 2)

        rr_ratio = round(tp_distance / sl_distance, 2)

        if rr_ratio < 1.8:  # V2: was 1.5
            trade_viable = False
            veto_reason = f"R:R {rr_ratio}:1 < 1.8:1 min"

    if trade_viable and atr > 0:
        # V2: Max risk 1.5% instead of 2%
        max_risk_usd = capital * 0.015
        point_value = 10.0
        sl_risk_per_contract = sl_distance * point_value

        if sl_risk_per_contract > 0:
            contracts = max(1, int(max_risk_usd / sl_risk_per_contract))
            sl_risk_usd = round(sl_risk_per_contract * contracts, 2)

            # Hard cap: never risk more than 2%
            if sl_risk_usd > capital * 0.02:
                contracts = max(1, contracts - 1)
                sl_risk_usd = round(sl_risk_per_contract * contracts, 2)

        # Halve in elevated vol
        if vol_zscore > 1.5 and contracts > 1:
            contracts = max(1, contracts // 2)
            sl_risk_usd = round(sl_risk_per_contract * contracts, 2)

    return {
        "volatility_regime": vol_regime,
        "trade_viable": trade_viable,
        "veto_reason": veto_reason,
        "stop_loss": {"price": sl_price, "risk_usd": sl_risk_usd},
        "take_profit": {"price": tp_price, "risk_reward": f"{rr_ratio}:1" if rr_ratio else "N/A"},
        "position_size": {"contracts_mzs": contracts},
        "max_hold_bars": 3,  # V2: was 4
    }


# ─── Evaluate with trailing stop ────────────────────────────────────────

def _evaluate_trade_v2(signal, future_bars, capital):
    """V2 evaluation — adds breakeven trailing stop after 1 ATR MFE."""
    direction = signal.get("signal", "FLAT")
    entry = signal.get("entry")
    sl = signal.get("stop_loss")
    tp = signal.get("take_profit")
    contracts = signal.get("contracts", 0)
    atr = signal.get("_atr", 5)  # for trailing stop calc

    if direction == "FLAT" or not entry or contracts == 0:
        return {"traded": False, "pnl_usd": 0, "pnl_pct": 0}
    if future_bars.empty:
        return {"traded": False, "pnl_usd": 0, "pnl_pct": 0}

    point_value = 10.0
    max_bars = min(len(future_bars), signal.get("max_hold_bars", 3) or 3)

    exit_price = None
    hit_sl = False
    hit_tp = False
    hit_trailing = False
    max_favorable = 0
    max_adverse = 0
    active_sl = sl  # can be tightened by trailing stop

    for i in range(max_bars):
        bar = future_bars.iloc[i]
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])

        if direction == "LONG":
            # Check SL
            if active_sl and bar_low <= active_sl:
                hit_sl = True
                exit_price = active_sl
                if active_sl >= entry:
                    hit_trailing = True
                    hit_sl = False
                break
            # Check TP
            if tp and bar_high >= tp:
                hit_tp = True
                exit_price = tp
                break
            mfe = bar_high - entry
            max_favorable = max(max_favorable, mfe)
            max_adverse = min(max_adverse, bar_low - entry)
            # V2: Trailing stop — move SL to breakeven after 1 ATR profit
            if mfe >= atr and active_sl < entry:
                active_sl = entry + 0.25  # breakeven + 1 tick

        else:  # SHORT
            if active_sl and bar_high >= active_sl:
                hit_sl = True
                exit_price = active_sl
                if active_sl <= entry:
                    hit_trailing = True
                    hit_sl = False
                break
            if tp and bar_low <= tp:
                hit_tp = True
                exit_price = tp
                break
            mfe = entry - bar_low
            max_favorable = max(max_favorable, mfe)
            max_adverse = min(max_adverse, entry - bar_high)
            if mfe >= atr and active_sl > entry:
                active_sl = entry - 0.25

    if exit_price is None:
        exit_price = float(future_bars.iloc[max_bars - 1]["close"])

    pnl_pts = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
    pnl_usd = pnl_pts * point_value * contracts

    exit_type = "timeout"
    if hit_tp:
        exit_type = "tp"
    elif hit_trailing:
        exit_type = "trailing_be"
    elif hit_sl:
        exit_type = "sl"

    return {
        "traded": True,
        "direction": direction,
        "entry": round(entry, 2),
        "exit": round(exit_price, 2),
        "exit_type": exit_type,
        "contracts": contracts,
        "pnl_pts": round(pnl_pts, 2),
        "pnl_usd": round(pnl_usd, 2),
        "pnl_pct": round(pnl_usd / capital * 100, 2),
        "mfe": round(max_favorable, 2),
        "mae": round(max_adverse, 2),
        "bars_held": min(max_bars, i + 1) if exit_price != float(future_bars.iloc[max_bars - 1]["close"]) else max_bars,
    }


# ─── Main ────────────────────────────────────────────────────────────────

def run_backtest_v2(days=200, capital=1000.0, sample_hour=11, verbose=True):
    os.makedirs(_OUT_DIR, exist_ok=True)

    if verbose:
        print(f"[BT-V2] {days} days, ${capital:,.0f}, sample h{sample_hour} CT")
        print(f"[BT-V2] Fetching bars...")

    bars = fetch_intraday_bars(interval="60m", use_cache=True, cache_max_age_min=120)
    if bars.empty:
        return {"error": "No bars"}

    feat = build_intraday_features(bars, interval="60m")
    rth = feat[feat["is_rth"] == 1].copy() if "is_rth" in feat.columns else feat.copy()
    unique_days = sorted(rth["date_ct"].unique())
    test_days = unique_days[-(days + 5):-5] if len(unique_days) > days + 5 else unique_days[:-5]

    if not test_days:
        return {"error": "Not enough days"}

    if verbose:
        print(f"[BT-V2] {len(test_days)} days: {test_days[0]} to {test_days[-1]}\n")

    trades = []
    running_capital = capital
    peak_capital = capital
    max_drawdown = 0
    equity_curve = [{"day": "start", "capital": capital}]

    for i, day in enumerate(test_days):
        day_bars = rth[rth["date_ct"] == day]
        sample_bars = day_bars[day_bars["hour_ct"] == sample_hour]
        if sample_bars.empty:
            for h in [sample_hour - 1, sample_hour + 1, sample_hour + 2]:
                sample_bars = day_bars[day_bars["hour_ct"] == h]
                if not sample_bars.empty:
                    break
        if sample_bars.empty:
            continue

        row = sample_bars.iloc[-1]
        sample_idx = feat.index.get_loc(sample_bars.index[-1])
        session_bars = day_bars[day_bars.index <= sample_bars.index[-1]]

        # V2 Agents
        trend_dir, trend_str, trend_score = _detect_trend_v2(row)
        setup = _detect_setup_v2(row, session_bars, trend_dir, trend_score)
        trend = {"trend": trend_dir, "trend_strength": trend_str, "setup": setup, "_score": trend_score}

        risk = rule_risk_agent_v2(row, trend, running_capital)

        # Synthesize
        close = float(row["close"])
        atr = float(row.get("atr_14", 5) or 5)

        if not risk["trade_viable"] or setup["direction"] == "FLAT":
            signal = {"signal": "FLAT", "entry": None, "stop_loss": None,
                      "take_profit": None, "contracts": 0, "confidence": "LOW",
                      "max_hold_bars": 3, "_atr": atr}
        else:
            signal = {
                "signal": setup["direction"],
                "entry": round(close, 2),
                "stop_loss": risk["stop_loss"]["price"],
                "take_profit": risk["take_profit"]["price"],
                "contracts": risk["position_size"]["contracts_mzs"],
                "confidence": setup["confidence"],
                "max_hold_bars": risk["max_hold_bars"],
                "_atr": atr,
            }

        # Evaluate
        future = feat.iloc[sample_idx + 1:sample_idx + 20]
        evaluation = _evaluate_trade_v2(signal, future, running_capital)

        if evaluation.get("traded"):
            running_capital = round(running_capital + evaluation["pnl_usd"], 2)
            peak_capital = max(peak_capital, running_capital)
            dd = (peak_capital - running_capital) / peak_capital * 100
            max_drawdown = max(max_drawdown, dd)

        trade = {
            "day": str(day),
            "price": round(close, 2),
            "trend": trend_dir,
            "trend_str": trend_str,
            "setup": setup["type"],
            "signal": signal["signal"],
            "conf": setup["confidence"],
            "vol_regime": risk["volatility_regime"],
            "veto": risk.get("veto_reason"),
            "eval": evaluation,
            "cap": running_capital,
        }
        trades.append(trade)
        equity_curve.append({"day": str(day), "capital": running_capital})

        if verbose:
            sig = signal["signal"]
            pnl = evaluation.get("pnl_usd", 0)
            icon = "  "
            if evaluation.get("traded"):
                icon = " W" if pnl > 0 else " L"
                exit_t = evaluation.get("exit_type", "?")[:3]
            else:
                exit_t = "   "
            print(f"  [{i+1:>3}/{len(test_days)}] {day} "
                  f"| {close:>8.2f} "
                  f"| {trend_dir:>7} "
                  f"| {setup['type'][:8]:>8} "
                  f"| {sig:>5} {setup['confidence'][:3]} "
                  f"|{icon} {exit_t} ${pnl:>+8.2f} "
                  f"| ${running_capital:>9,.2f}")

    # ── Stats ────────────────────────────────────────────────────────────
    active = [t for t in trades if t["eval"].get("traded")]
    flat = [t for t in trades if not t["eval"].get("traded")]
    wins = [t for t in active if t["eval"]["pnl_usd"] > 0]
    losses = [t for t in active if t["eval"]["pnl_usd"] <= 0]

    total_pnl = sum(t["eval"]["pnl_usd"] for t in active)
    avg_win = np.mean([t["eval"]["pnl_usd"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["eval"]["pnl_usd"] for t in losses]) if losses else 0
    gross_w = sum(t["eval"]["pnl_usd"] for t in wins)
    gross_l = abs(sum(t["eval"]["pnl_usd"] for t in losses))
    pf = round(gross_w / gross_l, 2) if gross_l > 0 else float("inf")

    daily_rets = [t["eval"]["pnl_pct"] / 100 for t in active]
    sharpe = (np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)) if len(daily_rets) > 1 and np.std(daily_rets) > 0 else 0

    # Exit type breakdown
    exit_counts = {}
    for t in active:
        et = t["eval"].get("exit_type", "timeout")
        exit_counts[et] = exit_counts.get(et, 0) + 1

    # Setup breakdown
    setup_stats = {}
    for t in active:
        st = t["setup"]
        if st not in setup_stats:
            setup_stats[st] = {"n": 0, "w": 0, "pnl": 0}
        setup_stats[st]["n"] += 1
        setup_stats[st]["pnl"] = round(setup_stats[st]["pnl"] + t["eval"]["pnl_usd"], 2)
        if t["eval"]["pnl_usd"] > 0:
            setup_stats[st]["w"] += 1

    # Confidence breakdown
    conf_stats = {}
    for t in active:
        c = t["conf"]
        if c not in conf_stats:
            conf_stats[c] = {"n": 0, "w": 0, "pnl": 0}
        conf_stats[c]["n"] += 1
        conf_stats[c]["pnl"] = round(conf_stats[c]["pnl"] + t["eval"]["pnl_usd"], 2)
        if t["eval"]["pnl_usd"] > 0:
            conf_stats[c]["w"] += 1

    stats = {
        "capital_initial": capital,
        "capital_final": running_capital,
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round((running_capital - capital) / capital * 100, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "total_samples": len(trades),
        "active": len(active),
        "flat": len(flat),
        "flat_rate": round(len(flat) / len(trades) * 100, 1) if trades else 0,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(active) * 100, 1) if active else 0,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": pf if pf != float("inf") else "inf",
        "sharpe": round(sharpe, 2),
        "exits": exit_counts,
        "by_setup": setup_stats,
        "by_confidence": conf_stats,
        "period": {"start": str(test_days[0]), "end": str(test_days[-1])},
    }

    results = {
        "timestamp": datetime.now().isoformat(),
        "type": "rule_based_v2",
        "config": {"days": days, "capital": capital, "sample_hour": sample_hour},
        "changes_vs_v1": [
            "Breakout: vol_zscore > 0.8 + ret_3 alignment required",
            "SL: 1.2x ATR (was 1.5x)",
            "TP: 3.0x ATR (was 2.5x)",
            "Max risk: 1.5% (was 2%)",
            "Veto ALL low-confidence signals",
            "Added mean-reversion setup",
            "Trailing stop: SL -> breakeven after 1 ATR MFE",
            "Max hold: 3 bars (was 4)",
        ],
        "statistics": stats,
        "equity_curve": equity_curve,
        "trades": trades,
    }

    try:
        with open(_BT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"[BT-V2] Save error: {e}")

    if verbose:
        print(f"\n{'='*72}")
        print(f"  BACKTEST V2 — {stats['period']['start']} to {stats['period']['end']}")
        print(f"{'='*72}")
        print(f"  Capital:       ${capital:,.2f} -> ${running_capital:,.2f} ({stats['total_return_pct']:+.1f}%)")
        print(f"  P&L:           ${total_pnl:+,.2f}")
        print(f"  Max Drawdown:  {max_drawdown:.1f}%")
        print(f"  Trades:        {len(active)} activos / {len(flat)} flat ({stats['flat_rate']:.0f}%)")
        print(f"  Win Rate:      {stats['win_rate']:.1f}% ({len(wins)}W / {len(losses)}L)")
        print(f"  Avg Win:       ${avg_win:+.2f} | Avg Loss: ${avg_loss:+.2f}")
        print(f"  Profit Factor: {pf}")
        print(f"  Sharpe:        {sharpe:.2f}")
        print(f"  Exits:         {exit_counts}")

        if setup_stats:
            print(f"\n  --- By Setup ---")
            for st, ss in sorted(setup_stats.items(), key=lambda x: -x[1]["n"]):
                wr = round(ss['w'] / ss['n'] * 100, 0) if ss['n'] else 0
                print(f"    {st:>14}: {ss['n']:>2}T  {wr:>3.0f}%WR  ${ss['pnl']:>+9.2f}")

        if conf_stats:
            print(f"\n  --- By Confidence ---")
            for c, cs in sorted(conf_stats.items()):
                wr = round(cs['w'] / cs['n'] * 100, 0) if cs['n'] else 0
                print(f"    {c:>8}: {cs['n']:>2}T  {wr:>3.0f}%WR  ${cs['pnl']:>+9.2f}")

        # Compare V1 vs V2
        print(f"\n  --- V1 vs V2 ---")
        print(f"  {'Metric':<20} {'V1':>12} {'V2':>12} {'Delta':>12}")
        print(f"  {'Return':<20} {'-10.8%':>12} {stats['total_return_pct']:>+11.1f}% {'':>12}")
        print(f"  {'Max DD':<20} {'68.3%':>12} {max_drawdown:>11.1f}% {'':>12}")
        print(f"  {'Win Rate':<20} {'50.0%':>12} {stats['win_rate']:>11.1f}% {'':>12}")
        print(f"  {'Profit Factor':<20} {'0.89':>12} {pf:>12} {'':>12}")
        print(f"  {'Active Trades':<20} {'28':>12} {len(active):>12} {'':>12}")

        print(f"\n{'='*72}")

        # Verdict
        print("\n  --- VEREDICTO V2 ---")
        if len(active) < 5:
            print("  INSUFICIENTE: Muy pocos trades.")
        elif stats["win_rate"] >= 55 and total_pnl > 0 and max_drawdown < 15:
            print("  PROMETEDOR: WR>55%, PnL+, DD<15%.")
            print("  -> RECOMENDACION: Cuenta demo con 1 MZS.")
        elif total_pnl > 0 and max_drawdown < 25:
            print("  CAUTELOSO: P&L positivo, DD moderado.")
            print("  -> RECOMENDACION: Paper trade 30 dias mas.")
        elif total_pnl > 0 and max_drawdown < 35:
            print("  MARGINAL: P&L positivo pero DD preocupante.")
            print("  -> Ajustar sizing o continuar optimizando.")
        else:
            print("  NO LISTO: No mejoro suficiente vs V1.")
            print("  -> Revisar logica de setups.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=200)
    parser.add_argument("--capital", type=float, default=1000)
    parser.add_argument("--sample-hour", type=int, default=11)
    args = parser.parse_args()
    run_backtest_v2(days=args.days, capital=args.capital, sample_hour=args.sample_hour)
