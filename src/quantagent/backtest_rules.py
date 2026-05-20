"""
QuantAgent-lite Rule-Based Backtest — No LLM calls.

Approximates the Trend + Risk agent logic using programmatic rules
derived from the same indicators the LLM agents receive.

This is a FREE, INSTANT alternative to the LLM backtest for initial
feasibility assessment before committing API credits.

Trend Agent Rules (approximated):
  - Trend direction: EMA cross sign + price vs VWAP
  - Trend strength: magnitude of EMA cross + ret_12
  - Momentum: RSI zones + ret acceleration (ret_1 vs ret_3)
  - Setup detection:
    * Breakout: price breaks session high/low with volume
    * Pullback: trending + RSI reversion toward 50
    * Reversal: RSI extreme + opposing candle body
    * Range/Compression: narrow ATR + mixed signals
  - Confidence: combined score of alignment across indicators

Risk Agent Rules (approximated):
  - Volatility regime: vol_zscore_30 thresholds
  - SL: ATR-based (1.5x ATR from entry)
  - TP: ATR-based (2.5x ATR from entry for 1.67:1 R:R)
  - Position sizing: risk 2% of capital, divide by SL distance
  - Vetoes: vol_zscore > 2, mins_to_close < 30, R:R < 1.5

Usage:
  python -m src.quantagent.backtest_rules [--days 60] [--capital 1000]
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
_BT_FILE = os.path.join(_OUT_DIR, "backtest_rules_results.json")


# ─── Trend Agent (rule-based) ───────────────────────────────────────────

def _detect_trend(row, session_bars):
    """Determine trend direction and strength from indicators."""
    ema_cross = row.get("ema_cross", 0) or 0
    vwap_dist = row.get("vwap_dist", 0) or 0
    ret_12 = row.get("ret_12", 0) or 0
    rsi = row.get("rsi_14", 50) or 50

    # Direction scoring
    score = 0
    score += np.sign(ema_cross) * min(abs(ema_cross) * 300, 2)  # EMA cross: up to ±2
    score += np.sign(vwap_dist) * min(abs(vwap_dist) * 200, 1.5)  # VWAP dist: up to ±1.5
    score += np.sign(ret_12) * min(abs(ret_12) * 100, 1)  # 12-bar return: up to ±1

    # RSI bias
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

    # Strength
    abs_score = abs(score)
    if abs_score > 2.5:
        strength = "strong"
    elif abs_score > 1.5:
        strength = "moderate"
    else:
        strength = "weak"

    return direction, strength, score


def _detect_momentum(row):
    """Assess momentum state."""
    rsi = row.get("rsi_14", 50) or 50
    ret_1 = abs(row.get("ret_1", 0) or 0)
    ret_3 = abs(row.get("ret_3", 0) or 0)

    # Acceleration: is the move speeding up or slowing down?
    if ret_3 > 0:
        accel_ratio = (ret_1 * 3) / ret_3  # normalized to same period
    else:
        accel_ratio = 1.0

    if accel_ratio > 1.5 and (rsi > 60 or rsi < 40):
        return "accelerating"
    elif accel_ratio > 0.8:
        return "steady"
    elif accel_ratio > 0.3:
        return "decelerating"
    else:
        return "reversing"


def _detect_setup(row, session_bars, trend_dir, trend_score):
    """Detect trading setup type and generate signal."""
    rsi = row.get("rsi_14", 50) or 50
    body_pct = row.get("body_pct", 0) or 0
    atr = row.get("atr_14", 0) or 0
    close = float(row["close"])
    vol_zscore = row.get("vol_zscore_30", 0) or 0
    ema_cross = row.get("ema_cross", 0) or 0
    vwap_dist = row.get("vwap_dist", 0) or 0
    cum_delta_5 = row.get("cum_delta_5", 0) or 0
    ret_1 = row.get("ret_1", 0) or 0

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

    # 1. BREAKOUT: price near session extreme + strong trend + volume
    near_high = close >= sess_high - atr * 0.3
    near_low = close <= sess_low + atr * 0.3
    vol_active = vol_zscore > 0.5

    if near_high and trend_dir == "UP" and abs(trend_score) > 1.5 and vol_active:
        setup_type = "breakout"
        direction = "LONG"
        confidence = "HIGH" if abs(trend_score) > 2.5 and rsi < 75 else "MEDIUM"
        description = f"Breakout alcista: precio cerca de high sesion {sess_high:.2f}, trend score {trend_score:.1f}"

    elif near_low and trend_dir == "DOWN" and abs(trend_score) > 1.5 and vol_active:
        setup_type = "breakout"
        direction = "SHORT"
        confidence = "HIGH" if abs(trend_score) > 2.5 and rsi > 25 else "MEDIUM"
        description = f"Breakout bajista: precio cerca de low sesion {sess_low:.2f}, trend score {trend_score:.1f}"

    # 2. PULLBACK: trending + RSI pulling back toward neutral
    elif trend_dir == "UP" and 40 < rsi < 55 and ema_cross > 0.0005 and vwap_dist > -0.002:
        setup_type = "pullback"
        direction = "LONG"
        confidence = "MEDIUM" if abs(trend_score) > 1.5 else "LOW"
        description = f"Pullback en tendencia alcista: RSI {rsi:.1f} retrocediendo, EMA cross positivo"

    elif trend_dir == "DOWN" and 45 < rsi < 60 and ema_cross < -0.0005 and vwap_dist < 0.002:
        setup_type = "pullback"
        direction = "SHORT"
        confidence = "MEDIUM" if abs(trend_score) > 1.5 else "LOW"
        description = f"Pullback en tendencia bajista: RSI {rsi:.1f} retrocediendo, EMA cross negativo"

    # 3. REVERSAL: RSI extreme + opposing candle + delta divergence
    elif rsi > 72 and body_pct < -0.3 and cum_delta_5 < 0:
        setup_type = "reversal"
        direction = "SHORT"
        confidence = "MEDIUM" if rsi > 78 else "LOW"
        description = f"Reversal bajista: RSI {rsi:.1f} sobrecompra + vela bajista + delta negativo"

    elif rsi < 28 and body_pct > 0.3 and cum_delta_5 > 0:
        setup_type = "reversal"
        direction = "LONG"
        confidence = "MEDIUM" if rsi < 22 else "LOW"
        description = f"Reversal alcista: RSI {rsi:.1f} sobreventa + vela alcista + delta positivo"

    # 4. COMPRESSION: very narrow range relative to ATR
    elif sess_range < atr * 1.2 and abs(ema_cross) < 0.001:
        setup_type = "compression"
        direction = "FLAT"
        description = f"Compresion: rango sesion {sess_range:.2f} < 1.2x ATR {atr:.2f}"

    # 5. RANGE: no clear direction
    else:
        setup_type = "range"
        direction = "FLAT"
        description = f"Rango sin setup claro: trend score {trend_score:.1f}, RSI {rsi:.1f}"

    return {
        "type": setup_type,
        "direction": direction,
        "confidence": confidence,
        "description": description,
        "entry_zone": {
            "low": round(close - atr * 0.2, 2) if direction != "FLAT" else None,
            "high": round(close + atr * 0.2, 2) if direction != "FLAT" else None,
        },
    }


def rule_trend_agent(row, session_bars):
    """Rule-based Trend Agent."""
    trend_dir, trend_str, trend_score = _detect_trend(row, session_bars)
    momentum = _detect_momentum(row)
    setup = _detect_setup(row, session_bars, trend_dir, trend_score)

    close = float(row["close"])
    atr = row.get("atr_14", 0) or 0
    vwap = row.get("vwap_session", close) or close

    return {
        "trend": trend_dir,
        "trend_strength": trend_str,
        "momentum": momentum,
        "key_levels": {
            "resistance": [round(close + atr, 2), round(close + atr * 2, 2)],
            "support": [round(close - atr, 2), round(close - atr * 2, 2)],
            "vwap": round(vwap, 2),
        },
        "setup": setup,
        "reasoning": setup["description"],
        "_trend_score": round(trend_score, 2),
    }


# ─── Risk Agent (rule-based) ────────────────────────────────────────────

def rule_risk_agent(row, trend_output, capital):
    """Rule-based Risk Agent."""
    atr = float(row.get("atr_14", 0) or 0)
    vol_zscore = float(row.get("vol_zscore_30", 0) or 0)
    mins_to_close = float(row.get("mins_to_close", 999) or 999)
    close = float(row["close"])

    setup = trend_output.get("setup", {})
    direction = setup.get("direction", "FLAT")
    confidence = setup.get("confidence", "LOW")
    entry_zone = setup.get("entry_zone", {})

    # Volatility regime
    if vol_zscore > 2.5:
        vol_regime = "extreme"
    elif vol_zscore > 1.5:
        vol_regime = "elevated"
    elif vol_zscore > 0.5:
        vol_regime = "normal"
    else:
        vol_regime = "low"

    # ATR assessment
    atr_30 = float(row.get("atr_14", atr))  # approximate
    atr_assessment = f"ATR actual {atr:.2f} — vol regime {vol_regime}"

    # Vetoes
    trade_viable = True
    veto_reason = None

    if direction == "FLAT":
        trade_viable = False
        veto_reason = "Trend Agent emite FLAT — sin setup"

    elif mins_to_close < 30:
        trade_viable = False
        veto_reason = f"Solo {mins_to_close:.0f} min para cierre RTH — no abrir posiciones"

    elif vol_zscore > 2.5:
        trade_viable = False
        veto_reason = f"Volatilidad extrema (z-score {vol_zscore:.1f}) — demasiado riesgo"

    elif confidence == "LOW" and vol_regime != "low":
        trade_viable = False
        veto_reason = f"Confianza LOW con volatilidad {vol_regime} — R:R desfavorable"

    elif atr < 0.5:
        trade_viable = False
        veto_reason = f"ATR demasiado bajo ({atr:.2f}) — spread comerfa la ganancia"

    # SL/TP calculation
    sl_price = None
    tp_price = None
    sl_risk_usd = None
    rr_ratio = None
    contracts = 0

    if trade_viable and atr > 0:
        # SL: 1.5x ATR from entry (structure-based approximation)
        sl_distance = atr * 1.5

        # TP: target 2.0-2.5x ATR for 1.5:1+ R:R
        tp_distance = atr * 2.5

        entry = close  # simplified: enter at current price

        if direction == "LONG":
            sl_price = round(entry - sl_distance, 2)
            tp_price = round(entry + tp_distance, 2)
        elif direction == "SHORT":
            sl_price = round(entry + sl_distance, 2)
            tp_price = round(entry - tp_distance, 2)

        rr_ratio = round(tp_distance / sl_distance, 2) if sl_distance > 0 else 0

        # Validate R:R
        if rr_ratio < 1.5:
            trade_viable = False
            veto_reason = f"R:R insuficiente ({rr_ratio}:1, min 1.5:1)"

    if trade_viable and atr > 0:
        # Position sizing: risk max 2% of capital
        max_risk_usd = capital * 0.02
        # MZS: 1 point = $10/contract
        point_value = 10.0
        sl_risk_per_contract = sl_distance * point_value

        if sl_risk_per_contract > 0:
            contracts = max(1, int(max_risk_usd / sl_risk_per_contract))
            sl_risk_usd = round(sl_risk_per_contract * contracts, 2)

            # Extra safety: don't risk more than 3% even with rounding
            if sl_risk_usd > capital * 0.03:
                contracts = max(1, contracts - 1)
                sl_risk_usd = round(sl_risk_per_contract * contracts, 2)

        # Halve sizing in elevated/extreme vol
        if vol_zscore > 1.5 and contracts > 1:
            contracts = max(1, contracts // 2)
            sl_risk_usd = round(sl_risk_per_contract * contracts, 2)

    return {
        "volatility_regime": vol_regime,
        "atr_assessment": atr_assessment,
        "trade_viable": trade_viable,
        "veto_reason": veto_reason,
        "stop_loss": {
            "price": sl_price,
            "type": "atr_structure",
            "risk_usd": sl_risk_usd,
        },
        "take_profit": {
            "price": tp_price,
            "risk_reward": f"{rr_ratio}:1" if rr_ratio else "N/A",
        },
        "position_size": {
            "contracts_mzs": contracts,
            "risk_pct": round((sl_risk_usd / capital) * 100, 2) if sl_risk_usd and capital else 0,
        },
        "session_risk": "high" if mins_to_close < 60 else ("moderate" if vol_zscore > 1 else "low"),
        "max_hold_bars": 3 if vol_regime in ("elevated", "extreme") else 4,
    }


# ─── Synthesize Signal ──────────────────────────────────────────────────

def _synthesize(trend, risk, close):
    """Combine trend + risk into final signal."""
    setup = trend.get("setup", {})
    direction = setup.get("direction", "FLAT")
    confidence = setup.get("confidence", "LOW")

    if not risk.get("trade_viable", False) or direction == "FLAT":
        return {
            "signal": "FLAT",
            "reason": risk.get("veto_reason") or setup.get("description", "No setup"),
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "contracts": 0,
            "confidence": "LOW",
            "risk_reward": None,
            "max_hold_bars": None,
            "volatility_regime": risk.get("volatility_regime"),
        }

    sl = risk["stop_loss"]["price"]
    tp = risk["take_profit"]["price"]
    contracts = risk["position_size"]["contracts_mzs"]

    return {
        "signal": direction,
        "reason": setup.get("description", ""),
        "entry": round(close, 2),
        "stop_loss": sl,
        "take_profit": tp,
        "contracts": contracts,
        "confidence": confidence,
        "risk_reward": risk["take_profit"].get("risk_reward"),
        "max_hold_bars": risk.get("max_hold_bars", 4),
        "volatility_regime": risk.get("volatility_regime"),
    }


# ─── Evaluate Trade ─────────────────────────────────────────────────────

def _evaluate_trade(signal, future_bars, capital):
    """Evaluate signal against actual subsequent bars."""
    direction = signal.get("signal", "FLAT")
    entry = signal.get("entry")
    sl = signal.get("stop_loss")
    tp = signal.get("take_profit")
    contracts = signal.get("contracts", 0)

    if direction == "FLAT" or not entry or contracts == 0:
        return {"traded": False, "pnl_usd": 0, "pnl_pct": 0}

    if future_bars.empty:
        return {"traded": False, "pnl_usd": 0, "pnl_pct": 0, "reason": "no future bars"}

    point_value = 10.0
    max_bars = min(len(future_bars), signal.get("max_hold_bars", 4) or 4)

    exit_price = None
    hit_sl = False
    hit_tp = False
    max_favorable = 0
    max_adverse = 0

    for i in range(max_bars):
        bar = future_bars.iloc[i]
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])

        if direction == "LONG":
            if sl and bar_low <= sl:
                hit_sl, exit_price = True, sl
                break
            if tp and bar_high >= tp:
                hit_tp, exit_price = True, tp
                break
            max_favorable = max(max_favorable, bar_high - entry)
            max_adverse = min(max_adverse, bar_low - entry)
        else:  # SHORT
            if sl and bar_high >= sl:
                hit_sl, exit_price = True, sl
                break
            if tp and bar_low <= tp:
                hit_tp, exit_price = True, tp
                break
            max_favorable = max(max_favorable, entry - bar_low)
            max_adverse = min(max_adverse, entry - bar_high)

    if exit_price is None:
        exit_price = float(future_bars.iloc[max_bars - 1]["close"])

    pnl_pts = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
    pnl_usd = pnl_pts * point_value * contracts

    return {
        "traded": True,
        "direction": direction,
        "entry": round(entry, 2),
        "exit": round(exit_price, 2),
        "hit_sl": hit_sl,
        "hit_tp": hit_tp,
        "sl": round(sl, 2) if sl else None,
        "tp": round(tp, 2) if tp else None,
        "contracts": contracts,
        "pnl_pts": round(pnl_pts, 2),
        "pnl_usd": round(pnl_usd, 2),
        "pnl_pct": round(pnl_usd / capital * 100, 2),
        "mfe": round(max_favorable, 2),
        "mae": round(max_adverse, 2),
        "bars_held": min(max_bars, i + 1) if (hit_sl or hit_tp) else max_bars,
    }


# ─── Main Backtest ──────────────────────────────────────────────────────

def run_backtest(
    days: int = 60,
    capital: float = 1000.0,
    sample_hour: int = 11,
    verbose: bool = True,
) -> dict:
    """Run rule-based backtest over historical 60m RTH bars."""
    os.makedirs(_OUT_DIR, exist_ok=True)

    if verbose:
        print(f"[BT-Rules] Starting: {days} days, ${capital:,.0f} capital, sample h{sample_hour} CT")
        print(f"[BT-Rules] Fetching 60m bars...")

    bars = fetch_intraday_bars(interval="60m", use_cache=True, cache_max_age_min=120)
    if bars.empty:
        return {"error": "No bars available"}

    if verbose:
        print(f"[BT-Rules] Building features on {len(bars)} bars...")

    feat = build_intraday_features(bars, interval="60m")
    rth_feat = feat[feat["is_rth"] == 1].copy() if "is_rth" in feat.columns else feat.copy()

    unique_days = sorted(rth_feat["date_ct"].unique())
    # Reserve last 5 days for evaluation of final trades
    test_days = unique_days[-(days + 5):-5] if len(unique_days) > days + 5 else unique_days[:-5]

    if not test_days:
        return {"error": f"Not enough days. Have {len(unique_days)}, need {days + 5}"}

    if verbose:
        print(f"[BT-Rules] {len(test_days)} test days: {test_days[0]} to {test_days[-1]}")
        print()

    trades = []
    running_capital = capital
    peak_capital = capital
    max_drawdown = 0
    equity_curve = [{"day": "start", "capital": capital}]

    for i, day in enumerate(test_days):
        day_bars = rth_feat[rth_feat["date_ct"] == day]

        # Find sample bar
        sample_bars = day_bars[day_bars["hour_ct"] == sample_hour]
        if sample_bars.empty:
            for h in [sample_hour - 1, sample_hour + 1, sample_hour + 2]:
                sample_bars = day_bars[day_bars["hour_ct"] == h]
                if not sample_bars.empty:
                    break
        if sample_bars.empty:
            continue

        sample_row = sample_bars.iloc[-1]
        sample_idx = feat.index.get_loc(sample_bars.index[-1])

        # Session bars up to this point
        session_bars = day_bars[day_bars.index <= sample_bars.index[-1]]

        # Run rule-based agents
        trend = rule_trend_agent(sample_row, session_bars)
        risk = rule_risk_agent(sample_row, trend, running_capital)
        signal = _synthesize(trend, risk, float(sample_row["close"]))

        # Evaluate against future bars
        future_start = sample_idx + 1
        future_end = min(sample_idx + 20, len(feat))
        future = feat.iloc[future_start:future_end]
        evaluation = _evaluate_trade(signal, future, running_capital)

        # Update capital
        if evaluation.get("traded"):
            running_capital = round(running_capital + evaluation["pnl_usd"], 2)
            peak_capital = max(peak_capital, running_capital)
            dd = (peak_capital - running_capital) / peak_capital * 100
            max_drawdown = max(max_drawdown, dd)

        trade = {
            "day": str(day),
            "hour_ct": int(sample_row.get("hour_ct", 0)),
            "price": round(float(sample_row["close"]), 2),
            "trend": trend["trend"],
            "trend_strength": trend["trend_strength"],
            "trend_score": trend.get("_trend_score", 0),
            "momentum": trend["momentum"],
            "setup_type": trend["setup"]["type"],
            "signal": signal["signal"],
            "confidence": signal.get("confidence", "LOW"),
            "volatility_regime": risk["volatility_regime"],
            "veto_reason": risk.get("veto_reason"),
            "evaluation": evaluation,
            "capital_after": running_capital,
        }
        trades.append(trade)
        equity_curve.append({"day": str(day), "capital": running_capital})

        if verbose:
            sig = signal["signal"]
            pnl = evaluation.get("pnl_usd", 0)
            icon = "  "
            if evaluation.get("traded"):
                icon = " W" if pnl > 0 else " L"
            setup = trend["setup"]["type"][:6]
            conf = signal.get("confidence", "?")[:3]
            print(f"  [{i+1:>3}/{len(test_days)}] {day} h{int(sample_row['hour_ct']):02d} "
                  f"| {float(sample_row['close']):>8.2f} "
                  f"| {trend['trend']:>7} {trend['trend_strength'][:3]} "
                  f"| {setup:>6} "
                  f"| {sig:>5} {conf} "
                  f"|{icon} ${pnl:>+8.2f} "
                  f"| cap=${running_capital:>9,.2f}")

    # ── Statistics ───────────────────────────────────────────────────────
    active = [t for t in trades if t["evaluation"].get("traded")]
    flat = [t for t in trades if not t["evaluation"].get("traded")]
    wins = [t for t in active if t["evaluation"]["pnl_usd"] > 0]
    losses = [t for t in active if t["evaluation"]["pnl_usd"] <= 0]

    total_pnl = sum(t["evaluation"]["pnl_usd"] for t in active)

    avg_win = np.mean([t["evaluation"]["pnl_usd"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["evaluation"]["pnl_usd"] for t in losses]) if losses else 0

    gross_wins = sum(t["evaluation"]["pnl_usd"] for t in wins)
    gross_losses = abs(sum(t["evaluation"]["pnl_usd"] for t in losses))
    profit_factor = round(gross_wins / gross_losses, 2) if gross_losses > 0 else float("inf")

    daily_rets = [t["evaluation"]["pnl_pct"] / 100 for t in active]
    sharpe = (np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)) if len(daily_rets) > 1 and np.std(daily_rets) > 0 else 0

    # Breakdown by setup type
    setup_stats = {}
    for t in active:
        st = t["setup_type"]
        if st not in setup_stats:
            setup_stats[st] = {"trades": 0, "wins": 0, "pnl": 0}
        setup_stats[st]["trades"] += 1
        setup_stats[st]["pnl"] = round(setup_stats[st]["pnl"] + t["evaluation"]["pnl_usd"], 2)
        if t["evaluation"]["pnl_usd"] > 0:
            setup_stats[st]["wins"] += 1

    # Breakdown by confidence
    conf_stats = {}
    for t in active:
        c = t["confidence"]
        if c not in conf_stats:
            conf_stats[c] = {"trades": 0, "wins": 0, "pnl": 0}
        conf_stats[c]["trades"] += 1
        conf_stats[c]["pnl"] = round(conf_stats[c]["pnl"] + t["evaluation"]["pnl_usd"], 2)
        if t["evaluation"]["pnl_usd"] > 0:
            conf_stats[c]["wins"] += 1

    # SL/TP hit rates
    sl_hits = sum(1 for t in active if t["evaluation"].get("hit_sl"))
    tp_hits = sum(1 for t in active if t["evaluation"].get("hit_tp"))
    timeout_exits = len(active) - sl_hits - tp_hits

    stats = {
        "capital_initial": capital,
        "capital_final": running_capital,
        "total_pnl_usd": round(total_pnl, 2),
        "total_return_pct": round((running_capital - capital) / capital * 100, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "total_samples": len(trades),
        "active_trades": len(active),
        "flat_signals": len(flat),
        "flat_rate_pct": round(len(flat) / len(trades) * 100, 1) if trades else 0,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(active) * 100, 1) if active else 0,
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "profit_factor": profit_factor if profit_factor != float("inf") else "inf",
        "sharpe_annualized": round(sharpe, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "exit_stats": {
            "sl_hits": sl_hits,
            "tp_hits": tp_hits,
            "timeout_exits": timeout_exits,
        },
        "by_setup": setup_stats,
        "by_confidence": conf_stats,
        "period": {
            "start": str(test_days[0]),
            "end": str(test_days[-1]),
            "days": len(test_days),
        },
    }

    results = {
        "timestamp": datetime.now().isoformat(),
        "type": "rule_based_backtest",
        "config": {"days": days, "capital": capital, "sample_hour_ct": sample_hour},
        "statistics": stats,
        "equity_curve": equity_curve,
        "trades": trades,
    }

    # Save
    try:
        with open(_BT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        if verbose:
            print(f"\n[BT-Rules] Saved to {_BT_FILE}")
    except Exception as e:
        print(f"[BT-Rules] Save error: {e}")

    # Print summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"  BACKTEST RULE-BASED — {stats['period']['start']} to {stats['period']['end']}")
        print(f"{'='*70}")
        print(f"  Capital:       ${capital:,.2f} -> ${running_capital:,.2f} ({stats['total_return_pct']:+.1f}%)")
        print(f"  P&L:           ${total_pnl:+,.2f}")
        print(f"  Max Drawdown:  {max_drawdown:.1f}%")
        print(f"  Trades:        {len(active)} activos / {len(flat)} flat ({stats['flat_rate_pct']:.0f}% flat)")
        print(f"  Win Rate:      {stats['win_rate_pct']:.1f}% ({len(wins)}W / {len(losses)}L)")
        print(f"  Avg Win:       ${avg_win:+.2f}")
        print(f"  Avg Loss:      ${avg_loss:+.2f}")
        print(f"  Profit Factor: {profit_factor}")
        print(f"  Sharpe:        {sharpe:.2f}")
        print(f"  Exits:         {tp_hits} TP / {sl_hits} SL / {timeout_exits} timeout")

        if setup_stats:
            print(f"\n  --- By Setup Type ---")
            for st, ss in sorted(setup_stats.items(), key=lambda x: -x[1]["trades"]):
                wr = round(ss['wins'] / ss['trades'] * 100, 0) if ss['trades'] else 0
                print(f"    {st:>12}: {ss['trades']}T {wr:.0f}%WR  ${ss['pnl']:+.2f}")

        if conf_stats:
            print(f"\n  --- By Confidence ---")
            for c, cs in sorted(conf_stats.items()):
                wr = round(cs['wins'] / cs['trades'] * 100, 0) if cs['trades'] else 0
                print(f"    {c:>8}: {cs['trades']}T {wr:.0f}%WR  ${cs['pnl']:+.2f}")

        print(f"\n{'='*70}")

        # Verdict
        print("\n  --- VEREDICTO ---")
        if len(active) < 5:
            print("  INSUFICIENTE: Muy pocos trades para evaluar.")
            print("  -> No hay suficiente evidencia para decidir.")
        elif stats["win_rate_pct"] >= 50 and total_pnl > 0 and max_drawdown < 15:
            print("  PROMETEDOR: Win rate >50%, P&L positivo, DD controlado.")
            print("  -> RECOMENDACION: Probar en cuenta demo con 1 MZS minimo.")
        elif total_pnl > 0 and max_drawdown < 25:
            print("  CAUTELOSO: P&L positivo pero metricas mixtas.")
            print("  -> RECOMENDACION: Paper trade 30 dias mas antes de demo.")
        elif total_pnl <= 0 and max_drawdown < 15:
            print("  MARGINALMENTE NEGATIVO: Perdida pequena, DD controlado.")
            print("  -> RECOMENDACION: Ajustar reglas de entrada, no ir a demo aun.")
        else:
            print("  NO LISTO: P&L negativo y/o drawdown excesivo.")
            print("  -> RECOMENDACION: Revisar estrategia completa antes de demo.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantAgent Rule-Based Backtest")
    parser.add_argument("--days", type=int, default=60, help="RTH days to test")
    parser.add_argument("--capital", type=float, default=1000, help="Starting capital USD")
    parser.add_argument("--sample-hour", type=int, default=11, help="CT hour to sample")
    args = parser.parse_args()

    run_backtest(days=args.days, capital=args.capital, sample_hour=args.sample_hour)
