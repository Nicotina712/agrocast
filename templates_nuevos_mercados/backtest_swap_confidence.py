"""
Confidence-Aware Cluster Swap — Exploration
===========================================
The RR-based quality swap (backtest_cluster_policy.py) never fired because RR
is homogeneous within each cluster. A real quality swap needs a signal that
DIFFERENTIATES trades inside a cluster — in live trading that's the LLM
confidence (LOW/MEDIUM/HIGH). The backtest has no LLM, so we build a technical
PROXY for confidence and test whether swapping on it beats the hard cap.

Pipeline:
  1. For every instrument: fetch bars, compute indicators, generate signals,
     simulate trades — AND score each trade's conviction at entry.
  2. VALIDATE the score: bucket all trades into score quartiles and check
     whether higher-score trades actually win more / earn more. If the score
     isn't predictive, swapping on it is pointless.
  3. Replay the portfolio under:
        A · HARD-CAP        (live behaviour)
        B · CONFIDENCE-SWAP (close weaker, open stronger when new score is
                             materially higher) — swapped-out trades are closed
                             at their REAL price at swap time (no break-even
                             approximation; we keep the bars to recompute P&L).

Signal score (all ATR-normalized → comparable across instruments):
    trend_sep  = |ema_fast - ema_slow| / atr      (clean trend separation)
    slope_norm = |Δema_fast|           / atr      (momentum thrust)
    vwap_dist  = |close - vwap|        / atr       (displacement confirmation)
    score      = trend_sep + slope_norm + vwap_dist

Usage:
  python backtest_swap_confidence.py
  python backtest_swap_confidence.py --bars 8000 --swap-margin 0.30
"""

import os, sys, argparse, warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# stdout UTF-8 wrapping handled by backtest_portfolio on import — do NOT re-wrap.

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(_HERE)
for p in [_HERE, _MVP_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

import backtest_portfolio as bp
from portfolio_guard import _get_cluster_name


# ── trade generation WITH per-trade conviction score + retained bars ──────────

def _gen_scored_trades(inst, n_bars, bars_store):
    """Generate trades for one instrument and attach a conviction score to each.
    Stores the indicator DataFrame in bars_store[sym] so the swap replay can
    recompute a truncated exit P&L from real prices."""
    sym    = inst["sym"]
    folder = inst["folder"]
    cfg    = bp._load_cfg(folder)
    o      = bp.OPTIMIZED_CONFIGS.get(sym, {})
    tf     = o.get("tf") or (getattr(cfg, "TIMEFRAME", "M5") if cfg else "M5")

    bars = bp._mt5_bars(sym, tf, n_bars)
    if bars is None or len(bars) < (o.get("es", bp.EMA_SLOW) + 20):
        print(f"    {sym:<9} SKIP (no bars)")
        return []

    # run strategy with this instrument's optimized params, capturing trades
    trades = bp._run_strategy(
        bars, cfg, sym,
        ema_fast=o.get("ef"), ema_slow=o.get("es"),
        sl_mult=o.get("sl"),  tp_mult=o.get("tp"), cooldown=o.get("cd"),
        rsi_lo_long=o.get("rll", 38), rsi_hi_long=o.get("rhl", 65),
        rsi_lo_short=o.get("rls", 35), rsi_hi_short=o.get("rhs", 62),
        verbose=False,
    )

    # recompute indicators with the SAME params used above so we can score
    _ef0, _es0 = bp.EMA_FAST, bp.EMA_SLOW
    bp.EMA_FAST = o.get("ef") or bp.EMA_FAST
    bp.EMA_SLOW = o.get("es") or bp.EMA_SLOW
    try:
        df = bp._compute_indicators(bars)
    finally:
        bp.EMA_FAST, bp.EMA_SLOW = _ef0, _es0

    bars_store[sym] = df
    cluster = _get_cluster_name(sym)

    out = []
    for t in trades:
        i = t["entry_i"]
        row = df.iloc[i]
        atr = row["atr"]
        if atr is None or atr <= 0 or np.isnan(atr):
            score = 0.0
        else:
            trend_sep  = abs(row["ema_fast"] - row["ema_slow"]) / atr
            slope_norm = abs(row["ema_fast_slope"]) * row["close"] / atr
            vwap_dist  = abs(row["close"] - row["vwap"]) / atr
            score = float(trend_sep + slope_norm + vwap_dist)
        out.append({
            "sym":        sym,
            "cluster":    cluster,
            "entry_i":    i,
            "entry_time": pd.Timestamp(t["entry_time"]),
            "exit_time":  pd.Timestamp(t["exit_time"]),
            "entry_px":   t["entry_px"],
            "sl_px":      t["sl_px"],
            "tp_px":      t["tp_px"],
            "direction":  t["direction"],
            "risk_usd":   t["risk_usd"],
            "rr":         t["rr"],
            "pnl_usd":    t["pnl_usd"],
            "outcome":    t["outcome"],
            "score":      round(score, 3),
        })
    print(f"    {sym:<9} cluster={(cluster or '—'):<8} trades={len(out)}  "
          f"score[min/med/max]={min((x['score'] for x in out), default=0):.2f}/"
          f"{np.median([x['score'] for x in out]) if out else 0:.2f}/"
          f"{max((x['score'] for x in out), default=0):.2f}")
    return out


# ── score validation: is conviction predictive? ──────────────────────────────

def _validate_score(intents):
    print("\n" + "=" * 64)
    print("  SCORE VALIDATION — do higher-conviction trades win more?")
    print("=" * 64)
    scored = [t for t in intents if t["score"] > 0]
    if len(scored) < 20:
        print("  Too few scored trades to validate.")
        return False
    scores = np.array([t["score"] for t in scored])
    q1, q2, q3 = np.quantile(scores, [0.25, 0.5, 0.75])

    def bucket(s):
        if s <= q1: return 0
        if s <= q2: return 1
        if s <= q3: return 2
        return 3

    labels = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    print(f"  {'Bucket':<14} {'Ops':>5} {'WR%':>7} {'AvgP&L':>9} {'TotP&L':>9}")
    print("  " + "-" * 50)
    wr_by_q = {}
    for q in range(4):
        grp = [t for t in scored if bucket(t["score"]) == q]
        if not grp:
            continue
        pnls = [t["pnl_usd"] for t in grp]
        wins = sum(1 for p in pnls if p > 0)
        wr   = wins / len(grp) * 100
        wr_by_q[q] = wr
        print(f"  {labels[q]:<14} {len(grp):>5} {wr:>6.1f}% "
              f"${np.mean(pnls):>7.2f} ${sum(pnls):>7,.0f}")
    predictive = (wr_by_q.get(3, 0) > wr_by_q.get(0, 100))
    print("  " + "-" * 50)
    if predictive:
        print(f"  → Top quartile WR ({wr_by_q.get(3,0):.1f}%) > bottom "
              f"({wr_by_q.get(0,0):.1f}%): conviction score IS predictive. "
              f"Swapping on it can help.")
    else:
        print(f"  → Top quartile WR ({wr_by_q.get(3,0):.1f}%) NOT above bottom "
              f"({wr_by_q.get(0,0):.1f}%): score is weak. Swap unlikely to help.")
    return predictive


# ── replays ───────────────────────────────────────────────────────────────────

def _replay_hard_cap(intents):
    selected, occ = [], {}
    for t in sorted(intents, key=lambda x: x["entry_time"]):
        cl = t["cluster"]
        if cl is None:
            selected.append(t); continue
        if occ.get(cl) is not None and t["entry_time"] < occ[cl]:
            continue
        selected.append(t); occ[cl] = t["exit_time"]
    return selected


def _truncated_pnl(trade, bars_store, cut_time):
    """Real P&L if `trade` is force-closed at the close price at/just before
    cut_time (market exit on swap). Falls back to break-even if unresolvable."""
    df = bars_store.get(trade["sym"])
    if df is None:
        return 0.0
    idx = df.index.searchsorted(cut_time, side="right") - 1
    if idx <= trade["entry_i"]:
        return 0.0  # swapped out before any new bar → break-even
    exit_px = float(df["close"].iloc[idx])
    ep, sl  = trade["entry_px"], trade["sl_px"]
    denom   = (ep - sl) if trade["direction"] == "LONG" else (sl - ep)
    if denom == 0:
        return 0.0
    if trade["direction"] == "LONG":
        pnl_r = (exit_px - ep) / denom
    else:
        pnl_r = (ep - exit_px) / denom
    return round(pnl_r * trade["risk_usd"], 2)


def _replay_conf_swap(intents, bars_store, swap_margin=0.20):
    """B — close the weaker open trade and take the newcomer when its conviction
    score exceeds the occupant's by >swap_margin. Swapped-out trade is closed at
    its REAL price at swap time."""
    selected, occupant = [], {}
    n_swaps = 0
    for t in sorted(intents, key=lambda x: x["entry_time"]):
        cl = t["cluster"]
        if cl is None:
            selected.append(t); continue
        cur = occupant.get(cl)
        cur_open = cur is not None and t["entry_time"] < cur["exit_time"]
        if not cur_open:
            selected.append(t); occupant[cl] = t; continue
        if cur["score"] > 0 and t["score"] > cur["score"] * (1.0 + swap_margin):
            # swap: truncate occupant at real price, take newcomer
            cur["pnl_usd"]  = _truncated_pnl(cur, bars_store, t["entry_time"])
            cur["exit_time"] = t["entry_time"]
            cur["outcome"]   = "SWAP"
            selected.append(t); occupant[cl] = t
            n_swaps += 1
        # else drop newcomer
    return selected, n_swaps


def _pm(selected):
    if not selected:
        return dict(n=0, wr=0, pnl=0, pf=0, dd=0, sharpe=0, calmar=0)
    merged = sorted(selected, key=lambda t: t["exit_time"])
    pnls   = [t["pnl_usd"] for t in merged]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    cum    = np.cumsum(pnls); peak = np.maximum.accumulate(cum)
    dd     = float((cum - peak).min())
    pf     = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")
    sharpe = (np.mean(pnls) / np.std(pnls) * np.sqrt(252)) if len(pnls) > 1 and np.std(pnls) > 0 else 0
    calmar = (sum(pnls) / abs(dd)) if dd != 0 else 0
    return dict(n=len(merged), wr=round(len(wins)/len(merged)*100,1),
                pnl=round(sum(pnls),2), pf=round(pf,2) if pf!=float("inf") else 999,
                dd=round(dd,2), sharpe=round(sharpe,2), calmar=round(calmar,2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=5000)
    ap.add_argument("--swap-margin", type=float, default=0.20)
    args = ap.parse_args()

    print("\n" + "=" * 64)
    print("  CONFIDENCE-AWARE CLUSTER SWAP — EXPLORATION")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Bars/instrument: {args.bars:,} | Swap margin: +{args.swap_margin:.0%} score")
    print("=" * 64)

    bars_store = {}
    intents = []
    print("\n  Generating scored trades...")
    for inst in bp.INSTRUMENTS:
        intents.extend(_gen_scored_trades(inst, args.bars, bars_store))

    if not intents:
        print("\n  No trades — MT5 connected?")
        return

    predictive = _validate_score(intents)

    hard = _replay_hard_cap(intents)
    # deep-copy intents for swap (it mutates occupant dicts)
    import copy
    swap_intents = copy.deepcopy(intents)
    swap, n_swaps = _replay_conf_swap(swap_intents, bars_store, args.swap_margin)

    a, b = _pm(hard), _pm(swap)
    print("\n" + "=" * 64)
    print("  POLICY COMPARISON")
    print("=" * 64)
    print(f"  {'Policy':<22} {'Ops':>5} {'WR%':>6} {'NetP&L':>10} {'PF':>6} {'MaxDD':>9} {'Sharpe':>7} {'Calmar':>7}")
    print("  " + "-" * 64)
    print(f"  {'A · HARD-CAP (live)':<22} {a['n']:>5} {a['wr']:>6} "
          f"${a['pnl']:>8,.0f} {a['pf']:>6} ${a['dd']:>7,.0f} {a['sharpe']:>7} {a['calmar']:>7}")
    print(f"  {'B · CONFIDENCE-SWAP':<22} {b['n']:>5} {b['wr']:>6} "
          f"${b['pnl']:>8,.0f} {b['pf']:>6} ${b['dd']:>7,.0f} {b['sharpe']:>7} {b['calmar']:>7}")

    print("\n" + "=" * 64)
    print("  READING THE RESULT")
    print("=" * 64)
    print(f"  Swaps executed: {n_swaps}")
    print(f"  ΔNet P&L  (swap − hard): ${b['pnl']-a['pnl']:+,.0f}")
    print(f"  ΔSharpe   (swap − hard): {b['sharpe']-a['sharpe']:+.2f}")
    print(f"  ΔMaxDD    (swap − hard): ${b['dd']-a['dd']:+,.0f}")
    print()
    if not predictive:
        print("  Score wasn't predictive → swap result is not trustworthy as a")
        print("  basis for live LLM-confidence swapping. Need a better signal.")
    elif n_swaps == 0:
        print("  Score predictive but margin too high → no swaps fired. Try a")
        print("  lower --swap-margin.")
    elif b["sharpe"] > a["sharpe"] and b["dd"] >= a["dd"]:
        print("  → Confidence swap improves Sharpe without worsening drawdown.")
        print("    Worth implementing in live with LLM confidence as the score.")
    else:
        print("  → Swap does not clearly beat the hard cap on risk-adjusted terms.")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
