"""
Weekend vs Weekday — Crypto Edge Analysis
=========================================
Question: do BTCUSD / ETHUSD trades taken on weekends actually make money, or
is weekend trading net-negative (thin liquidity, wider spreads, gap risk)?

The academic literature says crypto weekends are LOWER liquidity / volume /
volatility with NO compensating return premium. This checks whether that shows
up in our own strategy's simulated trades.

Method: generate trades per crypto instrument with its optimized config
(reusing backtest_portfolio), then bucket each trade by the weekday of its
ENTRY timestamp (UTC). Reports:
  • per day-of-week: trades, WR, avg R, expectancy $, total P&L
  • weekend (Sat+Sun) vs weekday aggregate, with Sharpe

Read-only. Touches nothing in the live system or the soja codebase.

Usage:
  python backtest_weekend_crypto.py
  python backtest_weekend_crypto.py --bars 8000
"""

import os, sys, argparse, warnings
from datetime import datetime
from collections import defaultdict

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

CRYPTO = ["BTCUSD", "ETHUSD"]
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _gen_trades(sym, n_bars):
    inst = next(i for i in bp.INSTRUMENTS if i["sym"] == sym)
    o = bp.OPTIMIZED_CONFIGS.get(sym, {})
    m, trades = bp.backtest_instrument(
        inst, n_bars=n_bars, verbose=False,
        tf_override=o.get("tf"), ema_fast_ov=o.get("ef"), ema_slow_ov=o.get("es"),
        sl_mult_ov=o.get("sl"), tp_mult_ov=o.get("tp"), cooldown_ov=o.get("cd"),
        rsi_lo_long=o.get("rll", 38), rsi_hi_long=o.get("rhl", 65),
        rsi_lo_short=o.get("rls", 35), rsi_hi_short=o.get("rhs", 62),
    )
    out = []
    for t in trades:
        ts = pd.Timestamp(t["entry_time"])
        # R-multiple from outcome
        if t["outcome"] == "WIN":
            r = t["rr"]
        elif t["outcome"] == "LOSS":
            r = -1.0
        else:  # TIME — recover R from pnl/risk
            r = t["pnl_usd"] / t["risk_usd"] if t["risk_usd"] else 0.0
        sl_dist = abs(t["entry_px"] - t["sl_px"])
        out.append(dict(sym=sym, dow=ts.weekday(), ts=ts, r=float(r),
                        pnl=t["pnl_usd"], risk=t["risk_usd"],
                        entry_px=t["entry_px"], sl_dist=sl_dist))
    return out


def _apply_spread(rows, weekday_bps, weekend_mult):
    """Return NEW rows with a round-trip spread cost subtracted.
    1 R == risk_usd (position sized so SL distance = risk). A full round-trip
    spread costs (spread_price / sl_dist) R. Weekend trades pay weekend_mult×.
    """
    out = []
    for x in rows:
        bps  = weekday_bps * (weekend_mult if x["dow"] >= 5 else 1.0)
        cost_price = x["entry_px"] * bps / 10000.0      # one full spread, round trip
        cost_r     = (cost_price / x["sl_dist"]) if x["sl_dist"] > 0 else 0.0
        net_r      = x["r"] - cost_r
        out.append({**x, "r": net_r, "pnl": net_r * x["risk"]})
    return out


def _stats(rows):
    if not rows:
        return None
    rs   = [x["r"] for x in rows]
    pnls = [x["pnl"] for x in rows]
    wins = [r for r in rs if r > 0]
    sharpe = (np.mean(pnls) / np.std(pnls) * np.sqrt(252)) if len(pnls) > 1 and np.std(pnls) > 0 else 0
    return dict(n=len(rows), wr=round(len(wins)/len(rows)*100, 1),
                avg_r=round(float(np.mean(rs)), 2),
                exp=round(float(np.mean(pnls)), 2),
                tot=round(float(np.sum(pnls)), 2),
                sharpe=round(float(sharpe), 2))


def _wk_split(rows):
    return [x for x in rows if x["dow"] < 5], [x for x in rows if x["dow"] >= 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=5000)
    ap.add_argument("--weekday-bps", type=float, default=5.0,
                    help="round-trip spread on weekdays, in bps of price (default 5)")
    ap.add_argument("--weekend-mult", type=float, default=3.0,
                    help="weekend spread multiplier for the MAIN table (default 3x)")
    args = ap.parse_args()

    print("\n" + "=" * 68)
    print("  WEEKEND vs WEEKDAY — CRYPTO EDGE ANALYSIS (spread-adjusted)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  {args.bars:,} bars/sym")
    print(f"  Spread: weekday {args.weekday_bps:.0f} bps | weekend {args.weekday_bps*args.weekend_mult:.0f} bps (x{args.weekend_mult:.0f})")
    print("=" * 68)

    gross_rows = []
    for sym in CRYPTO:
        rows = _gen_trades(sym, args.bars)
        print(f"  {sym}: {len(rows)} trades")
        gross_rows.extend(rows)

    if not gross_rows:
        print("\n  No trades — MT5 connected?")
        return

    # apply spread haircut for the main tables
    all_rows = _apply_spread(gross_rows, args.weekday_bps, args.weekend_mult)

    # ── per day-of-week ──
    print("\n" + "-" * 68)
    print("  BY DAY OF WEEK (entry, UTC)")
    print("-" * 68)
    print(f"  {'Day':<5} {'Ops':>5} {'WR%':>7} {'AvgR':>7} {'Exp$':>9} {'TotP&L':>10}")
    by_dow = defaultdict(list)
    for x in all_rows:
        by_dow[x["dow"]].append(x)
    for d in range(7):
        s = _stats(by_dow.get(d, []))
        if s:
            mark = "  <- weekend" if d >= 5 else ""
            print(f"  {DOW[d]:<5} {s['n']:>5} {s['wr']:>6.1f}% {s['avg_r']:>7} "
                  f"${s['exp']:>7.2f} ${s['tot']:>8,.0f}{mark}")

    # ── weekend vs weekday ──
    wknd = [x for x in all_rows if x["dow"] >= 5]
    week = [x for x in all_rows if x["dow"] < 5]
    sw, sk = _stats(wknd), _stats(week)
    print("\n" + "-" * 68)
    print("  AGGREGATE")
    print("-" * 68)
    print(f"  {'Bucket':<10} {'Ops':>5} {'WR%':>7} {'AvgR':>7} {'Exp$':>9} {'TotP&L':>10} {'Sharpe':>8}")
    for label, s in [("Weekday", sk), ("Weekend", sw)]:
        if s:
            print(f"  {label:<10} {s['n']:>5} {s['wr']:>6.1f}% {s['avg_r']:>7} "
                  f"${s['exp']:>7.2f} ${s['tot']:>8,.0f} {s['sharpe']:>8}")

    # ── sensitivity sweep: weekend spread multiplier ──
    print("\n" + "-" * 68)
    print("  SENSITIVITY — weekend edge vs spread multiplier")
    print(f"  (weekday spread fixed at {args.weekday_bps:.0f} bps)")
    print("-" * 68)
    print(f"  {'WkndMult':>9} {'WkndBps':>8} {'WkndExp$':>10} {'WkndWR%':>9} {'WkndTotP&L':>12}")
    for mult in [1.0, 2.0, 3.0, 5.0]:
        adj = _apply_spread(gross_rows, args.weekday_bps, mult)
        _, wk_w = _wk_split(adj)
        s = _stats(wk_w)
        if s:
            print(f"  {mult:>8.0f}x {args.weekday_bps*mult:>7.0f} "
                  f"${s['exp']:>8.2f} {s['wr']:>8.1f}% ${s['tot']:>10,.0f}")

    # ── verdict ──
    print("\n" + "=" * 68)
    print("  VERDICT")
    print("=" * 68)
    if not sw:
        print("  No weekend trades in sample.")
    elif sw["exp"] <= 0 and sk["exp"] > 0:
        print(f"  Weekend expectancy is NEGATIVE (${sw['exp']}/trade) while weekday")
        print(f"  is positive (${sk['exp']}/trade). Weekend crypto trading is a drag —")
        print("  consider pausing crypto on weekends rather than adding instruments.")
    elif sw["exp"] > 0 and sw["sharpe"] >= sk["sharpe"] * 0.7:
        print(f"  Weekend holds up (${sw['exp']}/trade, Sharpe {sw['sharpe']} vs weekday")
        print(f"  {sk['sharpe']}). Weekend trading is defensible on these data.")
    elif sw["exp"] > 0:
        print(f"  Weekend is positive (${sw['exp']}/trade) but weaker risk-adjusted")
        print(f"  (Sharpe {sw['sharpe']} vs weekday {sk['sharpe']}). Marginal — keep but")
        print("  don't expand weekend exposure.")
    else:
        print(f"  Both buckets weak. Weekend exp ${sw['exp']}, weekday ${sk['exp']}.")
    print("  NOTE: spread IS modeled above. Remaining unmodeled risk: thin weekend")
    print("  bars compress high/low ranges -> may understate SL hits (inflated WR).")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
