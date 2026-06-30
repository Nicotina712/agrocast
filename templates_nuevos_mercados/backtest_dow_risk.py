"""
Day-of-Week Risk Weighting — Crypto
===================================
The weekend analysis (backtest_weekend_crypto.py) showed, spread-adjusted, that
the crypto strategy makes its money on weekends + Monday and LOSES on Wednesday.
This tests whether scaling per-trade RISK by day-of-week improves the crypto
book's risk-adjusted return — instead of adding new instruments.

Two layers, because tuning weights on the same data you measured the pattern on
is circular (overfitting):

  1. PRINCIPLE-BASED schemes (fixed, low-DoF, motivated by the mechanism:
     quiet markets favour this mean-reversion strategy) applied to full data.
  2. OUT-OF-SAMPLE test: derive which days are positive from the first 60% of
     trades, apply weights to the last 40%, and compare vs flat risk. This is
     the honest test of whether day weighting generalises.

All P&L is spread-adjusted (reuses backtest_weekend_crypto). Scaling a trade's
risk by w just scales its P&L by w (position sized so 1R = risk_usd).

Read-only. Touches nothing in the live system or the soja codebase.

Usage:
  python backtest_dow_risk.py
  python backtest_dow_risk.py --bars 8000 --train-frac 0.6
"""

import os, sys, argparse, warnings
from datetime import datetime

warnings.filterwarnings("ignore")

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(_HERE)
for p in [_HERE, _MVP_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import backtest_portfolio as bp
from backtest_weekend_crypto import _gen_trades, _apply_spread, CRYPTO, DOW

# Mon=0 ... Sun=6
SCHEMES = {
    "Flat (1.0 all)":        {d: 1.0 for d in range(7)},
    # overweight quiet/strong days, halve the worst (Wed), keep rest neutral
    "Heuristic 1.5/0.5":     {0: 1.5, 1: 1.0, 2: 0.5, 3: 1.0, 4: 1.0, 5: 1.5, 6: 1.5},
    # same but simply SKIP Wednesday
    "Skip-Wed":              {0: 1.0, 1: 1.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0},
    # combine: overweight weekend+Mon, skip Wed
    "Wknd+Mon up, Wed off":  {0: 1.5, 1: 1.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 1.5, 6: 1.5},
}


def _metrics(rows, weights):
    """rows: list of dicts with ts, dow, pnl, r. weights: dow->multiplier.
    Returns portfolio metrics on the weighted, time-ordered P&L stream."""
    seq = sorted(rows, key=lambda x: x["ts"])
    pnls = [x["pnl"] * weights.get(x["dow"], 1.0) for x in seq]
    pnls = [p for p, x in zip(pnls, seq) if weights.get(x["dow"], 1.0) > 0]  # drop skipped
    if not pnls:
        return None
    wins = [p for p in pnls if p > 0]
    cum  = np.cumsum(pnls); peak = np.maximum.accumulate(cum)
    dd   = float((cum - peak).min())
    sharpe = (np.mean(pnls) / np.std(pnls) * np.sqrt(252)) if len(pnls) > 1 and np.std(pnls) > 0 else 0
    calmar = (sum(pnls) / abs(dd)) if dd != 0 else 0
    return dict(n=len(pnls), wr=round(len(wins)/len(pnls)*100, 1),
                pnl=round(float(sum(pnls)), 2), dd=round(dd, 2),
                sharpe=round(float(sharpe), 2), calmar=round(float(calmar), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=5000)
    ap.add_argument("--weekday-bps", type=float, default=5.0)
    ap.add_argument("--weekend-mult", type=float, default=3.0)
    ap.add_argument("--train-frac", type=float, default=0.6)
    args = ap.parse_args()

    print("\n" + "=" * 70)
    print("  DAY-OF-WEEK RISK WEIGHTING — CRYPTO")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  {args.bars:,} bars/sym")
    print(f"  Spread: weekday {args.weekday_bps:.0f} bps | weekend x{args.weekend_mult:.0f}")
    print("=" * 70)

    gross = []
    for sym in CRYPTO:
        gross.extend(_gen_trades(sym, args.bars))
    if not gross:
        print("\n  No trades — MT5 connected?"); return
    rows = _apply_spread(gross, args.weekday_bps, args.weekend_mult)
    print(f"  Trades: {len(rows)} (spread-adjusted)")

    # ── 1. principle-based schemes on full data (IN-SAMPLE — directional) ──
    print("\n" + "-" * 70)
    print("  IN-SAMPLE: fixed schemes on full data (beware overfitting)")
    print("-" * 70)
    print(f"  {'Scheme':<24} {'Ops':>5} {'WR%':>7} {'NetP&L':>9} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8}")
    flat = _metrics(rows, SCHEMES["Flat (1.0 all)"])
    for name, w in SCHEMES.items():
        m = _metrics(rows, w)
        if m:
            print(f"  {name:<24} {m['n']:>5} {m['wr']:>6.1f}% ${m['pnl']:>7,.0f} "
                  f"${m['dd']:>6,.0f} {m['sharpe']:>8} {m['calmar']:>8}")

    # ── 2. OUT-OF-SAMPLE test (the honest one) ──
    print("\n" + "-" * 70)
    print(f"  OUT-OF-SAMPLE: weights learned on first {args.train_frac:.0%}, tested on rest")
    print("-" * 70)
    seq = sorted(rows, key=lambda x: x["ts"])
    cut = int(len(seq) * args.train_frac)
    train, test = seq[:cut], seq[cut:]
    print(f"  Train: {len(train)} trades ({train[0]['ts'].date()} -> {train[-1]['ts'].date()})")
    print(f"  Test : {len(test)} trades ({test[0]['ts'].date()} -> {test[-1]['ts'].date()})")

    # learn: per-day mean pnl on train -> weight 1.5 if >0, 0.5 if <=0
    learned = {}
    for d in range(7):
        day = [x["pnl"] for x in train if x["dow"] == d]
        mean = np.mean(day) if day else 0.0
        learned[d] = 1.5 if mean > 0 else 0.5
    learn_str = " ".join(f"{DOW[d]}:{learned[d]}" for d in range(7))
    print(f"  Learned weights: {learn_str}")

    # fixed weekend-only overweight (the part that was independently robust)
    wknd_only = {d: (1.5 if d >= 5 else 1.0) for d in range(7)}

    flat_test = _metrics(test, SCHEMES["Flat (1.0 all)"])
    wtd_test  = _metrics(test, learned)
    wknd_test = _metrics(test, wknd_only)
    print(f"\n  {'On TEST set':<24} {'Ops':>5} {'WR%':>7} {'NetP&L':>9} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8}")
    for name, m in [("Flat risk", flat_test),
                    ("Day-weighted (learned)", wtd_test),
                    ("Weekend-only 1.5x", wknd_test)]:
        if m:
            print(f"  {name:<24} {m['n']:>5} {m['wr']:>6.1f}% ${m['pnl']:>7,.0f} "
                  f"${m['dd']:>6,.0f} {m['sharpe']:>8} {m['calmar']:>8}")

    # ── verdict ──
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    if flat_test and wtd_test:
        ds = wtd_test["sharpe"] - flat_test["sharpe"]
        dd = wtd_test["dd"] - flat_test["dd"]
        print(f"  OOS Sharpe: flat {flat_test['sharpe']} -> weighted {wtd_test['sharpe']} ({ds:+.2f})")
        print(f"  OOS MaxDD : flat ${flat_test['dd']:,.0f} -> weighted ${wtd_test['dd']:,.0f} ({dd:+,.0f})")
        if ds > 0 and dd >= 0:
            print("  -> Day-weighting GENERALISES: better Sharpe out-of-sample without")
            print("     worse drawdown. Worth implementing as a risk multiplier by weekday.")
        elif ds > 0:
            print("  -> Better OOS Sharpe but deeper drawdown — mixed; size carefully.")
        else:
            print("  -> Day-weighting does NOT generalise out-of-sample. The in-sample")
            print("     gain was likely overfitting. Keep flat risk.")
    print("  CAVEAT: small sample; weekend WR may be inflated by thin-bar ranges.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
