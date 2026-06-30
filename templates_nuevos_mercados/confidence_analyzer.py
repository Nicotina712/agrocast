"""
Confidence Analyzer — Win-Rate by LLM Confidence Level
======================================================
Tracks whether the LLM's stated confidence (LOW / MEDIUM / HIGH) actually
predicts outcomes, as live trades accumulate. This is the prerequisite for
ever building a confidence-aware cluster swap (see backtest_swap_confidence.py):
we must FIRST prove HIGH-confidence trades outperform LOW ones with real data.

Source: each robot's artifacts/<sym>/paper_trades.jsonl logs every signal that
became a trade, tagged with confidence + entry/sl/tp. Those logs do NOT record
the outcome, so we resolve each trade's result the same way the backtest does:
fetch the real subsequent price bars from MT5 and check whether SL or TP was
touched first (first-touch, worst-case on same-bar ties).

Outputs:
  • per-confidence bucket: trades, win-rate, avg R-multiple, expectancy ($)
  • per-symbol breakdown
  • data-sufficiency gauge vs the ~100-150 trades/robot target

Read-only. Touches nothing in the live system or the soja codebase.

Usage:
  python confidence_analyzer.py
  python confidence_analyzer.py --resolve-tf 5m --max-hold-bars 288
  python confidence_analyzer.py --sym BTCUSD
"""

import os, sys, json, glob, argparse, warnings
from datetime import datetime, timezone, timedelta
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

TARGET_PER_ROBOT = 125   # midpoint of the ~100-150 prerequisite
CONF_ORDER = ["LOW", "MEDIUM", "HIGH"]


def _local_utc_offset():
    """This machine's local-time -> UTC offset (paper_trades timestamps are
    written in system-local naive time; MT5 bars are tz-aware UTC)."""
    off = datetime.now().astimezone().utcoffset()
    return off or timedelta(0)


def _load_trades(artifacts_dir, only_sym=None):
    """Read every paper_trades.jsonl; return list of trade dicts (LONG/SHORT)."""
    trades = []
    for f in glob.glob(os.path.join(artifacts_dir, "*", "paper_trades.jsonl")):
        folder = os.path.basename(os.path.dirname(f))
        sym = folder.upper()
        if only_sym and sym != only_sym.upper():
            continue
        for line in open(f, encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("signal") not in ("LONG", "SHORT"):
                continue
            if not all(d.get(k) is not None for k in ("entry", "sl", "tp")):
                continue
            trades.append({
                "sym":        sym,
                "ts":         pd.Timestamp(d["timestamp"]),   # naive local
                "signal":     d["signal"],
                "entry":      float(d["entry"]),
                "sl":         float(d["sl"]),
                "tp":         float(d["tp"]),
                "confidence": (d.get("confidence") or "?").upper(),
                "rr":         d.get("rr"),
                "risk_usd":   d.get("risk_usd") or bp.MAX_RISK_USD,
            })
    return trades


def _resolve_outcome(trade, bars, local_off, max_hold_bars):
    """Determine WIN/LOSS/TIME/OPEN by first-touch against real bars.
    Returns (outcome, r_multiple) or (None, None) if unresolvable."""
    if bars is None or len(bars) == 0:
        return None, None
    entry_utc = (trade["ts"] - local_off).tz_localize("UTC")
    # bars index is tz-aware UTC
    try:
        start = bars.index.searchsorted(entry_utc, side="right")
    except Exception:
        return None, None
    if start >= len(bars):
        return None, None  # trade newer than available bars

    ep, sl, tp = trade["entry"], trade["sl"], trade["tp"]
    long = trade["signal"] == "LONG"
    sl_dist = abs(ep - sl)
    if sl_dist == 0:
        return None, None
    rr = trade["rr"] or round(abs(tp - ep) / sl_dist, 2)

    window = bars.iloc[start:start + max_hold_bars]
    for _, row in window.iterrows():
        hi, lo = row["high"], row["low"]
        hit_tp = (long and hi >= tp) or (not long and lo <= tp)
        hit_sl = (long and lo <= sl) or (not long and hi >= sl)
        if hit_sl and hit_tp:
            return "LOSS", -1.0          # worst-case: assume SL on same bar
        if hit_tp:
            return "WIN", float(rr)
        if hit_sl:
            return "LOSS", -1.0
    # neither touched within window
    if start + max_hold_bars >= len(bars):
        return "OPEN", None              # still running / not enough data yet
    # time-stop: mark-to-close at last window bar
    exit_px = float(window["close"].iloc[-1])
    r = ((exit_px - ep) if long else (ep - exit_px)) / sl_dist
    return "TIME", float(r)


def _bucket_stats(rows):
    """rows: list of (r_multiple, risk_usd, outcome). Returns metrics dict."""
    resolved = [(r, ru) for (r, ru, o) in rows if r is not None]
    if not resolved:
        return None
    rs   = [r for r, _ in resolved]
    usd  = [r * ru for r, ru in resolved]
    wins = [r for r in rs if r > 0]
    return dict(
        n=len(resolved),
        wr=round(len(wins) / len(resolved) * 100, 1),
        avg_r=round(float(np.mean(rs)), 2),
        exp_usd=round(float(np.mean(usd)), 2),
        tot_usd=round(float(np.sum(usd)), 2),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=os.path.join(_MVP_ROOT, "artifacts"))
    ap.add_argument("--resolve-tf", default="5m", help="bar TF for outcome resolution")
    ap.add_argument("--max-hold-bars", type=int, default=288, help="288×5m = 24h cap")
    ap.add_argument("--fetch-bars", type=int, default=4000, help="bars to fetch per symbol")
    ap.add_argument("--sym", default=None)
    args = ap.parse_args()

    print("\n" + "=" * 70)
    print("  CONFIDENCE ANALYZER — does LLM confidence predict outcomes?")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  resolve TF: {args.resolve_tf}")
    print("=" * 70)

    trades = _load_trades(args.artifacts, args.sym)
    if not trades:
        print("\n  No paper trades found.")
        return
    local_off = _local_utc_offset()
    print(f"\n  Loaded {len(trades)} trades  |  local UTC offset: {local_off}")

    # fetch bars per symbol once
    bars_cache = {}
    for sym in sorted({t["sym"] for t in trades}):
        bars_cache[sym] = bp._mt5_bars(sym, args.resolve_tf, args.fetch_bars)

    # resolve every trade
    by_conf   = defaultdict(list)   # confidence -> [(r, risk_usd, outcome)]
    by_sym    = defaultdict(lambda: defaultdict(list))  # sym -> conf -> rows
    n_open = n_unres = 0
    for t in trades:
        outcome, r = _resolve_outcome(t, bars_cache.get(t["sym"]), local_off, args.max_hold_bars)
        if outcome is None:
            n_unres += 1; continue
        if outcome == "OPEN":
            n_open += 1; continue
        by_conf[t["confidence"]].append((r, t["risk_usd"], outcome))
        by_sym[t["sym"]][t["confidence"]].append((r, t["risk_usd"], outcome))

    # ── by confidence ──
    print("\n" + "-" * 70)
    print("  BY CONFIDENCE LEVEL")
    print("-" * 70)
    print(f"  {'Conf':<8} {'Trades':>7} {'WR%':>7} {'AvgR':>7} {'Exp$':>9} {'TotP&L':>10}")
    present = [c for c in CONF_ORDER if c in by_conf] + \
              [c for c in by_conf if c not in CONF_ORDER]
    for c in present:
        s = _bucket_stats(by_conf[c])
        if s:
            print(f"  {c:<8} {s['n']:>7} {s['wr']:>6.1f}% {s['avg_r']:>7} "
                  f"${s['exp_usd']:>7.2f} ${s['tot_usd']:>8,.0f}")
    if n_open or n_unres:
        print(f"\n  (excluded: {n_open} still-open, {n_unres} unresolvable/too-new)")

    # ── by symbol ──
    print("\n" + "-" * 70)
    print("  BY SYMBOL x CONFIDENCE")
    print("-" * 70)
    for sym in sorted(by_sym):
        parts = []
        for c in present:
            if c in by_sym[sym]:
                s = _bucket_stats(by_sym[sym][c])
                if s:
                    parts.append(f"{c[:1]}:{s['n']}t/{s['wr']:.0f}%")
        print(f"  {sym:<10} {'  '.join(parts)}")

    # ── data sufficiency ──
    print("\n" + "-" * 70)
    print("  DATA SUFFICIENCY (target ~125 resolved trades/robot)")
    print("-" * 70)
    n_robots = len({t['sym'] for t in trades})
    total_resolved = sum(len(v) for v in by_conf.values())
    need = TARGET_PER_ROBOT * n_robots
    pct = total_resolved / need * 100 if need else 0
    bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
    print(f"  Resolved trades: {total_resolved} / ~{need}  [{bar}] {pct:.0f}%")
    high = sum(len(by_conf.get(c, [])) for c in ("HIGH",))
    low  = sum(len(by_conf.get(c, [])) for c in ("LOW",))
    print(f"  HIGH-confidence resolved: {high}   LOW-confidence resolved: {low}")

    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    if total_resolved < 30:
        print("  Far too few trades to conclude anything. Keep accumulating.")
    elif high < 15 or low < 15:
        print("  Not enough HIGH and/or LOW trades to compare against MEDIUM.")
        print("  The LLM is barely differentiating confidence — that itself is a")
        print("  finding: a confidence swap needs spread in the confidence signal.")
    else:
        hi_s = _bucket_stats(by_conf["HIGH"]); lo_s = _bucket_stats(by_conf["LOW"])
        if hi_s and lo_s and hi_s["wr"] > lo_s["wr"] and hi_s["exp_usd"] > lo_s["exp_usd"]:
            print(f"  HIGH ({hi_s['wr']}% WR, ${hi_s['exp_usd']}/trade) beats "
                  f"LOW ({lo_s['wr']}% WR, ${lo_s['exp_usd']}/trade).")
            print("  Confidence IS predictive -> a confidence-aware swap is now worth")
            print("  backtesting on these real outcomes.")
        else:
            print("  HIGH does not clearly beat LOW. Confidence not yet predictive.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
