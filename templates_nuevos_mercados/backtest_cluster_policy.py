"""
Cluster-Cap Policy Backtest
===========================
Answers ONE question: does the cluster_cap guard help or hurt the portfolio?

The standard portfolio backtest (backtest_portfolio.py) runs every instrument
INDEPENDENTLY — it never simulates the cluster_cap interaction. This script
generates the per-instrument trades once, tags each with its correlation
cluster, then RE-PLAYS the whole portfolio chronologically under three
competing policies:

  C · NO-CAP        every signal is taken (the implicit ceiling)
  A · HARD-CAP      max 1 open position per cluster — drop conflicting signals
                    (this is what the live robots do today)
  B · QUALITY-SWAP  if a materially better signal (higher RR) arrives while the
                    cluster is occupied, close the weaker trade and take the
                    stronger one.  *** see CAVEAT on the swap P&L approximation ***

Clusters (from portfolio_guard.py):
  equity → US500 | USTEC | US30
  energy → WTI_N6 | BRENT_N6
  crypto → BTCUSD | ETHUSD
  (XAUUSD, UK100 are clusterless → always admitted)

Usage:
  python backtest_cluster_policy.py                 # 5000 bars, swap margin 20%
  python backtest_cluster_policy.py --bars 8000
  python backtest_cluster_policy.py --swap-margin 0.30
"""

import os, sys, argparse, warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# NOTE: stdout UTF-8 wrapping is handled by backtest_portfolio on import.
# Do NOT wrap here too — double-wrapping the same buffer closes it prematurely.

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(_HERE)
for p in [_HERE, _MVP_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

import backtest_portfolio as bp
from portfolio_guard import _get_cluster_name


# ── per-instrument trade generation ──────────────────────────────────────────

def _gen_trades_for_instrument(inst, n_bars):
    """Run the optimized strategy for one instrument; return its trade list,
    each trade tagged with cluster + entry/exit timestamps."""
    sym = inst["sym"]
    cfg = bp.OPTIMIZED_CONFIGS.get(sym, {})
    m, trades = bp.backtest_instrument(
        inst, n_bars=n_bars, verbose=False,
        tf_override  = cfg.get("tf"),
        ema_fast_ov  = cfg.get("ef"),
        ema_slow_ov  = cfg.get("es"),
        sl_mult_ov   = cfg.get("sl"),
        tp_mult_ov   = cfg.get("tp"),
        cooldown_ov  = cfg.get("cd"),
        rsi_lo_long  = cfg.get("rll", 38),
        rsi_hi_long  = cfg.get("rhl", 65),
        rsi_lo_short = cfg.get("rls", 35),
        rsi_hi_short = cfg.get("rhs", 62),
    )
    cluster = _get_cluster_name(sym)
    out = []
    for t in trades:
        out.append({
            "sym":        sym,
            "cluster":    cluster,           # None for XAUUSD / UK100
            "entry_time": pd.Timestamp(t["entry_time"]),
            "exit_time":  pd.Timestamp(t["exit_time"]),
            "rr":         t["rr"],
            "pnl_usd":    t["pnl_usd"],
            "outcome":    t["outcome"],
            "direction":  t["direction"],
        })
    return out


# ── policy replays ────────────────────────────────────────────────────────────
# Each replay consumes the chronologically-sorted list of trade "intents" and
# returns the SELECTED trades (those actually taken under that policy).

def _replay_no_cap(intents):
    """C — take everything. (ceiling)"""
    return list(intents)


def _replay_hard_cap(intents):
    """A — max 1 open position per cluster. Drop signals that arrive while
    their cluster is already occupied. Clusterless symbols always pass."""
    selected = []
    occupied_until = {}   # cluster -> exit_time of current occupant
    for t in sorted(intents, key=lambda x: x["entry_time"]):
        cl = t["cluster"]
        if cl is None:
            selected.append(t)
            continue
        busy_until = occupied_until.get(cl)
        if busy_until is not None and t["entry_time"] < busy_until:
            continue  # cluster busy → drop
        selected.append(t)
        occupied_until[cl] = t["exit_time"]
    return selected


def _replay_quality_swap(intents, swap_margin=0.20):
    """B — quality-aware swap. If a new signal arrives while its cluster is
    occupied but its RR exceeds the occupant's RR by >swap_margin, CLOSE the
    occupant early and take the stronger one. Otherwise behave like hard-cap.

    CAVEAT: closing a trade early needs its intra-trade P&L, which the merged
    intent list does not carry. We approximate a swapped-out trade's realized
    P&L as 0 (break-even). This biases B and makes it EXPLORATORY only — use it
    to gauge direction, not as a precise figure.
    """
    # We process in entry_time order, but to "close early" a previously selected
    # trade we keep a handle on the current occupant per cluster.
    selected = []
    occupant = {}   # cluster -> ref to the selected-trade dict currently open
    for t in sorted(intents, key=lambda x: x["entry_time"]):
        cl = t["cluster"]
        if cl is None:
            selected.append(t)
            continue
        cur = occupant.get(cl)
        cur_open = cur is not None and t["entry_time"] < cur["exit_time"]
        if not cur_open:
            selected.append(t)
            occupant[cl] = t
            continue
        # cluster occupied — is the newcomer materially better?
        if t["rr"] > cur["rr"] * (1.0 + swap_margin):
            # swap: truncate occupant at this point, approximate its P&L = 0
            cur["exit_time"] = t["entry_time"]
            cur["pnl_usd"]   = 0.0
            cur["outcome"]   = "SWAP"
            selected.append(t)
            occupant[cl] = t
        # else: drop newcomer (hard-cap behaviour)
    return selected


# ── portfolio metrics on a selected trade list ────────────────────────────────

def _portfolio_metrics(selected):
    if not selected:
        return dict(n=0, wr=0, pnl=0, pf=0, dd=0, sharpe=0, calmar=0)
    merged = sorted(selected, key=lambda t: t["exit_time"])
    pnls   = [t["pnl_usd"] for t in merged]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    cum    = np.cumsum(pnls)
    peak   = np.maximum.accumulate(cum)
    dd     = float((cum - peak).min())
    pf     = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")
    sharpe = (np.mean(pnls) / np.std(pnls) * np.sqrt(252)) if len(pnls) > 1 and np.std(pnls) > 0 else 0
    calmar = (sum(pnls) / abs(dd)) if dd != 0 else 0
    return dict(
        n=len(merged),
        wr=round(len(wins) / len(merged) * 100, 1),
        pnl=round(sum(pnls), 2),
        pf=round(pf, 2) if pf != float("inf") else 999,
        dd=round(dd, 2),
        sharpe=round(sharpe, 2),
        calmar=round(calmar, 2),
    )


def _cluster_breakdown(intents):
    """How many trade-intents fall in each cluster (shows where conflicts live)."""
    from collections import Counter
    c = Counter((t["cluster"] or "—none—") for t in intents)
    return dict(c)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Cluster-cap policy comparison backtest")
    ap.add_argument("--bars", type=int, default=5000, help="bars per instrument")
    ap.add_argument("--swap-margin", type=float, default=0.20,
                    help="RR improvement required to swap (B), e.g. 0.20 = +20%%")
    args = ap.parse_args()

    print("\n" + "=" * 64)
    print("  CLUSTER-CAP POLICY BACKTEST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Bars/instrument: {args.bars:,} | Swap margin (B): +{args.swap_margin:.0%} RR")
    print("=" * 64)

    # 1. generate per-instrument trades
    all_intents = []
    print("\n  Generating per-instrument trades (optimized configs)...")
    for inst in bp.INSTRUMENTS:
        trades = _gen_trades_for_instrument(inst, args.bars)
        cl = _get_cluster_name(inst["sym"]) or "—"
        print(f"    {inst['sym']:<9} cluster={cl:<8} trades={len(trades)}")
        all_intents.extend(trades)

    if not all_intents:
        print("\n  No trades generated — is MT5 connected?")
        return

    print(f"\n  Total trade-intents: {len(all_intents)}")
    print(f"  Cluster distribution: {_cluster_breakdown(all_intents)}")

    # 2. replay under each policy
    policies = [
        ("C · NO-CAP       ", _replay_no_cap(all_intents)),
        ("A · HARD-CAP (live)", _replay_hard_cap(all_intents)),
        ("B · QUALITY-SWAP  ", _replay_quality_swap(all_intents, args.swap_margin)),
    ]

    print("\n" + "=" * 64)
    print("  POLICY COMPARISON")
    print("=" * 64)
    print(f"  {'Policy':<20} {'Ops':>5} {'WR%':>6} {'NetP&L':>10} {'PF':>6} {'MaxDD':>9} {'Sharpe':>7} {'Calmar':>7}")
    print("  " + "-" * 62)

    rows = []
    for label, sel in policies:
        m = _portfolio_metrics(sel)
        rows.append((label, m))
        print(f"  {label:<20} {m['n']:>5} {m['wr']:>6} "
              f"${m['pnl']:>8,.0f} {m['pf']:>6} ${m['dd']:>7,.0f} "
              f"{m['sharpe']:>7} {m['calmar']:>7}")

    # 3. verdict
    print("\n" + "=" * 64)
    print("  READING THE RESULT")
    print("=" * 64)
    c_m = rows[0][1]; a_m = rows[1][1]; b_m = rows[2][1]
    dropped = c_m["n"] - a_m["n"]
    pnl_cost = c_m["pnl"] - a_m["pnl"]
    print(f"  Hard-cap dropped {dropped} trades vs no-cap "
          f"({dropped/max(c_m['n'],1)*100:.0f}% of signals).")
    print(f"  P&L difference (no-cap − hard-cap): ${pnl_cost:+,.0f}")
    print(f"  Sharpe: no-cap {c_m['sharpe']} | hard-cap {a_m['sharpe']} | swap {b_m['sharpe']}")
    print(f"  MaxDD : no-cap ${c_m['dd']:,.0f} | hard-cap ${a_m['dd']:,.0f} | swap ${b_m['dd']:,.0f}")
    print()
    if a_m["sharpe"] >= c_m["sharpe"] and a_m["dd"] >= c_m["dd"]:
        print("  → HARD-CAP wins on risk-adjusted terms: it sacrifices some raw P&L")
        print("    but improves Sharpe and/or cuts drawdown. Keeping it is justified.")
    elif c_m["sharpe"] > a_m["sharpe"] and c_m["dd"] >= a_m["dd"] * 1.1:
        print("  → NO-CAP looks better on Sharpe AND drawdown isn't much worse —")
        print("    the cap may be costing more than it protects. Worth relaxing.")
    else:
        print("  → MIXED: trade-off between P&L and drawdown. See numbers above.")
    if b_m["sharpe"] > max(a_m["sharpe"], c_m["sharpe"]):
        print("  → QUALITY-SWAP shows the best Sharpe (EXPLORATORY — swap P&L is")
        print("    approximated at break-even; treat as directional only).")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
