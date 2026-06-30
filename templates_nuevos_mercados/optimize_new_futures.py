"""
Optimize + OOS-validate candidate new futures before adding them to the portfolio.

Honest protocol (no peeking):
  1. Fetch bars once per (symbol, timeframe).
  2. Split by TIME: first 70% = TRAIN, last 30% = TEST.
  3. Grid-search params on TRAIN only.
  4. Select the config with the best TRAIN Sharpe (min trade count enforced).
  5. Report that config's TEST (out-of-sample) metrics — the number that matters.
  6. ACCEPT only if OOS Sharpe and profit factor clear a bar.

Risk is fixed at $200/trade (= 2% of a $10k account) to match the live setting the
user wants to keep; Sharpe/WR/PF/Calmar are risk-invariant anyway.

Run with the MT5 interpreter:
  C:\\Users\\Lenovo\\AppData\\Local\\Programs\\Python\\Python312\\python.exe optimize_new_futures.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import backtest_portfolio as bp

bp.MAX_RISK_USD = 200.0
TRAIN_FRAC = 0.70
MIN_TRAIN_TRADES = 20
MIN_TEST_TRADES  = 8
ACCEPT_OOS_SHARPE = 1.0
ACCEPT_OOS_PF     = 1.10

# candidate (symbol, [timeframes to try])
CANDIDATES = [
    ("Wheat_N6",  ["15m", "30m"]),
    ("Coffee_N6", ["15m", "30m"]),
    ("HK50",      ["15m", "30m"]),
    ("DE40",      ["15m", "30m"]),
    ("Corn_N6",   ["15m", "30m"]),
    ("XAGUSD",    ["15m", "30m"]),
    ("STOXX50",   ["15m", "30m"]),
    ("Cocoa_N6",  ["15m", "30m"]),
]

EMA_COMBOS = [(20, 50), (20, 100), (10, 50)]
RR_COMBOS  = [(1.5, 3.0), (1.5, 3.75), (1.5, 4.5)]
RSI_COMBOS = [(38, 65, 35, 62), (42, 60, 40, 58), (45, 55, 45, 55)]
COOLDOWNS  = [3, 5]


def _bt(bars, sym, ef, es, sl, tp, rsi, cd):
    rll, rhl, rls, rhs = rsi
    trades = bp._run_strategy(bars, None, sym, ema_fast=ef, ema_slow=es,
                              sl_mult=sl, tp_mult=tp, cooldown=cd,
                              rsi_lo_long=rll, rsi_hi_long=rhl,
                              rsi_lo_short=rls, rsi_hi_short=rhs, verbose=False)
    return bp._metrics(trades, sym)


def optimize_symbol(sym, timeframes):
    best = None   # best by TRAIN sharpe
    for tf in timeframes:
        bars = bp._mt5_bars(sym, tf, 5000)
        if bars is None or len(bars) < 400:
            print(f"  {sym} {tf}: insufficient bars"); continue
        split = int(len(bars) * TRAIN_FRAC)
        tr, te = bars.iloc[:split], bars.iloc[split:]
        for ef, es in EMA_COMBOS:
            for sl, tp in RR_COMBOS:
                for rsi in RSI_COMBOS:
                    for cd in COOLDOWNS:
                        mtr = _bt(tr, sym, ef, es, sl, tp, rsi, cd)
                        if not mtr or mtr["n_trades"] < MIN_TRAIN_TRADES:
                            continue
                        if best is None or mtr["sharpe"] > best["train"]["sharpe"]:
                            mte = _bt(te, sym, ef, es, sl, tp, rsi, cd)
                            best = dict(tf=tf, ef=ef, es=es, sl=sl, tp=tp, rsi=rsi, cd=cd,
                                        train=mtr, test=mte)
    return best


def main():
    print("\n" + "=" * 92)
    print("  NEW-FUTURES OPTIMIZER + OOS VALIDATION  (train 70% / test 30%, risk $200/trade)")
    print("=" * 92)
    print(f"  {'SYM':<10}{'cfg':<34}{'TR_Shrp':>8}{'TR_n':>6}{'OOS_Shrp':>9}{'OOS_PF':>8}{'OOS_n':>6}{'OOS_P&L':>9}  verdict")
    print("  " + "-" * 90)
    accepted = []
    for sym, tfs in CANDIDATES:
        b = optimize_symbol(sym, tfs)
        if b is None:
            print(f"  {sym:<10}  no config with >= {MIN_TRAIN_TRADES} train trades"); continue
        tr, te = b["train"], b["test"]
        rll, rhl, rls, rhs = b["rsi"]
        cfgstr = f"tf={b['tf']} ema={b['ef']}/{b['es']} tp={b['tp']} rsi={rll}-{rhl} cd={b['cd']}"
        ok = (te["n_trades"] >= MIN_TEST_TRADES and te["sharpe"] >= ACCEPT_OOS_SHARPE
              and te["profit_factor"] >= ACCEPT_OOS_PF)
        verdict = "ACCEPT" if ok else "reject"
        print(f"  {sym:<10}{cfgstr:<34}{tr['sharpe']:>8.2f}{tr['n_trades']:>6}"
              f"{te['sharpe']:>9.2f}{te['profit_factor']:>8.2f}{te['n_trades']:>6}{te['pnl']:>9.0f}  {verdict}")
        if ok:
            accepted.append((sym, b))

    print("\n  " + "=" * 90)
    if accepted:
        print("  ACCEPTED (survive OOS) — OPTIMIZED_CONFIGS entries:")
        for sym, b in accepted:
            rll, rhl, rls, rhs = b["rsi"]
            print(f'    "{sym}": dict(tf="{b["tf"]}", ef={b["ef"]}, es={b["es"]}, '
                  f'sl={b["sl"]}, tp={b["tp"]}, cd={b["cd"]}, '
                  f'rll={rll}, rhl={rhl}, rls={rls}, rhs={rhs}),')
    else:
        print("  No candidate survived OOS validation.")
    print("  " + "=" * 90 + "\n")


if __name__ == "__main__":
    main()
