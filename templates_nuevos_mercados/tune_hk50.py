"""
HK50 fine-tuning — wider grid-search validated by WALK-FORWARD consistency.

Honest protocol (no cherry-picking a lucky OOS window):
  1. Fetch bars once per timeframe.
  2. For each candidate config, split the series into K sequential folds (walk-forward).
  3. Run the mechanical EMA/RSI strategy in each fold (HK50 prime-session mask applied
     via the real HK50 config → CT_OFFSET_HOURS=+8, 09:30-16:00 HKT).
  4. A config is ROBUST only if it clears a minimum trade count AND a minimum fraction
     of folds with positive Sharpe. We rank robust configs by MEAN fold Sharpe.
  5. Report the best robust config + full-sample metrics, and compare to the current
     live HK50 config. If nothing clears the gate → recommend RETIRE.

Risk fixed at $200/trade (Sharpe/PF/WR are risk-invariant; only $ scale changes).

Run with the MT5 interpreter:
  C:\\Users\\Lenovo\\AppData\\Local\\Programs\\Python\\Python312\\python.exe tune_hk50.py
"""
import os, sys, math
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import backtest_portfolio as bp

bp.MAX_RISK_USD = 200.0

SYM       = "HK50"
N_BARS    = 5000
K_FOLDS   = 5
MIN_TRADES_TOTAL = 40     # over the whole sample
MIN_TRADES_FOLD  = 5      # per fold to count its Sharpe as meaningful
MIN_POS_FOLDS    = 4      # of K_FOLDS must have positive Sharpe to be "robust"

TIMEFRAMES = ["15m", "30m", "1h"]
EMA_COMBOS = [(10, 50), (20, 50), (20, 100), (10, 30), (50, 200)]
RR_COMBOS  = [(1.5, 3.0), (1.5, 3.75), (1.5, 4.5), (2.0, 4.0)]
RSI_COMBOS = [(38, 65, 35, 62), (42, 60, 40, 58), (45, 55, 45, 55), (40, 70, 30, 60)]
COOLDOWNS  = [3, 5, 8]

# current live config for comparison
CURRENT = dict(tf="30m", ef=10, es=50, sl=1.5, tp=4.5, cd=5,
               rsi=(42, 60, 40, 58))

_CFG = bp._load_cfg("HK50")   # real HK50 config → correct prime-session mask


def _bt_metrics(bars, ef, es, sl, tp, rsi, cd):
    rll, rhl, rls, rhs = rsi
    trades = bp._run_strategy(bars, _CFG, SYM, ema_fast=ef, ema_slow=es,
                              sl_mult=sl, tp_mult=tp, cooldown=cd,
                              rsi_lo_long=rll, rsi_hi_long=rhl,
                              rsi_lo_short=rls, rsi_hi_short=rhs, verbose=False)
    return bp._metrics(trades, SYM)


def _walk_forward(bars, ef, es, sl, tp, rsi, cd):
    """Return (mean_fold_sharpe, pos_folds, n_folds_valid, total_trades, full_metrics, fold_sharpes)."""
    n = len(bars)
    fold_len = n // K_FOLDS
    fold_sharpes = []
    for k in range(K_FOLDS):
        a = k * fold_len
        b = (k + 1) * fold_len if k < K_FOLDS - 1 else n
        seg = bars.iloc[a:b]
        m = _bt_metrics(seg, ef, es, sl, tp, rsi, cd)
        if m and m["n_trades"] >= MIN_TRADES_FOLD:
            fold_sharpes.append(m["sharpe"])
        else:
            fold_sharpes.append(None)
    valid = [s for s in fold_sharpes if s is not None]
    pos   = sum(1 for s in valid if s > 0)
    mean_sh = float(np.mean(valid)) if valid else -99.0
    full = _bt_metrics(bars, ef, es, sl, tp, rsi, cd)
    total_trades = full["n_trades"] if full else 0
    return mean_sh, pos, len(valid), total_trades, full, fold_sharpes


def main():
    print("\n" + "=" * 96)
    print("  HK50 FINE-TUNE — wider grid validated by walk-forward consistency (K=%d folds)" % K_FOLDS)
    print("  Gate: >= %d total trades AND >= %d/%d folds with positive Sharpe" %
          (MIN_TRADES_TOTAL, MIN_POS_FOLDS, K_FOLDS))
    print("=" * 96)

    # baseline: current live config
    bars_by_tf = {}
    for tf in TIMEFRAMES:
        b = bp._mt5_bars(SYM, tf, N_BARS)
        if b is not None and len(b) >= 400:
            bars_by_tf[tf] = b
            print(f"  fetched {tf}: {len(b)} bars  {b.index[0]} -> {b.index[-1]}")
        else:
            print(f"  {tf}: insufficient bars — skipped")

    cur_bars = bars_by_tf.get(CURRENT["tf"])
    if cur_bars is not None:
        ms, pos, nv, tt, full, fs = _walk_forward(
            cur_bars, CURRENT["ef"], CURRENT["es"], CURRENT["sl"], CURRENT["tp"],
            CURRENT["rsi"], CURRENT["cd"])
        fss = ", ".join(f"{s:.2f}" if s is not None else "—" for s in fs)
        print("\n  CURRENT live config (tf=%s ema=%d/%d tp=%.2f cd=%d rsi=%s):" %
              (CURRENT["tf"], CURRENT["ef"], CURRENT["es"], CURRENT["tp"], CURRENT["cd"], CURRENT["rsi"][:2]))
        print(f"    full-sample Sharpe {full['sharpe']:.2f} | PF {full['profit_factor']:.2f} | "
              f"P&L ${full['pnl']:.0f} | n {tt} | folds [{fss}] pos {pos}/{nv}")

    results = []
    total_cfgs = 0
    for tf, bars in bars_by_tf.items():
        for ef, es in EMA_COMBOS:
            if ef >= es:
                continue
            for sl, tp in RR_COMBOS:
                for rsi in RSI_COMBOS:
                    for cd in COOLDOWNS:
                        total_cfgs += 1
                        ms, pos, nv, tt, full, fs = _walk_forward(bars, ef, es, sl, tp, rsi, cd)
                        if tt < MIN_TRADES_TOTAL or full is None:
                            continue
                        results.append(dict(tf=tf, ef=ef, es=es, sl=sl, tp=tp, rsi=rsi, cd=cd,
                                            mean_sh=ms, pos=pos, nv=nv, tt=tt,
                                            full_sh=full["sharpe"], full_pf=full["profit_factor"],
                                            full_pnl=full["pnl"], full_dd=full["max_dd"], fs=fs))

    print(f"\n  Scanned {total_cfgs} configs across {len(bars_by_tf)} timeframes; "
          f"{len(results)} passed the min-trade gate.\n")

    robust = [r for r in results if r["pos"] >= MIN_POS_FOLDS]
    robust.sort(key=lambda r: r["mean_sh"], reverse=True)

    print("  " + "-" * 94)
    print("  TOP ROBUST CONFIGS (>= %d/%d positive folds), ranked by mean fold Sharpe:" %
          (MIN_POS_FOLDS, K_FOLDS))
    print("  %-4s %-8s %-7s %-4s %-22s %7s %7s %7s %6s %8s  folds" %
          ("tf", "ema", "tp", "cd", "rsi", "meanSh", "fullSh", "fullPF", "n", "P&L"))
    if not robust:
        print("    NONE cleared the robustness gate.")
    for r in robust[:12]:
        fss = "/".join(f"{s:.1f}" if s is not None else "—" for s in r["fs"])
        print("  %-4s %-8s %-7s %-4d %-22s %7.2f %7.2f %7.2f %6d %8.0f  [%s] %d/%d" % (
            r["tf"], f"{r['ef']}/{r['es']}", r["tp"], r["cd"], str(r["rsi"]),
            r["mean_sh"], r["full_sh"], r["full_pf"], r["tt"], r["full_pnl"], fss, r["pos"], r["nv"]))

    # also show overall best by full-sample Sharpe (the overfit pick, for contrast)
    by_full = sorted(results, key=lambda r: r["full_sh"], reverse=True)
    print("\n  (for contrast) best by FULL-SAMPLE Sharpe — the overfit-prone pick:")
    for r in by_full[:3]:
        fss = "/".join(f"{s:.1f}" if s is not None else "—" for s in r["fs"])
        print("    tf=%s ema=%d/%d tp=%.2f cd=%d rsi=%s | fullSh %.2f | folds [%s] pos %d/%d" % (
            r["tf"], r["ef"], r["es"], r["tp"], r["cd"], str(r["rsi"][:2]),
            r["full_sh"], fss, r["pos"], r["nv"]))

    print("\n  " + "=" * 92)
    if robust:
        b = robust[0]
        rll, rhl, rls, rhs = b["rsi"]
        print("  RECOMMENDED robust HK50 config (replace OPTIMIZED_CONFIGS + config.py):")
        print('    "HK50": dict(tf="%s", ef=%d, es=%d, sl=%.1f, tp=%.2f, cd=%d, rll=%d, rhl=%d, rls=%d, rhs=%d),'
              % (b["tf"], b["ef"], b["es"], b["sl"], b["tp"], b["cd"], rll, rhl, rls, rhs))
        print("    mean fold Sharpe %.2f | %d/%d positive folds | full Sharpe %.2f | PF %.2f | P&L $%.0f"
              % (b["mean_sh"], b["pos"], b["nv"], b["full_sh"], b["full_pf"], b["full_pnl"]))
    else:
        print("  No HK50 config cleared walk-forward robustness → RECOMMENDATION: RETIRE HK50.")
    print("  " + "=" * 92 + "\n")


if __name__ == "__main__":
    main()
