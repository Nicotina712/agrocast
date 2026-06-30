"""
Portfolio Backtest Engine — All 9 Instruments
==============================================
Fetches historical OHLCV bars from MT5 and simulates the robot strategy
using TECHNICAL signals only (EMA/RSI/VWAP/ATR — no LLM calls).

Signal logic mirrors what the agents produce:
  LONG  : EMA20 > EMA50 · price > VWAP · RSI 40-65 · bullish bar
  SHORT : EMA20 < EMA50 · price < VWAP · RSI 35-60 · bearish bar
  SL    = entry ± ATR × SL_ATR_MULT
  TP    = entry ± ATR × TP_ATR_MULT   (ensures RR >= RR_MIN)

Trade simulation: enter at bar close, check each subsequent bar's
High/Low to find first SL or TP touch (realistic fill).

Usage:
  python backtest_portfolio.py                  # all instruments, 2000 bars
  python backtest_portfolio.py --bars 5000      # more history
  python backtest_portfolio.py --sym BTCUSD     # single instrument
  python backtest_portfolio.py --report html    # open HTML report after
"""

import os, sys, io, json, argparse, warnings
from datetime import datetime, timezone, timedelta, time as dtime

warnings.filterwarnings("ignore")

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(_HERE)
for p in [_HERE, _MVP_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

# ── instrument registry ──────────────────────────────────────────────────────

INSTRUMENTS = [
    dict(sym="XAUUSD",   name="Gold",      folder="XAUUSD",   color="#FFD700"),
    dict(sym="BTCUSD",   name="Bitcoin",   folder="BTCUSD",   color="#f7931a"),
    dict(sym="ETHUSD",   name="Ethereum",  folder="ETHUSD",   color="#627eea"),
    dict(sym="US500",    name="S&P 500",   folder="US500",    color="#00c853"),
    dict(sym="USTEC",    name="Nasdaq",    folder="USTEC",    color="#00b0ff"),
    dict(sym="US30",     name="Dow Jones", folder="US30",     color="#ff6f00"),
    dict(sym="WTI_N6",   name="WTI Oil",   folder="WTI_N6",   color="#795548"),
    dict(sym="BRENT_N6", name="Brent Oil", folder="BRENT_N6", color="#546e7a"),
    dict(sym="UK100",    name="FTSE 100",  folder="UK100",    color="#e91e63"),
    dict(sym="Corn_N6",  name="Corn",      folder="Corn_N6",  color="#cddc39"),  # added 2026-05-31 (OOS+walk-forward validated)
    dict(sym="HK50",     name="Hang Seng", folder="HK50",     color="#e91e63"),  # added 2026-05-31; re-tuned same day → robust (5/5 WF folds, meanSh 4.14)
]

# ── per-instrument optimized configs  (--optimized flag) ─────────────────────
# Reflects actual robot configs after grid-search on 2026-05-25.
# Cooldown: 2 × CYCLE_MINUTES converted to bars per timeframe.
#   5m  TF + 15-min cycle → 2×15=30 min → 6 bars
#   15m TF + 15-min cycle → 2×15=30 min → 2 bars
#   30m TF + 30-min cycle → 2×30=60 min → 2 bars
#   1h  TF + 60-min cycle → 2×60=120 min → 2 bars

OPTIMIZED_CONFIGS = {
    "XAUUSD":   dict(tf="15m", ef=20, es=50,  sl=1.5, tp=3.75, cd=3,   # optimized 2026-05-29
                     rll=45, rhl=55, rls=45, rhs=55),
    "BTCUSD":   dict(tf="30m", ef=10, es=50,  sl=1.5, tp=3.75, cd=5,  # optimized 2026-05-29
                     rll=42, rhl=60, rls=40, rhs=58),
    "ETHUSD":   dict(tf="15m", ef=20, es=100, sl=1.5, tp=3.75, cd=5,  # optimized 2026-05-29
                     rll=42, rhl=60, rls=40, rhs=58),
    "US500":    dict(tf="30m", ef=10, es=50,  sl=1.5, tp=3.0, cd=5,   # optimized 2026-05-29
                     rll=45, rhl=55, rls=45, rhs=55),
    "USTEC":    dict(tf="30m", ef=20, es=50,  sl=1.5, tp=3.0, cd=5,   # optimized 2026-05-29
                     rll=42, rhl=60, rls=40, rhs=58),
    "US30":     dict(tf="5m",  ef=20, es=50,  sl=1.5, tp=3.0, cd=6,
                     rll=38, rhl=65, rls=35, rhs=62),
    "WTI_N6":   dict(tf="15m", ef=20, es=100, sl=1.5, tp=4.5, cd=2,
                     rll=45, rhl=55, rls=45, rhs=55),   # grid-search best
    "BRENT_N6": dict(tf="15m", ef=20, es=100, sl=1.5, tp=4.5, cd=3,   # optimized 2026-05-29
                     rll=45, rhl=55, rls=45, rhs=55),
    "UK100":    dict(tf="5m",  ef=20, es=100, sl=1.5, tp=3.75, cd=3,  # optimized 2026-05-29
                     rll=38, rhl=65, rls=35, rhs=62),
    "Corn_N6":  dict(tf="15m", ef=20, es=50,  sl=1.5, tp=4.5, cd=5,   # added 2026-05-31: OOS Sharpe 3.16 / PF 1.60, 5/5 walk-forward folds positive
                     rll=42, rhl=60, rls=40, rhs=58),
    "HK50":     dict(tf="30m", ef=10, es=30,  sl=1.5, tp=3.0, cd=8,   # re-tuned 2026-05-31 (tune_hk50.py): wide grid + 5-fold WF → meanSh 4.14, 5/5 folds, fullSh 4.31, PF 1.76 (old tp=4.5 cfg was -0.20)
                     rll=38, rhl=65, rls=35, rhs=62),
}

# ── strategy parameters ──────────────────────────────────────────────────────

SL_ATR_MULT   = 1.5    # SL = entry ± ATR × 1.5
TP_ATR_MULT   = 3.0    # TP = entry ± ATR × 3.0   → RR = 2.0
RR_MIN        = 1.5    # skip trade if RR < this
MAX_HOLD_BARS = 48     # force-close after N bars (time stop)
COOLDOWN_BARS = 3      # bars to wait after a trade closes before re-entering
EMA_FAST      = 20
EMA_SLOW      = 50
RSI_PERIOD    = 14
ATR_PERIOD    = 14
ATR_MEAN_BARS = 50     # rolling bars for ATR mean (informational)
MAX_RISK_USD  = 20.26  # ~2% of $1,013 account

# ── helpers ──────────────────────────────────────────────────────────────────

def _load_cfg(folder):
    """Load instrument config from its folder."""
    import importlib
    f = os.path.join(_HERE, folder)
    if f not in sys.path:
        sys.path.insert(0, f)
    try:
        import config as c
        importlib.reload(c)
        return c
    except Exception:
        return None
    finally:
        if f in sys.path:
            sys.path.remove(f)


def _mt5_bars(sym, timeframe, n_bars):
    """Fetch n_bars of history from MT5 using the instrument's mt5_bridge."""
    # Try to use the XAUUSD mt5_bridge (all share the same implementation)
    folder = os.path.join(_HERE, "XAUUSD")
    if folder not in sys.path:
        sys.path.insert(0, folder)
    try:
        import mt5_bridge as _b, importlib
        importlib.reload(_b)
        if not (_b.is_connected() or _b.initialize()):
            return None
        bars = _b.fetch_mt5_bars(timeframe, n_bars, sym)
        return bars
    except Exception as e:
        print(f"  [mt5] {e}")
        return None
    finally:
        if folder in sys.path:
            sys.path.remove(folder)


def _prime_mask(bars, cfg):
    """Boolean mask: True for bars inside the robot's prime session."""
    if cfg is None:
        return pd.Series(True, index=bars.index)

    trade_wknd = getattr(cfg, "TRADE_WEEKENDS", False)
    open_ct    = getattr(cfg, "PRIME_OPEN_CT",  None) or getattr(cfg, "RTH_OPEN_CT",  None)
    close_ct   = getattr(cfg, "PRIME_CLOSE_CT", None) or getattr(cfg, "RTH_CLOSE_CT", None)
    ct_offset  = getattr(cfg, "CT_OFFSET_HOURS", -5)

    if open_ct is None or close_ct is None:
        return pd.Series(True, index=bars.index)

    o_hm = open_ct.hour  * 60 + open_ct.minute
    c_hm = close_ct.hour * 60 + close_ct.minute

    def _in_prime(ts):
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            ct = ts.tz_convert("UTC").replace(tzinfo=timezone.utc) + timedelta(hours=ct_offset)
        else:
            ct = ts + timedelta(hours=ct_offset)
        wd  = ct.weekday()
        hm  = ct.hour * 60 + ct.minute
        if not trade_wknd and wd >= 5:
            return False
        return o_hm <= hm <= c_hm

    return pd.Series([_in_prime(ts) for ts in bars.index], index=bars.index)


# ── technical indicators ──────────────────────────────────────────────────────

def _compute_indicators(bars):
    """Add EMA, RSI, ATR, VWAP columns to a copy of bars."""
    df = bars.copy()
    c  = df["close"]
    h  = df["high"]
    lo = df["low"]

    # EMAs
    df["ema_fast"] = c.ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = c.ewm(span=EMA_SLOW, adjust=False).mean()

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    # ATR
    tr = pd.concat([
        h - lo,
        (h - c.shift()).abs(),
        (lo - c.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"]      = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    df["atr_mean"] = df["atr"].rolling(ATR_MEAN_BARS, min_periods=10).mean()

    # EMA slope (1-bar change, normalised by price)
    df["ema_fast_slope"] = df["ema_fast"].diff() / c

    # Session VWAP (reset daily)
    vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
    vol     = df[vol_col].replace(0, 1)
    dates   = pd.Series(df.index).apply(lambda t: t.date() if hasattr(t,"date") else t).values
    df["_date"] = dates
    vwap_vals = []
    cum_pv = cum_v = 0
    cur_d  = None
    for i, row in df.iterrows():
        if row["_date"] != cur_d:
            cum_pv = cum_v = 0
            cur_d  = row["_date"]
        tp = (row["high"] + row["low"] + row["close"]) / 3
        cum_pv += tp * row[vol_col]
        cum_v  += row[vol_col]
        vwap_vals.append(cum_pv / cum_v if cum_v else row["close"])
    df["vwap"] = vwap_vals
    df.drop(columns=["_date"], inplace=True)

    return df


# ── signal generation ────────────────────────────────────────────────────────

def _generate_signals(df, prime_mask,
                      rsi_lo_long=38, rsi_hi_long=65,
                      rsi_lo_short=35, rsi_hi_short=62):
    """
    Generate LONG / SHORT / FLAT for each bar.
    Requires EMA_SLOW warmup + COOLDOWN buffer.
    """
    signals = ["FLAT"] * len(df)
    cooldown = 0

    for i in range(EMA_SLOW + 5, len(df)):
        if cooldown > 0:
            cooldown -= 1
            continue

        if not prime_mask.iloc[i]:
            continue

        row  = df.iloc[i]
        prev = df.iloc[i - 1]

        atr  = row["atr"]
        if atr == 0 or np.isnan(atr):
            continue

        ema_f = row["ema_fast"]
        ema_s = row["ema_slow"]
        rsi   = row["rsi"]
        close = row["close"]
        vwap  = row["vwap"]
        o_    = row["open"]

        # Trend direction
        bull = ema_f > ema_s
        bear = ema_f < ema_s

        # Momentum: RSI in acceptable range
        rsi_long  = rsi_lo_long  < rsi < rsi_hi_long
        rsi_short = rsi_lo_short < rsi < rsi_hi_short

        # Price vs VWAP
        above_vwap = close > vwap
        below_vwap = close < vwap

        # Bar direction filter (body confirmation)
        bullish_bar = close > o_
        bearish_bar = close < o_

        # Minimum ATR (avoid dead markets)
        atr_ok = atr > (close * 0.0001)   # at least 0.01% of price

        if (bull and rsi_long and above_vwap and bullish_bar and atr_ok):
            signals[i] = "LONG"
        elif (bear and rsi_short and below_vwap and bearish_bar and atr_ok):
            signals[i] = "SHORT"

    return signals


# ── trade simulation ─────────────────────────────────────────────────────────

def _simulate(df, signals, sym_name=""):
    """
    Simulate bar-by-bar execution.
    Returns list of trade dicts with pnl, rr, duration_bars, etc.
    """
    trades = []
    in_trade = False
    entry_i  = None
    direction = None
    entry_px  = sl_px = tp_px = None
    risk_usd  = rr = None

    for i, sig in enumerate(signals):
        if in_trade:
            row = df.iloc[i]
            h, lo = row["high"], row["low"]

            hit_tp = (direction == "LONG"  and h >= tp_px) or \
                     (direction == "SHORT" and lo <= tp_px)
            hit_sl = (direction == "LONG"  and lo <= sl_px) or \
                     (direction == "SHORT" and h >= sl_px)

            bars_held = i - entry_i
            time_stop = bars_held >= MAX_HOLD_BARS

            if hit_tp or hit_sl or time_stop:
                if time_stop and not hit_tp and not hit_sl:
                    exit_px = row["close"]
                    if direction == "LONG":
                        pnl_r = (exit_px - entry_px) / (entry_px - sl_px)
                    else:
                        pnl_r = (entry_px - exit_px) / (sl_px - entry_px)
                    pnl_usd = pnl_r * risk_usd
                    outcome = "TIME"
                else:
                    # SL/TP: worst-case assume SL if both hit same bar
                    if hit_sl and hit_tp:
                        hit_tp = False
                    if hit_tp:
                        exit_px = tp_px
                        pnl_usd = risk_usd * rr
                        outcome = "WIN"
                    else:
                        exit_px = sl_px
                        pnl_usd = -risk_usd
                        outcome = "LOSS"

                trades.append({
                    "entry_i":    entry_i,
                    "exit_i":     i,
                    "entry_time": df.index[entry_i],
                    "exit_time":  df.index[i],
                    "direction":  direction,
                    "entry_px":   entry_px,
                    "sl_px":      sl_px,
                    "tp_px":      tp_px,
                    "exit_px":    exit_px,
                    "rr":         rr,
                    "risk_usd":   risk_usd,
                    "pnl_usd":    round(pnl_usd, 2),
                    "outcome":    outcome,
                    "bars_held":  bars_held,
                })
                in_trade = False
                # reset cooldown counter
                signals[i + 1: i + 1 + COOLDOWN_BARS] = ["FLAT"] * min(COOLDOWN_BARS, len(signals) - i - 1)

        elif sig in ("LONG", "SHORT") and not in_trade:
            row  = df.iloc[i]
            atr  = row["atr"]
            if atr <= 0 or np.isnan(atr):
                continue

            ep  = row["close"]
            sl  = ep - atr * SL_ATR_MULT if sig == "LONG" else ep + atr * SL_ATR_MULT
            tp  = ep + atr * TP_ATR_MULT  if sig == "LONG" else ep - atr * TP_ATR_MULT

            sl_dist = abs(ep - sl)
            tp_dist = abs(tp - ep)
            if sl_dist == 0:
                continue

            rr_ = round(tp_dist / sl_dist, 2)
            if rr_ < RR_MIN:
                continue

            # Estimate lots / risk
            # Simplified: use MAX_RISK_USD directly as risk per trade
            in_trade  = True
            entry_i   = i
            direction = sig
            entry_px  = ep
            sl_px     = sl
            tp_px     = tp
            risk_usd  = MAX_RISK_USD
            rr        = rr_

    return trades


# ── metrics ──────────────────────────────────────────────────────────────────

def _metrics(trades, sym=""):
    if not trades:
        return {
            "sym": sym, "n_trades": 0, "win_rate": 0, "pnl": 0,
            "profit_factor": 0, "avg_rr": 0, "max_dd": 0,
            "sharpe": 0, "calmar": 0, "avg_bars": 0,
        }

    pnls   = [t["pnl_usd"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    cum_pnl = np.cumsum(pnls)
    peak    = np.maximum.accumulate(cum_pnl)
    dd      = cum_pnl - peak
    max_dd  = float(dd.min())

    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")

    # Sharpe: ratio of mean to std of daily trade returns
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
    else:
        sharpe = 0

    calmar = (sum(pnls) / abs(max_dd)) if max_dd != 0 else 0

    return {
        "sym":           sym,
        "n_trades":      len(trades),
        "n_long":        sum(1 for t in trades if t["direction"] == "LONG"),
        "n_short":       sum(1 for t in trades if t["direction"] == "SHORT"),
        "n_win":         len(wins),
        "n_loss":        len(losses),
        "n_time":        sum(1 for t in trades if t["outcome"] == "TIME"),
        "win_rate":      round(len(wins) / len(trades) * 100, 1),
        "pnl":           round(sum(pnls), 2),
        "avg_trade":     round(np.mean(pnls), 2),
        "gross_profit":  round(sum(wins), 2),
        "gross_loss":    round(sum(losses), 2),
        "profit_factor": round(pf, 2) if pf != float("inf") else 999,
        "avg_rr":        round(np.mean([t["rr"] for t in trades]), 2),
        "max_dd":        round(max_dd, 2),
        "sharpe":        round(sharpe, 2),
        "calmar":        round(calmar, 2),
        "avg_bars":      round(np.mean([t["bars_held"] for t in trades]), 1),
        "equity_curve":  list(cum_pnl.round(2)),
    }


# ── single-instrument backtest ────────────────────────────────────────────────

def _run_strategy(bars, cfg, sym,
                  ema_fast=None, ema_slow=None,
                  sl_mult=None, tp_mult=None, cooldown=None,
                  rsi_lo_long=38, rsi_hi_long=65,
                  rsi_lo_short=35, rsi_hi_short=62,
                  verbose=False):
    """Core strategy runner — accepts explicit params, no globals mutation."""
    global EMA_FAST, EMA_SLOW, SL_ATR_MULT, TP_ATR_MULT, COOLDOWN_BARS
    ef  = ema_fast  or EMA_FAST
    es  = ema_slow  or EMA_SLOW
    slm = sl_mult   or SL_ATR_MULT
    tpm = tp_mult   or TP_ATR_MULT
    cd  = cooldown  or COOLDOWN_BARS

    _ef0, _es0, _sl0, _tp0, _cd0 = EMA_FAST, EMA_SLOW, SL_ATR_MULT, TP_ATR_MULT, COOLDOWN_BARS
    EMA_FAST, EMA_SLOW, SL_ATR_MULT, TP_ATR_MULT, COOLDOWN_BARS = ef, es, slm, tpm, cd
    try:
        df   = _compute_indicators(bars)
        mask = _prime_mask(df, cfg)
        sigs = _generate_signals(df, mask,
                                 rsi_lo_long=rsi_lo_long, rsi_hi_long=rsi_hi_long,
                                 rsi_lo_short=rsi_lo_short, rsi_hi_short=rsi_hi_short)
        if verbose:
            prime_pct = mask.mean() * 100
            sig_count = {s: sigs.count(s) for s in ("LONG","SHORT","FLAT")}
            print(f"  Prime-session bars: {mask.sum()} / {len(mask)} ({prime_pct:.1f}%)")
            print(f"  Signals: {sig_count}")
        trades = _simulate(df, sigs, sym)
        return trades
    finally:
        EMA_FAST, EMA_SLOW, SL_ATR_MULT, TP_ATR_MULT, COOLDOWN_BARS = _ef0, _es0, _sl0, _tp0, _cd0


def backtest_instrument(inst, n_bars=2000, verbose=True,
                        tf_override=None, ema_fast_ov=None, ema_slow_ov=None,
                        rsi_lo_long=38, rsi_hi_long=65, rsi_lo_short=35, rsi_hi_short=62,
                        sl_mult_ov=None, tp_mult_ov=None, cooldown_ov=None,
                        bars_df=None):
    """
    Run backtest for a single instrument.
    bars_df: pre-fetched DataFrame (skip MT5 fetch — used by optimizer to reuse bars).
    """
    sym    = inst["sym"]
    folder = inst["folder"]

    if verbose:
        print(f"\n{'='*55}")
        print(f"  {sym} — {inst['name']}")
        print(f"{'='*55}")

    cfg = _load_cfg(folder)
    tf  = tf_override or (getattr(cfg, "TIMEFRAME", "M5") if cfg else "M5")

    # Fetch bars (or reuse)
    if bars_df is not None:
        bars = bars_df
    else:
        if verbose:
            print(f"  Fetching {n_bars} × {tf} bars from MT5...")
        bars = _mt5_bars(sym, tf, n_bars)
        if bars is None or len(bars) < (ema_slow_ov or EMA_SLOW) + 20:
            print(f"  SKIP: not enough bars ({len(bars) if bars is not None else 0})")
            return None, []
        if verbose:
            print(f"  Got {len(bars)} bars | {bars.index[0]} → {bars.index[-1]}")

    trades = _run_strategy(bars, cfg, sym,
                           ema_fast=ema_fast_ov, ema_slow=ema_slow_ov,
                           sl_mult=sl_mult_ov, tp_mult=tp_mult_ov, cooldown=cooldown_ov,
                           rsi_lo_long=rsi_lo_long, rsi_hi_long=rsi_hi_long,
                           rsi_lo_short=rsi_lo_short, rsi_hi_short=rsi_hi_short,
                           verbose=verbose)
    if verbose:
        print(f"  Trades simulated: {len(trades)}")

    m = _metrics(trades, sym)
    if verbose and trades:
        print(f"  Win rate: {m['win_rate']}%  |  P&L: ${m['pnl']:+,.2f}"
              f"  |  PF: {m['profit_factor']}  |  MaxDD: ${m['max_dd']:.2f}"
              f"  |  Sharpe: {m['sharpe']}")

    return m, trades


# ── parameter optimizer ───────────────────────────────────────────────────────

def optimize_instrument(sym, n_bars=5000):
    """
    Grid-search key parameters for a single instrument.
    Tests all combinations and prints a ranked table.
    """
    inst = next((i for i in INSTRUMENTS if i["sym"].upper() == sym.upper()), None)
    if not inst:
        print(f"Unknown symbol: {sym}")
        return

    print(f"\n{'='*65}")
    print(f"  OPTIMIZER — {sym} | {n_bars:,} bars")
    print(f"{'='*65}")

    cfg = _load_cfg(inst["folder"])

    # Parameter grid (use mt5_bridge format: "5m", "15m", "30m", "1h")
    timeframes  = ["5m", "15m", "30m"]
    ema_combos  = [(20, 50), (20, 100), (10, 50)]
    rr_combos   = [(1.5, 3.0), (1.5, 3.75), (1.5, 4.5)]   # (sl_mult, tp_mult)
    rsi_combos  = [
        (38, 65, 35, 62),   # loose  (original)
        (42, 60, 40, 58),   # medium
        (45, 55, 45, 55),   # tight (momentum zone only)
    ]
    cooldowns   = [3, 5]

    results = []
    total = len(timeframes) * len(ema_combos) * len(rr_combos) * len(rsi_combos) * len(cooldowns)
    done  = 0

    # Cache bars per timeframe (avoid re-fetching)
    bars_cache = {}

    for tf in timeframes:
        if tf not in bars_cache:
            print(f"  Fetching {n_bars} × {tf} bars...")
            bars = _mt5_bars(sym, tf, n_bars)
            if bars is None or len(bars) < 120:
                print(f"  SKIP {tf}: not enough bars")
                total -= len(ema_combos)*len(rr_combos)*len(rsi_combos)*len(cooldowns)
                continue
            bars_cache[tf] = bars
            print(f"  {tf}: {len(bars)} bars | {bars.index[0]} -> {bars.index[-1]}")

        bars = bars_cache[tf]

        for (ef, es) in ema_combos:
            for (sl_m, tp_m) in rr_combos:
                for (rll, rhl, rls, rhs) in rsi_combos:
                    for cd in cooldowns:
                        done += 1
                        rr = round(tp_m / sl_m, 1)
                        label = f"TF={tf} EMA={ef}/{es} RR={rr} RSI={rll}-{rhl}/{rls}-{rhs} CD={cd}"
                        trades_ = _run_strategy(
                            bars, cfg, sym,
                            ema_fast=ef, ema_slow=es,
                            sl_mult=sl_m, tp_mult=tp_m,
                            rsi_lo_long=rll, rsi_hi_long=rhl,
                            rsi_lo_short=rls, rsi_hi_short=rhs,
                            cooldown=cd,
                        )
                        m = _metrics(trades_, sym)
                        if m and m["n_trades"] >= 10:
                            results.append({**m, "label": label,
                                           "tf": tf, "ema_fast": ef, "ema_slow": es,
                                           "rr": rr, "sl_mult": sl_m, "tp_mult": tp_m,
                                           "cd": cd})
                        sys.stdout.write(f"\r  {done}/{total} combinations tested...   ")
                        sys.stdout.flush()

    print(f"\n\n  {'='*63}")
    print(f"  TOP 10 RESULTS — {sym} (by Sharpe, min 10 trades)")
    print(f"  {'='*63}")
    print(f"  {'#':>2}  {'P&L':>8}  {'WR':>6}  {'PF':>5}  {'Sharpe':>7}  {'Trades':>6}  Config")
    print(f"  {'-'*63}")

    ranked = sorted(results, key=lambda r: (r["sharpe"], r["pnl"]), reverse=True)
    for i, r in enumerate(ranked[:10], 1):
        sign = "+" if r["pnl"] >= 0 else ""
        print(f"  {i:>2}  {sign}${r['pnl']:>7,.2f}  {r['win_rate']:>5.1f}%"
              f"  {r['profit_factor']:>5.2f}  {r['sharpe']:>7.2f}  {r['n_trades']:>6}  {r['label']}")

    if ranked:
        best = ranked[0]
        print(f"\n  BEST CONFIG for {sym}:")
        print(f"    Timeframe  : {best['tf']}")
        print(f"    EMA        : {best['ema_fast']} / {best['ema_slow']}")
        print(f"    RR         : {best['rr']}:1  (SL×{best['sl_mult']} / TP×{best['tp_mult']})")
        print(f"    Cooldown   : {best['cd']} bars")
        print(f"    Result     : {best['win_rate']}% WR | ${best['pnl']:+,.2f} P&L | Sharpe {best['sharpe']}")
    print(f"  {'='*63}\n")
    return ranked


# ── portfolio aggregation ────────────────────────────────────────────────────

def portfolio_summary(all_metrics, all_trades):
    """Combine all instruments into portfolio-level metrics."""
    valid  = [m for m in all_metrics if m and m["n_trades"] > 0]
    if not valid:
        return {}

    total_pnl = sum(m["pnl"] for m in valid)

    # Combined equity curve (sum across instruments by trade index)
    # Merge all trades by time, sort, compute running portfolio PnL
    merged = sorted(
        [t for trades in all_trades for t in trades],
        key=lambda t: str(t["exit_time"])
    )
    if merged:
        portfolio_pnls = [t["pnl_usd"] for t in merged]
        cum = np.cumsum(portfolio_pnls)
        peak = np.maximum.accumulate(cum)
        dd   = cum - peak
        max_dd_port = float(dd.min())
        sharpe_port = (np.mean(portfolio_pnls) / np.std(portfolio_pnls) * np.sqrt(252)
                       if np.std(portfolio_pnls) > 0 else 0)
    else:
        max_dd_port = 0
        sharpe_port = 0
        cum         = np.array([])
        merged      = []

    all_wins   = sum(m["n_win"]    for m in valid)
    all_losses = sum(m["n_loss"]   for m in valid)
    all_trades_ = sum(m["n_trades"] for m in valid)
    gp = sum(m["gross_profit"] for m in valid)
    gl = sum(m["gross_loss"]   for m in valid)

    return {
        "n_instruments":  len(valid),
        "n_trades_total": all_trades_,
        "n_win":          all_wins,
        "n_loss":         all_losses,
        "win_rate":       round(all_wins / all_trades_ * 100, 1) if all_trades_ else 0,
        "total_pnl":      round(total_pnl, 2),
        "gross_profit":   round(gp, 2),
        "gross_loss":     round(gl, 2),
        "profit_factor":  round(gp / abs(gl), 2) if gl != 0 else 999,
        "max_drawdown":   round(max_dd_port, 2),
        "sharpe":         round(sharpe_port, 2),
        "calmar":         round(total_pnl / abs(max_dd_port), 2) if max_dd_port != 0 else 0,
        "equity_curve":   list(cum.round(2)) if len(cum) > 0 else [],
    }


# ── HTML report ───────────────────────────────────────────────────────────────

def _eq_sparkline(curve, color="#00c853", h=60, w=200):
    """Generate a tiny inline SVG equity curve."""
    if not curve or len(curve) < 2:
        return "<span style='color:#555'>no data</span>"
    mn, mx = min(curve), max(curve)
    rng = mx - mn or 1
    pts = []
    for i, v in enumerate(curve):
        x = i / (len(curve) - 1) * w
        y = h - (v - mn) / rng * h
        pts.append(f"{x:.1f},{y:.1f}")
    final_color = "#00c853" if curve[-1] >= 0 else "#f44336"
    return (f'<svg width="{w}" height="{h}" style="vertical-align:middle">'
            f'<polyline points="{" ".join(pts)}" fill="none" '
            f'stroke="{final_color}" stroke-width="1.5"/></svg>')


def generate_html_report(all_metrics, port_summary, n_bars, out_path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pp  = port_summary

    # Portfolio equity curve sparkline
    port_eq = _eq_sparkline(pp.get("equity_curve",[]), w=400, h=80)

    pnl_color = "#00c853" if pp.get("total_pnl",0) >= 0 else "#f44336"

    # Instrument rows
    rows = ""
    for m in all_metrics:
        if not m:
            continue
        inst = next((i for i in INSTRUMENTS if i["sym"] == m["sym"]), {})
        color = inst.get("color","#888")
        eq_svg = _eq_sparkline(m.get("equity_curve",[]), color=color, w=180, h=50)
        p_c  = "#00c853" if m["pnl"] >= 0 else "#f44336"
        dd_c = "#f44336" if m["max_dd"] < -50 else "#ffa000" if m["max_dd"] < 0 else "#00c853"
        pf_c = "#00c853" if m["profit_factor"] >= 1.5 else "#ffa000" if m["profit_factor"] >= 1.0 else "#f44336"
        sh_c = "#00c853" if m["sharpe"] >= 1 else "#ffa000" if m["sharpe"] >= 0 else "#f44336"

        rows += f"""<tr>
          <td style="color:{color};font-weight:700">{m['sym']}</td>
          <td style="text-align:right">{m['n_trades']}</td>
          <td style="text-align:right;color:#00c853">{m['n_win']}</td>
          <td style="text-align:right;color:#f44336">{m['n_loss']}</td>
          <td style="text-align:right;color:#ffa000">{m['n_time']}</td>
          <td style="text-align:right;font-weight:700">{m['win_rate']}%</td>
          <td style="text-align:right;color:{p_c};font-weight:700">${m['pnl']:+,.2f}</td>
          <td style="text-align:right">${m['avg_trade']:+,.2f}</td>
          <td style="text-align:right;color:{pf_c};font-weight:700">{m['profit_factor']}</td>
          <td style="text-align:right">{m['avg_rr']}</td>
          <td style="text-align:right;color:{dd_c}">${m['max_dd']:.2f}</td>
          <td style="text-align:right;color:{sh_c}">{m['sharpe']}</td>
          <td>{eq_svg}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio Backtest Report</title>
<style>
  *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0 }}
  body {{ background:#0d1117; color:#e6edf3; font-family:system-ui,sans-serif; padding:28px; font-size:14px }}
  h1   {{ font-size:22px; margin-bottom:6px }}
  .sub {{ color:#666; font-size:12px; margin-bottom:28px }}
  h2   {{ font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:1px;
           color:#666; margin:0 0 14px }}
  .summary {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:32px }}
  .sc  {{ background:#161b22; border:1px solid #30363d; border-radius:10px;
           padding:16px 20px; flex:1; min-width:120px }}
  .sc .sv {{ font-size:26px; font-weight:700; line-height:1 }}
  .sc .sl {{ font-size:11px; color:#888; margin-top:5px; text-transform:uppercase }}
  .eq-wrap {{ background:#161b22; border:1px solid #30363d; border-radius:10px;
               padding:20px; margin-bottom:32px; text-align:center }}
  .tbl-wrap {{ background:#161b22; border:1px solid #30363d; border-radius:10px;
                overflow:hidden; margin-bottom:32px }}
  table {{ width:100%; border-collapse:collapse }}
  th {{ background:#21262d; color:#888; font-size:11px; text-transform:uppercase;
        letter-spacing:.6px; padding:10px 14px; text-align:left; white-space:nowrap }}
  td {{ padding:9px 14px; border-bottom:1px solid #21262d; font-size:13px }}
  tr:last-child td {{ border-bottom:none }}
  tr:hover td {{ background:#1c2128 }}
  .note {{ font-size:11px; color:#555; border:1px solid #21262d; border-radius:8px;
            padding:12px 16px; line-height:1.6 }}
  .legend {{ display:flex; gap:20px; flex-wrap:wrap; margin-bottom:14px; font-size:12px; color:#888 }}
  .leg {{ display:flex; align-items:center; gap:6px }}
  .dot {{ width:10px; height:10px; border-radius:50% }}
</style>
</head>
<body>

<h1>📊 Portfolio Backtest Report</h1>
<div class="sub">Generado: {now} &nbsp;·&nbsp; Barras por instrumento: {n_bars:,}
&nbsp;·&nbsp; Señales: técnicas (EMA{EMA_FAST}/EMA{EMA_SLOW}/RSI/VWAP/ATR)
&nbsp;·&nbsp; SL: {SL_ATR_MULT}×ATR &nbsp;·&nbsp; TP: {TP_ATR_MULT}×ATR (RR≈{TP_ATR_MULT/SL_ATR_MULT:.1f})</div>

<h2>Portfolio Consolidado — {pp.get('n_instruments',0)} Instrumentos</h2>
<div class="summary">
  <div class="sc"><div class="sv">{pp.get('n_trades_total',0)}</div><div class="sl">Operaciones</div></div>
  <div class="sc"><div class="sv" style="color:#00c853">{pp.get('n_win',0)}</div><div class="sl">Wins</div></div>
  <div class="sc"><div class="sv" style="color:#f44336">{pp.get('n_loss',0)}</div><div class="sl">Losses</div></div>
  <div class="sc"><div class="sv">{pp.get('win_rate',0)}%</div><div class="sl">Win Rate</div></div>
  <div class="sc"><div class="sv" style="color:{pnl_color}">${pp.get('total_pnl',0):+,.2f}</div><div class="sl">P&amp;L Total</div></div>
  <div class="sc"><div class="sv">{pp.get('profit_factor',0)}</div><div class="sl">Profit Factor</div></div>
  <div class="sc"><div class="sv" style="color:#f44336">${pp.get('max_drawdown',0):.2f}</div><div class="sl">Max Drawdown</div></div>
  <div class="sc"><div class="sv">{pp.get('sharpe',0)}</div><div class="sl">Sharpe Ratio</div></div>
  <div class="sc"><div class="sv">{pp.get('calmar',0)}</div><div class="sl">Calmar Ratio</div></div>
</div>

<h2>Curva de Equity del Portfolio</h2>
<div class="eq-wrap">
  {port_eq}
  <div style="font-size:11px;color:#555;margin-top:8px">
    Operaciones ordenadas cronológicamente — todos los instrumentos combinados
  </div>
</div>

<h2>Resultados por Instrumento</h2>
<div class="tbl-wrap">
  <table>
    <thead><tr>
      <th>Instrumento</th>
      <th style="text-align:right">Ops</th>
      <th style="text-align:right">Win</th>
      <th style="text-align:right">Loss</th>
      <th style="text-align:right">Time</th>
      <th style="text-align:right">Win%</th>
      <th style="text-align:right">P&amp;L</th>
      <th style="text-align:right">Avg/Op</th>
      <th style="text-align:right">Prof.Factor</th>
      <th style="text-align:right">Avg R:R</th>
      <th style="text-align:right">Max DD</th>
      <th style="text-align:right">Sharpe</th>
      <th>Equity Curve</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>

<div class="legend">
  <span class="leg"><span class="dot" style="background:#00c853"></span> Profit Factor ≥ 1.5 (bueno)</span>
  <span class="leg"><span class="dot" style="background:#ffa000"></span> Profit Factor 1.0–1.5 (marginal)</span>
  <span class="leg"><span class="dot" style="background:#f44336"></span> Profit Factor &lt; 1.0 (negativo)</span>
</div>

<div class="note">
  <b>Metodología:</b> Señales técnicas (EMA{EMA_FAST}/EMA{EMA_SLOW} trend + RSI{RSI_PERIOD} momentum + VWAP + ATR{ATR_PERIOD}).
  Solo se opera dentro de la sesión prime de cada robot.
  SL = {SL_ATR_MULT}×ATR, TP = {TP_ATR_MULT}×ATR (RR ~ {TP_ATR_MULT/SL_ATR_MULT:.0f}:1).
  Stop por tiempo: {MAX_HOLD_BARS} barras. Riesgo fijo: ${MAX_RISK_USD}/operación.
  <b>Sin LLM</b> — los resultados reales del robot pueden diferir.
  Sin slippage, sin comisiones (spread ya incluido en precios MT5).
</div>

</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Portfolio Backtest Engine")
    parser.add_argument("--bars",     type=int, default=2000,
                        help="Bars of history per instrument (default 2000)")
    parser.add_argument("--sym",      type=str, default=None,
                        help="Test single instrument only (e.g. BTCUSD)")
    parser.add_argument("--report",   action="store_true",
                        help="Open HTML report in browser after run")
    parser.add_argument("--optimize",  type=str, default=None, metavar="SYM",
                        help="Run parameter grid-search for a single instrument (e.g. BTCUSD)")
    parser.add_argument("--optimized", action="store_true",
                        help="Use per-instrument optimized configs (OPTIMIZED_CONFIGS dict)")
    args = parser.parse_args()

    # ── Optimizer mode ────────────────────────────────────────────────────────
    if args.optimize:
        optimize_instrument(args.optimize.upper(), n_bars=args.bars)
        return

    mode_label = "OPTIMIZED (per-instrument)" if args.optimized else "BASELINE (uniform)"
    print("\n" + "="*55)
    print("  Portfolio Backtest Engine")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode  : {mode_label}")
    print(f"  Bars  : {args.bars:,} | Risk: ${MAX_RISK_USD}/trade")
    print("="*55)

    instruments = INSTRUMENTS
    if args.sym:
        instruments = [i for i in INSTRUMENTS if i["sym"].upper() == args.sym.upper()]
        if not instruments:
            print(f"Unknown symbol: {args.sym}")
            return

    all_metrics = []
    all_trades  = []

    for inst in instruments:
        if args.optimized:
            cfg = OPTIMIZED_CONFIGS.get(inst["sym"], {})
            m, trades = backtest_instrument(
                inst, n_bars=args.bars, verbose=True,
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
        else:
            m, trades = backtest_instrument(inst, n_bars=args.bars, verbose=True)
        all_metrics.append(m)
        all_trades.append(trades)

    print("\n" + "="*55)
    print("  PORTFOLIO SUMMARY")
    print("="*55)

    pp = portfolio_summary(all_metrics, all_trades)
    if pp:
        pnl_s = f"+${pp['total_pnl']:,.2f}" if pp['total_pnl'] >= 0 else f"-${abs(pp['total_pnl']):,.2f}"
        print(f"  Instruments  : {pp['n_instruments']}")
        print(f"  Total trades : {pp['n_trades_total']:,}")
        print(f"  Win rate     : {pp['win_rate']}%")
        print(f"  Net P&L      : {pnl_s}")
        print(f"  Profit Factor: {pp['profit_factor']}")
        print(f"  Max Drawdown : ${pp['max_drawdown']:.2f}")
        print(f"  Sharpe       : {pp['sharpe']}")
        print(f"  Calmar       : {pp['calmar']}")
    print("="*55)

    # Save HTML report
    out = os.path.join(_HERE, "backtest_report.html")
    generate_html_report(all_metrics, pp, args.bars, out)
    print(f"\n  Report saved: {out}")

    if args.report:
        import webbrowser
        webbrowser.open(f"file:///{out}")
    else:
        print("  Run with --report to open in browser automatically.")


if __name__ == "__main__":
    main()
