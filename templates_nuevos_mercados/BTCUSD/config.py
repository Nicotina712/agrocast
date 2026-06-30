"""
BTCUSD Bitcoin Robot — Configuration
Cuenta Demo ICMarkets · 24/7 (crypto never sleeps)
"""

import os
from datetime import time

# ─── Paths ────────────────────────────────────────────────────────────────────

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))

ARTIFACTS_DIR   = os.path.join(_MVP_ROOT, "artifacts", "btcusd")

SIGNAL_FILE     = os.path.join(ARTIFACTS_DIR, "live_signal.json")
PAPER_LOG_FILE  = os.path.join(ARTIFACTS_DIR, "paper_trades.jsonl")
EXEC_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "executions.jsonl")
LIVE_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "live_log.jsonl")
LIVE_STATE_FILE = os.path.join(ARTIFACTS_DIR, "live_state.json")
MODEL_PATH      = os.path.join(ARTIFACTS_DIR, "model.joblib")

# ─── Instrument ───────────────────────────────────────────────────────────────

SYMBOL           = "BTCUSD"
FALLBACK_SYMBOLS = ["BTCUSD", "BTC/USD", "BTCUSDT", "BTCUSD.", "Bitcoin"]

# ─── Session ──────────────────────────────────────────────────────────────────
# Bitcoin trades 24/7 — no hard RTH window.
# We define a "prime session" for new signal generation: NY morning + afternoon.
# Signals can also fire on weekends (crypto never closes).

try:
    from zoneinfo import ZoneInfo as _ZI
    from datetime import datetime as _dtz
    CT_OFFSET_HOURS = int(_dtz.now(_ZI("America/Chicago")).utcoffset().total_seconds() / 3600)
except Exception:
    CT_OFFSET_HOURS = -5  # fallback CDT
PRIME_OPEN_CT   = time(3, 0)  # 03:00 CT — London open, captures EU institutional flow (was 07:00)
PRIME_CLOSE_CT  = time(22, 0) # 22:00 CT — wind-down before Asian open

TRADE_WEEKENDS  = True         # Crypto trades Sat/Sun
NO_NEW_SIGNALS_MINS = 30       # Stop new signals 30 min before prime close

# ─── Data / Model ─────────────────────────────────────────────────────────────
# ⚙️  Optimized 2026-05-25 via grid-search (162 combos × 5000 bars)
# Was: 5m TF, EMA 20/50 → WR 29.7%, P&L -$379, Sharpe -1.53
# Now: 30m TF, EMA 10/50 → WR 34.6%, P&L +$502, Sharpe 2.71  ✅

# 2026-06-22: migrado 30m -> 1h (misma EMA 9/21). Validacion TF neto de costos: 1h = +0.224R/PF1.25/
# minPF0.84 vs 30m = +0.058R/PF1.11. El 1h captura mejor el trend con menos ruido. (1d 20/100 = -1.64, descartado.)
TIMEFRAME      = "1h"
N_BARS_LIVE    = 150           # >=110 requerido por mechanical_signal
MIN_BARS_REQ   = 60
CYCLE_MINUTES  = 60

RETRAIN_TIMEFRAME    = "1h"
N_BARS_TRAIN         = 5000
RETRAIN_HORIZON_BARS = 6       # 3 hours ahead on 30m (was 12 × 5m = 60 min)

# ─── Strategy parameters (walk-forward 2026-06-08) ───────────────────────────
# BTC 30m: EMA 9/21 valido mejor (meanPF 1.39, consist 62-80%) que 10/50 (1.28) y 20/100 (1.06).
# 30m ya era el TF correcto (vs 15m consist 33%, 1h 50%). Mechanical-primary trend + LLM-veto.
EMA_FAST     = 9       # era 10 — 9/21 mejor en walk-forward
EMA_SLOW     = 21      # era 50
SL_ATR_MULT  = 2.0     # era 1.5
TP_ATR_MULT  = 4.0     # RR=2.0 (era 3.75)
RSI_LONG_LO  = 45      # zona momentum trend mecánico
RSI_LONG_HI  = 55
RSI_SHORT_LO = 45
RSI_SHORT_HI = 55
COOLDOWN_BARS= 3       # 3×30m = 1.5h cooldown

# ─── Risk / Sizing ────────────────────────────────────────────────────────────
# BTCUSD: 1 lot = 1 BTC
# 0.01 lot = 0.01 BTC
# P&L: at 0.01 lot, $1 BTC move = $0.01 P/L
# Typical setup stop: $1,500–3,000 BTC points
# 0.01 lot × $1,500 stop = $15 risk  ← safe for $1,013 account
#
# calc_lots: lots = MAX_RISK_USD / stop_usd
#            where stop_usd = stop_points × LOT_SIZE_PER_POINT × lots_per_unit
# For BTCUSD: 1 lot × N points = N USD P/L
# So: stop_usd = stop_points × volume
# Simplification: lots = MAX_RISK_USD / stop_points (at 1 lot = $1/pt at full lot)
#                      = MAX_RISK_USD / stop_points
# Minimum viable: 0.01 lots

CAPITAL_USD    = 1200.0
MAX_RISK_PCT   = 0.015   # Tier C — Sharpe <1.5 (was 2%)
MAX_RISK_USD   = round(CAPITAL_USD * MAX_RISK_PCT, 2)   # ~$15.20

LOT_SIZE_PTS   = 1.0    # 1 full lot moves $1 per point
MIN_LOT        = 0.01
MAX_LOTS       = 0.05   # Cap at 0.05 BTC per trade (conservative)
LOT_STEP       = 0.01
DEFAULT_VOLUME = 0.01   # Always use minimum lot until track record established


def calc_lots(stop_points: float) -> float:
    """
    Compute trade size so risk ≈ MAX_RISK_USD.
    stop_points: distance in USD from entry to stop-loss.
    Returns lots rounded to LOT_STEP, clipped to [MIN_LOT, MAX_LOTS].
    """
    # RIESGO DINAMICO 2026-06-13: presupuesto = balance real MT5 x risk_pct (fallback CAPITAL_USD)
    try:
        from portfolio_guard import live_risk_usd as _lru
        MAX_RISK_USD = _lru(CAPITAL_USD, MAX_RISK_PCT)
    except Exception:
        MAX_RISK_USD = round(CAPITAL_USD * MAX_RISK_PCT, 2)
    if stop_points <= 0:
        return MIN_LOT
    raw = MAX_RISK_USD / stop_points   # lots × stop_points = MAX_RISK_USD
    rounded = round(round(raw / LOT_STEP) * LOT_STEP, 2)
    return max(MIN_LOT, min(MAX_LOTS, rounded))


# ─── Execution ────────────────────────────────────────────────────────────────

EXECUTE_TRADES       = True    # ✅ demo activado 2026-05-27
MIN_SL_PCT           = 0.8    # minimum SL as % of entry price
MAX_LLM_CALLS_PER_DAY = 30   # raised: max observed 26 calls/day (24/7 session)
MAX_ENTRY_SLIP_PCT    = 0.5    # veto si precio adverso > X% del entry LLM
