"""
ETHUSD Ethereum Robot — Configuration
Cuenta Demo ICMarkets · 24/7 (crypto never sleeps)
"""

import os
from datetime import time

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))

ARTIFACTS_DIR   = os.path.join(_MVP_ROOT, "artifacts", "ethusd")
SIGNAL_FILE     = os.path.join(ARTIFACTS_DIR, "live_signal.json")
PAPER_LOG_FILE  = os.path.join(ARTIFACTS_DIR, "paper_trades.jsonl")
EXEC_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "executions.jsonl")
LIVE_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "live_log.jsonl")
LIVE_STATE_FILE = os.path.join(ARTIFACTS_DIR, "live_state.json")
MODEL_PATH      = os.path.join(ARTIFACTS_DIR, "model.joblib")

# ─── Instrument ───────────────────────────────────────────────────────────────

SYMBOL           = "ETHUSD"
FALLBACK_SYMBOLS = ["ETHUSD", "ETH/USD", "ETHUSD.", "Ethereum"]

# ─── Session ──────────────────────────────────────────────────────────────────
# ETH trades 24/7 like BTC.
# Same prime session concept: 03:00-22:00 CT (London open through NY close).
try:
    from zoneinfo import ZoneInfo as _ZI
    from datetime import datetime as _dtz
    CT_OFFSET_HOURS = int(_dtz.now(_ZI("America/Chicago")).utcoffset().total_seconds() / 3600)
except Exception:
    CT_OFFSET_HOURS = -5  # fallback CDT
PRIME_OPEN_CT    = time(3, 0)   # 03:00 CT — London open (was 07:00)
PRIME_CLOSE_CT   = time(22, 0)
TRADE_WEEKENDS   = True
NO_NEW_SIGNALS_MINS = 30

# ─── Data / Model ─────────────────────────────────────────────────────────────

# ⚙️  Optimized 2026-05-29 via grid-search (162 combos × 5000 bars)
# Was: 30m TF, EMA 20/100, RSI 38-65 → Sharpe 0.28
# Now: 15m TF, EMA 20/100, RSI 42-60/40-58 → Sharpe 1.40  ✅
# ⚙️ Re-arquitecturado 2026-06-08 (walk-forward 8 folds): ETH funciona en 1h, NO en 15m.
# 15m trend = ruido (consist 50%, whipsaw). 1h trend 20/100 = robusto (meanPF 1.49, consist 83%, minPF 0.91).
# El problema NO era la estrategia sino el TIMEFRAME. Ahora mechanical-primary trend + LLM-veto.
TIMEFRAME         = "1h"     # era 15m — cripto en 15m whipsaquea; en 1h las tendencias son limpias
N_BARS_LIVE       = 250      # 250×1h = ~10 dias de barras
MIN_BARS_REQ      = 110      # EMA100 necesita warmup
CYCLE_MINUTES     = 30       # cicla cada 30min sobre barras de 1h (re-evalua 2x por barra)

RETRAIN_TIMEFRAME    = "1h"
N_BARS_TRAIN         = 5000
RETRAIN_HORIZON_BARS = 4    # 4h ahead on 1h

# ─── Optimized signal parameters (1h walk-forward 2026-06-08) ─────────────────
EMA_FAST         = 20
EMA_SLOW         = 100
SL_ATR_MULT      = 2.0     # validado 1h
TP_ATR_MULT      = 4.0     # RR=2.0 (meanPF 1.60, exp +0.38R)
RSI_LONG_LO      = 45      # zona momentum (trend mecánico)
RSI_LONG_HI      = 55
RSI_SHORT_LO     = 45
RSI_SHORT_HI     = 55
COOLDOWN_BARS    = 3       # 3×1h = 3h cooldown

# ─── Risk / Sizing ────────────────────────────────────────────────────────────
# ETHUSD: 1 lot = 1 ETH ≈ $2,100
# 0.01 lot = 0.01 ETH ≈ $21 notional (very small!)
# P&L at 0.01 lot: $1 ETH move = $0.01 profit/loss
# Typical setup stop: $80-200 points
# 0.01 lot × $100 stop = $1 risk → need more lots
# lots = MAX_RISK_USD / stop_points
# E.g. $100 stop → lots = 20.26/100 = 0.20 lots
# E.g. $200 stop → lots = 20.26/200 = 0.10 lots

CAPITAL_USD    = 1_013.0
MAX_RISK_PCT   = 0.015   # Tier C — Sharpe <1.5 (was 2%)
MAX_RISK_USD   = round(CAPITAL_USD * MAX_RISK_PCT, 2)   # ~$15.20

LOT_SIZE_PTS   = 1.0     # 1 full lot × 1 point = $1 P&L
MIN_LOT        = 0.01
MAX_LOTS       = 0.50    # ETH is cheaper per lot, allow up to 0.50
LOT_STEP       = 0.01
DEFAULT_VOLUME = 0.10    # Default 0.10 ETH (risk ~$10 at $100 stop)


def calc_lots(stop_points: float) -> float:
    """
    Compute trade size so risk ≈ MAX_RISK_USD.
    stop_points: distance in USD from entry to stop-loss.
    """
    if stop_points <= 0:
        return MIN_LOT
    raw     = MAX_RISK_USD / stop_points
    rounded = round(round(raw / LOT_STEP) * LOT_STEP, 2)
    return max(MIN_LOT, min(MAX_LOTS, rounded))


# ─── Execution ────────────────────────────────────────────────────────────────

EXECUTE_TRADES         = True   # ✅ demo activado 2026-05-25
MIN_SL_PCT           = 0.8    # minimum SL as % of entry price (piso = 0.8% = ~$14 @ $1774)
MAX_SL_PCT           = 2.0    # NUEVO: techo SL — LLM no puede poner SL > 2% (era sin limite → 4.6%!)
                               # 2% @ $1774 = 35 pts → lots = 15.20/35 = 0.43 (vs 0.25 anterior)
SL_ATR_MULT          = 2.0    # raised 1.5→2.0: ATR M15 mediana=6.88pts, sweep rate 7%→3%
TP_ATR_MULT          = 5.0    # RR=2.5 mantenido (era 3.75, ajustado a 5.0 con nuevo SL_ATR)
MAX_LLM_CALLS_PER_DAY = 34   # raised: max observed 30 calls/day (24/7 session)
MAX_ENTRY_SLIP_PCT    = 0.4    # veto si precio adverso > X% del entry LLM
MAX_HOLD_HOURS        = 36     # cierre por tiempo (subido 12→36: en 1h los trends respiran mas).
                               # Corta los holds extremos (era 85h) pero deja correr el edge trend de 1h.
