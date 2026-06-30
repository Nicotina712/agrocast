"""
HK50 Hang Seng Index Robot — Configuration
Cuenta Demo ICMarkets · Hong Kong session (CT)
Added 2026-05-31 — candidato condicional (walk-forward 4/5 folds fuertes, 1 plano;
TRAIN Sharpe 3.09 / OOS Sharpe 0.95). En producción VIGILADO, pendiente de afinar.

ICMarkets opera HK50 en dos ventanas (basado en análisis de volumen MT5):
  Sesión A (HK día):   22:00-03:00 CT  (12:00-17:00 HKT, afternoon + cierre)
  Sesión B (HK noche): 04:00-14:00 CT  (17:15-03:00 HKT, night futures session)
PRIME_OPEN/CLOSE cubren ambas con una ventana 22:00-14:00 CT que cruza medianoche.
_is_prime en live_runner maneja el midnight-crossing.
"""

import os
from datetime import time

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))

ARTIFACTS_DIR   = os.path.join(_MVP_ROOT, "artifacts", "hk50")
SIGNAL_FILE     = os.path.join(ARTIFACTS_DIR, "live_signal.json")
PAPER_LOG_FILE  = os.path.join(ARTIFACTS_DIR, "paper_trades.jsonl")
EXEC_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "executions.jsonl")
LIVE_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "live_log.jsonl")
LIVE_STATE_FILE = os.path.join(ARTIFACTS_DIR, "live_state.json")
MODEL_PATH      = os.path.join(ARTIFACTS_DIR, "model.joblib")

# ─── Instrument ───────────────────────────────────────────────────────────────

SYMBOL           = "HK50"
FALLBACK_SYMBOLS = ["HK50", "HSI", "HK50.cash", "HangSeng", "HSI50"]

# ─── Session ──────────────────────────────────────────────────────────────────
# ICMarkets HK50 CFD: basado en análisis de volumen MT5 (72h), dos picos:
#   22:00-03:00 CT  (HK sesión día: tarde HKT, máximo volumen)
#   04:00-14:00 CT  (HK sesión noche: futures overnight, volumen moderado-alto)
# Ventana unificada 22:00 CT → 14:00 CT cruza medianoche.
# _is_prime() en live_runner detecta midnight-crossing automáticamente.

try:
    from zoneinfo import ZoneInfo as _ZI
    from datetime import datetime as _dtz
    CT_OFFSET_HOURS = int(_dtz.now(_ZI("America/Chicago")).utcoffset().total_seconds() / 3600)
except Exception:
    CT_OFFSET_HOURS = -5  # fallback CDT
# 2026-06-10 AUDITORIA SESIONES: la ventana 22:00-14:00 CT estaba AL REVES —
# dentro de HKEX el edge VWAP era ~0 (PF 1.00) y las 11h restantes (03-14 CT,
# HKEX CERRADO) eran expR NEGATIVO (-0.12). Corregido a la sesión HKEX real:
# 09:30-16:00 HKT = 20:30-03:00 CT (cruza medianoche; _is_prime lo soporta).
PRIME_OPEN_CT    = time(20, 30)  # 20:30 CT = 09:30 HKT — apertura HKEX
PRIME_CLOSE_CT   = time(3, 0)    # 03:00 CT = 16:00 HKT — cierre HKEX
TRADE_WEEKENDS   = False
NO_NEW_SIGNALS_MINS = 20

# ─── Data / Model ─────────────────────────────────────────────────────────────
# ⚙️  Optimized 2026-05-31 (train 70% / test 30% + walk-forward 5 folds)
# Config: 30m TF, EMA 10/50, tp=4.5, RSI 42-60 long / 40-58 short, cd=5
# TRAIN Sharpe 3.09 → OOS Sharpe 0.95; walk-forward 4/5 fuertes, 1 plano → VIGILAR.

TIMEFRAME         = "30m"
N_BARS_LIVE       = 72
MIN_BARS_REQ      = 60
CYCLE_MINUTES     = 30

RETRAIN_TIMEFRAME    = "30m"
N_BARS_TRAIN         = 5000
RETRAIN_HORIZON_BARS = 6

# ─── Strategy parameters (re-tuned 2026-05-31, tune_hk50.py: wide grid + 5-fold WF) ──
# meanSh 4.14 | 5/5 positive folds | fullSh 4.31 | PF 1.76 | P&L $7335 @ $200/trade
# (replaced original tp=4.5/ema10-50/cd5 cfg which scored -0.20, 2/5 folds)
EMA_FAST     = 10
EMA_SLOW     = 30
SL_ATR_MULT  = 1.5
TP_ATR_MULT  = 3.0     # RR = 2.0
RSI_LONG_LO  = 38
RSI_LONG_HI  = 65
RSI_SHORT_LO = 35
RSI_SHORT_HI = 62
COOLDOWN_BARS= 8

# ─── Risk / Sizing ────────────────────────────────────────────────────────────
# HK50: 1 lot ≈ $1/pt P&L (HKD/USD nuance aside, demo). ATR 30m: 80-300 pts.
# Avoid stops < 40 pts (HSI wicks aggressively).

CAPITAL_USD    = 1_013.0
MAX_RISK_PCT   = 0.03   # Tier A
MAX_RISK_USD   = round(CAPITAL_USD * MAX_RISK_PCT, 2)   # ~$30.39

LOT_SIZE_PTS   = 1.0
MIN_LOT        = 0.1
MAX_LOTS       = 2.0
LOT_STEP       = 0.1
DEFAULT_VOLUME = 0.2


def calc_lots(stop_points: float) -> float:
    """Compute trade size so risk ≈ MAX_RISK_USD. stop_points in index points."""
    if stop_points <= 0:
        return MIN_LOT
    raw     = MAX_RISK_USD / stop_points
    rounded = round(round(raw / LOT_STEP) * LOT_STEP, 1)
    return max(MIN_LOT, min(MAX_LOTS, rounded))


# ─── Execution ────────────────────────────────────────────────────────────────

EXECUTE_TRADES         = True    # ✅ demo (vigilado)
MIN_SL_PCT           = 0.3
MAX_LLM_CALLS_PER_DAY = 16   # raised: was hitting limit exactly (10/10)
MAX_ENTRY_SLIP_PCT    = 0.4    # veto si precio adverso > X% del entry LLM
