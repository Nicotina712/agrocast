"""
UK100 FTSE 100 Robot — Configuration
Cuenta Demo ICMarkets · London session
"""

import os
from datetime import time

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))

ARTIFACTS_DIR   = os.path.join(_MVP_ROOT, "artifacts", "uk100")
SIGNAL_FILE     = os.path.join(ARTIFACTS_DIR, "live_signal.json")
PAPER_LOG_FILE  = os.path.join(ARTIFACTS_DIR, "paper_trades.jsonl")
EXEC_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "executions.jsonl")
LIVE_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "live_log.jsonl")
LIVE_STATE_FILE = os.path.join(ARTIFACTS_DIR, "live_state.json")
MODEL_PATH      = os.path.join(ARTIFACTS_DIR, "model.joblib")

# ─── Instrument ───────────────────────────────────────────────────────────────

SYMBOL           = "UK100"
FALLBACK_SYMBOLS = ["UK100", "FTSE100", "UK100.", "FTSE"]

# ─── Session ──────────────────────────────────────────────────────────────────
# UK100 (FTSE 100): London Stock Exchange hours.
# LSE: 08:00-16:30 BST = 02:00-10:30 CT (CDT). DST-aware offset prevents winter drift.
# Close corrected 11:30→10:30 CT: post-LSE-close was thin futures market only.
try:
    from zoneinfo import ZoneInfo as _ZI
    from datetime import datetime as _dtz
    CT_OFFSET_HOURS = int(_dtz.now(_ZI("America/Chicago")).utcoffset().total_seconds() / 3600)
except Exception:
    CT_OFFSET_HOURS = -5  # fallback CDT
PRIME_OPEN_CT    = time(4, 0)    # 10:00 BST = 04:00 CT — LSE liquidity open (was 02:00, pre-open noise)
PRIME_CLOSE_CT   = time(16, 0)   # 22:00 BST = 16:00 CT — cierre NY + post-LSE futures (MT5 vol activo hasta 16:00 CT; era 12:30)
TRADE_WEEKENDS   = False
NO_NEW_SIGNALS_MINS = 30

# ─── Data / Model ─────────────────────────────────────────────────────────────

# ⚙️  Optimized 2026-05-29 via grid-search (162 combos × 5000 bars)
# Was: 5m EMA=20/50, RSI=38-65, RR=2.0 → Sharpe (prev)
# Now: 5m EMA=20/100, RSI=38-65/35-62, RR=2.5, CD=3 → Sharpe 2.0  ✅
# 2026-06-01: SL_ATR_MULT 1.5→2.5 (stop too tight for pre-LSE noise); PRIME_OPEN 02:00→04:00 CT
# 2026-06-09: MIGRADO a VWAP MEAN-REVERSION 30m (era 5m-trend, archetype equivocado).
# FTSE es índice mean-reverting (igual que HK50). Walk-forward 8 folds: VWAP-rev 30m
# th0.8 SL2/TP4 → meanPF 2.88, consist 88%, minPF 0.94, expR +0.66 (253 trades).
# 5m-trend anterior era marginal (meanPF 1.03, expR neg) + overtrading + chase.
TIMEFRAME         = "30m"
N_BARS_LIVE       = 200    # VWAP rolling 96 + warmup → necesita >=110 para mechanical_signal
MIN_BARS_REQ      = 120
CYCLE_MINUTES     = 30

RETRAIN_TIMEFRAME    = "30m"
N_BARS_TRAIN         = 5000
RETRAIN_HORIZON_BARS = 12

# ─── Optimized signal parameters (VWAP mean-reversion) ───────────────────────
EMA_FAST         = 20     # (no usado en vwap_reversion; se mantiene por compat)
EMA_SLOW         = 100
VWAP_TH          = 0.8    # fade solo extensiones >0.8% del VWAP (selectivo, anti-chase)
SL_ATR_MULT      = 2.0    # VWAP-rev validado: SL 2.0 / TP 4.0 (RR 2.0)
TP_ATR_MULT      = 4.0    # RR=2.0
RSI_LONG_LO      = 38
RSI_LONG_HI      = 65
RSI_SHORT_LO     = 35
RSI_SHORT_HI     = 62
COOLDOWN_BARS    = 3

# ─── Risk / Sizing ────────────────────────────────────────────────────────────
# UK100 CFD: 1 lot = $1/pt P&L (approx)
# ATR 5m: 10-40 pts depending on London session volatility
# stop minimum: 15 pts
# lots = MAX_RISK_USD / stop_points
# E.g. 25pt stop → lots = 20.26/25 = 0.81 lots
# E.g. 40pt stop → lots = 20.26/40 = 0.51 lots

CAPITAL_USD    = 1200.0
MAX_RISK_PCT   = 0.015   # Tier C — Sharpe <1.5 (was 2%)
MAX_RISK_USD   = round(CAPITAL_USD * MAX_RISK_PCT, 2)   # ~$15.20

LOT_SIZE_PTS   = 1.0
MIN_LOT        = 0.1
MAX_LOTS       = 2.0
LOT_STEP       = 0.1
DEFAULT_VOLUME = 0.5


def calc_lots(stop_points: float) -> float:
    """Compute trade size so risk ≈ MAX_RISK_USD."""
    # RIESGO DINAMICO 2026-06-13: presupuesto = balance real MT5 x risk_pct (fallback CAPITAL_USD)
    try:
        from portfolio_guard import live_risk_usd as _lru
        MAX_RISK_USD = _lru(CAPITAL_USD, MAX_RISK_PCT)
    except Exception:
        MAX_RISK_USD = round(CAPITAL_USD * MAX_RISK_PCT, 2)
    if stop_points <= 0:
        return MIN_LOT
    raw     = MAX_RISK_USD / stop_points
    rounded = round(round(raw / LOT_STEP) * LOT_STEP, 1)
    return max(MIN_LOT, min(MAX_LOTS, rounded))


# ─── Execution ────────────────────────────────────────────────────────────────

EXECUTE_TRADES         = True   # ✅ demo activado 2026-05-25
MIN_SL_PCT           = 0.25   # raised 0.2→0.25%: 2x ATR M15 mediana @ 10300 = 24.5pts = 0.24% → piso real
MAX_LLM_CALLS_PER_DAY = 36   # raised 20→36: sesion 04:00-16:00 CT = 12h = 48 ciclos max
MAX_ENTRY_SLIP_PCT    = 0.4    # veto si precio adverso > X% del entry LLM


# Cierre pre-finde en positivo (06-29): indice gapea el lunes y el gap atraviesa el SL.
# Backtest: cerrar GANADORES el viernes antes del cierre semanal mejora/de-riesga (no toca perdedores).
WEEKEND_PROFIT_CLOSE = True
WEEKEND_FLAT_CT      = time(17, 0)   # viernes >=17:00 CT (mercado cierra ~18:00 CT); solo si P&L>0
