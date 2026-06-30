"""
USTEC Nasdaq 100 Robot — Configuration
Cuenta Demo ICMarkets · Market hours Mon-Fri
"""

import os
from datetime import time

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))

ARTIFACTS_DIR   = os.path.join(_MVP_ROOT, "artifacts", "ustec")
SIGNAL_FILE     = os.path.join(ARTIFACTS_DIR, "live_signal.json")
PAPER_LOG_FILE  = os.path.join(ARTIFACTS_DIR, "paper_trades.jsonl")
EXEC_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "executions.jsonl")
LIVE_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "live_log.jsonl")
LIVE_STATE_FILE = os.path.join(ARTIFACTS_DIR, "live_state.json")
MODEL_PATH      = os.path.join(ARTIFACTS_DIR, "model.joblib")

# ─── Instrument ───────────────────────────────────────────────────────────────

SYMBOL           = "USTEC"
FALLBACK_SYMBOLS = ["USTEC", "NAS100", "NDX", "NASDAQ", "USTEC."]

# ─── Session ──────────────────────────────────────────────────────────────────
# USTEC (Nasdaq 100) trades Mon-Fri.
# Prime session: 08:30-15:00 CT (09:30-16:00 ET).

try:
    from zoneinfo import ZoneInfo as _ZI
    from datetime import datetime as _dtz
    CT_OFFSET_HOURS = int(_dtz.now(_ZI("America/Chicago")).utcoffset().total_seconds() / 3600)
except Exception:
    CT_OFFSET_HOURS = -5  # fallback CDT
PRIME_OPEN_CT    = time(8, 30)   # 09:30 ET = 08:30 CT
PRIME_CLOSE_CT   = time(15, 0)   # 16:00 ET = 15:00 CT — cierre cash NASDAQ (auditoria sesiones 06-10; igualado a US30)
TRADE_WEEKENDS   = False
NO_NEW_SIGNALS_MINS = 15   # was 30 — U-shape: closing 30min is highest volume, keep signals active

# ─── Data / Model ─────────────────────────────────────────────────────────────

# ⚙️  Optimized 2026-05-29 via grid-search (162 combos × 5000 bars)
# Was: 5m TF, EMA 20/100, RSI 38-65 → Sharpe (prev)
# Now: 30m TF, EMA 20/50, RSI 42-60/40-58 → Sharpe 4.19  ✅
TIMEFRAME         = "30m"
N_BARS_LIVE       = 150   # migrado mech-primary 2026-06-26: >=110 requerido por mechanical_signal (era 72, LLM-only)
MIN_BARS_REQ      = 50
CYCLE_MINUTES     = 30

RETRAIN_TIMEFRAME    = "30m"
N_BARS_TRAIN         = 5000
RETRAIN_HORIZON_BARS = 6

# ─── Optimized signal parameters ─────────────────────────────────────────────
EMA_FAST         = 20
EMA_SLOW         = 50
SL_ATR_MULT      = 2.0     # migrado 2026-06-26 mech-primary: WF mejor config 20/50 SL2.0/TP4.0
TP_ATR_MULT      = 4.0     # RR=2.0 (era 1.5/3.0 en la era LLM-primary)
RSI_LONG_LO      = 42
RSI_LONG_HI      = 60
RSI_SHORT_LO     = 40
RSI_SHORT_HI     = 58
COOLDOWN_BARS    = 5

# ─── Risk / Sizing ────────────────────────────────────────────────────────────
# USTEC: 1 lot = $1/pt P&L (CFD)
# ATR 5m: typically 15-50 pts (high-beta tech index)
# stop minimum: 15 pts (tech wicks aggressively)
# lots = MAX_RISK_USD / stop_points
# E.g. 20pt stop → lots = 20.26/20 = 1.0 lot
# E.g. 40pt stop → lots = 20.26/40 = 0.5 lots

CAPITAL_USD    = 1200.0
MAX_RISK_PCT   = 0.015   # Tier C — Sharpe <1.5 (was 2%)
MAX_RISK_USD   = round(CAPITAL_USD * MAX_RISK_PCT, 2)   # ~$15.20

LOT_SIZE_PTS   = 1.0     # 1 lot × 1 pt = $1 P&L
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

EXECUTE_TRADES         = True    # ✅ demo activado 2026-05-27
MIN_SL_PCT           = 0.4    # was 0.3 — NQ volatility ~33% mayor que SPX; stop mínimo mayor
MAX_LLM_CALLS_PER_DAY = 16   # 30min cycle, prime 08:30-16:00 = 15 ciclos max
VOL_ZSCORE_THRESHOLD  = 2.0  # raised 1.5→2.0: apertura NYSE tiene ATR alto normal, no ruido
MAX_ENTRY_SLIP_PCT    = 0.4    # veto si precio adverso > X% del entry LLM
