"""
US30 Dow Jones Robot — Configuration
Cuenta Demo ICMarkets · Market hours Mon-Fri (no weekends)
"""

import os
from datetime import time

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))

ARTIFACTS_DIR   = os.path.join(_MVP_ROOT, "artifacts", "us30")
SIGNAL_FILE     = os.path.join(ARTIFACTS_DIR, "live_signal.json")
PAPER_LOG_FILE  = os.path.join(ARTIFACTS_DIR, "paper_trades.jsonl")
EXEC_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "executions.jsonl")
LIVE_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "live_log.jsonl")
LIVE_STATE_FILE = os.path.join(ARTIFACTS_DIR, "live_state.json")
MODEL_PATH      = os.path.join(ARTIFACTS_DIR, "model.joblib")

# ─── Instrument ───────────────────────────────────────────────────────────────

SYMBOL           = "US30"
FALLBACK_SYMBOLS = ["US30", "DJ30", "DJIA", "DowJones", "US30."]

# ─── Session ──────────────────────────────────────────────────────────────────
# US30 market hours: 09:30-16:00 ET = 08:30-15:00 CT
# No weekend trading (equity index)

CT_OFFSET_HOURS     = -5
PRIME_OPEN_CT       = time(8, 30)    # 09:30 ET = 08:30 CT
PRIME_CLOSE_CT      = time(15, 0)    # 16:00 ET = 15:00 CT
TRADE_WEEKENDS      = False
NO_NEW_SIGNALS_MINS = 15   # was 30 — U-shape: closing 30min is highest volume, keep signals active

# ─── Data / Model ─────────────────────────────────────────────────────────────

# 2026-06-16: MIGRADO LLM-puro -> MECANICO-PRIMARIO (VWAP-rev 1h) + LLM-veto.
# El Dow es MEAN-REVERTING en sesion cash. Busqueda+validacion: VWAP-rev 1h th1.0 SL1.5/TP3.0,
# expR_net +0.35, meanPF 2.21, consist 75%, minPF 0.82, placebo edge +0.37 (señal produccion rolling-96).
TIMEFRAME         = "1h"     # era 5m (LLM-puro); el edge mecanico vive en 1h
N_BARS_LIVE       = 150      # >=110 para mechanical_signal + ventana VWAP rolling-96
MIN_BARS_REQ      = 110
CYCLE_MINUTES     = 60       # alineado a la barra 1h

RETRAIN_TIMEFRAME    = "1h"
N_BARS_TRAIN         = 5000
RETRAIN_HORIZON_BARS = 6

# ─── Arquitectura mecanica (VWAP mean-reversion) ──────────────────────────────
SL_ATR_MULT       = 1.5
TP_ATR_MULT       = 3.0      # RR = 2.0
VWAP_TH           = 1.0      # fade solo extensiones > 1.0% del VWAP rolling-96

# ─── Risk / Sizing ────────────────────────────────────────────────────────────
# US30 CFD: 1 lot = 1 point = $1 P&L (standard CFD approximation)
# ATR 5m: typically 30-100 pts (large absolute numbers)
# stop minimum: 30 pts to avoid tight stop outs
# lots = MAX_RISK_USD / stop_points
# E.g. 50pt stop  → lots = 20.26/50  = 0.40 lots
# E.g. 100pt stop → lots = 20.26/100 = 0.20 lots

CAPITAL_USD    = 1200.0
MAX_RISK_PCT   = 0.02
MAX_RISK_USD   = round(CAPITAL_USD * MAX_RISK_PCT, 2)   # ~$20.26

LOT_SIZE_PTS   = 1.0     # 1 full lot × 1 point = $1 P&L
MIN_LOT        = 0.1
MAX_LOTS       = 1.0
LOT_STEP       = 0.1
DEFAULT_VOLUME = 0.3     # Default 0.3 lots


def calc_lots(stop_points: float) -> float:
    """
    Compute trade size so risk ≈ MAX_RISK_USD.
    stop_points: distance in US30 points from entry to stop-loss.
    """
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
MIN_SL_PCT           = 0.35    # minimum SL as % of entry price
MAX_LLM_CALLS_PER_DAY = 36   # raised 20→36: sesion 08:00-16:00 CT = 8h = 32 ciclos max
MAX_ENTRY_SLIP_PCT    = 0.4    # veto si precio adverso > X% del entry LLM
