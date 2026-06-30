"""
US500 S&P 500 Robot — Configuration
Cuenta Demo ICMarkets · Market hours Mon-Fri
"""

import os
from datetime import time

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))

ARTIFACTS_DIR   = os.path.join(_MVP_ROOT, "artifacts", "us500")
SIGNAL_FILE     = os.path.join(ARTIFACTS_DIR, "live_signal.json")
PAPER_LOG_FILE  = os.path.join(ARTIFACTS_DIR, "paper_trades.jsonl")
EXEC_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "executions.jsonl")
LIVE_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "live_log.jsonl")
LIVE_STATE_FILE = os.path.join(ARTIFACTS_DIR, "live_state.json")
MODEL_PATH      = os.path.join(ARTIFACTS_DIR, "model.joblib")

# ─── Instrument ───────────────────────────────────────────────────────────────

SYMBOL           = "US500"
FALLBACK_SYMBOLS = ["US500", "US500.", "SP500", "SPX500", "S&P500"]

# ─── Session ──────────────────────────────────────────────────────────────────
# US500 (S&P 500) trades Mon-Fri.
# Prime session: 08:30-15:00 CT (09:30-16:00 ET).

CT_OFFSET_HOURS  = -5
PRIME_OPEN_CT    = time(8, 30)    # 09:30 ET = 08:30 CT
PRIME_CLOSE_CT   = time(15, 0)   # 16:00 ET = 15:00 CT — cierre cash NYSE (auditoria sesiones 06-10; 15-16 CT era post-cierre, igualado a US30)
TRADE_WEEKENDS   = False
NO_NEW_SIGNALS_MINS = 15   # was 30 — U-shape: closing 30min is highest volume, keep signals active

# ─── Data / Model ─────────────────────────────────────────────────────────────
# ⚙️  Optimized 2026-05-29 via grid-search (162 combos × 5000 bars)
# Was: 5m TF, EMA 20/50, RSI loose → Sharpe -1.92, P&L -$142
# Now: 30m TF, EMA 10/50, RSI tight (45-55) → Sharpe 6.45, P&L +$519  ✅

TIMEFRAME         = "1h"    # migrado 2026-06-26 mech-primary VWAP-rev (S&P mean-revierte en 1h, como US30)
N_BARS_LIVE       = 150     # >=110 requerido por mechanical_signal + ventana VWAP rolling-96
MIN_BARS_REQ      = 60
CYCLE_MINUTES     = 60      # alineado a la barra 1h

RETRAIN_TIMEFRAME    = "30m"
N_BARS_TRAIN         = 5000
RETRAIN_HORIZON_BARS = 6    # 3h ahead on 30m (was 12×5m = 60min)

# ─── Strategy parameters (backtest-validated) ────────────────────────────────
EMA_FAST     = 10      # (no usado en VWAP-rev; se conserva para retrain/legacy)
EMA_SLOW     = 50
SL_ATR_MULT  = 2.0     # migrado 2026-06-26 VWAP-rev 1h: WF th0.3 SL2.0/TP4.0 (meanPF 2.05, minPF 0.89)
TP_ATR_MULT  = 4.0     # RR = 2.0 (era 1.5/3.0)
VWAP_TH      = 0.3     # umbral de extensión VWAP para el fade (validado WF)
RSI_LONG_LO  = 45      # tight momentum zone (was 38) — reduces false signals
RSI_LONG_HI  = 55
RSI_SHORT_LO = 45
RSI_SHORT_HI = 55
COOLDOWN_BARS= 5       # bars between trades (2.5h buffer on 30m)

# ─── Risk / Sizing ────────────────────────────────────────────────────────────
# US500: 1 lot = $1/pt P&L
# Typical 5m ATR: 5-20 pts depending on session
# Avoid stops < 8 pts (index wicks aggressively)
# lots = MAX_RISK_USD / stop_points
# E.g. 15pt stop → lots = 20.26/15 = 1.35 lots

CAPITAL_USD    = 1200.0
MAX_RISK_PCT   = 0.03   # Tier A — Sharpe >2.5 (was 2%)
MAX_RISK_USD   = round(CAPITAL_USD * MAX_RISK_PCT, 2)   # ~$30.39

LOT_SIZE_PTS   = 1.0     # 1 lot × 1 pt = $1 P&L
MIN_LOT        = 0.1
MAX_LOTS       = 2.0    # was 5.0 — reduced: con stop 5pts, 5.0 lots = $25 riesgo > $20 presupuesto
LOT_STEP       = 0.1
DEFAULT_VOLUME = 1.0


def calc_lots(stop_points: float) -> float:
    """
    Compute trade size so risk ≈ MAX_RISK_USD.
    stop_points: distance in index points from entry to stop-loss.
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
MIN_SL_PCT           = 0.3    # minimum SL as % of entry price
MAX_LLM_CALLS_PER_DAY = 24   # 30min cycle, prime 08:30-16:00 = 15 ciclos max, margen extra
VOL_ZSCORE_THRESHOLD  = 2.0  # raised 1.5→2.0: apertura NYSE tiene ATR alto normal, no ruido
MAX_ENTRY_SLIP_PCT    = 0.4    # veto si precio adverso > X% del entry LLM
