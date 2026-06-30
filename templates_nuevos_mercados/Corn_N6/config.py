"""
Corn_N6 CBOT Corn Robot — Configuration
Cuenta Demo ICMarkets · CBOT grain session
Added 2026-05-31 — OOS-validated (walk-forward 5/5 folds positive, Sharpe TR 1.96 / OOS 3.16, PF 1.60)
"""

import os
from datetime import time

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))

ARTIFACTS_DIR   = os.path.join(_MVP_ROOT, "artifacts", "corn_n6")
SIGNAL_FILE     = os.path.join(ARTIFACTS_DIR, "live_signal.json")
PAPER_LOG_FILE  = os.path.join(ARTIFACTS_DIR, "paper_trades.jsonl")
EXEC_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "executions.jsonl")
LIVE_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "live_log.jsonl")
LIVE_STATE_FILE = os.path.join(ARTIFACTS_DIR, "live_state.json")
MODEL_PATH      = os.path.join(ARTIFACTS_DIR, "model.joblib")

# ─── Instrument ───────────────────────────────────────────────────────────────

SYMBOL           = "Corn_U6"   # rolado 2026-06-23: N6 (Jul) vencio -> U6 (Sep)
FALLBACK_SYMBOLS = ["Corn_U6", "CORN_U6", "Corn_N6", "ZC", "CORN", "XCBT_CORN"]

# ─── Session ──────────────────────────────────────────────────────────────────
# CBOT Corn day session: 08:30-13:20 CT (09:30-14:20 ET) — main grain liquidity.
# Overnight Globex thinner; trade the day session only.

try:
    from zoneinfo import ZoneInfo as _ZI
    from datetime import datetime as _dtz
    CT_OFFSET_HOURS = int(_dtz.now(_ZI("America/Chicago")).utcoffset().total_seconds() / 3600)
except Exception:
    CT_OFFSET_HOURS = -5  # fallback CDT
PRIME_OPEN_CT    = time(8, 30)    # 08:30 CT — CBOT grain open
PRIME_CLOSE_CT   = time(13, 20)   # 13:20 CT — CBOT grain close
TRADE_WEEKENDS   = False
NO_NEW_SIGNALS_MINS = 20

# ─── Data / Model ─────────────────────────────────────────────────────────────
# ⚙️  Optimized 2026-05-31 (train 70% / test 30% + walk-forward 5 folds)
# Config: 15m TF, EMA 20/50, tp=4.5, RSI 42-60 long / 40-58 short, cd=5
# TRAIN Sharpe 1.96 → OOS Sharpe 3.16, PF 1.60 — 5/5 walk-forward folds positive ✅

# 2026-06-22: migrado 15m -> 4h. Validacion TF neto de costos: 15m = -0.87R (PIERDE, el spread 0.20%
# mata el edge intradia); 4h = +0.05R/PF1.16 (unico TF positivo). Granos CFD solo sobreviven en TF alto.
TIMEFRAME         = "4h"
N_BARS_LIVE       = 150   # >=110 requerido por mechanical_signal
MIN_BARS_REQ      = 60
CYCLE_MINUTES     = 240

RETRAIN_TIMEFRAME    = "4h"
N_BARS_TRAIN         = 5000
RETRAIN_HORIZON_BARS = 8

# ─── Strategy parameters (OOS-validated) ─────────────────────────────────────
EMA_FAST     = 20
EMA_SLOW     = 50
SL_ATR_MULT  = 1.5
TP_ATR_MULT  = 4.5     # RR = 3.0
RSI_LONG_LO  = 42
RSI_LONG_HI  = 60
RSI_SHORT_LO = 40
RSI_SHORT_HI = 58
COOLDOWN_BARS= 5

# ─── Risk / Sizing ────────────────────────────────────────────────────────────
# Corn_N6 CFD (ICMarkets): priced in cents/bushel. ATR 15m typically 2-8 cents.
# lots = MAX_RISK_USD / stop_usd
# Avoid stops < 1.5 cents (grains wick on weather/USDA headlines).

CAPITAL_USD    = 1200.0
MAX_RISK_PCT   = 0.03   # Tier A
MAX_RISK_USD   = round(CAPITAL_USD * MAX_RISK_PCT, 2)   # ~$30.39

LOT_SIZE_PTS   = 1.0
MIN_LOT        = 0.1
MAX_LOTS       = 20.0
LOT_STEP       = 0.1
DEFAULT_VOLUME = 0.5


def calc_lots(stop_points: float) -> float:
    """Compute trade size so risk ≈ MAX_RISK_USD. stop_points in price units."""
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

EXECUTE_TRADES         = True    # ✅ demo
MIN_SL_PCT           = 0.2
MAX_LLM_CALLS_PER_DAY = 16   # raised: gate firing at 11h mid prime session
MAX_ENTRY_SLIP_PCT    = 0.4    # veto si precio adverso > X% del entry LLM
