"""
XAUUSD Gold — Central Configuration
All Gold-specific parameters in one place.

To change the system behavior, edit ONLY this file.
"""

from datetime import time

# ─── MT5 / Symbol ──────────────────────────────────────────────────────────
SYMBOL = "XAUUSD"
FALLBACK_SYMBOLS = ["XAUUSD", "XAUUSDm", "GOLD", "Gold", "XAUUSD."]

# ─── Trading Session ────────────────────────────────────────────────────────
# London-NY overlap + full COMEX floor session for Gold
# CT = Chicago Time (CDT = UTC-5 / CST = UTC-6, auto-detected via zoneinfo)
RTH_OPEN_CT  = time(3, 0)    # Asia/Europe open: 03:00 CT = 08:00 UTC. MT5 vol activo desde 03:00 CT (era 07:00)
RTH_CLOSE_CT = time(15, 0)   # 16:00 ET / 21:00 UTC — COMEX electronic close. MT5 vol cae a 0 después (era 13:00 CT)
try:
    from zoneinfo import ZoneInfo as _ZI
    from datetime import datetime as _dtz
    CT_OFFSET_HOURS = int(_dtz.now(_ZI("America/Chicago")).utcoffset().total_seconds() / 3600)
except Exception:
    CT_OFFSET_HOURS = -5  # fallback CDT

# ─── Bars & Features ────────────────────────────────────────────────────────
# 2026-06-09: MIGRADO a MECH-PRIMARY trend 4h EMA50/200 LONG-ONLY + LLM-veto.
# Walk-forward 8 folds (2022-2026): meanPF 3.13, consist 88%, minPF 0.88, expR +1.38 (N=207).
# Edge SOLO en LONG (SHORT PF 0.74) — estable: mitad antigua PF 1.37, reciente 1.78.
# Si oro entra en bear (EMA50<200) el robot queda FLAT, no shortea (fallo seguro).
TIMEFRAME      = "4h"
N_BARS_LIVE    = 280     # EMA200 + warmup → necesita >=220 barras 4h (~47 dias)
N_BARS_TRAIN   = 5000    # historical bars for retraining
MIN_BARS_REQ   = 230     # minimum bars needed (EMA200 estable)

# ─── Mechanical signal (trend 4h LONG-only) ─────────────────────────────────
EMA_FAST       = 50
EMA_SLOW       = 200
SL_ATR_MULT    = 2.0     # validado: SL 2.0 / TP 6.0 ATR (RR 3)
TP_ATR_MULT    = 6.0
LONG_ONLY      = True    # SHORT mecánico pierde (PF 0.74) — vetar shorts siempre

# ─── Contract Specs (ICMarkets XAUUSD) ─────────────────────────────────────
# 1 standard lot = 100 oz. 0.01 lot = 1 oz.
# P&L: price move of $1 × lots × 100 = USD profit/loss
# Example: 0.01 lot, Gold moves +$10 → P&L = $10 × 0.01 × 100 = $10
LOT_SIZE_OZ    = 100     # oz per standard lot
MIN_LOT        = 0.01    # ICMarkets minimum
LOT_STEP       = 0.01
POINT_DIGITS   = 2       # Gold quoted to 2 decimal places (e.g., 2345.67)

def lot_value_per_point(lots: float) -> float:
    """USD profit per $1 move in Gold at given lot size."""
    return lots * LOT_SIZE_OZ  # e.g., 0.01 lot → $1/point

# ─── Risk Management ────────────────────────────────────────────────────────
CAPITAL_USD    = 1200.0   # demo account balance
MAX_RISK_PCT     = 0.03     # Tier A — Sharpe >2.5 (was 2%)
MAX_RISK_USD     = CAPITAL_USD * MAX_RISK_PCT   # ~$20.26
MIN_RR           = 1.5      # minimum Risk:Reward ratio
MAX_LOTS         = 0.10     # never exceed 0.10 lots (10 oz) — safety cap
VOL_ZSCORE_LIMIT = 2.0      # reduce sizing if vol_zscore > this

# ─── Position sizing helper ─────────────────────────────────────────────────
def calc_lots(stop_points: float) -> float:
    """
    Calculate lot size given stop distance in points.
    risk_usd = stop_points × lots × LOT_SIZE_OZ
    → lots = MAX_RISK_USD / (stop_points × LOT_SIZE_OZ)
    """
    # RIESGO DINAMICO 2026-06-13: presupuesto = balance real MT5 x risk_pct (fallback CAPITAL_USD)
    try:
        from portfolio_guard import live_risk_usd as _lru
        MAX_RISK_USD = _lru(CAPITAL_USD, MAX_RISK_PCT)
    except Exception:
        MAX_RISK_USD = round(CAPITAL_USD * MAX_RISK_PCT, 2)
    if stop_points <= 0:
        return MIN_LOT
    raw = MAX_RISK_USD / (stop_points * LOT_SIZE_OZ)
    # Round down to nearest LOT_STEP
    lots = max(MIN_LOT, min(MAX_LOTS, int(raw / LOT_STEP) * LOT_STEP))
    return round(lots, 2)

# ─── Live Runner ────────────────────────────────────────────────────────────
CYCLE_MINUTES        = 60    # 4h bars → ciclo 60min sobra (señal cambia cada 4h)
MAX_LLM_CALLS_PER_DAY = 36   # raised 20→36: sesion 03:00-15:00 CT = 12h = 48 ciclos max
MAX_ENTRY_SLIP_PCT    = 0.4    # veto si precio adverso > X% del entry LLM
EXECUTE_TRADES       = True    # ✅ demo activado 2026-05-27
MIN_SL_PCT           = 0.50    # Gold ATR típico $15–25; 0.15% era demasiado estrecho
DEFAULT_VOLUME       = 0.01  # default lot size (1 oz)

# ─── Model / Retrainer ──────────────────────────────────────────────────────
RETRAIN_INTERVAL     = "5m"   # bars used for ML retraining
HORIZON_BARS         = 12     # 12 × 5m = 60 min forward prediction
EMBARGO_BARS         = 24
N_SPLITS             = 5
MIN_BARS_RETRAIN     = 2000

# ─── Artifact paths ─────────────────────────────────────────────────────────
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))

ARTIFACTS_DIR        = os.path.join(_MVP_ROOT, "artifacts", "xauusd")
MODEL_PATH           = os.path.join(ARTIFACTS_DIR, "model_xauusd.joblib")
METRICS_PATH         = os.path.join(ARTIFACTS_DIR, "train_metrics.json")
SIGNAL_FILE          = os.path.join(ARTIFACTS_DIR, "live_signal.json")
PAPER_LOG_FILE       = os.path.join(ARTIFACTS_DIR, "paper_trades.jsonl")
EXEC_LOG_FILE        = os.path.join(ARTIFACTS_DIR, "execution_log.jsonl")
PERF_FILE            = os.path.join(ARTIFACTS_DIR, "performance.json")
LIVE_STATE_FILE      = os.path.join(ARTIFACTS_DIR, "live_state.json")
LIVE_LOG_FILE        = os.path.join(ARTIFACTS_DIR, "live_log.jsonl")
RETRAIN_LOG_FILE     = os.path.join(ARTIFACTS_DIR, "retrain_log.jsonl")
DRIFT_FILE           = os.path.join(ARTIFACTS_DIR, "drift_report.json")
