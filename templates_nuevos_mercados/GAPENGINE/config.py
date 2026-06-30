"""GAPENGINE — robot de gaps overnight en mega-caps (robot #13).
Validado 2026-06-12 (stocks_study.py + sim PEAD walk-forward, N=1.937 eventos, 2.7 anios):
  - GAP UP >=3%  -> LONG al open, hold 5 dias: expR +0.22, WF meanPF 1.59, minPF 1.07, consist 100% (N=1008)
  - GAP DOWN <=-3% -> LONG (fade del panico), hold 5: expR +0.13, minPF 1.08, consist 100% (N=929)
  - SHORT de gaps: REFUTADO (consist 17%) — solo LONG.
Fase 2 (pendiente): veto por debate multi-agente (src/intel) con noticias del ticker.
"""
import os

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))

ARTIFACTS_DIR   = os.path.join(_MVP_ROOT, "artifacts", "gapengine")
LIVE_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "live_log.jsonl")
LIVE_STATE_FILE = os.path.join(ARTIFACTS_DIR, "live_state.json")
SIGNAL_FILE     = os.path.join(ARTIFACTS_DIR, "live_signal.json")
PAPER_LOG_FILE  = os.path.join(ARTIFACTS_DIR, "paper_trades.jsonl")

BASKET = ["AAPL.NAS", "NVDA.NAS", "TSLA.NAS", "MSFT.NAS", "AMZN.NAS",
          "AMD.NAS", "NFLX.NAS", "COIN.NAS", "MSTR.NAS", "GOOG.NAS"]

GAP_MIN_PCT     = 3.0     # |gap| minimo para operar
SL_ATR_MULT     = 1.5     # SL en ATRs DIARIOS
HOLD_TRADING_DAYS = 5     # salida al close del 5to dia habil
MAX_CONCURRENT  = 3       # cap de posiciones simultaneas (earnings season)
MAX_RISK_PCT    = 0.017   # 1.7% del balance real (~$20 sobre $1200); riesgo dinamico
CAPITAL_USD     = 1200.0  # fallback si MT5 no responde
MIN_LOT  = 0.1
LOT_STEP = 0.1
MAX_LOTS = 50.0

# Ventana de operacion (CT): deteccion al open, salidas antes del close
SESSION_OPEN_CT  = (8, 30)
SESSION_CLOSE_CT = (15, 0)
ENTRY_WINDOW_MIN = 20      # solo entrar en los primeros 20 min de sesion
EXIT_AFTER_CT    = (14, 30)  # cerrar holds vencidos desde 14:30 CT

EXECUTE_TRADES = True
OUR_MAGIC      = 20260615
POLL_SECONDS   = 120


def calc_lots(stop_dollars: float) -> float:
    """Acciones CFD: contrato=1, P&L = lots x movimiento$. lots = riesgo/stop$."""
    if stop_dollars <= 0:
        return MIN_LOT
    try:
        from portfolio_guard import live_risk_usd as _lru
        MAX_RISK_USD = _lru(CAPITAL_USD, MAX_RISK_PCT)
    except Exception:
        MAX_RISK_USD = round(CAPITAL_USD * MAX_RISK_PCT, 2)
    raw = MAX_RISK_USD / stop_dollars
    return max(MIN_LOT, min(MAX_LOTS, round(round(raw / LOT_STEP) * LOT_STEP, 1)))
