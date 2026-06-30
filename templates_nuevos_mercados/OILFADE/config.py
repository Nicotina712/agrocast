"""OILFADE — Oil Flush-Fade hibrido (mecanica + inteligencia de noticias).
Validado 2026-06-11 (news_shock_hybrid_study + revalidacion sin lookahead):
  fade de shocks 15m de WTI/BRENT SIN confirmacion cross-asset en la hora previa
  (= flush de liquidez sin noticia) -> expR +0.63, WF meanPF 3.60, minPF 1.01,
  consist 100%, N=52 (~3 trades/semana). Shocks CONFIRMADOS (= noticia real)
  NO se fadean (expR +0.09, consist 50%).
Capa de noticias: si data/news_portfolio/<sym>_latest.json esta fresco (<2h) y hay
articulo de magnitud>=3 conf>=0.6 reciente -> VETO adicional (noticia que el precio
cruzado aun no reflejo).
"""
import os

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))

ARTIFACTS_DIR   = os.path.join(_MVP_ROOT, "artifacts", "oilfade")
LIVE_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "live_log.jsonl")
LIVE_STATE_FILE = os.path.join(ARTIFACTS_DIR, "live_state.json")
SIGNAL_FILE     = os.path.join(ARTIFACTS_DIR, "live_signal.json")
PAPER_LOG_FILE  = os.path.join(ARTIFACTS_DIR, "paper_trades.jsonl")
NEWS_DIR        = os.path.join(_MVP_ROOT, "data", "news_portfolio")

OIL_SYMBOLS  = ["WTI_Q6"]   # 2026-06-26: sacado BRENT (BRENT_Q6 contrato muerto + Brent es follower de WTI, su shock es eco del de WTI -> redundante). 06-16: WTI_N6 expiro y resolvia a soja -> WTI_Q6
SENTINELS    = ["US500", "USTEC", "XAUUSD", "BTCUSD", "UK100", "CHINA50"]   # 2026-06-26: sacado ETHUSD (robot retirado; sentinel de menos)

# Deteccion de shock (criterio EXACTO del estudio)
SHOCK_RANGE_MULT   = 4.0    # rango barra > 4x mediana rolling
MEDIAN_WINDOW      = 400    # barras 15m para la mediana
BODY_MIN_FRAC      = 0.5    # cuerpo >= 50% del rango (direccional)
CONFIRM_LOOKBACK_BARS = 4   # 1h: shocks de vigias en las ultimas 4 barras = confirmado -> skip
COOLDOWN_BARS      = 8      # 2h entre trades del mismo simbolo

# Trade
SL_ATR_MULT   = 1.5
TP_ATR_MULT   = 3.0          # RR 2
MAX_RISK_PCT  = 0.02         # 2% del balance real (~$24 sobre $1200); riesgo dinamico
CAPITAL_USD   = 1200.0       # fallback si MT5 no responde
CONTRACT_USD_PER_PT = 100.0  # WTI/BRENT ICMarkets: ~$100 por punto por lote
MIN_LOT  = 0.1
LOT_STEP = 0.1
MAX_LOTS = 2.0
MAX_TRADES_PER_DAY = 3

# News veto
NEWS_MAX_AGE_H   = 2.0
NEWS_MAG_MIN     = 3
NEWS_CONF_MIN    = 0.60

EXECUTE_TRADES = True
OUR_MAGIC      = 20260613
POLL_SECONDS   = 60


def calc_lots(stop_points: float) -> float:
    if stop_points <= 0:
        return MIN_LOT
    try:
        from portfolio_guard import live_risk_usd as _lru
        MAX_RISK_USD = _lru(CAPITAL_USD, MAX_RISK_PCT)
    except Exception:
        MAX_RISK_USD = round(CAPITAL_USD * MAX_RISK_PCT, 2)
    raw = MAX_RISK_USD / (stop_points * CONTRACT_USD_PER_PT)
    lots = max(MIN_LOT, min(MAX_LOTS, round(round(raw / LOT_STEP) * LOT_STEP, 1)))
    return lots
