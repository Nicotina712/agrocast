"""EVENTBREAK — breakout de evento macro bidireccional (PRODUCCION DEMO, 2026-06-17).
⚠️ Estrategia NO validada (la busqueda dio expR+ pero NO robusta). Se corre en DEMO por decision
del usuario para GENERAR DATO REAL de ejecucion (fills/spread/slippage) ancladio al minuto del evento.
Mecanica: al minuto de cada evento de data/macro_events.json, ancla el precio y arma gatillos
±K_ATR*ATR; el primer cruce dentro de WATCH_MIN dispara orden real (BUY arriba / SELL abajo).
SL/TP por ATR. Un trade por (evento,simbolo).
"""
import os

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))

ARTIFACTS_DIR   = os.path.join(_MVP_ROOT, "artifacts", "eventbreak")
LIVE_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "live_log.jsonl")
LIVE_STATE_FILE = os.path.join(ARTIFACTS_DIR, "live_state.json")
SIGNAL_FILE     = os.path.join(ARTIFACTS_DIR, "live_signal.json")
PAPER_LOG_FILE  = os.path.join(ARTIFACTS_DIR, "paper_trades.jsonl")
EVENTS_FILE     = os.path.join(_MVP_ROOT, "data", "macro_events.json")

# ─── Estrategia (config menos-mala de la busqueda: k1.0 SL2/TP4, mejor expR) ───
K_ATR         = 1.0     # gatillo = ancla ± K_ATR * ATR
SL_ATR_MULT   = 2.0
TP_ATR_MULT   = 4.0     # RR = 2.0
WATCH_MIN     = 90      # minutos tras el evento en los que se vigila el breakout
ATR_TIMEFRAME = "15m"
ATR_BARS      = 100
# ─── AJUSTES aprendidos de los 7 trades reales del FOMC (2026-06-17) ───────────
# Diagnostico: entrabamos en el SPIKE inicial (head-fake/barrido) que revierte; sin time-stop
# (perdedores vivian 5h); y 4-7 posiciones correlacionadas por evento.
ARM_DELAY_MIN   = 15    # esperar 15min tras el evento antes de anclar (deja pasar el barrido inicial)
MAX_HOLD_MIN    = 90    # time-stop: cerrar la posicion si sigue abierta a los 90min (corta sangrados)
MAX_CONCURRENT  = 2     # maximo de posiciones EVENTBREAK simultaneas (evita ráfaga correlacionada)

# ─── Riesgo / ejecucion ───────────────────────────────────────────────────────
CT_OFFSET_HOURS = -5
CAPITAL_USD     = 1200.0
MAX_RISK_PCT    = 0.02      # riesgo alto actual (dinamico via live_risk_usd)
RISK_CAP_PCT    = 0.03      # SKIP el trade si el lote-minimo del broker fuerza riesgo > 3% del balance
                           # (ej: WTI/BRENT vol_min=1.0 = ~13% en cuenta chica -> se saltea; auto-habilita en prop)
EXECUTE_TRADES  = True
OUR_MAGIC       = 20260617
POLL_SECONDS    = 60
