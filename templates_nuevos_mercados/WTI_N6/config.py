"""
WTI_N6 WTI Crude Oil Robot — Configuration
Cuenta Demo ICMarkets · NYMEX session
"""

import os
from datetime import time

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))

ARTIFACTS_DIR   = os.path.join(_MVP_ROOT, "artifacts", "wti_n6")
SIGNAL_FILE     = os.path.join(ARTIFACTS_DIR, "live_signal.json")
PAPER_LOG_FILE  = os.path.join(ARTIFACTS_DIR, "paper_trades.jsonl")
EXEC_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "executions.jsonl")
LIVE_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "live_log.jsonl")
LIVE_STATE_FILE = os.path.join(ARTIFACTS_DIR, "live_state.json")
MODEL_PATH      = os.path.join(ARTIFACTS_DIR, "model.joblib")

# ─── Instrument ───────────────────────────────────────────────────────────────

# 2026-06-16: WTI_N6 NO existe en el broker (N6 expiro) -> el bridge caia al fallback
# de soja (Sbean_N6) y el robot operaba SOJA. Contrato activo = WTI_Q6 (igual que BRENT_Q6).
SYMBOL           = "WTI_Q6"
FALLBACK_SYMBOLS = ["WTI_Q6", "WTI_N6", "WTIUSD", "USOIL", "WTI", "CrudeOil"]

# ─── Session ──────────────────────────────────────────────────────────────────
# WTI Crude: main liquidity 09:00-14:30 ET = 08:00-13:30 CT
# NYMEX pit session closes 14:30 ET. Post that: electronic only, thinner.

try:
    from zoneinfo import ZoneInfo as _ZI
    from datetime import datetime as _dtz
    CT_OFFSET_HOURS = int(_dtz.now(_ZI("America/Chicago")).utcoffset().total_seconds() / 3600)
except Exception:
    CT_OFFSET_HOURS = -5  # fallback CDT
PRIME_OPEN_CT    = time(6, 0)    # 07:00 ET = 06:00 CT — London/ICE oil open. MT5 vol sube fuerte desde 06:00 CT (era 08:00)
PRIME_CLOSE_CT   = time(16, 0)   # 17:00 ET = 16:00 CT — cierre NYMEX electrónico. MT5 vol activo hasta 16:00 CT (era 13:30)
TRADE_WEEKENDS   = False
NO_NEW_SIGNALS_MINS = 20

# ─── Data / Model ─────────────────────────────────────────────────────────────
# ⚙️  Optimized 2026-05-25 via grid-search (162 combos × 5000 bars)
# Was: 5m TF, EMA 20/50 → WR 26.9%, P&L -$283, Sharpe -2.58
# Now: 15m TF, EMA 20/100 → WR 48.7%, P&L +$457, Sharpe 5.21  ✅

TIMEFRAME         = "15m"
N_BARS_LIVE       = 150        # >=110 requerido por mechanical_signal (ema100) — bug dormido corregido 2026-06-19: con 80 devolvia None SIEMPRE
MIN_BARS_REQ      = 60
CYCLE_MINUTES     = 15         # keep 15 min cycle (aligns to 15m bars)

RETRAIN_TIMEFRAME    = "15m"
N_BARS_TRAIN         = 5000
RETRAIN_HORIZON_BARS = 8       # 2 hours ahead on 15m (was 12 × 5m = 60 min)

# ─── Strategy parameters (backtest-validated) ────────────────────────────────
EMA_FAST     = 20
EMA_SLOW     = 100     # was 50 — slower trend filter reduces noise for oil
SL_ATR_MULT  = 2.0     # raised 1.5→2.0: ATR M15 mediana=0.48pts, 1.5x deja sweep rate 7.0% → 2x baja a 2.9%
TP_ATR_MULT  = 6.0     # RR=3.0 mantenido (TP_ATR = SL_ATR x 3.0)
RSI_LONG_LO  = 45      # tight momentum zone (was 38)
RSI_LONG_HI  = 55
RSI_SHORT_LO = 45
RSI_SHORT_HI = 55
COOLDOWN_BARS        = 8     # raised 3→8: 8×15m=2h cooldown tras cada señal
                              # 3 bars (45min) permitia revenge trading consecutivo
MAX_LOSSES_PER_DAY   = 1     # kill switch: tras 1 SL hit en el dia, parar
H1_TREND_FILTER      = True  # bloquear LONGs si H1 EMA9<EMA21; SHORTs si EMA9>EMA21

# ─── Risk / Sizing ────────────────────────────────────────────────────────────
# WTI_Q6 CFD (ICMarkets): contract_size=100 (1 lote = 100 barriles), vol_min=1.0,
# vol_step=1.0 → SOLO lotes enteros, mínimo 1. Riesgo por 1 lote = stop_usd × 100.
# ⚠️ 2026-06-23 BUG CORREGIDO: calc_lots asumía $1/lote/$1move (contract_size ausente)
#   → sobredimensionaba ×100 → pedía 20 lotes (cap) → todas las órdenes 'No money' (retcode 10019)
#   desde el rolado a Q6 (~16-jun). Ahora usa contract_size real + tope por margen.
# DECISION USUARIO 06-23: operar en demo aunque 1 lote ≈ 4-6% de la cuenta (lote mín
#   indivisible, igual que XNGUSD). Validar a riesgo alto valida también para menor.

CAPITAL_USD    = 1200.0
MAX_RISK_PCT   = 0.015  # presupuesto nominal; el lote mín del contrato puede excederlo (aceptado)
MAX_RISK_USD   = round(CAPITAL_USD * MAX_RISK_PCT, 2)

# Fallbacks si MT5 no responde (specs reales de WTI_Q6 a 06-23)
CONTRACT_SIZE  = 100.0
MIN_LOT        = 1.0
MAX_LOTS       = 20.0   # tope duro; el tope real lo pone el margen libre
LOT_STEP       = 1.0
DEFAULT_VOLUME = 1.0
MARGIN_BUFFER  = 0.80   # usar como máx 80% del margen libre
HARD_CAP_PCT   = 0.10   # techo duro: si ni el lote mín entra en 10% del equity → NO operar


def calc_lots(stop_points: float) -> float:
    """Lotes para riesgo ≈ MAX_RISK_USD. stop_points en USD (distancia de precio).
    Riesgo por lote = stop_points × contract_size. Respeta vol_min/step del broker y
    topa por margen libre. Si el lote mínimo ya excede el presupuesto, devuelve el mínimo
    (decisión: operar igual en demo)."""
    import math
    try:
        from portfolio_guard import live_risk_usd as _lru
        risk_usd = _lru(CAPITAL_USD, MAX_RISK_PCT)
    except Exception:
        risk_usd = round(CAPITAL_USD * MAX_RISK_PCT, 2)
    # specs reales del broker
    cs, vmin, vstep, vmax = CONTRACT_SIZE, MIN_LOT, LOT_STEP, MAX_LOTS
    margin_per_min = None
    try:
        import MetaTrader5 as mt5
        si = mt5.symbol_info(SYMBOL)
        if si:
            cs    = si.trade_contract_size or cs
            vmin  = si.volume_min  or vmin
            vstep = si.volume_step or vstep
            vmax  = min(MAX_LOTS, si.volume_max or MAX_LOTS)
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick:
            margin_per_min = mt5.order_calc_margin(mt5.ORDER_TYPE_SELL, SYMBOL, vmin, tick.ask or tick.bid)
    except Exception:
        pass
    if stop_points <= 0:
        return vmin
    risk_per_lot = stop_points * cs
    raw = risk_usd / risk_per_lot if risk_per_lot > 0 else vmin
    # redondear HACIA ABAJO al step, piso en vmin
    lots = math.floor(raw / vstep) * vstep
    lots = max(vmin, lots)
    # techo duro por trade: si ni el lote mínimo entra en HARD_CAP_PCT del equity → no operar
    try:
        import MetaTrader5 as mt5
        eq = mt5.account_info().equity
        if risk_per_lot * vmin > eq * HARD_CAP_PCT:
            return 0.0
    except Exception:
        pass
    # tope por margen libre (evita 'No money')
    try:
        import MetaTrader5 as mt5
        acc = mt5.account_info()
        if acc and margin_per_min and margin_per_min > 0:
            affordable = math.floor((acc.margin_free * MARGIN_BUFFER) / margin_per_min) * vmin
            if affordable < vmin:
                affordable = 0.0  # ni el mínimo entra → no operar
            lots = min(lots, affordable)
    except Exception:
        pass
    return min(lots, vmax)


# ─── Execution ────────────────────────────────────────────────────────────────

EXECUTE_TRADES         = True    # ✅ demo activado 2026-05-27
MIN_SL_PCT           = 0.8    # raised 0.2→0.8%: 2x ATR M15 mediana @ $95 = 0.95pts = 1.0% → piso real
MAX_LLM_CALLS_PER_DAY = 32   # raised 16→32: sesion 06:00-16:00 CT = 10h = 40 ciclos max
MAX_ENTRY_SLIP_PCT    = 0.35  # veto si precio adverso > X% del entry LLM
ORDER_DEVIATION       = 50   # MT5 max slippage en puntos ($0.01/pt → $0.50 max)
                              # 5 era demasiado ajustado → órdenes rechazadas → fills tardíos a peor precio
NO_TRADE_OPEN_MINS   = 30    # bloquear entradas en los primeros 30min de prime (06:00-06:30 CT)
                              # Análisis ATR: 07:00 CT tiene z>1.5 el 50% del tiempo → apertura ICE London violenta
