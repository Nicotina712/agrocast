"""
BRENT Crude Oil Robot — Configuration
Cuenta Demo ICMarkets · London/NYMEX sessions
Contrato activo: Q6 (agosto 2026). M6 (junio) como fallback hasta su vencimiento ~12 jun 2026.
"""

import os
from datetime import time

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))

ARTIFACTS_DIR   = os.path.join(_MVP_ROOT, "artifacts", "brent_n6")
SIGNAL_FILE     = os.path.join(ARTIFACTS_DIR, "live_signal.json")
PAPER_LOG_FILE  = os.path.join(ARTIFACTS_DIR, "paper_trades.jsonl")
EXEC_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "executions.jsonl")
LIVE_LOG_FILE   = os.path.join(ARTIFACTS_DIR, "live_log.jsonl")
LIVE_STATE_FILE = os.path.join(ARTIFACTS_DIR, "live_state.json")
MODEL_PATH      = os.path.join(ARTIFACTS_DIR, "model.joblib")

# ─── Instrument ───────────────────────────────────────────────────────────────

SYMBOL           = "BRENT_U6"   # rolado 2026-06-26: Q6 no existe en broker; M6(jun) expiró -> U6(sep) es el activo
FALLBACK_SYMBOLS = ["BRENT_U6", "BRENT_Q6", "BRENT_M6", "BRTUSD", "UKOIL", "BRENT", "BrentOil"]
# Rollover estimado Q6 → U6: ~agosto 2026

# ─── Session ──────────────────────────────────────────────────────────────────
# Brent crude benchmark.
# London open 08:00 BST = 02:00 CT CDT (best Brent liquidity — was 03:00, missed open).
# NYMEX/ICE close overlap 14:30 ET = 13:30 CT.
try:
    from zoneinfo import ZoneInfo as _ZI
    from datetime import datetime as _dtz
    CT_OFFSET_HOURS = int(_dtz.now(_ZI("America/Chicago")).utcoffset().total_seconds() / 3600)
except Exception:
    CT_OFFSET_HOURS = -5  # fallback CDT
PRIME_OPEN_CT    = time(6, 0)    # 06:00 CT — re-opt 2026-06-06: 04:00 metia ruido pre-market ICE (PF peor). 06:00 alineado con WTI mejora OOS PF a 1.68
PRIME_CLOSE_CT   = time(16, 0)   # 17:00 ET = 16:00 CT — cierre ICE Brent electrónico. MT5 vol activo hasta 16:00 CT (era 13:30)
TRADE_WEEKENDS   = False
NO_NEW_SIGNALS_MINS = 20

# ─── Data / Model ─────────────────────────────────────────────────────────────

TIMEFRAME         = "15m"   # was 5m — aligned to WTI_N6; 5m too granular for crude
N_BARS_LIVE       = 80      # ~20h of 15m bars (was 72×5m = 6h)
MIN_BARS_REQ      = 60
CYCLE_MINUTES     = 15

RETRAIN_TIMEFRAME    = "15m"  # was 5m
N_BARS_TRAIN         = 5000
RETRAIN_HORIZON_BARS = 8      # 2h ahead on 15m (was 12×5m = 60min)

# ─── Risk / Sizing ────────────────────────────────────────────────────────────
# BRENT_Q6 CFD (ICMarkets): contract_size=100 (1 lote = 100 barriles), vol_min=1.0,
# vol_step=1.0 → SOLO lotes enteros, mínimo 1. Riesgo por 1 lote = stop_usd × 100.
# ⚠️ 2026-06-23 BUG CORREGIDO (igual que WTI_N6): calc_lots asumía $1/lote/$1move
#   → sobredimensionaba ×100 → topaba en 20 lotes → 'No money'. Ahora usa contract_size
#   real + tope por margen + techo duro por trade. Ver WTI_N6/config.py.
# DECISION USUARIO 06-23: operar en demo aunque 1 lote ≈ 4-6% (lote mín indivisible).

CAPITAL_USD    = 1200.0
MAX_RISK_PCT   = 0.02   # presupuesto nominal; el lote mín del contrato puede excederlo (aceptado)
MAX_RISK_USD   = round(CAPITAL_USD * MAX_RISK_PCT, 2)

CONTRACT_SIZE  = 100.0
MIN_LOT        = 1.0
MAX_LOTS       = 20.0   # tope duro; el tope real lo pone el margen libre
LOT_STEP       = 1.0
DEFAULT_VOLUME = 1.0
MARGIN_BUFFER  = 0.80   # usar como máx 80% del margen libre
HARD_CAP_PCT   = 0.10   # techo duro: si ni el lote mín entra en 10% del equity → NO operar


def calc_lots(stop_points: float) -> float:
    """Lotes para riesgo ≈ MAX_RISK_USD. stop_points en USD (distancia de precio).
    Riesgo por lote = stop_points × contract_size. Respeta vol_min/step del broker,
    topa por margen libre y por techo duro por trade. Si ni el lote mín entra → 0."""
    import math
    try:
        from portfolio_guard import live_risk_usd as _lru
        risk_usd = _lru(CAPITAL_USD, MAX_RISK_PCT)
    except Exception:
        risk_usd = round(CAPITAL_USD * MAX_RISK_PCT, 2)
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
    lots = math.floor(raw / vstep) * vstep
    lots = max(vmin, lots)
    # techo duro por trade
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
                affordable = 0.0
            lots = min(lots, affordable)
    except Exception:
        pass
    return min(lots, vmax)


# ─── Execution ────────────────────────────────────────────────────────────────

# ─── Strategy parameters (backtest-validated) ────────────────────────────────
# ⚙️  Optimized 2026-05-29 via grid-search (162 combos × 5000 bars)
# Was: 5m EMA=20/50, RSI=38-65, RR=2.0, CD=6 → Sharpe -0.12, P&L -$17
# Now: 5m EMA=20/100, RSI=45-55, RR=3.0, CD=3 → Sharpe 2.50, P&L +$554  ✅
# ⚙️ Re-optimizado 2026-06-06 (IS/OOS validado): BRENT necesita EMA RAPIDA, no lenta como WTI.
# EMA20/100 SL2/TP6 04-16 → PF 0.90, Sharpe -0.67, -$84, MaxDD -$479 (ROTO)
# EMA9/21  SL1.5/TP4.5 06-16 → PF 1.17, Sharpe +1.05, +$153, MaxDD -$182 ✅ (IS PF1.41/OOS PF1.68)
EMA_FAST     = 9       # era 20 — BRENT mean-revierte mas rapido que WTI; EMA lenta era demasiado laggy
EMA_SLOW     = 21      # era 100
# 2026-06-11: SL 1.5→2.5 (mismo RR3). El ruido del Brent barre stops de 1.5×ATR
# sistemáticamente (spike 06-11: short correcto stopeado antes del -6%). Walk-forward:
# SL1.5 meanPF 1.06/consist 62% vs SL2.5 meanPF 1.26/consist 75%, expR +0.05→+0.15.
# Mismo riesgo $ por trade (calc_lots achica el lote al ensanchar el stop).
SL_ATR_MULT  = 2.5
TP_ATR_MULT  = 7.5     # RR=3.0 mantenido (TP=SL×3.0)
RSI_LONG_LO  = 45      # tight momentum zone
RSI_LONG_HI  = 55
RSI_SHORT_LO = 45
RSI_SHORT_HI = 55
COOLDOWN_BARS        = 8     # raised 3→8: 8×15m=2h cooldown tras cada señal
MAX_LOSSES_PER_DAY   = 1     # kill switch: tras 1 SL hit en el dia, parar
H1_TREND_FILTER      = True  # bloquear LONGs si H1 EMA9<EMA21; SHORTs si EMA9>EMA21

EXECUTE_TRADES         = True   # ✅ demo activado 2026-05-25
MIN_SL_PCT           = 0.8    # raised 0.2→0.8%: 2x ATR M15 mediana @ $96 = 0.95pts = 0.99% → piso real
MAX_LLM_CALLS_PER_DAY = 36   # raised 20→36: sesion 04:00-16:00 CT = 12h = 48 ciclos max
MAX_ENTRY_SLIP_PCT    = 0.4   # veto si precio adverso > X% del entry LLM
ORDER_DEVIATION       = 50   # MT5 max slippage en puntos ($0.01/pt → $0.50 max)
NO_TRADE_OPEN_MINS   = 30    # bloquear entradas en los primeros 30min de prime (04:00-04:30 CT)
                              # Apertura London Brent es estructuralmente volátil igual que WTI
