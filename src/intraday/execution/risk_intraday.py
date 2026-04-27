"""
src/intraday/execution/risk_intraday.py
Risk management intradía.

Reglas (configurables):
  RISK_PER_TRADE     = 0.01   (1% capital por trade en SL completo)
  DAILY_DD_STOP      = 0.02   (2% drawdown diario → freeze 24h)
  MAX_TRADES_DAY     = 6
  MAX_LOSSES_STREAK  = 3      (3 SLs consecutivos → freeze 4h)
  SL_ATR_MULT        = 1.5
  TP_ATR_MULT        = 2.5

Sizing:
  n_contracts = floor( (capital × risk_per_trade) / (sl_distance × point_value) )
  Cap a 1 si capital < $10k para MZS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from src.intraday.execution.slippage_model import ContractSpec, MZS


@dataclass
class RiskConfig:
    risk_per_trade:    float = 0.01
    daily_dd_stop:     float = 0.02
    max_trades_day:    int   = 6
    max_losses_streak: int   = 3
    sl_atr_mult:       float = 1.5
    tp_atr_mult:       float = 2.5
    freeze_after_streak_h: int = 4
    freeze_after_dd_h:     int = 24
    min_contracts:     int   = 1
    max_contracts:     int   = 5
    cap_for_one_contract: float = 10_000  # debajo de esto, max 1 contrato


@dataclass
class RiskState:
    capital:               float
    trades_today:          int = 0
    pnl_today:             float = 0.0
    losses_streak:         int = 0
    frozen_until:          Optional[datetime] = None
    closed_trades:         list = field(default_factory=list)
    starting_capital_today: float = 0.0
    last_session_date:     Optional[str] = None


def reset_daily_if_needed(state: RiskState, now: datetime) -> None:
    today = now.date().isoformat()
    if state.last_session_date != today:
        state.starting_capital_today = state.capital
        state.trades_today = 0
        state.pnl_today = 0.0
        state.losses_streak = 0   # streak es un concepto intra-día
        state.last_session_date = today


def is_frozen(state: RiskState, now: datetime) -> bool:
    return state.frozen_until is not None and now < state.frozen_until


def compute_position_size(
    capital: float,
    sl_distance_price: float,
    contract: ContractSpec = MZS,
    cfg: Optional[RiskConfig] = None,
) -> int:
    """
    Sizing basado en riesgo fijo por trade.
      sl_distance_price: |entry - SL| en unidades de precio (no ticks)
    """
    cfg = cfg or RiskConfig()
    import math
    if (sl_distance_price is None
            or (isinstance(sl_distance_price, float) and math.isnan(sl_distance_price))
            or sl_distance_price <= 0):
        return 0
    risk_dollars = capital * cfg.risk_per_trade
    risk_per_contract = sl_distance_price * contract.point_value_usd
    if risk_per_contract <= 0 or math.isnan(risk_per_contract):
        return 0
    n = int(risk_dollars // risk_per_contract)
    if capital < cfg.cap_for_one_contract:
        n = min(n, 1)
    n = max(n, cfg.min_contracts) if risk_per_contract <= risk_dollars else 0
    n = min(n, cfg.max_contracts)
    return max(0, n)


def compute_sl_tp(
    entry: float,
    atr: float,
    side: str,
    cfg: Optional[RiskConfig] = None,
    expected_vol_swing: Optional[float] = None,
) -> tuple[float, float]:
    """
    Retorna (stop_loss, take_profit). Si expected_vol_swing del swing es alto
    (>20%), ensancha stops 30% para no ser barrido por noise macro.
    """
    cfg = cfg or RiskConfig()
    sl_mult = cfg.sl_atr_mult
    tp_mult = cfg.tp_atr_mult
    if expected_vol_swing and expected_vol_swing > 0.20:
        sl_mult *= 1.3
        tp_mult *= 1.3

    if side.upper() == "BUY":
        return (entry - sl_mult * atr, entry + tp_mult * atr)
    elif side.upper() == "SELL":
        return (entry + sl_mult * atr, entry - tp_mult * atr)
    raise ValueError(f"side inválido: {side}")


def can_open_trade(state: RiskState, now: datetime, cfg: Optional[RiskConfig] = None) -> tuple[bool, str]:
    cfg = cfg or RiskConfig()
    reset_daily_if_needed(state, now)
    if is_frozen(state, now):
        return (False, f"frozen until {state.frozen_until}")
    if state.trades_today >= cfg.max_trades_day:
        return (False, f"max trades del día alcanzado ({cfg.max_trades_day})")
    dd = (state.pnl_today / state.starting_capital_today) if state.starting_capital_today else 0
    if dd <= -cfg.daily_dd_stop:
        state.frozen_until = now + timedelta(hours=cfg.freeze_after_dd_h)
        return (False, f"daily DD stop alcanzado ({dd:.2%}); freeze 24h")
    if state.losses_streak >= cfg.max_losses_streak:
        state.frozen_until = now + timedelta(hours=cfg.freeze_after_streak_h)
        return (False, f"{cfg.max_losses_streak} losses seguidos; freeze 4h")
    return (True, "OK")


def register_trade_outcome(state: RiskState, pnl: float, closed_at: datetime) -> None:
    state.capital += pnl
    state.pnl_today += pnl
    state.trades_today += 1
    state.closed_trades.append({"pnl": pnl, "closed_at": closed_at})
    if pnl < 0:
        state.losses_streak += 1
    else:
        state.losses_streak = 0
