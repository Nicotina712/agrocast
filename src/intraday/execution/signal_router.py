"""
src/intraday/execution/signal_router.py
Convierte probabilidad del modelo intradía → orden ejecutable.

Reglas:
  prob_up >= 0.62 AND swing_bias_today >= 0  → BUY
  prob_up <= 0.38 AND swing_bias_today <= 0  → SELL
  else                                       → HOLD

Filtros adicionales:
  - no_trade flag (calendar) → HOLD
  - vol_zscore_30 < 0.0 → HOLD (mercado muy quieto, no operar)
  - swing_age_hours > 36 → ignorar bias swing (modelo standalone)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class RouterConfig:
    p_buy_threshold:  float = 0.62
    p_sell_threshold: float = 0.38
    require_swing_alignment: bool = True
    min_vol_zscore: float = -0.5
    swing_max_age_h: float = 36
    use_limit_orders: bool = True
    limit_offset_ticks: float = 0.25  # mejor que cruzar el spread


@dataclass
class TradeSignal:
    side: str               # "BUY" / "SELL" / "HOLD"
    prob_up: float
    reason: str
    order_type: str = "HOLD"   # "LMT" / "MKT" / "HOLD"
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    n_contracts: int = 0
    timestamp: Optional[datetime] = None


def route_signal(
    prob_up: float,
    bar: dict,
    cfg: Optional[RouterConfig] = None,
) -> TradeSignal:
    """
    bar dict requiere:
      datetime, close, atr_14, vol_zscore_30, no_trade,
      swing_bias_today, swing_age_hours
    """
    cfg = cfg or RouterConfig()

    if bar.get("no_trade", 0) == 1:
        return TradeSignal("HOLD", prob_up, "no_trade window (WASDE / weekend)")

    if bar.get("vol_zscore_30", 0) < cfg.min_vol_zscore:
        return TradeSignal("HOLD", prob_up, f"vol_zscore<{cfg.min_vol_zscore}")

    swing_bias = bar.get("swing_bias_today", 0)
    swing_age  = bar.get("swing_age_hours", 999)
    if swing_age > cfg.swing_max_age_h:
        swing_bias = 0  # ignorar prior

    side = "HOLD"
    reason = ""
    if prob_up >= cfg.p_buy_threshold:
        if cfg.require_swing_alignment and swing_bias < 0:
            reason = f"prob_up={prob_up:.2f} pero swing_bias=SELL → bloqueado"
        else:
            side = "BUY"; reason = f"prob_up={prob_up:.2f} >= {cfg.p_buy_threshold}"
    elif prob_up <= cfg.p_sell_threshold:
        if cfg.require_swing_alignment and swing_bias > 0:
            reason = f"prob_up={prob_up:.2f} pero swing_bias=BUY → bloqueado"
        else:
            side = "SELL"; reason = f"prob_up={prob_up:.2f} <= {cfg.p_sell_threshold}"
    else:
        reason = f"prob_up={prob_up:.2f} en zona neutra"

    if side == "HOLD":
        return TradeSignal("HOLD", prob_up, reason)

    return TradeSignal(
        side=side,
        prob_up=prob_up,
        reason=reason,
        order_type="LMT" if cfg.use_limit_orders else "MKT",
        limit_price=float(bar["close"]),
        timestamp=bar.get("datetime"),
    )
