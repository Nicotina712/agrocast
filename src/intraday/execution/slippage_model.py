"""
src/intraday/execution/slippage_model.py
Modelo de costos para backtest realista de futuros agrícolas.

CONVENCIÓN DE UNIDADES (importante):
  Usamos la convención de yfinance/CME display: precios en CENTS por bushel.
  Ej: ZS=F a 1178.5 → 1178.5 USc/bu = $11.785/bu.

Especs MZS (Micro Soybean):
  contract_size       = 500 bushels
  tick_size (display) = 0.50 cents/bu  ("medio centavo")
  tick_value          = 0.50 × 500 × $0.01 = $2.50/tick
  point_value         = $5 por cada 1.0 de movimiento de precio (1 cent/bu)
  spread típico       = 1.5 ticks ≈ 0.75 cents

Especs ZS (Standard Soybean):
  contract_size       = 5000 bushels
  tick_size (display) = 0.25 cents/bu  ("cuarto de centavo")
  tick_value          = 0.25 × 5000 × $0.01 = $12.50/tick
  point_value         = $50 por cada 1.0 de movimiento de precio
  spread típico       = 1 tick ≈ 0.25 cents

Comisiones (Tradovate retail, Apr 2026):
  MZS: 0.74 USD/lado → 1.48 USD/round-trip
  ZS:  1.99 USD/lado → 3.98 USD/round-trip
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContractSpec:
    symbol:        str
    contract_size: int
    tick_size:     float
    tick_value:    float
    typical_spread_ticks: float
    commission_round_trip_usd: float
    point_value_usd: float   # = tick_value / tick_size


MZS = ContractSpec(
    symbol="MZS", contract_size=500,
    tick_size=0.50, tick_value=2.50,        # display units: cents/bu
    typical_spread_ticks=1.5,
    commission_round_trip_usd=1.48,
    point_value_usd=5.0,                    # USD por 1.0 cent/bu
)

ZS = ContractSpec(
    symbol="ZS", contract_size=5000,
    tick_size=0.25, tick_value=12.50,       # display units: cents/bu
    typical_spread_ticks=1.0,
    commission_round_trip_usd=3.98,
    point_value_usd=50.0,                   # USD por 1.0 cent/bu
)


def compute_round_trip_cost(
    contract: ContractSpec,
    n_contracts: int = 1,
    extra_slippage_ticks: float = 1.0,
) -> dict:
    """
    Costo total round-trip (entry + exit) en USD.

    Componentes:
      spread_cost     = (spread_ticks/2) · tick_value · 2 sides
      slippage_cost   = extra_slippage_ticks · tick_value · 2 sides (adverso)
      commission_cost = commission_round_trip · n_contracts
    """
    spread = contract.typical_spread_ticks * contract.tick_value * n_contracts
    slip   = extra_slippage_ticks * contract.tick_value * 2 * n_contracts
    comm   = contract.commission_round_trip_usd * n_contracts
    total  = spread + slip + comm
    return {
        "spread_usd":     round(spread, 2),
        "slippage_usd":   round(slip, 2),
        "commission_usd": round(comm, 2),
        "total_usd":      round(total, 2),
        "total_ticks":    round(total / contract.tick_value, 2),
    }


def fill_price(
    side: str,
    quoted_price: float,
    contract: ContractSpec,
    extra_slippage_ticks: float = 1.0,
) -> float:
    """
    Precio de fill considerando spread cruzado + slippage adverso.

    side: "BUY"  → paga ask + slip
          "SELL" → recibe bid - slip
    quoted_price: precio mid o último trade
    """
    half_spread_ticks = contract.typical_spread_ticks / 2
    adverse = (half_spread_ticks + extra_slippage_ticks) * contract.tick_size
    if side.upper() == "BUY":
        return round(quoted_price + adverse, 4)
    elif side.upper() == "SELL":
        return round(quoted_price - adverse, 4)
    raise ValueError(f"side inválido: {side}")


def pnl_usd(
    entry_price: float,
    exit_price: float,
    side: str,
    contract: ContractSpec,
    n_contracts: int = 1,
) -> float:
    """PnL en USD para una operación cerrada (sin comisiones aún)."""
    delta = (exit_price - entry_price) if side.upper() == "BUY" else (entry_price - exit_price)
    return round(delta * contract.point_value_usd * n_contracts, 2)


if __name__ == "__main__":
    print("MZS round-trip 1 contrato:", compute_round_trip_cost(MZS, 1))
    print("ZS round-trip 1 contrato: ", compute_round_trip_cost(ZS, 1))
    p_in  = fill_price("BUY",  1178.50, MZS)
    p_out = fill_price("SELL", 1180.00, MZS)
    print(f"MZS BUY @ {p_in}, SELL @ {p_out} → "
          f"PnL gross = ${pnl_usd(p_in, p_out, 'BUY', MZS)}")
