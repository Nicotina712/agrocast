"""
src/trader/costs.py
Modelo de costos para ZS (Soybeans) y MZS (Mini Soybeans).

Componentes:
  - Spread bid-ask: típico 0.25-0.5 cents en horario líquido (ZS), 1-2 cents en MZS
  - Comisión:       ~$2.50 por contrato ZS por lado (full round-trip $5), $1.20 MZS
  - Slippage:       ejecución vs mid; depende de tamaño y volatilidad

Multiplicadores:
  - Día normal:     1.0x
  - Pre-WASDE:      1.5x (spread se ensancha 30 min antes del reporte)
  - Día WASDE:      2.0x
  - Día Acreage:    2.5x
  - Roll window:    1.3x (5-9 del mes, Goldman roll)

estimate_round_trip_cost_pct devuelve costo estimado como % del precio,
para que el target del modelo lo pueda restar.
"""

# Specs de contrato ZS (full size)
ZS_CONTRACT_SIZE   = 5000      # bushels
ZS_TICK_SIZE_USC   = 0.25      # USc/bu
ZS_TICK_VALUE_USD  = 12.50     # USD por tick
ZS_COMMISSION_USD  = 2.50      # comisión por lado (round-trip = 2x)

# MZS (mini)
MZS_CONTRACT_SIZE  = 500
MZS_TICK_SIZE_USC  = 0.50
MZS_TICK_VALUE_USD = 2.50
MZS_COMMISSION_USD = 1.20

# Spreads típicos en USc/bu
ZS_BASE_SPREAD_USC  = 0.25     # 1 tick mid→bid o mid→ask
MZS_BASE_SPREAD_USC = 0.50

# Slippage base (USc/bu)
BASE_SLIPPAGE_USC = 0.25


def round_trip_cost_usd(
    price_usc: float,
    contract: str = "ZS",
    cost_multiplier: float = 1.0,
) -> dict:
    """
    Costo round-trip de 1 contrato.

    price_usc: precio actual en USc/bu (para reporting)
    contract:  "ZS" o "MZS"
    cost_multiplier: multiplicador del cost model (event_cost_mult del calendario)
    """
    if contract == "MZS":
        size       = MZS_CONTRACT_SIZE
        spread_usc = MZS_BASE_SPREAD_USC
        comm_per_side = MZS_COMMISSION_USD
    else:
        size       = ZS_CONTRACT_SIZE
        spread_usc = ZS_BASE_SPREAD_USC
        comm_per_side = ZS_COMMISSION_USD

    spread_eff = spread_usc * cost_multiplier
    slip_eff   = BASE_SLIPPAGE_USC * cost_multiplier

    spread_usd = spread_eff * size / 100         # USc → USD
    slip_usd   = slip_eff   * size / 100
    comm_usd   = comm_per_side * 2               # round-trip

    total_usd = spread_usd + slip_usd + comm_usd
    total_pct = (total_usd / (price_usc / 100 * size)) if price_usc > 0 else 0.0

    return {
        "contract":         contract,
        "spread_usd":       round(spread_usd, 2),
        "slippage_usd":     round(slip_usd, 2),
        "commission_usd":   round(comm_usd, 2),
        "total_usd":        round(total_usd, 2),
        "total_pct":        round(total_pct, 5),
        "cost_multiplier":  cost_multiplier,
    }


def estimate_round_trip_cost_pct(cost_multiplier: float = 1.0,
                                  contract: str = "ZS") -> float:
    """
    Costo round-trip aproximado como fracción del precio.
    Para precio típico $13/bu (1300 USc): ~0.05% día normal, 0.13% día WASDE.
    Útil para `target_net = ret_7d - cost_pct` en el entrenamiento.
    """
    # precio referencia 1300 USc/bu (~ $13/bu)
    return round_trip_cost_usd(1300.0, contract=contract,
                                cost_multiplier=cost_multiplier)["total_pct"]
