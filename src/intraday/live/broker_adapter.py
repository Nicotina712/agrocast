"""
src/intraday/live/broker_adapter.py
Adapter abstracto para brokers de futuros (Fase 2 — STUB).

Implementaciones concretas previstas:
  - TradovateAdapter   (REST + WS, retail-friendly)
  - InteractiveBrokersAdapter (ibapi, más establecido)
  - CQGAdapter         (institucional)

Esta clase define el contrato que debe cumplir cualquier broker para que
predict_intraday + signal_router puedan operar sin saber detalles del broker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Order:
    side:       str            # BUY / SELL
    quantity:   int
    order_type: str            # LMT / MKT / STP
    limit_price: Optional[float] = None
    stop_price:  Optional[float] = None
    symbol:     str = "MZS"
    tif:        str = "DAY"    # DAY / GTC


@dataclass
class Position:
    symbol:     str
    quantity:   int            # signed (positive long, negative short)
    avg_price:  float
    unrealized_pnl: float


class BrokerAdapter(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_positions(self) -> list[Position]: ...

    @abstractmethod
    def send_order(self, order: Order) -> str:
        """Returns broker order_id."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> None: ...

    @abstractmethod
    def stream_bars(self, symbol: str, interval: str, on_bar): ...


class StubBroker(BrokerAdapter):
    """Stub para Fase 0/1: loggea pero no manda nada."""
    def connect(self):     print("[StubBroker] connect")
    def disconnect(self):  print("[StubBroker] disconnect")
    def get_positions(self): return []
    def send_order(self, order):
        print(f"[StubBroker] WOULD SEND: {order}")
        return "stub-order-id"
    def cancel_order(self, order_id):
        print(f"[StubBroker] cancel {order_id}")
    def stream_bars(self, symbol, interval, on_bar):
        raise NotImplementedError("Fase 2")
