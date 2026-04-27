"""
src/intraday/data/bar_builder.py
Tick → Bar aggregation (Fase 2 — STUB).

En Fase 0/1 no usamos: yfinance ya entrega barras agregadas.
Necesario en Fase 2 cuando recibimos tick stream del broker en vivo.
"""

from __future__ import annotations


def aggregate_ticks_to_bars(ticks, interval_seconds: int = 300):
    raise NotImplementedError("Fase 2 — usar tick stream del broker live")
