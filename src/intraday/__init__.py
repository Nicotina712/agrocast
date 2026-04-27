"""
src/intraday — módulo paralelo de trading intradía.

Arquitectura: NO modifica src/model/ (swing 14d). Consume signals.csv del swing
como prior bayesiano (daily_bias, expected_vol) y opera sobre microestructura
de barras intradía (1m/5m/15m).

Fase 0 actual: yfinance 5m/15m (ZS=F), sin DOM.
"""
