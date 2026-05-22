# Prompt para iniciar analisis de nuevo mercado

Copia y pega este prompt en un nuevo chat de Claude Code para arrancar:

---

## PROMPT

```
Soy un trader que ya tiene un robot de trading intradiario funcionando con futuros de soja (ZS) en CBOT. El sistema usa:

1. MetaTrader 5 como fuente de datos en tiempo real y ejecucion
2. 36 indicadores tecnicos (RSI, ATR, EMA, VWAP, volatilidad, etc.)
3. Dos agentes Claude (Trend Agent + Risk Agent) que analizan y deciden trades
4. Ejecucion automatica en MT5
5. Retrainer semanal con walk-forward CV
6. Execution tracker con metricas de slippage y PnL

La arquitectura completa esta documentada en:
C:\Users\Lenovo\OneDrive\Escritorio\MVP\templates_nuevos_mercados\LOGICA_ROBOT_TRADING.md

El codigo fuente del sistema de soja esta en:
C:\Users\Lenovo\OneDrive\Escritorio\MVP\src\

Archivos clave a leer para entender la arquitectura:
- src/intraday/data/mt5_bridge.py (conexion MT5)
- src/quantagent/live_runner.py (loop principal)
- src/quantagent/agents.py (agentes IA + prompts)
- src/quantagent/runner.py (pipeline completo)
- src/intraday/features/microstructure.py (36 features)
- src/intraday/model/retrainer.py (aprendizaje continuo)
- src/quantagent/execution_tracker.py (feedback loop)

Quiero replicar este sistema para operar [INSTRUMENTO] en [MERCADO].

Mi broker en MT5 es ICMarkets (cuenta demo con $1,013 USD).

Necesito que:
1. Analices si [INSTRUMENTO] es viable para esta estrategia intradiaria
2. Identifiques las diferencias clave vs soja (horarios, volatilidad, spread, liquidez)
3. Definas que parametros hay que adaptar (timeframe, R:R, sizing, horarios RTH)
4. Identifiques las fuentes de datos fundamentales equivalentes
5. Me des un plan de implementacion paso a paso
6. Crees el codigo adaptado en una nueva carpeta dentro de:
   C:\Users\Lenovo\OneDrive\Escritorio\MVP\templates_nuevos_mercados\[NOMBRE_MERCADO]\

Todo el codigo nuevo va en esa carpeta, NO tocar nada del sistema de soja existente.
```

---

## COMO USARLO

1. Reemplaza `[INSTRUMENTO]` con lo que quieras operar:
   - **Forex**: EURUSD, GBPUSD, USDJPY
   - **Indices**: NAS100 (Nasdaq), US500 (S&P), US30 (Dow)
   - **Commodities**: XAUUSD (Oro), CL (Petroleo), NG (Gas Natural)
   - **Crypto**: BTCUSD, ETHUSD

2. Reemplaza `[MERCADO]` con el exchange/mercado correspondiente

3. Pega el prompt en un nuevo chat de Claude Code

---

## MERCADOS RECOMENDADOS PARA REPLICAR

### Tier 1 — Mas facil de adaptar (similar a soja)
| Mercado | Simbolo MT5 | Por que |
|---------|------------|---------|
| Oro (Gold) | XAUUSD | Alta liquidez, tendencias claras, volatilidad predecible |
| Petroleo (Crude) | CL o OilUS | Commodity como soja, ciclos fundamentales similares |
| Nasdaq 100 | NAS100 | Tendencial, alta liquidez, horarios claros |

### Tier 2 — Requiere mas adaptacion
| Mercado | Simbolo MT5 | Por que |
|---------|------------|---------|
| EUR/USD | EURUSD | Mercado mas liquido del mundo, spread minimo, 24h |
| Bitcoin | BTCUSD | 24/7, alta vol, requiere adaptar horarios completamente |
| S&P 500 | US500 | Similar a Nasdaq pero menos volatil |

### Tier 3 — Avanzado
| Mercado | Simbolo MT5 | Por que |
|---------|------------|---------|
| Gas Natural | NG | Extremadamente volatil, requiere risk management especial |
| GBP/JPY | GBPJPY | "The Beast" — alta volatilidad, alto riesgo |

---

## CHECKLIST ANTES DE EMPEZAR

- [ ] Verificar que el instrumento esta disponible en tu MT5 (ICMarkets)
- [ ] Verificar spread y comisiones del instrumento
- [ ] Tener creditos en Anthropic API (para los agentes Claude)
- [ ] Leer LOGICA_ROBOT_TRADING.md para entender la arquitectura completa
