# Robot de Trading Intradiario con IA — Documento de Arquitectura

## Para replicar en cualquier mercado financiero

---

## 1. QUE ES ESTO

Un sistema automatizado que:
1. Lee precios en tiempo real desde MetaTrader 5
2. Calcula 36 indicadores tecnicos sobre las barras
3. Dos agentes de IA (Claude) analizan el mercado y deciden si comprar, vender, o no hacer nada
4. Ejecuta la orden automaticamente en MT5
5. Aprende de sus propios errores cada semana

**Resultado**: un trader que opera 24/5, no tiene emociones, y mejora con el tiempo.

---

## 2. ARQUITECTURA (4 Capas)

```
CAPA 1: DATOS EN TIEMPO REAL
    MetaTrader 5 Terminal (local)
         |
    MT5 Python Bridge (mt5_bridge.py)
         |
    Barras OHLCV cada N minutos
         |
CAPA 2: INTELIGENCIA
    Feature Engine (36 indicadores)
         |
    +--- Trend Agent (Claude): "Para donde va el precio?"
    +--- Risk Agent (Claude):  "Cuanto arriesgo? Donde pongo stop?"
         |
    Senal: LONG / SHORT / FLAT + entry, SL, TP, contratos
         |
CAPA 3: EJECUCION
    MT5 place_order() --> orden real en el broker
         |
    Execution Tracker: mide slippage, latencia, PnL real
         |
CAPA 4: APRENDIZAJE
    Paper Log: registra cada trade y su resultado
         |
    Retrainer Semanal: re-entrena modelo XGBoost con datos nuevos
         |
    Drift Detection: alerta si el modelo deja de funcionar
```

---

## 3. COMPONENTES DETALLADOS

### 3.1 MT5 Bridge (`mt5_bridge.py`)
**Que hace**: Conecta Python con el terminal MetaTrader 5 que corre en tu PC.

**Funciones clave**:
- `fetch_mt5_bars(interval, n_bars)` — Trae barras OHLCV historicas
- `get_live_tick()` — Precio actual (bid/ask) en tiempo real
- `place_order(direction, volume, sl, tp)` — Coloca ordenes de mercado
- `get_positions()` — Posiciones abiertas
- `get_account_info()` — Balance, equity, margin

**Para adaptar a otro mercado**:
- Cambiar `DEFAULT_SYMBOL` al simbolo de tu instrumento en MT5
- Ajustar `FALLBACK_SYMBOLS` con nombres alternativos del broker
- Verificar `volume_min`, `volume_step` del nuevo simbolo
- Detectar `filling_mode` (IOC/FOK) — ya lo hace automatico

### 3.2 Feature Engine (`microstructure.py`)
**Que hace**: Transforma barras OHLCV en 36 indicadores que los agentes IA pueden interpretar.

**Indicadores incluidos**:

| Categoria | Indicadores | Para que sirve |
|-----------|------------|----------------|
| Momentum | RSI(14), EMA(9/26), EMA cross, momentum_k | Direccion y fuerza del movimiento |
| Volatilidad | ATR(14), realized vol 30, range_pct, vol z-score | Regimen de volatilidad |
| Retornos | ret_1, ret_3, ret_12 (log-returns) | Velocidad del movimiento |
| Flujo/Presion | body_pct, upper/lower wick, cum_delta_proxy | Presion compradora/vendedora |
| VWAP | vwap_session, vwap_dist | Precio justo intrasesion |
| Tiempo | hour_sin/cos, minute_of_session, mins_to_close | Estacionalidad intradiaria |

**Para adaptar a otro mercado**:
- Los indicadores son universales, no requieren cambio
- Ajustar horarios de sesion RTH (`_RTH_OPEN_CT`, `_RTH_CLOSE_CT`) segun el mercado
- Ajustar timezone si no es CT (Chicago Time)
- Considerar agregar indicadores especificos del mercado (ej: funding rate para crypto)

### 3.3 Agentes de IA (`agents.py`)
**Que hace**: Dos agentes Claude especializados que "razonan" sobre los datos.

**Trend Agent** — Analiza:
- Tendencia: alcista, bajista, lateral
- Momentum: acelerando o desacelerando
- Estructura: soportes, resistencias, compresion/expansion
- Setup: breakout, pullback, reversal, range, none
- Timing: momento de la sesion
- Contexto fundamental: alineado, neutral, conflictivo

**Risk Agent** — Evalua:
- Volatilidad: regimen actual vs historico
- R:R minimo 1.5:1 para tomar el trade
- Stop loss basado en estructura (no porcentaje fijo)
- Position sizing basado en ATR y capital
- Tiempo restante de sesion
- Max perdida por trade: 2% del capital

**Para adaptar a otro mercado**:
- Reescribir los system prompts con el contexto del nuevo mercado
- Ajustar el contrato (tick size, tick value, contract size)
- Ajustar capital de referencia
- Ajustar horarios de sesion
- Agregar/quitar reglas de riesgo segun el mercado

### 3.4 Live Runner (`live_runner.py`)
**Que hace**: El loop principal que corre cada N minutos.

**Ciclo de vida**:
```
Cada 15 min durante RTH:
  1. Conectar a MT5
  2. Traer 500 barras 60min
  3. Calcular 36 features
  4. Evaluar paper trades pendientes
  5. Llamar Trend Agent + Risk Agent (2 LLM calls)
  6. Sintetizar senal (LONG/SHORT/FLAT)
  7. Si --execute: colocar orden en MT5
  8. Loguear todo
  9. Dormir hasta proximo ciclo

Fuera de RTH:
  - Modo monitor (no gasta LLM calls)
  - Ciclo cada 30 min
  - Solo registra precio y posiciones
```

**Controles de costo**:
- Max 6 LLM calls/dia (~$0.72/dia)
- No opera ultimos 30 min de sesion
- No opera fines de semana

**Para adaptar a otro mercado**:
- Cambiar `RTH_START_CT` y `RTH_END_CT` a los horarios del nuevo mercado
- Cambiar `CT_OFFSET_HOURS` si el mercado opera en otra zona horaria
- Ajustar `CYCLE_MINUTES` segun volatilidad del mercado (crypto = 5min, forex = 15min, acciones = 30min)
- Ajustar `MAX_LLM_CALLS_PER_DAY` segun presupuesto

### 3.5 Retrainer (`retrainer.py`)
**Que hace**: Cada semana re-entrena el modelo XGBoost con datos nuevos.

**Innovaciones**:
- **Walk-forward CV**: entrena con pasado, valida con futuro (no random split)
- **Embargo**: gap entre train/test para evitar data leakage
- **Paper trade feedback**: trades erroneos pesan 1.5x en el re-entrenamiento
- **Drift detection**: si AUC cae >5% o modelo tiene >7 dias, dispara re-entrenamiento
- **Backup automatico**: guarda modelo anterior antes de sobreescribir

**Para adaptar a otro mercado**:
- `HORIZON_BARS`: horizonte de prediccion (depende del timeframe)
- `EMBARGO_BARS`: 2x horizon (regla general)
- `MIN_BARS_RETRAIN`: minimo de barras para entrenar (2000+ recomendado)

### 3.6 Execution Tracker (`execution_tracker.py`)
**Que hace**: Compara trades esperados vs trades reales.

**Metricas**:
- Slippage: diferencia entre precio senalado vs precio ejecutado
- Latencia: tiempo entre senal y ejecucion
- Paper vs Real PnL: comparacion sistematica
- Fill rate: que % de senales se ejecutaron
- Dashboard JSON para visualizacion

---

## 4. CONTEXTO FUNDAMENTAL (Opcional pero potente)

Ademas de precio, el sistema puede incorporar datos fundamentales como CONTEXTO (no como veto):

**En soja usamos**:
- WASDE (USDA supply/demand)
- COT positioning (Commitment of Traders)
- China crush margin / demanda
- Implied volatility (CVOL)
- Daily swing model (SMA cross)
- Active shocks

**Para otro mercado, equivalentes**:
| Soja | Crypto | Forex | Acciones |
|------|--------|-------|----------|
| WASDE | Halving cycle | NFP/Fed minutes | Earnings calendar |
| COT | Funding rate | COT forex | Short interest |
| China demand | Exchange flows | Trade balance | Sector rotation |
| CVOL | BTC IV (Deribit) | FX IV | VIX / skew |
| Seasonal | On-chain metrics | Carry trade | Insider buying |

**Regla de oro**: Los fundamentals INFORMAN al agente, NUNCA vetan un setup tecnico fuerte. El price action manda en intradiario.

---

## 5. COSTOS OPERATIVOS

| Concepto | Costo | Notas |
|----------|-------|-------|
| Claude API (Anthropic) | ~$0.12/run x 6/dia = $0.72/dia | ~$15/mes |
| MT5 Terminal | $0 | Gratis |
| Broker demo | $0 | Para testing |
| Broker live | Variable | Comisiones + spread |
| Servidor | $0 | Corre en tu PC local |
| **Total mensual** | **~$15-20** | Solo API cost |

---

## 6. CHECKLIST PARA REPLICAR EN OTRO MERCADO

### Paso 1: Elegir el mercado
- [ ] Definir instrumento (ej: EURUSD, BTCUSD, NQ, CL)
- [ ] Verificar que el broker lo tenga en MT5
- [ ] Anotar: tick size, tick value, contract size, volume_min
- [ ] Anotar: horarios de sesion (apertura, cierre, pre/post market)
- [ ] Anotar: spread tipico y comision

### Paso 2: Adaptar datos
- [ ] Cambiar DEFAULT_SYMBOL en mt5_bridge.py
- [ ] Verificar que fetch_mt5_bars devuelve datos correctos
- [ ] Ajustar horarios RTH en live_runner.py y microstructure.py
- [ ] Ajustar timezone

### Paso 3: Adaptar agentes
- [ ] Reescribir TREND_AGENT_SYSTEM con contexto del nuevo mercado
- [ ] Reescribir RISK_AGENT_SYSTEM con parametros del nuevo contrato
- [ ] Ajustar capital de referencia y max risk por trade
- [ ] Ajustar R:R minimo segun caracteristicas del mercado

### Paso 4: Adaptar fundamental context
- [ ] Identificar 3-5 fuentes de datos fundamentales del nuevo mercado
- [ ] Crear funciones de carga en _build_fundamental_context()
- [ ] Configurar crons de actualizacion de datos

### Paso 5: Testing
- [ ] Correr diagnostic: `python -m src.quantagent.live_runner --diagnose`
- [ ] Correr 1 ciclo: `python -m src.quantagent.live_runner --force`
- [ ] Paper trading 2-4 semanas
- [ ] Verificar WR > 50%, PF > 1.5 en 20+ trades
- [ ] Correr con --execute en demo
- [ ] Si consistente: pasar a live con volumen minimo

---

## 7. RIESGOS Y MITIGACIONES

| Riesgo | Mitigacion |
|--------|-----------|
| Modelo deja de funcionar | Drift detection + retrain semanal |
| Perdida grande en 1 trade | Max 2% capital por trade, SL estructural |
| API de Claude cae | Signal = FLAT si no puede analizar |
| MT5 se desconecta | Reconexion automatica, fallback a yfinance |
| Overfitting del modelo | Walk-forward CV con embargo |
| Costos de API excesivos | Gate de max calls/dia |
| Flash crash | ATR-based sizing reduce contratos en vol extrema |

---

## 8. METRICAS DE EXITO

Para considerar el sistema rentable:
- **Win Rate > 55%** en 30+ trades
- **Profit Factor > 1.5** (ganancia bruta / perdida bruta)
- **Max Drawdown < 15%** del capital
- **Sharpe Ratio > 1.0** (retorno ajustado por riesgo)
- **Slippage promedio < 2 ticks** del instrumento
- **Paper vs Real PnL divergencia < 10%**
