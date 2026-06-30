# XAUUSD Gold — Trading Robot

Sistema intradiario de IA para Oro (XAUUSD) en ICMarkets, replicando la arquitectura del sistema de soja ZS.

---

## Por qué Oro (XAUUSD)

| Factor | Detalle |
|--------|---------|
| **Horario** | London-NY overlap: 07:00–11:30 CT (08:00–12:30 ET) — ventana de máximo volumen |
| **Volatilidad** | ATR 60m típico: $8–20 USD — suficiente para R:R 1.5:1 con stops de estructura |
| **Spread ICMarkets** | ~0.2–0.4 pts (raw spread) — muy competitivo |
| **Sizing con $1,013** | 0.01 lot = $1/punto → stop $15 = $15 riesgo = 1.5% capital ✅ |
| **Price action** | El Oro respeta VWAP, niveles redondos, y estructura técnica mejor que la mayoría |
| **Eventos** | Sin equivalente WASDE — menos sorpresas intradiarias destructivas |

---

## Diferencias vs Sistema Soja

| Parámetro | ZS Soja | XAUUSD Oro |
|-----------|---------|------------|
| Símbolo MT5 | `Sbean_N6` | `XAUUSD` |
| Ventana RTH | 08:30–13:20 CT | **07:00–11:30 CT** |
| Tick/punto | $1.25/tick MZS | **$1/punto @ 0.01 lot** |
| Capital ref | $10,000 | **$1,013** |
| Max riesgo/trade | $200 | **$20** |
| Sizing | 1–3 contratos | **0.01–0.02 lots** |
| Contexto fundamental | USDA/WASDE | **DXY inverso, yields, FOMC** |

---

## Archivos

```
XAUUSD/
├── config.py           # Todos los parámetros — editar aquí para cambiar comportamiento
├── agents.py           # Trend Agent + Risk Agent con prompts Gold-específicos
├── microstructure.py   # 36 indicadores con horarios sesión Gold
├── live_runner.py      # Loop principal — archivo a ejecutar
├── mt5_bridge.py       # Wrapper MT5 con símbolo XAUUSD
├── execution_tracker.py # Métricas de ejecución y PnL
└── retrainer.py        # Reentrenamiento semanal XGBoost
```

---

## Setup rápido

### 1. Verificar que XAUUSD está disponible en MT5

En MetaTrader 5 → Market Watch → buscar `XAUUSD`. Si no aparece:
- Click derecho → Show All → buscar XAUUSD
- O ir a File → Open an Account → ICMarkets → verificar símbolos disponibles

### 2. Configurar parámetros (opcional)

Editar `config.py` si necesitas ajustar:
```python
CT_OFFSET_HOURS = -5   # -5 CDT (verano), -6 CST (invierno)
EXECUTE_TRADES  = False # cambiar a True solo después de paper trading
```

### 3. Ejecutar diagnóstico

```bash
cd templates_nuevos_mercados/XAUUSD
python live_runner.py --diagnose
```

Debe mostrar la conexión MT5 y que `XAUUSD` está disponible.

### 4. Paper trading (PRIMERO hacer esto, mínimo 2 semanas)

```bash
# Un ciclo de prueba
python live_runner.py

# Loop continuo (paper trading)
python live_runner.py --loop
```

Los signals se guardan en:
- `artifacts/xauusd/live_signal.json` — último signal
- `artifacts/xauusd/paper_trades.jsonl` — historial de signals
- `artifacts/xauusd/live_log.jsonl` — log completo

### 5. Ver métricas de paper trading

```bash
python execution_tracker.py --report
```

### 6. Reentrenamiento semanal (cada domingo)

```bash
python retrainer.py
```

### 7. Ejecutar en vivo (solo después de 2+ semanas paper trading con resultados positivos)

```bash
python live_runner.py --loop --execute
```

---

## Cálculo de sizing — Explicación

En ICMarkets, XAUUSD:
- 1 lote estándar = 100 oz de Oro
- 0.01 lote = 1 oz
- Si Oro sube $1, 0.01 lot = $1 de ganancia

Con $1,013 de capital y 2% riesgo máximo = **$20.26 por trade**:

| Stop (puntos) | Lots calculados | Riesgo USD |
|---------------|-----------------|------------|
| $10           | 0.02 lots       | $20 ✅     |
| $15           | 0.01 lots       | $15 ✅     |
| $20           | 0.01 lots       | $20 ✅     |
| $25           | 0.01 lots       | $25 ⚠ (cap min lot) |

La función `calc_lots(stop_points)` en `config.py` hace este cálculo automáticamente.

---

## Sesión Gold — Horarios detallados

| Sesión | Hora CT | Hora ET | Característica |
|--------|---------|---------|----------------|
| Asia (monitoreo) | 19:00–02:00 | 20:00–03:00 | Baja liquidez, no operar |
| London | 02:00–07:00 | 03:00–08:00 | Empieza la acción |
| **London-NY overlap** | **07:00–11:30** | **08:00–12:30** | **OPERAR AQUÍ** |
| NY tarde | 11:30–16:00 | 12:30–17:00 | Liquidez decrece |

La primera hora del overlap (07:00–08:00 CT) suele ser la de mayor volatilidad y mejores setups.

---

## Contexto fundamental Gold

A diferencia de la soja (WASDE semanal), el Oro tiene drivers macro:

| Driver | Correlación | Cómo usarlo |
|--------|-------------|-------------|
| **DXY (Dólar)** | **Inversa** | DXY sube → sesgo bajista Gold |
| **Yields reales (TIPS/TNX)** | **Inversa** | Yields suben → presión bajista Gold |
| **VIX (miedo)** | Positiva | VIX sube → flujo hacia Gold |
| **FOMC/NFP/CPI** | Bipolar | Alta volatilidad — ver `_build_fundamental_context()` |

> **Nota**: El sistema actualmente usa contexto fundamental estático. Para enriquecerlo,
> editar `_build_fundamental_context()` en `live_runner.py` para incluir el DXY en tiempo real
> (disponible en MT5 como `USDX` o `DXY`).

---

## Checklist antes de activar ejecución real

- [ ] MT5 conectado con cuenta ICMarkets demo
- [ ] `XAUUSD` visible en Market Watch de MT5
- [ ] `python live_runner.py --diagnose` sin errores
- [ ] Mínimo 2 semanas de paper trading con win rate > 50% y R:R > 1.5
- [ ] Revisar `execution_tracker.py --report` y validar los signals
- [ ] Confirmar spread real del instrumento (puede variar en horarios de baja liquidez)
- [ ] Entender que esta es una cuenta DEMO — nunca activar `--execute` en cuenta real sin validar extensamente
