# AgroCast — Diseño del módulo Intradía

**Versión**: 0.1 (Fase 0 piloto)
**Fecha**: 2026-04-26
**Autor**: AgroCast core
**Estado**: implementación en curso

---

## 1. Objetivo

Extender AgroCast con un sistema **paralelo** de trading intradía sobre futuros
de soja CBOT (ZS) y micro (MZS), con horizonte 5 min – 4 h, manteniendo el
sistema swing actual (14d) **intacto**.

### No-objetivos
- No reemplazar el modelo swing.
- No operar HFT (<1s). Target latencia: **<500 ms** end-to-end.
- No desarrollar feed propio: usar yfinance (Fase 0) → CME DataMine (Fase 1)
  → broker live (Fase 2).

---

## 2. Principios de diseño

1. **Aislamiento total del swing**: el código intradía vive en `src/intraday/`
   y nunca importa nada de `src/model/`. La única dependencia es leer
   `artifacts/signals.csv` como input read-only.
2. **El swing como prior bayesiano**: el bias diario del swing
   (`signal`, `expected_return`, `expected_vol`) entra al modelo intradía como
   features de contexto, no como señal directa.
3. **Point-in-time correctness intradía**: el embargo entre train/test se
   escala al horizonte (no es 18 días, es 2-4× horizon).
4. **Costos siempre incluidos**: ningún backtest sin spread + slippage + fees.
5. **Streaming-ready**: los features se calculan rolling para que el mismo
   código sirva en backtest y en vivo (sin look-ahead).

---

## 3. Arquitectura

```
┌──────────────────────────────────────────────────────────────────┐
│  SWING (existente, intocado)                                     │
│   pipeline.py → features.csv → train.py → signals.csv            │
└──────────────────────┬───────────────────────────────────────────┘
                       │ (read-only)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  INTRADAY (nuevo)                                                │
│                                                                  │
│  data/                                                           │
│    tick_feed.py        — yfinance / DataMine / broker            │
│    bar_builder.py      — tick→bar (Fase 2)                       │
│    session_calendar.py — RTH, breaks, WASDE, festivos            │
│                                                                  │
│  features/                                                       │
│    microstructure.py   — VWAP, imbalance, cum_delta proxy        │
│    price_action.py     — RSI, ATR, momentum (delegado)           │
│    seasonality.py      — hora, día, sesgo AM/PM                  │
│    context_swing.py    — ⭐ daily_bias, expected_vol del swing    │
│    regime.py           — detector trend/range/shock              │
│                                                                  │
│  model/                                                          │
│    train_intraday.py   — XGBoost multi-horizonte                 │
│    predict_intraday.py — inferencia streaming                    │
│    audit_intraday.py   — walk-forward + leak checks              │
│                                                                  │
│  execution/                                                      │
│    signal_router.py    — prob → LMT/STP/OCO                      │
│    slippage_model.py   — k·spread + impact                       │
│    risk_intraday.py    — sizing, DD stops, kill switch           │
│                                                                  │
│  backtest/                                                       │
│    replay_engine.py    — event-driven con latency injection      │
│    metrics.py          — Sharpe, PF, WR, DD, expectancy          │
│                                                                  │
│  live/                                                           │
│    broker_adapter.py   — IB/Tradovate abstracto (stub Fase 0)    │
│    monitor.py          — drift, kill switch, PnL live            │
│    retrainer.py        — cron semanal                            │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Contrato Swing → Intraday

`src/intraday/features/context_swing.py` lee `artifacts/signals.csv` y expone:

```python
def attach_swing_context(intraday_df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas:
      swing_bias_today:   -1 / 0 / +1     (mapeo de signal SELL/HOLD/BUY)
      swing_expected_ret: float           (expected_return del swing)
      swing_expected_vol: float           (expected_vol — sizing y stops)
      swing_confidence:   float ∈ [0,1]
      swing_age_hours:    float           (cuán fresco es el signal)
    """
```

**Reglas de uso en el modelo intradía**:
- `swing_bias_today = +1` → solo permitir trades LONG (filtro direccional macro).
- `swing_expected_vol > p75` → stops más anchos, sizing más chico.
- `swing_age_hours > 36` → considerar señal stale, ignorar.

---

## 5. Decisiones técnicas confirmadas

| Decisión | Valor | Justificación |
|---|---|---|
| Horizonte target piloto | 60 min (12 barras × 5m) | Equilibrio entre señal/ruido |
| Bar interval principal | 5m | 100% cobertura RTH, 49 sesiones recientes |
| Bar interval contexto | 60m | 875 días de historia, drivers macro |
| Capital simulado | $10,000 USD | Permite operar 1-2 MZS realista |
| Instrumento piloto | MZS (micro) | Capital chico, tick=$2.50, riesgo controlado |
| Modelo | XGBoost binario + vol head | Reuso patrón swing, latencia <50ms |
| Embargo train/test | 2× horizonte = 24 barras (5m) | Evita data leakage en walk-forward |
| Risk/trade | 1% capital | Conservador, sobrevive racha 10 SLs |
| DD diario stop | 2% capital | Freeze 24h al alcanzarlo |
| Slippage modelado | 1 tick adverso + 0.5·spread | MZS ilíquido: tick = $2.50 ≈ 4 bps |
| Comisión | $1.50 round-trip MZS | Tradovate retail típico |
| Format datos | parquet (con fallback CSV) | 3-5× más rápido vs CSV |

---

## 6. Métricas y umbrales (gates de fase)

### Gate Fase 0 → Fase 1 (datos pagos)
Requiere TODAS:
- Sharpe walk-forward ≥ 0.8 (anualizado)
- Profit Factor ≥ 1.4
- Win Rate ≥ 48%
- Max DD < 15%
- Expectancy por trade > 1.5× (slippage + fees)

### Gate Fase 1 → Fase 2 (live)
Requiere TODAS las anteriores + paper trading 30 días con:
- Slippage real ≤ 1.3× modelado
- Drift detectado en <10% de features
- Win rate live ≥ 80% del backtest

---

## 7. Flujo de datos (modo backtest)

```
  signals.csv (swing)        bars 5m (yfinance)
         │                          │
         └──── context_swing ◄──────┘
                    │
                    ▼
            build_features (microstructure + seasonality + regime)
                    │
                    ▼
            target = sign(close[t+12] − close[t])
                    │
                    ▼
       walk-forward CV (window=20 sessions, embargo=24 bars)
                    │
                    ▼
            train_intraday → model_intraday.joblib
                    │
                    ▼
            predict → prob_up
                    │
                    ▼
            signal_router → orders LMT/SL/TP
                    │
                    ▼
            replay_engine → fills + slippage
                    │
                    ▼
            metrics → Sharpe/PF/WR/DD
```

---

## 8. Flujo de datos (modo live, Fase 2)

```
  broker websocket → tick → bar_builder → 5m bar closed
                                                │
                                                ▼
                            features (rolling, sin lookahead)
                                                │
                                                ▼
                              predict_intraday (latencia <50ms)
                                                │
                                                ▼
                              risk_intraday (sizing + checks)
                                                │
                                                ▼
                              signal_router → broker.send_order
                                                │
                                                ▼
                              monitor (PnL, drift, kill switch)
```

---

## 9. Estructura de archivos resultante

```
docs/
  intraday_design.md          ← este doc
src/intraday/
  __init__.py
  data/
    tick_feed.py              ✅ Fase 0
    session_calendar.py       ✅ Fase 0
    bar_builder.py            🔲 Fase 2 (stub)
  features/
    microstructure.py         ✅ Fase 0
    price_action.py           🔲 (delegado a microstructure por ahora)
    seasonality.py            🔲 (delegado a microstructure por ahora)
    context_swing.py          ✅ Fase 0
    regime.py                 ✅ Fase 0 (basico)
  model/
    train_intraday.py         ✅ Fase 0
    predict_intraday.py       ✅ Fase 0
    audit_intraday.py         🔲 Fase 1
  execution/
    signal_router.py          ✅ Fase 0
    slippage_model.py         ✅ Fase 0
    risk_intraday.py          ✅ Fase 0
  backtest/
    replay_engine.py          ✅ Fase 0
    metrics.py                ✅ Fase 0
  live/
    broker_adapter.py         🔲 Fase 2 (stub)
    monitor.py                🔲 Fase 2 (stub)
    retrainer.py              🔲 Fase 2 (stub)
artifacts/intraday/
  model_intraday.joblib
  intraday_signals.csv
  intraday_backtest.csv
  intraday_metrics.json
notebooks/
  intraday_data_quality.py    ✅ existente
  intraday_train_pilot.py     ✅ Fase 0 (orquestador end-to-end)
data/intraday/
  zsf_5m.parquet
  zsf_15m.parquet
  zsf_60m.parquet
```

---

## 10. Roadmap

| Fase | Duración | Costo | Entregable |
|---|---|---|---|
| **0**  | 2 sem  | $0      | Piloto backtest GO/NO-GO con yfinance + MZS |
| **1**  | 4-8 sem| ~$200/m | Datos pagos + paper trading 30 días |
| **2**  | 8-12 sem| ~$300/m| Live con kill switch, capital chico |
| **3**  | ongoing | ~$300/m| Escalado, regime detection avanzado, ZS |

---

## 11. Riesgos y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| yfinance suspende ZS=F intraday | Media | Alto | Fase 1 ya prevé migrar a CME DataMine |
| Modelo overfittea ventana 60d | Alta | Alto | Walk-forward + embargo + 60m con 875d para validar |
| Slippage real >> modelado | Media | Alto | Paper trading 30d antes de capital real |
| Spread MZS demasiado ancho | Media | Medio | Filtro: no operar si spread > 3× mediana |
| Drift de mercado post-entreno | Alta | Medio | Retraining semanal automático + monitor PSI |
| WASDE durante posición abierta | Cierta (mensual) | Alto | Kill rule: cerrar todo 30min antes de WASDE |
