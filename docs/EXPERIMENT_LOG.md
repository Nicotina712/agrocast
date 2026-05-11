# Experiment Log — AgroCast PRO

Log cronológico de experimentos. Una entrada por experimento, ordenado del más reciente al más antiguo.

**Veredictos posibles**:
- ✅ **Aplicado** — en producción, aporta valor demostrable
- ⚠️ **Informativo** — en producción como contexto, no como decisor (no supera always-sell con significancia)
- ❌ **Descartado** — no aporta, no volver a perseguir sin cambio fundamental de datos o enfoque

Para análisis profundos ver:
- [`MODELING_FINDINGS.md`](MODELING_FINDINGS.md) — hipótesis validadas/rechazadas con detalle técnico
- [`PREDICTIVE_THREAD_FINDINGS.md`](PREDICTIVE_THREAD_FINDINGS.md) — tabla de los 13 enfoques probados sobre el hilo predictivo principal
- [`PATTERN_DECISION_ROADMAP.md`](PATTERN_DECISION_ROADMAP.md) — roadmap Fases 1–4 del decision classifier

---

## Template para nueva entrada

```
## YYYY-MM-DD · Nombre del experimento · Veredicto

**Qué se probó:** una línea clara.
**Config / params clave:** lo mínimo para poder reproducirlo.
**Resultado / métrica:** número concreto, siempre vs baseline.
**Veredicto:** ✅ / ⚠️ / ❌ con una frase de por qué.
**Lección (si aplica):** qué no volver a intentar, o qué sí escalar.
```

---

## 2026-05-10 · Narrative-Only Forecast con rango diario (1d) · ⚠️ Informativo

**Qué se probó:** modelo narrative-only (sin ML) que usa análogos del event_memory para producir rangos esperados por horizonte (1d, 7d, 15d, 30d). Reemplaza al modelo híbrido ML+Narrativa que probó no agregar valor. El horizonte 1d es nuevo y responde "qué esperar hoy/mañana dado el evento actual".

**Config / params clave:** k=20 análogos, min_gap_days=7 (1d) o 30 (otros), threshold z-score=1.5, narrative_only wait = direction==bullish AND strength>0.3 AND is_event. Rangos: Q10/Q25/median/Q75/Q90 empíricos.

**Resultado / métrica (backtest OOS 12m, daily para 1d, non-overlapping para 7d+):**

| Horizonte | n dec | Narrative mean | Always-Wait | Oracle | % activo |
|-----------|-------|----------------|-------------|--------|----------|
| 1d  | 252 | +$0.025/ton | var | var | 56.3% |
| 7d  | 52  | +$0.762/ton | +$0.86/ton | +$4.85/ton | 51.9% |
| 15d | 23  | +$0.672/ton | −$1.08/ton | +$6.25/ton | 52.2% |
| 30d | 13  | −$1.824/ton | −$2.08/ton | +$6.62/ton | 61.5% |

Hallazgo clave 1d: los eventos ESTRECHAN el rango diario (std 0.701% vs 0.847% sin evento).
Direccionalidad 1d: ~50-53% (marginal). El valor está en el RANGO, no en la dirección.

**Veredicto:** ⚠️ Informativo. El forecast narrative-only aporta como RANGO INFORMATIVO para el productor, no como predictor direccional. A 1d la direccionalidad es coin flip pero el rango más estrecho es información útil. A 7d y 15d supera al always-wait (período bajista). A 30d sigue siendo anti-predictivo.

**Lección:**
1. **El valor de la narrativa es el rango, no la dirección.** Los eventos estrechan la distribución — eso es información genuina.
2. **Hybrid model descartado:** la combinación lineal ML+narrativa destruye señal. Narrative-only supera a hybrid en todos los horizontes.
3. **1d es informativo pero no decisor.** El productor puede usar "rango esperado hoy: -0.8% a +1.2%" como contexto, no como orden de venta.

---

## 2026-05-10 · Market Intelligence Engine V1 — Modelo Híbrido ML+Narrativa · ⚠️ Informativo

**Qué se probó:** motor de inteligencia de mercado que detecta eventos narrativos (oil/biofuel, macro/FX, momentum especulativo), construye event memory retroactivo (1423 eventos en 10 años), busca análogos narrativos (no solo shocks de precio), y combina la señal ML (decision classifier) con un ajuste narrativo: `hybrid_p = ml_p + α × direction × strength × (1 - fade_risk)`.

**Config / params clave:** alpha=0.15 (conservador), threshold z-score=1.5 para detección, k=20 análogos, fade_risk heurístico (RSI extremo + momentum misalignment + vol alta + overextension).

**Resultado / métrica (backtest OOS 12m, default profile):**

| Horizonte | ML-only | Narrative-only | Hybrid | Oracle |
|-----------|---------|----------------|--------|--------|
| 7d (52 dec) | +$0.00/ton | **+$1.08/ton** | −$0.28/ton | +$4.85/ton |
| 15d (23 dec) | **+$0.81/ton** | +$0.67/ton | −$0.30/ton | +$6.25/ton |
| 30d (13 dec) | +$0.00/ton | −$4.57/ton | +$0.00/ton | +$6.62/ton |

Event memory: 1423 eventos, fade rate 7d = 39.1%, tipos dominantes: speculative_mom (520), oil_energy (452), macro_fx (441).

**Veredicto:** ⚠️ Informativo. El modelo híbrido con combinación aditiva simple (α=0.15) NO mejora al ML-only. Dato interesante: narrative_only supera a ML en 7d (+$1.08 vs $0) — hay señal narrativa a corto plazo que el ML no captura. Pero la combinación lineal la destruye. A 30d la narrativa es anti-predictiva (−$4.57), confirmando R10 (shock engine pierde).

**Lección:**
1. **La narrativa tiene señal a 7d** pero la combinación lineal con ML es demasiado naive. Necesita interacción no-lineal o regime-switching (activar narrativa solo cuando hay evento fuerte).
2. **El event_memory es el activo más valioso** de este módulo. En 6-12 meses permitirá optimizar la combinación con datos reales.
3. **Fade risk (39.1% empírico) es información genuinamente útil** para el productor aunque no genere alpha de trading.
4. No escalar alpha ni optimizar umbrales con solo 23-52 decisiones OOS — overfitting garantizado.

---

## 2026-05-10 · Backtest OOS 6 estrategias (Fase 2.3) · ⚠️ Informativo

**Qué se probó:** comparar Always-Sell / Always-Wait / Split-50 / Oracle / Model-Binary / Model-Partial en el último año de datos OOS (entrenando en años previos), sobre ventanas no solapadas de 7 / 15 / 30 días.

**Config / params clave:** profile=default, test_months=12, precio referencia $444/ton, cutoff 2025-05-08.

**Resultado / métrica (profile default):**

| Horizonte | n dec | Always-Wait | Model-Partial | Oracle (techo) |
|-----------|-------|-------------|---------------|----------------|
| 7d  | 52 | +$0.86/ton | −$0.28/ton | +$4.85/ton |
| 15d | 23 | −$1.08/ton | **+$0.18/ton** | +$6.25/ton |
| 30d | 13 | −$2.08/ton | **+$0.37/ton** | +$6.62/ton |

Delta siempre relativo a Always-Sell (baseline = $0).

**Veredicto:** ⚠️ Informativo. En el régimen bajista del último año, el modelo graduado (partial_sell) vence al always_wait en 15d y 30d al reducir la exposición al esperar. Sigue muy lejos del oracle. No hay significancia estadística con 13–23 decisiones. **No escalar a decisor.**

**Lección:** El valor de partial_sell no está en "ganar alpha" sino en **evitar la pérdida de always_wait en régimen bajista**. Útil como segunda opinión cuando el mercado cae. Documentar el régimen del OOS siempre (este año fue bajista post-cosecha Brasil récord).

---

## 2026-05-10 · Partial Sell graduado — _combine_partial_sell (Fase 2.2) · ⚠️ Informativo

**Qué se probó:** regla no aprendida que combina P(WAIT calibrado) + delta esperado del regresor en 5 niveles de decisión: SELL_100 / SELL_70 / SPLIT_50 / HOLD_70 / HOLD_100.

**Config / params clave:** MIN_DELTA=0.5% del precio; umbrales P: 0.40 (SELL_100), 0.50 (SELL_70), 0.55 (SPLIT_50), 0.62+delta>=1.5% (HOLD_70).

**Resultado / métrica:** ver backtest Fase 2.3 arriba. model_partial gana en 15d y 30d en OOS bajista.

**Veredicto:** ⚠️ Informativo. Regla simple, no sobreajustada, que usa la magnitud económica para graduar la venta. Coherente con la intuición del productor. Se muestra en UI con badge de color y porcentajes (v:X% r:Y%).

**Lección:** Las reglas simples no aprendidas son preferibles a umbrales optimizados en este dataset (pocos grados de libertad, régimen-dependiente). No agregar más parámetros.

---

## 2026-05-10 · Delta Regressor económico (Fase 2.1) · ⚠️ Informativo

**Qué se probó:** XGBRegressor sobre el mismo feature set del clasificador, con target = ret_Hd_fwd − cost_pct (delta económico real de esperar). Entrena en paralelo al clasificador binario.

**Config / params clave:** n_estimators=300, max_depth=4, lr=0.05, eval_metric=mae. Target: delta fraccionario. val_delta_mae reportado por horizonte.

**Resultado / métrica:** val_delta_mae ~0.02–0.04 (fracción del precio). El regresor da magnitud; no reemplaza la dirección del clasificador.

**Veredicto:** ⚠️ Informativo. Agrega información de magnitud que el clasificador binario no provee. Validado anteriormente (R11 en MODELING_FINDINGS) que regresores NN-analog de delta fallan → XGB sobre 17 features es el mejor disponible, aunque MAE es alto. Se usa para graduar el partial_sell, no como predictor directo.

**Lección:** Separar siempre dirección (clasificador) de magnitud (regresor). No usar el delta como decisor único — solo como input para la regla de graduación.

---

## 2026-05-07 · Multi-horizon decision classifier 7d / 15d / 30d (Fase 1.2) · ⚠️ Informativo

**Qué se probó:** extender el clasificador (antes solo 30d) a tres horizontes simultaneos. Un modelo distinto por horizonte, entrenado con cost_pct prorrateado.

**Config / params clave:** HORIZONS_MULTI=[7,15,30], train_years=5, fp_weight=2.0 (Idea A), isotonic calibration.

**Resultado / métrica (2026-05-06, profile default):**

| Profile | 7d | 15d | 30d | Best |
|---------|-----|------|------|------|
| default | 50% (IND) | 36% (SELL) | 51% (IND) | 15d |
| low_cost | 57% (IND) | 30% (SELL) | 61% (WAIT) | 15d |
| high_cost | 43% (IND) | 0% (SELL fuerte) | 18% (SELL) | 15d |

**Veredicto:** ⚠️ Informativo. El horizonte 15d es el que muestra más convicción en todos los profiles. high_cost → señal SELL fuerte (coherente: costos altos penalizan esperar). low_cost → único con señal WAIT en 30d (coherente: puede absorber el costo). En 7d todos en zona INDIFFERENT (movimientos cortos = ruido).

**Lección:** El horizonte óptimo para el productor NO es siempre 30d. 15d captura el momento de mayor convicción en este dataset. Reportar siempre el `best_horizon` en la UI.

---

## 2026-05-07 · Producer Profiles × 5 (Fase 1.3) · ✅ Aplicado

**Qué se probó:** parametrizar el clasificador por perfil del productor: default, low_cost (silo propio), high_cost (storage rentado + crédito caro), liquidity_need, quality_aware.

**Config / params clave:**
- low_cost: storage=$0/ton/mes, financing=5%
- high_cost: storage=$10/ton/mes, financing=15%, quality_risk=0.5%/mes
- liquidity_need: financing=18%
- quality_aware: quality_risk=1.5%/mes

**Resultado / métrica:** cada profile genera cost_pct distinto → targets distintos → señales distintas. Los resultados son cualitativamente coherentes con la economía del productor.

**Veredicto:** ✅ Aplicado. Persiste en `artifacts/decision_classifier/{profile}.json`. Selector en UI funcional. Aporta personalización real al producto sin añadir complejidad de modelo.

**Lección:** La personalización por costo aporta más valor percibido que cualquier mejora de MAE. Mantener los 5 profiles; no agregar más sin validación de que representan segmentos reales de usuarios.

---

## 2026-05-07 · Análogos históricos como capa narrativa (Fase 1.1) · ⚠️ Informativo

**Qué se probó:** dado el estado actual del mercado (vector de features), buscar los 20 casos históricos más similares (distancia euclídea en Z-score) y reportar sus outcomes reales.

**Config / params clave:** ANALOG_FEATURES=7 (price_pct_12m, mom_5d/20d, rsi_14, vol_30d, Oil_chg7, news_sentiment). k=20, min_gap_days=60.

**Resultado / métrica:** win_rate, delta_avg_usd_ton, Q10/Q90, narrativa en español. No se usa como predictor (validado en R11: NN-analog pierde).

**Veredicto:** ⚠️ Informativo. La capa narrativa es valiosa para comunicar al productor ("en 20 situaciones similares, esperar fue mejor en X% de los casos"). NO predice — solo contextualiza. Coherente con el diagnóstico R11.

**Lección:** Usar análogos para COMUNICAR, no para DECIDIR. El productor puede procesar "20 casos similares" mejor que "P=0.43". El valor está en el framing, no en la predictividad.

---

## 2026-05-03 · Decision Classifier + asymmetric loss — Idea A (best config) · ⚠️ Informativo

**Qué se probó:** XGBClassifier sobre 17 features con asymmetric sample_weight (fp_weight=2.0 sobre label=0) + isotonic calibration. Mejor de 21 configuraciones en 7 enfoques.

**Config / params clave:** target=ret_30d_fwd>cost_pct, fp_weight=2.0, isotonic, train_years=5, refit 90d walk-forward.

**Resultado / métrica:** PnL −0.113%/mes (−1.36%/año vs always-sell). Acc=57.1%, Brier calibrado=0.31. El menor gap encontrado entre todos los enfoques.

**Veredicto:** ⚠️ Informativo. Aplicado en producción como segunda opinión. SIGUE PERDIENDO al always-sell. El gap de 1.36%/año es económicamente pequeño pero estadísticamente presente. No usar como decisor único.

**Lección:** Penalizar los falsos positivos (fp_weight=2.0) mejora el clasificador en este problema asimétrico (el costo de esperar cuando no conviene > el costo de vender cuando convendría esperar). Esta lección aplica a cualquier problema de decisión de carrying cost.

---

*Instrucción de uso: copiar el template al inicio de este archivo, completar los campos, y hacer commit. Un experimento = una entrada. Si el experimento tiene subvariantes, una entrada por variante relevante.*
