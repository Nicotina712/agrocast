# AgroCast PRO - Auditoría Integral del Producto

**Fecha**: 2026-05-18  
**Alcance**: Arquitectura, lógica de negocio, integridad de datos, contraste con evidencia académica  
**Versión auditada**: commit `d489a00` (main)

---

## 1. Descripción del Producto

### Qué es AgroCast PRO

Sistema de inteligencia de mercado para **soja CBOT** orientado a productores agropecuarios latinoamericanos y traders. Combina:

- **Intelligence Engine (IE)**: Debate multi-agente con 5 agentes Claude Sonnet (Bull Analyst, Bear Analyst, Risk Assessor, Technical Analyst, Fund Manager) que deliberan sobre datos de mercado y producen un veredicto unificado.
- **Modelo ML (XGBoost)**: Clasificador binario que predice dirección del precio a 14 días usando ~50+ features (técnicos, fundamentales, estacionales, noticias).
- **Signal Breakdown**: Score compuesto 0-100 ponderando IE (35%), China demand (20%), WASDE (15%), ML (15%), Technical (15%).
- **Market Synthesis**: Briefs ejecutivos generados por Claude Sonnet con toda la inteligencia disponible como contexto.
- **Paper Trading**: Simulador de operaciones que abre/cierra trades basado en la señal consolidada.
- **Producer Advisory**: Recomendación de venta/espera para productores, con cálculo de ROI de almacenamiento.

### Arquitectura técnica

- **Backend**: Flask (Python), desplegado en Render Free Tier (512MB RAM, 0.1 CPU compartido)
- **Frontend**: HTML/JS monolítico con 7 pestañas (Dashboard, Productor, Trader, Estacionalidad, Paper Trading, News Impact, Señales)
- **Pipeline de datos**: GitHub Actions ejecuta `src/pipeline.py` periódicamente, commitea artefactos a `main`
- **APIs externas**: CBOT/CME (yfinance), NewsAPI, GDELT, RSS feeds, USDA, Anthropic Claude

### Stack de datos (20 fuentes verificadas)

| Fuente | Frecuencia | Uso |
|--------|-----------|-----|
| raw_market.csv | Diaria (pipeline) | Precios históricos CBOT |
| features.csv | Diaria (pipeline) | Feature matrix para ML |
| IE verdict | 1x/día (weekdays) | Señal primaria del debate |
| Signal breakdown | 4h cache | Score compuesto |
| Market synthesis x2 | 1x/día | Briefs ejecutivos |
| News intel | 1x/día | Análisis LLM de noticias |
| China demand | 24h cache | Crush margin + importaciones |
| WASDE official | Mensual | Stocks mundiales |
| Brazil exports | 24h cache | Pace de exportaciones |
| Argentina supply | Diaria | Cepo, retenciones, CIARA |
| Basis Uruguay | 24h cache | Precio local vs CBOT |
| Multi commodity | 4h cache | RSI/Bollinger de soja/maíz/trigo |
| ML signals | Diaria (pipeline) | P(sube) del XGBoost |
| Forecast (legacy+horizons) | Diaria (pipeline) | Proyecciones de precio |
| Drift monitor | Pipeline | PSI de features |
| Narrative forecast | Pipeline | Rangos por análogos históricos |
| Event memory | Pipeline | Registro de eventos narrativos |
| Crop progress | Semanal (USDA) | Siembra/cosecha EE.UU. |
| Satellite | Pipeline | NDVI, soil moisture |
| Current contract | API/pipeline | Precio live CME front-month |

---

## 2. Hallazgos Críticos

### 2.1 SEVERIDAD ALTA - Imputación peligrosa de datos faltantes

**Archivo**: `src/pipeline.py`, línea 332  
**Código**: `features = features.ffill().bfill().fillna(0)`

**Problema**: La cadena `ffill() -> bfill() -> fillna(0)` garantiza que el modelo **nunca ve NaN**. Si una fuente externa falla silenciosamente (y todas pueden fallar — están envueltas en `try/except`), la columna entera queda en cero. El modelo entrena sobre datos degradados sin ninguna señal de alarma.

**Impacto**: El XGBoost no puede distinguir entre "dato no disponible" y "valor es 0". Para features como `enso_oni` o `cot_noncomm_long_pct`, el valor 0 tiene significado económico específico. Rellenar NaN con 0 introduce señales espurias.

**Evidencia académica**: La literatura de ML financiero recomienda que los valores faltantes se manejen con indicadores binarios de missingness, no con imputación incondicional (Gu, Kelly, Xiu, 2020 - "Empirical Asset Pricing via Machine Learning", Review of Financial Studies).

**Recomendación**: 
1. Agregar columnas indicadoras `_is_missing` para cada feature externa
2. Validar que la matriz de features tiene completitud mínima antes de entrenar
3. Loguear alertas cuando >5% de features sean imputadas

---

### 2.2 SEVERIDAD ALTA - Pesos fijos sin base empírica

**Archivos**: `src/trader/signal_breakdown.py`, `src/trader/ensemble.py`  
**Pesos actuales**: IE=35%, China=20%, WASDE=15%, ML=15%, Technical=15%

**Problema**: Los pesos de cada factor son constantes hardcodeados sin justificación empírica. No existe ningún proceso de calibración, backtesting de pesos, ni ajuste dinámico.

**Evidencia académica**: La literatura de modelos multi-factor en commodities usa consistentemente esquemas de pesos dinámicos:
- Filtros de Kalman (Harvey, 1989)
- Estimación Bayesiana con actualización secuencial
- Sequential Monte Carlo para calibración

Un estudio de 2025 sobre modelos multi-factor en commodities (arXiv:2501.15596) explícitamente usa Kalman filtering, no pesos fijos. No hay publicación académica que respalde una asignación fija como la utilizada.

**Impacto**: Los pesos no se adaptan a cambios de régimen. En un mercado dominado por weather events, la señal técnica (15%) es casi irrelevante; en un mercado lateral, los técnicos deberían dominar. La rigidez del esquema impide esta adaptación.

**Recomendación**:
1. Implementar rolling hit-rate adjustment: subir peso de factores que aciertan, bajar los que fallan
2. A largo plazo: Bayesian Model Averaging o ensemble stacking con pesos aprendidos
3. Como mínimo: documentar las razones de cada peso y re-evaluarlos trimestralmente

---

### 2.3 SEVERIDAD ALTA - Falta de early stopping en XGBoost

**Archivo**: `src/model/train_returns.py`  
**Configuración**: 300 árboles, max_depth=4, sin `early_stopping_rounds`

**Problema**: El modelo siempre entrena los 300 árboles completos. Con ~1250 filas de datos y potencialmente cientos de features, el ratio parámetros/observaciones es preocupante. No hay mecanismo para detener el entrenamiento cuando la validación deja de mejorar.

**Evidencia**: XGBoost documentation recomienda early stopping como práctica estándar. En datasets financieros con bajo signal-to-noise ratio, sobreajustar es el riesgo principal (Coqueret & Guida, 2020 - "Machine Learning for Factor Investing").

**Recomendación**: Agregar `early_stopping_rounds=20` y usar `eval_set` (que ya existe) para monitorear overfitting.

---

### 2.4 SEVERIDAD ALTA - Falta de embargo en Decision Classifier

**Archivo**: `src/model/decision_classifier.py`, líneas 155-156  
**Código**: `X_train, X_val = X.iloc[:split], X.iloc[split:]`

**Problema**: El split temporal no tiene gap (embargo) entre train y validation. El target es `ret_30d_fwd`, lo que significa que las últimas 30 filas de training tienen targets que solapan temporalmente con las primeras filas de validation. Esto infla las métricas de validación.

**Contraste**: El modelo principal XGBoost (`train_returns.py`) SÍ tiene embargo de 18 días. La inconsistencia sugiere que el decision classifier se desarrolló sin el mismo rigor.

**Recomendación**: Agregar embargo mínimo igual al horizonte del target (30 días).

---

### 2.5 SEVERIDAD MEDIA - Etiqueta "Bayesiano" injustificada

**Archivo**: `src/trader/ensemble.py`  
**Docstring**: "Bayesian-style ensemble"

**Realidad**: Es una media ponderada simple: `p_ens = (w_m * p_model + w_l * p_llm + w_ie * p_ie) / total_w`. No hay prior, no hay likelihood, no hay actualización posterior. El nombre es marketing, no matemáticas.

**Función de peso**: `_weight(hit_rate) = max(hit_rate - 0.50, 0)`. Esto crea una discontinuidad agresiva: un LLM con 48% hit rate (apenas debajo de azar) recibe peso 0, mientras que uno con 52% recibe peso 0.02. La diferencia operativa entre 48% y 52% no justifica esta asimetría.

**Recomendación**: 
1. Renombrar a "weighted ensemble" o "hit-rate-weighted ensemble"
2. Usar una función de peso suave (ej: sigmoid) en vez del threshold en 0.50

---

### 2.6 SEVERIDAD MEDIA - Intelligence Engine: riesgo de anchoring bias

**Archivos**: `src/intel/multi_agent_debate.py`, `src/intel/intelligence_engine.py`

**Lo positivo**: El debate multi-agente SÍ existe con 5 agentes reales (Bull, Bear, Risk, Technical, Fund Manager), cada uno con prompts diferenciados que fuerzan perspectivas opuestas. La arquitectura está inspirada en TradingAgents (ICML 2025), un framework académico legítimo.

**Problema**: Todos los agentes reciben **exactamente el mismo contexto de mercado**. El Bull Analyst y el Bear Analyst ven los mismos datos y solo difieren en el prompt que les pide buscar evidencia alcista vs bajista. No hay asimetría informacional real.

**Evidencia académica**: Un estudio de 2025 (arXiv:2602.14233) advierte que los LLMs en finanzas sufren de: (a) alucinación de datos financieros, (b) sobreconfianza en outputs, (c) errores de carry-over. Otro estudio (FinVision, ACM ICAIF 2024) confirma que el debate multi-agente mejora vs single-agent, pero requiere "substantial human oversight".

**Impacto del peso al 35%**: Dar al IE el mayor peso individual es agresivo. La literatura muestra que los LLMs pueden outperformear en sentimiento pero no en predicción cuantitativa.

**Recomendación**:
1. Reducir peso del IE a 25% y documentar por qué
2. Agregar validación: si el veredicto del Fund Manager contradice >3 de los 4 datos duros (precio, RSI, WASDE, COT), flaggear para revisión humana
3. Implementar IE accountability tracking para calibrar el peso empíricamente

---

### 2.7 SEVERIDAD MEDIA - Forecast 30d deshabilitado por sesgo sistemático

**Archivo**: `src/trader/signal_breakdown.py`, factor forecast con peso=0.00

**Problema**: El comentario en el código admite explícitamente que el forecast 30d "satura sistemáticamente por el cap diario +/-1% acumulado a 30 días, llegando a +12% típico y siendo scoreado como MUY ALCISTA (+1.0). Eso secuestraba el composite."

**Impacto**: Que un componente del sistema tuviera que ser deshabilitado por producir señales sistemáticamente erróneas revela problemas fundamentales en el modelo de forecast. El modelo Ridge+XGBoost subyacente tiene un sesgo alcista estructural que no fue corregido, solo silenciado.

**Recomendación**: 
1. Diagnosticar y corregir el sesgo del modelo de forecast (posible overfitting al trend alcista de largo plazo)
2. Si se rehabilita, usar pesos adaptativos basados en error reciente

---

### 2.8 SEVERIDAD MEDIA - Paper Trading sin costos de transacción

**Archivo**: `src/trader/paper_trading.py`

**Problemas detectados**:
1. **Sin slippage**: Trades entran y salen al precio exacto de cierre
2. **Sin comisiones**: P&L se calcula sin fees ($2.50-5.00 por contrato de soja es estándar)
3. **Sharpe asume 52 trades/año**: `np.sqrt(52)` pero la frecuencia real es desconocida
4. **SL/TP check limitado a 10 velas**: Si un trade lleva 14 días y tocó el SL en el día 3, no se detecta
5. **Buy & Hold benchmark mal calculado**: Usa precios de entry de trades, no precios de mercado

**Evidencia**: La literatura es clara en que backtesting/paper trading sin costos realistas sobreestima rendimientos en 2-5% anual para futures (Harvey, Liu, Zhu, 2016 - "...and the Cross-Section of Expected Returns"). Más del 70% de estrategias rentables en paper trading fallan en real.

**Recomendación**: 
1. Agregar slippage de 0.25-0.50 cents/bushel por operación
2. Agregar comisión fija de $5 por contrato round-trip
3. Limitar SL/TP check a todo el historial de precios desde apertura del trade
4. Corregir Sharpe ratio usando la frecuencia real de trades

---

### 2.9 SEVERIDAD MEDIA - Errores silenciados masivamente

**Patrón**: 213 instancias de `except Exception:` o `except:` en 73 archivos

**Pipeline** (`src/pipeline.py`): Cada paso externo (WASDE, CME, satellite, crop progress, etc.) está envuelto individualmente en `try/except Exception` que imprime un warning y continúa. Si datos críticos fallan, el pipeline sigue como si nada.

**Caso peor**: Si `raw_market.csv` se carga correctamente pero 5 fuentes externas fallan, las features de esas 5 fuentes quedan en 0 (por la imputación del hallazgo 2.1), el modelo entrena, produce una señal, y esa señal alimenta el trader y el productor — todo sin ninguna alarma.

**Recomendación**:
1. Clasificar errores en bloqueantes (datos de precio, features core) vs no-bloqueantes (satellite, crop progress)
2. Implementar un health score del pipeline: si <80% de features tienen datos frescos, no generar señal
3. Agregar alertas (Telegram/WhatsApp ya existen) para failures de pipeline

---

### 2.10 SEVERIDAD BAJA - Freshness thresholds no diferenciados

**Archivo**: `news_server.py`, endpoint `/api/data_freshness`

**Problema**: Los thresholds de frescura (6h/24h) son idénticos para todas las fuentes. WASDE se actualiza mensualmente y mostrará "stale" 28/30 días. Argentina supply tiene cache diario y siempre estará en "warning" por la tarde.

**Recomendación**: Thresholds por fuente: WASDE=35 días, IE=72h (cubre weekend), Argentina=48h, etc.

---

### 2.11 SEVERIDAD BAJA - Despliegue en Free Tier limita confiabilidad

**Archivos**: `render.yaml`, `Procfile`

- 512MB RAM con GC cada 20 requests
- 0.1 CPU compartido
- Sleep after 15min inactivity (cold start ~30s)
- Discrepancia entre Procfile (8 threads) y render.yaml (2 threads)
- `CORS_ORIGINS: "*"` permite requests desde cualquier origen

**Impacto**: Aceptable para MVP personal, pero cualquier producción requeriría tier pago.

---

## 3. Contraste con Evidencia Académica

### 3.1 Lo que SÍ está respaldado

| Componente | Evidencia | Fuente |
|-----------|-----------|--------|
| **Drivers de precio seleccionados** | China demand, WASDE, weather, Brazil/Argentina confirmados como drivers primarios de soja | Journal of Futures Markets 2025; PLOS ONE 2023 |
| **Drift monitoring con PSI** | PSI es el estándar de la industria financiera para detectar drift. Threshold de 0.25 como significativo | ResearchGate 2025; Expert-Driven Monitoring 2024 |
| **Debate multi-agente** | TradingAgents (ICML 2025) y FinVision (ACM ICAIF 2024) validan que multi-agent > single-agent para señales | arXiv:2412.20138; ACM ICAIF 2024 |
| **Estacionalidad como contexto** | Patrones estacionales documentados por CME Group, pero debilitándose por producción hemisferio sur | CME Group Education; Journal of Futures Markets 2024 |
| **Embargo en train/test split** | Correctamente implementado en XGBoost principal con 18 días | Coqueret & Guida 2020 |

### 3.2 Lo que está parcialmente respaldado

| Componente | Evidencia | Riesgo |
|-----------|-----------|--------|
| **XGBoost para precios** | Funciona pero no es state-of-the-art. Deep learning (GRU, LSTM) logra RNMSE ~0.06 vs ~0.15 de XGBoost | Scientific Reports 2025; peso de 15% es apropiado |
| **Ensemble LLM + ML** | Evidencia en equities pero no en commodities. FinSentLLM muestra +3-6% accuracy, pero commodities dependen más de supply/demand que de sentimiento | arXiv:2509.12638; transferibilidad no probada |
| **Análisis técnico** | RSI y Bollinger tienen soporte empírico moderado para commodities de corto plazo. Mejor para timing que para dirección | JMSR 2024; Sage Journals 2025 |
| **LLM accountability** | Verificar predicciones a 14 días es correcto. Pero banda neutral de +/-1.5% es generosa y la muestra es muy pequeña aún | Necesita >50 verificaciones para ser estadísticamente significativa |

### 3.3 Lo que NO está respaldado

| Componente | Problema | Recomendación |
|-----------|----------|---------------|
| **Pesos fijos del signal breakdown** | Ninguna publicación respalda pesos constantes. La literatura usa Kalman, Bayesian, o SMC | Implementar ajuste dinámico |
| **Formula de forecast 7d** | `price_return_est = 2 * expected_return * vol_7d * sqrt(7)` no tiene base teórica | Reemplazar con modelo calibrado |
| **Scaling factors arbitrarios** | `* 5` en ML factor, `/ 5` en technical, `/ 10` en forecast — sin justificación | Calibrar con distribución histórica de cada variable |
| **Paper trading como validación** | Sin costos, sin slippage, sin walk-forward OOS. >70% de estrategias paper-rentables fallan en real | Agregar costos realistas y walk-forward validation |
| **"Bayesian" ensemble** | Es un promedio ponderado simple. Nombrar algo "Bayesiano" sin prior/posterior es misleading | Renombrar o implementar Bayesian real |

---

## 4. Recomendaciones Priorizadas

### Prioridad 1 - CRÍTICAS (impacto directo en calidad de señal)

| # | Acción | Esfuerzo | Impacto |
|---|--------|---------|---------|
| 1 | **Validar feature matrix** antes de entrenar: rechazar si >5% imputado | 2h | Previene señales basadas en datos degradados |
| 2 | **Agregar early stopping** al XGBoost: `early_stopping_rounds=20` | 15min | Previene overfitting |
| 3 | **Agregar embargo** al decision classifier: gap >= horizonte del target | 30min | Corrige leak de datos futuros |
| 4 | **Agregar costos al paper trading**: slippage + comisiones | 1h | Métricas realistas de rendimiento |
| 5 | **Clasificar errores de pipeline** en bloqueantes vs no-bloqueantes | 2h | Evita señales degradadas silenciosas |

### Prioridad 2 - IMPORTANTES (mejoran rigor metodológico)

| # | Acción | Esfuerzo | Impacto |
|---|--------|---------|---------|
| 6 | **Pesos adaptativos en signal breakdown**: rolling hit-rate por factor | 4h | Pesos empíricos en vez de arbitrarios |
| 7 | **Renombrar "Bayesian ensemble"** a "weighted ensemble" | 5min | Honestidad metodológica |
| 8 | **Diagnosticar y corregir sesgo del forecast 30d** | 4h | Rehabilitar factor deshabilitado |
| 9 | **Freshness thresholds por fuente** en el indicador de stale data | 1h | UX correcta (WASDE no es stale a los 2 días) |
| 10 | **IE accountability tracker**: medir hit rate del debate vs random | 2h | Justificar peso del 35% con datos |

### Prioridad 3 - DESEABLES (mejoran robustez a largo plazo)

| # | Acción | Esfuerzo | Impacto |
|---|--------|---------|---------|
| 11 | **Reemplazar XGBoost por híbrido** (LSTM + XGBoost ensemble) | 1-2 semanas | ~2.5x mejora en RNMSE según literatura |
| 12 | **Walk-forward validation** del modelo | 1 día | Métricas OOS confiables |
| 13 | **Centralizar configuración** (thresholds, pesos, params) en un YAML | 3h | Mantenibilidad y trazabilidad |
| 14 | **Rate limiting y cost tracking** para Anthropic API | 2h | Control de costos operativos |
| 15 | **Seguridad**: remover CORS `*`, proteger `/export_brief` | 1h | Endurecimiento de API |

---

## 5. Conclusión

AgroCast PRO es un **MVP ambicioso y funcionalmente completo** que integra más de 20 fuentes de datos en un sistema de decisión multi-capa. El Intelligence Engine con debate multi-agente es un diferenciador legítimo con respaldo académico emergente (TradingAgents, ICML 2025). La selección de drivers de precio de soja es correcta y está alineada con la literatura.

Los problemas principales son de **rigor metodológico**, no de concepto:
- Los pesos del sistema son arbitrarios y deberían ser adaptativos
- La imputación de datos faltantes puede contaminar señales silenciosamente
- El paper trading sin costos da métricas optimistas
- Varios nombres técnicos (Bayesian, Intelligence Engine) prometen más de lo que entregan

El producto tiene una **base sólida** pero necesita las 5 correcciones de Prioridad 1 para que sus señales sean confiables. La buena noticia es que las correcciones más críticas (early stopping, embargo, validación de features) son cambios de pocas líneas de código con alto impacto.

**Fortaleza principal**: Diversificación de fuentes de información y transparencia operativa (accountability, drift monitoring, signal breakdown visible).  
**Debilidad principal**: Falta de calibración empírica de los parámetros del sistema.

---

*Auditoría generada el 2026-05-18. Metodología: revisión de código de 73+ archivos, contraste con 25+ papers académicos y fuentes de industria, y verificación de integridad de 20 fuentes de datos.*
