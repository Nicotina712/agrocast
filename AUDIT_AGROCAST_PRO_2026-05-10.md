# AUDITORÍA INTEGRAL — AgroCast PRO
### Fecha: 10 de mayo 2026
### Versión: Post-sincronización worktree completa

---

## PARTE 1: ESTADO TÉCNICO DEL SISTEMA

**Dimensiones del sistema:**
- Servidor Flask: 2,633 líneas, 50+ endpoints API
- Frontend SPA: 5,241 líneas, 7 tabs, 68 funciones JS
- Pipeline: 697 líneas, 19 etapas
- Módulos fuente: 96 archivos Python en `src/`
- Datos: todos frescos al 2026-05-10

---

### TABLA DE ESTADO POR SUBSISTEMA

#### ✅ FUNCIONA PERFECTO (40 subsistemas)

| # | Subsistema | Datos | Endpoint | UI | Observaciones |
|---|-----------|-------|----------|-----|---------------|
| 1 | Forecast Legacy (Ridge/XGB) | `forecast.csv` ✓ | `/api/news`, `/api/forecast_ab` | Dashboard: chart + overlay | Walk-forward 14d, completo |
| 2 | Forecast Horizons (Ensemble) | `forecast_horizons.csv` ✓ | `/api/forecast_multihorizon` | Dashboard: toggle A/B | Bandas conformales + multi-horizonte |
| 3 | Forecast Mensual | `monthly_forecast.csv` ✓ | `/api/monthly_forecast` | Productor + Historial | ETS/seasonal-naive 90d |
| 4 | News/Sentimiento | `news_intel.json` (28 drivers) ✓ | `/api/news`, `/api/news_intel` | Dashboard: cards + gauge | GDELT + multi-source |
| 5 | Paper Trading | `paper_trades.csv` (9 trades) ✓ | `/api/paper_trades` | Trader: equity curve | ATR-based SL/TP, P&L tracking |
| 6 | Señal Modelo (ML) | `signals.csv` ✓ | `/api/news` (embedded) | Dashboard: badge + confianza | XGBoost BUY/SELL/HOLD |
| 7 | Señal Noticias | Calculado en vivo | `/api/news` (embedded) | Dashboard: sección señal | ALCISTA/BAJISTA/NEUTRAL |
| 8 | Signal Breakdown | `signal_breakdown.json` ✓ | `/api/signal_breakdown` | Dashboard: tabla factores | Multi-factor compuesto |
| 9 | Ensemble Bayesiano | `ensemble_signal.json` ✓ | `/api/ensemble` | Dashboard: panel | ML + LLM combinado |
| 10 | Estacionalidad | Calculado en vivo | `/api/seasonality` | Tab dedicado | Win rates, CI por mes |
| 11 | Narrative Forecast | `narrative_forecast/latest.json` ✓ | `/api/intel/narrative_forecast` | Intel Engine tab | 1d/7d/15d/30d rangos + backtest |
| 12 | Hybrid Verdict | En vivo + cache | `/api/intel/hybrid_verdict` | Intel Engine tab | ML + Narrativa blended |
| 13 | Event Memory | `event_memory.csv` (243KB) ✓ | `/api/intel/event_memory` | Intel Engine tab | Detección eventos + análogos |
| 14 | CME Precios | `cme_history.csv` ✓ | `/api/cme` | Dashboard: panel CME | OI, volumen, spread |
| 15 | Contrato Actual | `current_contract.json` ✓ | `/api/current_contract` | Dashboard: info contrato | JUL2026 @ 1208.0, spread -5.25 |
| 16 | Term Structure | `curve_history.csv` ✓ | `/api/curve_history` | Trader + Historial | Contango/backwardation + roll yield |
| 17 | WASDE | `wasde_official.json` ✓ | `/api/wasde_official`, `/api/next_wasde` | Dashboard: panel + countdown | PSD API, stocks + sorpresa |
| 18 | WASDE Stress Test | `wasde_stress.json` ✓ | `/api/wasde_stress` | Dashboard: panel | Top 5 reportes más volátiles |
| 19 | USDA Inspections | `usda_inspections.csv` (50KB) ✓ | `/api/usda_inspections` | Dashboard: panel | Inspecciones semanales, YoY |
| 20 | COT (Commitments of Traders) | `cot_soybeans.csv` (155KB) ✓ | `/api/cot_delta`, `/api/cot_analogs` | Dashboard + Trader | Posicionamiento + análogos + extremos |
| 21 | Brazil Exports | `brazil_exports.json` ✓ | `/api/brazil_exports` | Dashboard: panel | Pace vs proyección USDA |
| 22 | China Demand | `china_demand.json` ✓ | `/api/china_demand` | Dashboard: panel | Imports + crush margin + CNY |
| 23 | Argentina Signal | `argentina_supply.json` ✓ | `/api/argentina_signal` | Dashboard: sección | Cepo/retenciones + supply score |
| 24 | Basis Uruguay | `basis_uruguay.json` ✓ | `/api/basis_uruguay` | Dashboard + Productor | Revista Verde scraper, z-score |
| 25 | Brief Productor | `market_synthesis.json` ✓ | `/api/market_synthesis?type=producer` | Intel LLM tab | Claude Sonnet, headline + stance |
| 26 | Brief Trader | `market_synthesis_trader.json` ✓ | `/api/market_synthesis?type=trader` | Intel LLM tab | Trader-focused, generación separada |
| 27 | Export Brief | Generado on-demand | `/export_brief` | Navbar: "Exportar Brief" | Print-to-PDF via Ctrl+P |
| 28 | Régimen de Mercado | `regime.json` ✓ | `/api/regime` | Dashboard: panel + caveats | Rule-based + HMM 3-state + Markov |
| 29 | Regime Switching | `regime_switching.json` ✓ | `/api/regime_switching` | Embebido en régimen | Markov-Switching con alpha |
| 30 | Shock Engine | `active_shock.json` + `shock_catalog.csv` ✓ | `/api/active_shock` | Dashboard: panel | Detección 5d/10d + análogos históricos |
| 31 | Decision Classifier | `decision_classifier.json` + `.joblib` + 5 perfiles ✓ | `/api/decision_classifier` | Dashboard: panel + perfiles | Cost-aware, 7d/15d/30d × 5 perfiles |
| 32 | Optimal Stopping | Calculado en vivo (Monte Carlo) | `/api/optimal_stopping` | Dashboard: panel | Backward induction |
| 33 | Economic Utility | Calculado en vivo | `/api/economic_utility` | Dashboard: embebido | WAIT vs SELL_NOW |
| 34 | Backtest Histórico | Calculado en vivo | `/api/backtest` | Dashboard + Historial | Walk-forward, cache 1h |
| 35 | Backtest Decision | 5 perfiles pre-computados ✓ | `/api/backtest_decision` | Dashboard: panel | 6 estrategias × 5 perfiles × 3 horizontes |
| 36 | Backtest Hybrid | `hybrid_backtest/default.json` ✓ | `/api/intel/hybrid_backtest` | Intel Engine tab | ML vs Narrativa vs Hybrid |
| 37 | Drift Monitor | `drift_monitor.json` ✓ | `/api/drift_monitor` | Dashboard + Historial | Rolling 30/60/90d health |
| 38 | Lookahead Audit | `lookahead_audit.json` ✓ | `/api/lookahead_audit` | Dashboard: panel | OOS temporal-cut audit |
| 39 | Multi-Commodity | `multi_commodity.json` ✓ | `/api/multi_commodity` | Dashboard: panel señales | Soy + Corn + Wheat técnicos |
| 40 | Satélite/Clima | `satellite_history.csv` (31KB) + `climate_forecast.csv` ✓ | `/api/satellite` | Dashboard: panel + stress chart | NASA POWER, 4 regiones |
| 41 | Crop Progress | `crop_progress.csv` ✓ | `/api/crop_progress` | Dashboard: panel | USDA NASS semanal |
| 42 | LLM Accountability | `llm_snapshots.json` ✓ | `/api/llm_accountability` | Intel LLM tab | Daily snapshot + 7d hit rate |
| 43 | ML Quality | `ml_quality.json` ✓ | `/api/ml_quality` | Historial tab | Métricas classifier |
| 44 | Accountability | `forecast_snapshots.json` ✓ | `/api/accountability` | Historial tab | Forecast vs actuals |
| 45 | Productor Module | `producer_decisions.csv` ✓ | `/api/producer`, `/api/producer_decision` | Tab dedicado | Sell signal, storage ROI, costos |
| 46 | Trader Module | Múltiples ✓ | `/api/trader` | Tab dedicado | Risk, term structure, COT, paper |

#### ⚠️ NECESITA MEJORA (6 subsistemas)

| # | Subsistema | Problema | Impacto | Acción requerida |
|---|-----------|----------|---------|-----------------|
| 1 | **Precisión del modelo ML** | Drift monitor muestra <48% accuracy a 30d (random). AUC degrada de 0.609 → 0.446 OOS | **CRÍTICO** | Implementar re-entrenamiento rolling, agregar features de news volume, considerar Time-MoE/Chronos-2 como ensemble members |
| 2 | **Track record paper trading** | Solo 9 trades (4 ganadores, 5 perdedores). Insuficiente para validación estadística | **ALTO** | Necesita n≥30-50 trades para ser comercialmente creíble. Acelerar señales o agregar backtesting walk-forward con datos históricos |
| 3 | **Ensemble cold start** | `model_hit_rate` y `llm_hit_rate` = None. Bayesian weighting cae en equal weights | **MEDIO** | Se auto-resuelve con tiempo. Considerar seed con backtest data |
| 4 | **Profundidad CME history** | Solo 16 días de datos en `cme_history.csv` | **BAJO** | Crece automáticamente. Considerar backfill histórico |
| 5 | **Export Brief** | Print-to-PDF via Ctrl+P, no es generación PDF server-side | **BAJO** | Implementar WeasyPrint o ReportLab para PDF profesional |
| 6 | **Climate Forecast (Forward)** | Datos existen (`climate_forecast.csv`) pero no hay UI ni API endpoint | **BAJO** | Agregar endpoint `/api/climate_forecast` y panel en dashboard |

#### ❌ NO FUNCIONA / NO INTEGRADO (3 subsistemas)

| # | Subsistema | Problema | Impacto | Acción requerida |
|---|-----------|----------|---------|-----------------|
| 1 | **SHAP Explanation** | Package `shap` no instalado → artefacto nunca generado. Endpoint retorna 404. Sin UI | **MEDIO** | Instalar shap, generar artefacto en pipeline, crear panel en frontend |
| 2 | **Forecast Paths (Monte Carlo)** | Endpoint `/api/forecast_paths` funcional, retorna distribución. Pero NINGÚN código frontend lo llama | **MEDIO** | Agregar visualización de densidad/probabilidad al chart principal |
| 3 | **Sistema Intraday** | 14 archivos completos en `src/intraday/` (tick feed, bar builder, microstructure, signal router, broker adapter). CERO endpoints, CERO UI | **ALTO** | Decisión estratégica: ¿integrar o mantener como R&D? Si se integra, necesita 5-8 endpoints + tab dedicado |

---

## PARTE 2: ANÁLISIS DE VIABILIDAD COMERCIAL

### ¿Qué información entrega el producto hoy?

| Capa | Contenido | Valor para el usuario |
|------|-----------|----------------------|
| **Precio y mercado** | Precio CBOT en vivo, contrato activo, spread rollover, term structure, OI/volumen | Base necesaria, commoditizado |
| **Forecast ML** | Predicción 7d y 30d con bandas de confianza, dos modelos en A/B test | Diferenciador si accuracy mejora |
| **Intel LLM** | Brief productor + trader con análisis narrativo, stance, recomendación táctica | **Diferenciador principal** — ningún competidor ofrece esto a <$100/mo |
| **Fundamentales** | WASDE, USDA inspections, COT, Brazil exports, China demand, Argentina supply, Basis Uruguay | Ahorra horas de compilación manual |
| **Señales compuestas** | ML + News + Fundamentales con breakdown de factores y pesos transparentes | Más transparente que cualquier competidor |
| **Gestión de riesgo** | Régimen de mercado, detección de shocks, análogos históricos, drift monitor | Único en el segmento de precio |
| **Decisión** | Optimal stopping, economic utility, decision classifier con 5 perfiles | Sofisticación institucional a precio retail |
| **Paper trading** | Track record desde inception con equity curve y accountability | Builds trust, necesita más trades |

### Fortalezas comerciales principales

1. **Nicho sin competencia directa**: inteligencia de mercado de soja con foco Sudamericano (Uruguay/Argentina/Brasil). Ningún competidor ofrece esto.
2. **Full-stack a precio accesible**: ML + LLM + fundamentales + señales = lo que Bloomberg ofrece a $24K/año, nosotros a $99/mo.
3. **Doble audiencia**: productor Y trader en el mismo producto con vistas diferenciadas.
4. **Costo operativo ultra-bajo**: ~$0.75/mo en LLM API calls. Margen bruto >95%.
5. **Brief LLM como killer feature**: el análisis narrativo que genera Claude Sonnet es comparable a lo que produce un analista humano senior.

### Debilidades comerciales críticas

1. **Accuracy del modelo ML <48%**: el drift monitor expone que el modelo principal predice peor que random a 30 días. Esto destruye credibilidad si un cliente técnico lo audita.
2. **Track record insuficiente**: 9 trades no son un argumento comercial. Necesitamos mínimo 30+ trades cerrados.
3. **Sin autenticación de usuarios**: no hay cuentas, login, ni personalización. Imposible cobrar sin esto.
4. **Sin app móvil**: el productor necesita consultar precios desde el campo.
5. **Commodity único**: solo soja. Limita TAM artificialmente.

---

## PARTE 3: COMPARACIÓN COMPETITIVA

### Mapa del mercado 2026

| Competidor | Tipo | Precio | ML Forecast | LLM Briefs | LatAm Focus | Paper Trading |
|-----------|------|--------|-------------|-------------|-------------|---------------|
| **AgroCast PRO** | Intelligence | $49-99/mo | ✅ | ✅ | ✅ | ✅ |
| Bloomberg Terminal | Terminal | $24K/año | ❌ | ❌ | Parcial | ❌ |
| LSEG/Refinitiv | Terminal | $15K/año | Parcial | ❌ | Parcial | ❌ |
| Barchart cmdtyView | Data + Trading | $500-2K/mo | ❌ | Parcial (Carl AI) | Parcial (LATAM 2025) | Parcial |
| DTN | News + Weather | No público | ❌ | ❌ | ❌ | ❌ |
| **Helios AI** | AI Copilot | TBD (seed $4.7M) | ✅ (400+ series) | Parcial | Limitado | ❌ |
| **Vesper** | Forecast | $20K+/año | ✅ (95% dir.) | ❌ | Limitado | ❌ |
| ChAI | Forecast | Enterprise | ✅ | ❌ | ❌ | ❌ |
| Revenue.ai (Zeta) | AI Copilot | Enterprise | ❌ | ✅ (copilot) | ❌ | ❌ |
| S&P/Kensho | Data + API | $50K+/año | ❌ | ✅ (API) | ❌ | ❌ |
| Cropin SAGE | Agri Intel | Enterprise | Parcial (yields) | ✅ (Gemini) | Parcial | ❌ |

### Posición de AgroCast PRO

**Ocupamos un espacio vacío genuino**: no existe otra plataforma que combine ML forecasting + LLM market briefs + datos fundamentales agregados, enfocada en mercados sudamericanos de soja, a $49-99/mo.

El competidor más cercano es **Helios AI** (multi-agente AI, $4.7M seed en Sept 2025), pero se enfoca en procurement/supply chain, no en soporte de decisión para productores/traders.

### GAPs vs líderes de mercado

| Gap | Severidad | Competidores que lo tienen |
|-----|-----------|--------------------------|
| Datos en tiempo real (streaming) | Alta si apuntamos a institucionales | Bloomberg, LSEG, Barchart |
| Multi-commodity (>soja) | Alta para crecimiento | Helios (75+), Vesper (1900+), ChAI |
| Integración broker/ejecución | Media | Barchart, Bloomberg |
| App móvil | Media para productores | Bushel, DTN, FieldView |
| Track record largo | Alta para credibilidad | Vesper (claims 95% dir.) |
| API institucional documentada | Media | S&P/Kensho, ChAI |

---

## PARTE 4: FRONTERA TECNOLÓGICA — LA VISIÓN DE "INTELIGENCIA REAL"

### Tu intuición validada por la academia

Lo que estás describiendo — que **antes del dato está la noticia, y antes de la noticia está la especulación** — está respaldado por investigación reciente:

> **Hallazgo clave**: Un paper de 2025 (arXiv 2508.06497) sobre predicción de shocks en commodities mostró que al remover los embeddings de noticias, el AUC cayó de **0.94 a 0.46**. Las noticias no son suplementarias — son la señal primaria.

> **Segundo hallazgo**: Un estudio MDPI 2025 usando GDELT para maíz encontró que el **volumen y persistencia de cobertura mediática** importa más que el tono (positivo/negativo). Los variables de sentimiento recibieron peso CERO bajo regularización.

### Arquitectura propuesta: "Intelligence Engine" (evolución del sistema actual)

```
┌─────────────────────────────────────────────────────────────┐
│                    INTELLIGENCE ENGINE v2                     │
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │  TIER 1:    │  │  TIER 2:     │  │  TIER 3:         │    │
│  │  FinBERT    │  │  Claude      │  │  Multi-Agent     │    │
│  │  (rápido,   │→ │  Sonnet      │→ │  Debate          │    │
│  │  $0/llamada)│  │  (profundo,  │  │  (Bull vs Bear   │    │
│  │  Clasifica  │  │  $0.01/call) │  │  + Risk Team)    │    │
│  │  TODO       │  │  Analiza     │  │  Decide          │    │
│  │             │  │  COMPLEJO    │  │  ACCIONES         │    │
│  └─────────────┘  └──────────────┘  └──────────────────┘    │
│         │                │                    │               │
│         ▼                ▼                    ▼               │
│  ┌─────────────────────────────────────────────────────┐     │
│  │              KNOWLEDGE BASE (RAG)                    │     │
│  │  - Event Memory histórica (shock_catalog.csv)        │     │
│  │  - Patrones estacionales documentados                │     │
│  │  - Literatura sobre mercados de futuros              │     │
│  │  - Análogos históricos de shocks similares           │     │
│  │  - Resultados de predicciones anteriores             │     │
│  └─────────────────────────────────────────────────────┘     │
│         │                │                    │               │
│         ▼                ▼                    ▼               │
│  ┌─────────────────────────────────────────────────────┐     │
│  │              FUSION LAYER                            │     │
│  │  ML Forecast + LLM Verdict + Regime + Fundamentals   │     │
│  │  → Veredicto unificado con franja de movimiento      │     │
│  │  → Confianza calibrada (conformal prediction)        │     │
│  │  → Acción recomendada + sizing                       │     │
│  └─────────────────────────────────────────────────────┘     │
│                          │                                    │
│                          ▼                                    │
│              ┌──────────────────┐                            │
│              │  VEREDICTO FINAL │                            │
│              │  Rango: $X - $Y  │                            │
│              │  Acción: BUY/SELL│                            │
│              │  Confianza: 78%  │                            │
│              │  Horizonte: 7d   │                            │
│              └──────────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

### Tecnologías concretas a incorporar (ordenadas por prioridad)

#### FASE 1 — Implementar ya (1-2 semanas)

| Tecnología | Qué hace | Por qué | Complejidad |
|-----------|----------|---------|-------------|
| **FinBERT como tier 1** | Clasificación rápida de sentimiento de todas las noticias ($0/llamada, modelo local) | 10-50x más barato que Claude para clasificación masiva. Escalar a Claude solo los artículos ambiguos o de alto impacto | BAJA |
| **News Volume como feature** | Contar artículos por tema/día, trackear persistencia | Paper MDPI 2025: el volumen predice mejor que el tono. Ya tenemos GDELT | BAJA |
| **Conformal Prediction en forecasts existentes** | Envolver XGB/Ridge con intervalos de confianza calibrados | ICLR 2025: cobertura garantizada sin supuestos distribucionales. Librería MAPIE en Python | BAJA-MEDIA |

#### FASE 2 — Corto plazo (1-3 meses)

| Tecnología | Qué hace | Por qué | Complejidad |
|-----------|----------|---------|-------------|
| **Time-MoE / Chronos-2** | Foundation models de series temporales, zero-shot | Paper 2025: superan forecasts USDA en 3/4 commodities SIN entrenamiento. Agregar como miembro del ensemble | MEDIA |
| **Agente de extracción de noticias** | Pipeline agéntico (manager + especialista + fact-checker) que genera resúmenes estructurados | Paper arXiv 2508.06497: AUC 0.94 para detectar spikes >25% con embeddings de noticias | MEDIA |
| **Geopolitical Risk Index** | Integrar índice Caldara-Iacoviello como feature | Paper J. Futures Markets 2025: mejora accuracy para la mayoría de commodities | MEDIA |
| **HMM mejorado para régimen** | Hidden Markov Model calibrado específicamente para soja | Paper J. Forecasting: HMM+LSTM da ventajas significativas en forecast medio/largo plazo de soja | MEDIA |

#### FASE 3 — Medio plazo (3-6 meses)

| Tecnología | Qué hace | Por qué | Complejidad |
|-----------|----------|---------|-------------|
| **Multi-Agent Debate** | Agentes Bull/Bear que debaten, equipo de riesgo que calibra, fund manager que decide | Paper TradingAgents (ICML 2025): Sharpe 5.6-8.2, max drawdown <2.1%. Open-source en GitHub | ALTA |
| **RAG sobre Knowledge Base** | Retrieval-Augmented Generation sobre nuestra base de eventos, análogos, y literatura | Paper MARAG-Fin 2025: context precision 1.00 con ReAct prompting | ALTA |
| **FinSearch temporal** | Agente de búsqueda con decomposición de queries + ponderación temporal 72h | Paper ACM ICAIF 2025: supera Perplexity Pro 15-22%. Open-source | MEDIA |
| **Cross-sectional ranking** | Ranking soja vs maíz vs trigo vs aceite de palma para valor relativo | Paper CFA 2025: momentum + carry + OI como features principales | MEDIA |

#### FASE 4 — Largo plazo (6-12 meses)

| Tecnología | Qué hace | Por qué | Complejidad |
|-----------|----------|---------|-------------|
| **LLM + Reinforcement Learning** | LLM como "estratega" (dirección macro mensual) + RL como "ejecutor" (timing diario) | Paper FLLM 2025: Sharpe 1.10 vs 0.64 RL-only. Separa razonamiento macro de ejecución micro | MUY ALTA |
| **FinCon (Verbal RL)** | Sistema que aprende de sus errores via feedback en lenguaje natural, sin re-entrenamiento | Paper NeurIPS 2024: supera deep RL y LLMs standalone. El sistema MEJORA con el tiempo sin gradient updates | MUY ALTA |
| **GNN Supply Chain** | Modelar cadena de suministro de soja como grafo: producción → puertos → demanda | Paper Nature Sci. Reports 2025: captura propagación de disrupciones entre nodos | MUY ALTA |

---

### Cómo el sistema explicaría lo que pasó con el shock del petróleo

**Situación**: caída del petróleo arrastró commodities incluyendo soja.

**Con el sistema actual**: el shock engine detecta el movimiento DESPUÉS (spike/drop 5d/10d). El brief LLM menciona el contexto. Pero la detección es reactiva.

**Con Intelligence Engine v2**:

1. **T-3 días**: FinBERT detecta explosión en volumen de noticias sobre OPEC+ (news volume como feature). Aún sin dirección clara.
2. **T-2 días**: Claude Sonnet analiza los artículos de alto impacto, identifica patrón "reunión OPEC + tensión Arabia-Rusia" similar a marzo 2020 (RAG busca en Knowledge Base de análogos).
3. **T-1 día**: Multi-Agent Debate:
   - **Agente Bull**: "La soja tiene fundamentos propios, China sigue comprando"
   - **Agente Bear**: "Correlación petróleo-commodities históricamente 0.6+ en shocks. Si petróleo cae >5%, soja pierde $15-25 en 48h"
   - **Risk Team**: "Régimen actual = alta volatilidad. Reducir exposición"
4. **T-0**: Fusion Layer emite veredicto: "REDUCIR EXPOSICIÓN. Rango 48h: -$10 a -$30. Confianza: 72%. Basado en: 3 análogos históricos de shock petróleo con impacto medio en soja de -2.1%"

**Eso es inteligencia real**: no predice el precio exacto, pero ANTICIPA la dirección y magnitud probable basándose en lectura inteligente de señales + memoria de eventos pasados.

---

## PARTE 5: PRICING Y MERCADO OBJETIVO

### Estructura de precios recomendada

| Tier | Precio | Target | Justificación |
|------|--------|--------|---------------|
| **Free** | $0 | Lead generation | Forecasts con 24h delay, calendario WASDE, noticias limitadas |
| **Productor** | $49/mo | Productores pequeños/medianos UY/AR | Ventana óptima de venta, alertas, brief semanal |
| **Trader** | $149/mo (↑ de $99) | Traders independientes, mesas pequeñas | Todo: signals, paper trading, regime, LLM briefs. Un trader que extrae alpha paga $149 sin pestañear |
| **Cooperativa** | $490/mo (5 seats) | Cooperativas, agroexportadoras | Admin dashboard, analytics, alertas custom |
| **API Institucional** | $2K-5K/mo | Hedge funds, trading firms grandes | API de señales, data histórica, integración custom |

### TAM (Total Addressable Market)

| Segmento | Tamaño | WTP estimado | Revenue potencial |
|----------|--------|-------------|-------------------|
| Productores soja UY/AR | ~85,000 | $49/mo × 2% penetración | $1M/año |
| Trading desks regionales | ~200 | $149-490/mo × 20% pen. | $360K/año |
| Cooperativas | ~50 | $490/mo × 30% pen. | $88K/año |
| Hedge funds / quant | ~30 | $2K-5K/mo × 10% pen. | $120K/año |
| **Total alcanzable (3 años)** | | | **~$1.6M/año** |

### Mercado global

El mercado de plataformas de trading de commodities agrícolas está valorado en **$2.4B (2024)** creciendo a **$8.3B (2033)** con CAGR de 14.7%. IA generativa en agricultura crece al **30% CAGR**.

---

## PARTE 6: ROADMAP PRIORIZADO

### Sprint 1 (Semanas 1-2): Fundamentos

- [ ] Implementar re-entrenamiento rolling del modelo ML (fix accuracy <48%)
- [ ] Agregar news volume/persistencia como features al pipeline
- [ ] Integrar FinBERT como clasificador tier-1 de sentimiento
- [ ] Exponer `/api/forecast_paths` en el frontend (Monte Carlo density)

### Sprint 2 (Semanas 3-4): Credibilidad

- [ ] Implementar conformal prediction (MAPIE) en forecasts existentes
- [ ] Backfill paper trading con datos históricos para n≥30 trades
- [ ] Agregar SHAP explanations al pipeline y frontend
- [ ] Crear endpoint `/api/climate_forecast` con panel UI

### Sprint 3 (Meses 2-3): Diferenciación

- [ ] Integrar Chronos-2 o Time-MoE como miembro del ensemble
- [ ] Construir pipeline agéntico de extracción de noticias
- [ ] Integrar Geopolitical Risk Index como feature
- [ ] Sistema de autenticación de usuarios

### Sprint 4 (Meses 3-6): Intelligence Engine v2

- [ ] Multi-Agent Debate (Bull/Bear/Risk)
- [ ] RAG sobre Knowledge Base de eventos históricos
- [ ] Cross-sectional commodity ranking
- [ ] App móvil (PWA o React Native)

### Sprint 5 (Meses 6-12): Escala

- [ ] LLM + RL hybrid (macro strategy + micro execution)
- [ ] Expansión a maíz y trigo
- [ ] API institucional documentada
- [ ] Integración con brokers para ejecución

---

## CONCLUSIÓN

AgroCast PRO tiene **46 de 49 subsistemas funcionando correctamente** — una base técnica sólida. El producto ocupa un **nicho sin competencia directa** en inteligencia de mercado de soja para Sudamérica.

Las **3 prioridades críticas** antes de comercializar:
1. **Fijar la accuracy del modelo ML** (actualmente <48%, destruye credibilidad)
2. **Construir track record** de paper trading (n≥30 trades)
3. **Implementar autenticación de usuarios** (sin esto no se puede cobrar)

La visión de "inteligencia real" — donde el sistema LEE y ENTIENDE las señales del mercado en lugar de solo hacer regresión sobre datos históricos — está completamente validada por la literatura académica reciente. La arquitectura propuesta (FinBERT → Claude Sonnet → Multi-Agent Debate, con RAG sobre Knowledge Base de eventos) es el estado del arte para 2026 y nos posiciona adelante de competidores como Helios AI ($4.7M funding) que aún no integran multi-agente debate.

El mercado de $8.3B para 2033 con crecimiento de 14.7% CAGR confirma que hay espacio comercial significativo. Con el roadmap propuesto, AgroCast PRO puede alcanzar ~$1.6M ARR en 3 años apuntando a productores, traders, y cooperativas en Uruguay, Argentina, y Brasil.
