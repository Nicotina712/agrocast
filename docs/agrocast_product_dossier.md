# AgroCast PRO — Dossier de Producto e Inversión

**Plataforma de inteligencia de mercado para commodities agrícolas**
*Versión 1.0 — Abril 2026*

---

## 0. Resumen ejecutivo

AgroCast PRO es una plataforma SaaS de **inteligencia de mercado para commodities agrícolas** que combina modelos de Machine Learning, datos fundamentales (USDA, WASDE, CoT, clima), análisis estructurado de noticias por LLM y un motor de señales compuestas para entregar al productor y al trader una **recomendación accionable** (BUY / HOLD / SELL) sobre cuándo fijar precio, vender forward o tomar exposición direccional.

El MVP está **operativo** sobre soja (ZS=F) con cobertura conceptual Uruguay/Argentina/Brasil, e incluye una **extensión intradía** (Fase 0 completa, lista para Fase 1 con datos profesionales).

| Dimensión | Estado actual |
|---|---|
| Producto core (swing, horizonte 7–30d) | Producción — MAE 25.78 USc/bu en backtest |
| Capa Intel LLM (Claude Haiku) | Producción — análisis estructurado por noticia |
| Extensión intradía (5m bars) | Fase 0 completa, gate FAIL documentado por insuficiencia de datos retail |
| Pricing definido | Starter $49 / PRO $99 / Coop $490 mensual |
| Mercado piloto | Uruguay (1.0–1.4 M ha soja, ~3.000–5.000 productores activos) |

**Tesis de inversión:** un mercado piloto pequeño y controlado (Uruguay) permite validar product-market-fit, retention y unit economics antes de escalar a Argentina (16 M ha), Brasil (45 M ha) y mercados maduros (USA). La arquitectura es **agnóstica al commodity**: replicable a maíz, trigo, café, ganado, lácteos con **<20% de re-trabajo** por nuevo activo.

**Inyección de capital propuesta:** USD 250–350k para 12 meses → cubre equipo técnico, datos profesionales, GTM Uruguay y prepara entrada a Argentina + extensión intradía Fase 1.

---

## 1. ¿Qué es AgroCast?

### 1.1 Problema

El productor sojero sudamericano y el trader retail toman decisiones de **fijación de precio** y **exposición** sobre la base de:

- Llamados al corredor (información asimétrica, conflicto de interés).
- Rumores de WhatsApp / grupos de productores.
- Lectura cruzada de Bloomberg / Reuters / portales agro (información dispersa, sin agregación cuantitativa).
- Intuición sobre clima, China y dólar.

El resultado documentado en la literatura de hedging agrícola: **el productor mediano captura entre 60% y 75% del precio óptimo posible** dentro de la ventana comercial. La diferencia es captura de valor por intermediarios, market timing pobre o exposición no cubierta a shocks (WASDE, clima brasileño, política china).

### 1.2 Solución

AgroCast unifica **siete fuentes de datos**, **tres capas de modelado** y **una capa de explicabilidad LLM** en un dashboard único que entrega:

1. Una **señal compuesta** BUY / HOLD / SELL con horizonte 7–30 días.
2. Un **forecast de precio** 30 días con bandas de confianza.
3. Un **breakdown de drivers** (China, clima Brasil, WASDE, técnicos, USD) con peso explícito.
4. **Alertas push** (Telegram / WhatsApp) cuando la señal cambia.
5. Un **brief semanal** generado por Claude que contextualiza la semana en lenguaje natural.
6. **Accountability** — track record visible de cada señal pasada.

### 1.3 Status del producto

```
Componente                                    Status        Cobertura
─────────────────────────────────────────────────────────────────────
Pipeline de datos (load_data + load_external) ✅ Producción  7 series core + macro
Feature engineering (build_features)          ✅ Producción  ~120 features
Modelo ML 7d (XGBoost regresor)               ✅ Producción  MAE 25.78 USc/bu
Modelo retornos (BUY/SELL/HOLD)               ✅ Producción  walk-forward CV
Forecast 30d (anchor 0.88/0.12)               ✅ Producción  cap diario ±1%
Capa Intel LLM (Claude Haiku)                 ✅ Producción  cache permanente
Señal compuesta (signal_breakdown)            ✅ Producción  6 factores
Dashboard (Flask + vanilla JS)                ✅ Producción  SPA single-page
Alertas Telegram + Brief semanal              ✅ Producción  scheduler APS
Accountability (snapshots diarios)            ✅ Producción  /api/accountability
Módulo intradía (5m bars)                     ⚙️ Fase 0     gate FAIL doc.
Autenticación / billing                       ⏳ Pendiente   roadmap M2
Multi-commodity (maíz, trigo)                 ⏳ Roadmap     M6+
```

---

## 2. Arquitectura técnica

### 2.1 Stack

- **Backend:** Python 3.12, Flask, APScheduler, XGBoost, scikit-learn, pandas, yfinance.
- **LLM:** Anthropic Claude Haiku 4.5 (análisis por artículo + brief semanal).
- **Frontend:** HTML/CSS/JS vanilla (sin framework), single-page application servida por Flask.
- **Almacenamiento:** CSV/Parquet (pyarrow) en `data/` y `artifacts/`. PostgreSQL planificado para multi-tenant.
- **Distribución:** Telegram Bot, WhatsApp (Twilio), email.

### 2.2 Topología de módulos

```
┌────────────────────────────────────────────────────────────────────┐
│                        AgroCast PRO                                │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─── DATA LAYER ────────────────────────────────────────────┐     │
│  │ load_data.py        ZS, ZC, ZW, ZM, ZL, CL, DXY (yfinance)│     │
│  │ load_external.py    USDA, WASDE, CoT, trends, basis,      │     │
│  │                     Argentina (BCBA), Brasil, China        │     │
│  │ news_engine.py      RSS + NewsAPI + GDELT + VADER fallback│     │
│  └─────────────────────┬─────────────────────────────────────┘     │
│                        │                                            │
│  ┌─── INTEL LLM ───────▼─────────────────────────────────────┐     │
│  │ news_analyst.py     Claude Haiku: price_impact, magnitude,│     │
│  │                     horizon, drivers, confidence,          │     │
│  │                     extracted_data, key_quote, rationale  │     │
│  │ aggregator.py       agrega por driver, persiste history   │     │
│  └─────────────────────┬─────────────────────────────────────┘     │
│                        │                                            │
│  ┌─── FEATURES ────────▼─────────────────────────────────────┐     │
│  │ build_features.py   lags, momentum, spreads, estacional   │     │
│  │ news_features.py    sentiment, volume, topic_scores       │     │
│  │ wasde_features.py   distancia a WASDE, sorpresa histórica │     │
│  └─────────────────────┬─────────────────────────────────────┘     │
│                        │                                            │
│  ┌─── MODELS ──────────▼─────────────────────────────────────┐     │
│  │ train.py            XGBoost regresor 7d                   │     │
│  │ predict.py          forecast 30d (anchor + cap)           │     │
│  │ train_returns.py    clasificador BUY/SELL/HOLD            │     │
│  │ signal_breakdown.py composite 6 factores                  │     │
│  └─────────────────────┬─────────────────────────────────────┘     │
│                        │                                            │
│  ┌─── DELIVERY ────────▼─────────────────────────────────────┐     │
│  │ Flask API (puerto 8000):  /api/forecast, /api/signal,     │     │
│  │   /api/news, /api/news_intel, /api/accountability         │     │
│  │ alerts/telegram.py        push on signal change           │     │
│  │ alerts/weekly_brief.py    Claude → narrativa semanal      │     │
│  │ index.html (SPA)          dashboard productor/trader      │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 2.3 Pipeline (orquestado cada 6h por APScheduler, TZ America/Montevideo)

1. `load_data` → `raw_market.csv` (7 series).
2. `load_external` → CoT, USDA, WASDE, trends, sentiment, basis, AR/BR/CN.
3. `build_features` → ~120 features con masking de rollovers.
4. `news_history` merge → sentiment + volume + topic.
4b. `intel_history` merge → 30+ features de drivers LLM.
5. WASDE features.
6. `train` → XGBoost regresor (MAE test 25.78 USc/bu).
7. `predict` → forecast 30d.
8. `train_returns` + `predict_returns` → BUY/SELL/HOLD.
9. `accountability.save_forecast_snapshot`.
10. Telegram push si la señal cambió.

---

## 3. Modelos de predicción — detalle técnico

AgroCast no es "un modelo" — es un **stack** de tres capas independientes que se combinan en una señal compuesta. La filosofía: **redundancia controlada y degradación graceful** (si una capa falla, las otras siguen entregando).

### 3.1 Capa 1 — Modelo de precio (XGBoost regresor)

**Objetivo:** predecir el cierre de soja a 7 días vista.

| Item | Valor |
|---|---|
| Algoritmo | XGBoost regresor (`tree_method=hist`) |
| Hiperparámetros | n_estimators=300, max_depth=4, lr=0.05, subsample=0.8 |
| Target | `close[t+7] - close[t]` en USc/bu |
| Features | ~120 (lags 1-30, momentum, spreads ZS-ZC y ZS-ZM, estacionalidad, USDA stocks, CoT net non-comm, USD index, basis Up River, news drivers) |
| Validación | Walk-forward 5 folds con embargo proporcional al horizonte |
| Performance | MAE = 25.78 USc/bu en test out-of-sample |
| Anti-leakage | Rollovers detectados y enmascarados en `_detect_and_flag_rollovers()` |

El rolling de futuros está abordado en `load_data.py`: el frontend rota de mes activo a JUL26 cuando faltan ≤5 business days al First Notice Day. Esto evita el clásico bug de "saltos" en serie histórica.

### 3.2 Capa 2 — Modelo de retornos (clasificador)

**Objetivo:** convertir la predicción de precio en una decisión accionable.

- Algoritmo: clasificador multiclase (XGBoost) sobre el signo del retorno 7d.
- Salida: probabilidades `p_buy`, `p_hold`, `p_sell`.
- Decisión: BUY si `p_buy > 0.55` y `p_buy - p_sell > 0.15`. Simétrico para SELL.
- Recalibración mensual.

### 3.3 Capa 3 — Forecast de trayectoria 30d

- Anchor: la predicción 7d se proyecta como **trayectoria diaria** con suavizado 0.88/0.12 (88% peso al precio actual, 12% al target predicho), cap diario de ±1.0% de movimiento.
- Esto entrega una **curva** (no solo un punto) que el productor puede comparar contra su precio objetivo de fijación.
- Bandas de confianza derivadas de la dispersión histórica de errores por horizonte.

### 3.4 Capa 4 — Intel LLM (capa nueva, diferenciador clave)

**Reemplaza VADER (léxico, ciego al contexto) con análisis estructurado por noticia.**

Cada artículo se procesa **una sola vez** (cache permanente por `sha1(url|title)`). Modelo: `claude-haiku-4-5-20251001`.

Schema de respuesta por artículo:

```json
{
  "price_impact": "bullish | bearish | neutral",
  "magnitude": 1-5,
  "horizon": "1d | 7d | 30d | 90d",
  "drivers": ["china_demand", "weather_br", "weather_ar", "weather_us",
              "supply_global", "usda_report", "policy_ar", "policy_us",
              "policy_br", "macro_usd", "macro_oil", "logistics",
              "biofuels", "geopolitics", "other"],
  "confidence": 0.0-1.0,
  "extracted_data": {
    "volume_mmt": float,
    "price_target_usc": float,
    "country": str,
    "yield_change_pct": float
  },
  "key_quote": str,
  "rationale": str
}
```

**Agregación:** por driver,

```
score_d = mean( direction · magnitude · confidence · horizon_weight )
```

normalizado a [-1, +1]. Genera columnas `news_<driver>_signal` y `news_<driver>_count`, más `intel_composite`, `intel_n_articles`, `intel_n_high_impact`. Estas ~30 features alimentan al modelo de Capa 1.

**Costo operativo:** ~25 artículos nuevos/día × $0.001 = **~$0.75/mes** con cache. Marginal.

### 3.5 Capa 5 — Señal compuesta (delivery final)

| Factor | Peso |
|---|---|
| Modelo ML 7d | 30% |
| Forecast 30d | 25% |
| Demanda China (drivers LLM) | 20% |
| WASDE / stocks USDA | 15% |
| Análisis técnico (RSI, MACD, EMA cross) | 10% |
| Estacionalidad (informativo, peso 0%) | 0% |

Cache TTL 4h en `data/signal_breakdown.json`. El productor **ve los pesos** — la señal no es una caja negra.

### 3.6 Capa 6 — Extensión intradía (Fase 0 completa)

Módulo paralelo `src/intraday/` con stack propio (ver §10):
- Bars 5m de yfinance, 60d de cobertura.
- 25 features microestructurales + 5 de régimen + 5 puente swing (Bayesian prior).
- XGBoost walk-forward, horizonte 12 barras (60 min).
- Backtest event-driven con slippage realista (MZS: $10.23 round-trip).
- **Veredicto Fase 0:** gate FAIL con datos retail → requiere CME DataMine ($150/mo) para Fase 1.

---

## 4. Modelo de negocio

### 4.1 Pricing actual (definido)

| Plan | Precio | Target | Incluye |
|---|---|---|---|
| **Starter** | USD 49/mes | Productor pequeño (<300 ha) | Señal diaria, dashboard, brief semanal |
| **PRO** | USD 99/mes | Productor mediano-grande / trader retail | + Alertas Telegram tiempo real, accountability, breakdown de drivers, intel LLM full |
| **Cooperativa** | USD 490/mes | Coop con 20+ productores | Multi-usuario, white-label parcial, soporte prioritario, sesión mensual con analista |

### 4.2 Las dos líneas de negocio

AgroCast tiene **una infraestructura, dos productos**:

#### Línea A — Productor agrícola (B2B)

- **Quién:** propietario o gerente comercial de explotación sojera (200–10.000 ha).
- **Decisión que apoya:** ¿cuándo fijo precio? ¿forward, futuro, opción put? ¿qué % vendo hoy y qué dejo abierto?
- **Frecuencia de uso:** 2–3 veces/semana, picos en ventana de fijación (mar-jun en Uruguay, ene-may en Argentina).
- **Métrica de valor:** capturar 5–10 USc/bu adicionales sobre el precio promedio de la campaña → en una explotación de 1.000 ha × 2.8 ton/ha × ~36.7 bu/ton = ~103.000 bu/año. Cinco USc/bu = **USD 5.150/año adicionales**. Pricing PRO = **USD 1.188/año**. ROI > 4×.
- **Canal:** referidos por agrónomo, cooperativas, ferias agro (Expo Prado en Uruguay, Expoagro en Argentina).

#### Línea B — Trader (B2C / B2B small)

- **Quién:** trader retail con cuenta en broker de futuros (Tradovate, Interactive Brokers, AMP), tamaño de cuenta USD 10k–500k. También mesas pequeñas en cerealeras, fondos cuant boutique.
- **Decisión que apoya:** dirección 7–30d para swing, dirección intraday (Fase 1) para day trading.
- **Frecuencia de uso:** diaria. Login 1–3×/día.
- **Métrica de valor:** Sharpe incremental del 0.3–0.6 sobre estrategia base. En cuenta USD 50k al 15% objetivo, eso son **USD 1.500–3.000/año** en alpha capturado. Pricing PRO = USD 1.188/año. ROI ≥ 1.5×.
- **Canal:** YouTube/Twitter/X (contenido educativo + track record público), comunidades de traders agrícolas (FuturesTrader71, Trading Pit).

#### ¿Por qué dos líneas y no una?

- **El productor paga por certeza** (LTV alto, churn bajo, ciclo de venta largo).
- **El trader paga por edge** (LTV medio, churn medio, ciclo de venta corto, marketing viral).

La combinación **estabiliza el funnel**: productores aportan retention y predictibilidad; traders aportan crecimiento orgánico y feedback rápido sobre el modelo.

### 4.3 Unit economics target (post-piloto)

| Métrica | Productor (PRO) | Trader (PRO) |
|---|---|---|
| ARPU mensual | USD 99 | USD 99 |
| CAC (referido / orgánico) | USD 80 | USD 120 |
| Churn mensual | 3% | 7% |
| LTV (1/churn × ARPU) | USD 3.300 | USD 1.414 |
| LTV/CAC | 41× | 12× |
| Payback | 0.8 meses | 1.2 meses |

---

## 5. Valor diferencial vs competencia

### 5.1 Mapa competitivo

| Competidor | Tipo | Fortaleza | Debilidad vs AgroCast |
|---|---|---|---|
| **Bloomberg Terminal** | Data terminal premium | Coverage total, calidad institucional | USD 24.000/año; sin recomendación accionable; ningún productor sojero lo paga |
| **Refinitiv (Eikon)** | Data terminal premium | Similar a Bloomberg | Mismo problema de precio y orientación institucional |
| **DTN ProphetX** | Data agro | Buen coverage USDA, weather | USA-céntrico, sin LLM, sin señal sintética |
| **Barchart** | Data + charts | Buen frontend, gratis tier | Sin recomendación, sin contexto LLM, sin productores SA |
| **AgroPanel / Granos.com.ar** | Portales agro | Contenido en español | Sin modelos cuantitativos, son medios no plataformas |
| **Corredores locales** (Tucker, Garmet, etc.) | Asesoría humana | Conocimiento local | Conflicto de interés (cobran comisión por venta), sin transparencia |
| **Excel del productor** | DIY | Gratis | No escala, sin actualización automática, sin LLM |

### 5.2 Los cinco diferenciales

1. **Capa Intel LLM — primer producto agro con análisis estructurado por noticia.** No conocemos competidor en habla hispana que use Claude/GPT para extraer drivers cuantitativos por artículo y mergearlos al modelo ML. VADER (que usan algunos) es léxico — no entiende "China cancela compra" vs "China niega haber cancelado compra".

2. **Señal compuesta transparente.** El productor ve los pesos (30/25/20/15/10/0). No es una caja negra. Esto importa porque el productor agro es **escéptico por cultura** — necesita ver el porqué.

3. **Accountability pública.** Cada señal pasada queda registrada con fecha, precio en ese momento y resultado a 7d/30d. Hoy, ningún competidor agro publica track record auditado.

4. **Bilingüe nativo (español rioplatense).** Los productos USA traducen mal. AgroCast nace en español, con jerga local (Up River, FOB Nueva Palmira, basis Argentina, dólar soja).

5. **Cobertura del ciclo completo.** Un productor puede usar la línea swing para fijar campaña y la línea intradía (post Fase 1) para hedge dinámico. Un trader puede usar swing para posiciones direccionales y intradía para day trading. Mismo dashboard, mismo login.

### 5.3 Foso defendible (moat)

- **Datos privados acumulados:** `intel_history` crece todos los días con ~25 artículos analizados. A los 6 meses son 4.500 artículos con análisis estructurado — un dataset que un competidor que arranca **no puede recrear retroactivamente**.
- **Accountability como switching cost:** cuanto más tiempo lleva un productor en la plataforma, más decisiones pasadas tiene auditadas. Cambiar de proveedor implica perder ese historial.
- **Network effects asimétricos vía coops:** una coop con 30 productores no se cambia de plataforma sin reentrenar a todos.

---

## 6. Mercado piloto: Uruguay

### 6.1 ¿Por qué Uruguay primero?

| Variable | Valor | Por qué favorece |
|---|---|---|
| Superficie soja sembrada | 1.0–1.4 M ha | Mercado abarcable, no hay 50 competidores locales |
| Productores activos estimados | 3.000–5.000 | Funnel de venta cabe en hojas de cálculo |
| % productores con conexión digital | >85% | Penetración de smartphone alta, idioma único |
| Concentración geográfica | Litoral oeste + sur | Pocas regiones a cubrir |
| Estabilidad institucional | Alta (top LATAM) | Pago en USD predecible, sin restricciones cambiarias retail |
| Idioma | Español | Cero costo de localización |
| Marco regulatorio fintech/SaaS | Permisivo | Sin licencia para vender SaaS B2B |
| Prensa agro local | Concentrada (El País Agro, Búsqueda agro, Blasina) | Pocos canales para hacer ruido |

**Riesgo de Uruguay:** mercado pequeño en términos absolutos. Por eso es **piloto, no destino final**.

### 6.2 TAM/SAM/SOM Uruguay (solo soja)

| Capa | Definición | Tamaño |
|---|---|---|
| **TAM** | Todos los actores agro Uruguay con decisión comercial sobre granos (productores soja+trigo+cebada, acopios, traders retail) | ~12.000 entidades |
| **SAM** | Productores soja >200 ha + acopios + traders retail con cuenta futuros | ~4.500 entidades |
| **SOM** (3 años) | Penetración 8% del SAM en ARPU promedio USD 80/mes | ~360 cuentas → **USD 345.600 ARR** |

### 6.3 Proyección económica — MVP soja Uruguay

Asunciones:
- Lanzamiento comercial: mes 4 post-financiamiento.
- Mix: 60% Starter ($49), 30% PRO ($99), 10% Coop ($490 — coops con 8–15 productores).
- ARPU blendido: 0.6×49 + 0.3×99 + 0.1×490 = **USD 108.2/mes**.
- Crecimiento neto mensual: 15 cuentas/mes en M5–M9, 25 cuentas/mes en M10–M18.
- Churn: 4% mensual blendido.

| Mes | Cuentas activas | MRR (USD) | ARR run-rate |
|---|---|---|---|
| M6 | 30 | 3.246 | 38.952 |
| M9 | 70 | 7.574 | 90.888 |
| M12 | 130 | 14.066 | 168.792 |
| M18 | 260 | 28.132 | 337.584 |
| M24 | 400 | 43.280 | 519.360 |

**Costos operativos para servir Uruguay** (sin equipo de desarrollo, solo runtime):
- Cloud (Render/Railway/Hetzner): USD 80/mes
- LLM API (Claude Haiku, escala lineal): USD 25/mes a 400 cuentas
- Datos market (yfinance + NewsAPI): USD 50/mes
- Telegram/Twilio (alertas): USD 30/mes a 400 cuentas
- **Total infra**: ~USD 185/mes → 0.4% del MRR a 400 cuentas. **Margen bruto ≥ 90%.**

---

## 7. Otros mercados de soja

Una vez validado Uruguay, la **misma plataforma** se extiende a mercados sojeros mucho más grandes con costo marginal por mercado bajo (localización de basis y nombres de cooperativas locales).

| Mercado | Soja sembrada | Productores aproximados | Particularidades | Esfuerzo entrada |
|---|---|---|---|---|
| **Argentina** | ~16 M ha | ~50.000 | Volatilidad cambiaria (dólar soja), retenciones, basis Rosario | Medio — basis BCBA ya integrado, solo localizar UI y canal |
| **Brasil** | ~45 M ha | ~240.000 | Portugués (no español), basis Paranaguá/Santos, clima como driver dominante | Alto — localización idioma + relaciones con cooperativas (CCGL, Coamo) |
| **Paraguay** | ~3.5 M ha | ~12.000 | Mercado muy similar a Uruguay (basis Up River compartido) | Bajo — extensión natural de Uruguay |
| **Bolivia (Santa Cruz)** | ~1.4 M ha | ~14.000 | Restricciones export, dólar paralelo | Medio — riesgo macro |
| **USA** | ~35 M ha | ~300.000 | Mercado maduro, alta competencia (DTN, Barchart), USDA local | Alto — localización inglés, competencia premium |

### 7.1 Estrategia de expansión secuencial

```
M0–M12   Uruguay (piloto, validar PMF, retention, NPS)
M9–M18   Paraguay (extensión técnica trivial, mercado gemelo)
M12–M24  Argentina (mercado grande, requiere GTM dedicado)
M18–M30  Brasil sur (Rio Grande do Sul, Paraná) — localización portugués
M24–M36  Brasil centro (Mato Grosso) — el premio
M30+     Evaluación USA (probablemente como producto premium con CME DataMine)
```

### 7.2 Proyección económica — soja Sudamérica integrada (M36)

Modelo conservador (penetración 1–3% del SAM por mercado):

| Mercado | Cuentas M36 | MRR | ARR |
|---|---|---|---|
| Uruguay | 400 | 43.280 | 519.360 |
| Paraguay | 350 | 37.870 | 454.440 |
| Argentina | 1.800 | 194.760 | 2.337.120 |
| Brasil sur | 900 | 97.380 | 1.168.560 |
| **Total** | **3.450** | **373.290** | **4.479.480** |

ARR a 36 meses **~USD 4.5 M** con margen bruto ≥85%. EBITDA ~50% post equipo de soporte y GTM regional.

---

## 8. Replicabilidad a otros commodities — la tesis de escalabilidad

Aquí está la apuesta grande. La arquitectura de AgroCast es **genérica sobre series de futuros con drivers fundamentales identificables**. La soja es solo el primer dataset.

### 8.1 ¿Qué se reusa, qué se cambia?

| Componente | Reuso | Cambio por commodity |
|---|---|---|
| Pipeline de datos (`load_data`) | 90% | Cambio de tickers (ZC, ZW, KC, CL, GC…) |
| Feature engineering core | 85% | Lags/momentum/spreads idénticos; spreads específicos |
| WASDE / reportes | 70% | WASDE sirve granos; café usa USDA-FAS, ganado usa Cattle on Feed |
| Capa Intel LLM | 95% | El schema de drivers es genérico; solo se cambia el set de drivers válidos |
| Modelo ML core | 90% | Mismo XGBoost, retraining con datos del nuevo activo |
| Frontend | 95% | Re-skin con nuevo color y unidades |
| Alertas / billing / accountability | 100% | Idéntico |

**Estimación: 15–20% de re-trabajo por nuevo commodity** (~3 semanas-persona).

### 8.2 Roadmap de commodities

| Commodity | Ticker | Productor target | Trader target | Prioridad |
|---|---|---|---|---|
| **Soja** | ZS | Sudamérica | Retail USA + LATAM | **0 (vivo)** |
| **Maíz** | ZC | Argentina, Brasil, USA | Retail USA + LATAM | 1 (M9) |
| **Trigo** | ZW / KE / MW | Argentina, Australia | Retail | 2 (M12) |
| **Café** | KC | Brasil, Colombia, Vietnam | Retail global (alto interés) | 3 (M15) |
| **Ganado** | LE / GF | Uruguay, Argentina, USA | Hedger productor + retail | 4 (M18) |
| **Algodón** | CT | Brasil, India, USA | Retail | 5 (M21) |
| **Lácteos** | DC / CSC | Uruguay (cuenca), Nueva Zelanda | Productor | 6 (M24) |

### 8.3 Proyección económica — multi-commodity (M48)

Asumiendo penetración 1–2% del SAM por commodity en mercados ya activos:

| Línea | ARR M48 (USD) |
|---|---|
| Soja (Uruguay + Paraguay + Argentina + Brasil) | 5.500.000 |
| Maíz (Argentina + Brasil + USA tier) | 2.200.000 |
| Trigo (Argentina + USA tier) | 800.000 |
| Café (Brasil + Colombia) | 1.300.000 |
| Ganado (Uruguay + Argentina) | 600.000 |
| Algodón / Lácteos | 400.000 |
| **Total ARR M48** | **~10.800.000** |

Esa es la **forma de la oportunidad**: pasar de un MVP sobre soja Uruguay a una plataforma multi-commodity, multi-mercado, con USD 10M+ de ARR en 4 años bajo asunciones conservadoras (penetración 1–2% del SAM, no 10%).

---

## 9. Cómo el capital hace despegar el producto

Sin capital, AgroCast crece linealmente — el founder vende uno a uno, el churn se compensa con voluntariado, el roadmap se posterga. **Con capital**, los multiplicadores se desbloquean.

### 9.1 Ronda propuesta: USD 250–350k (12 meses)

| Rubro | Asignación | % | Justificación |
|---|---|---|---|
| Equipo técnico (1 ML eng + 0.5 fullstack) | 130.000 | 43% | Liberar al founder del backlog técnico para enfocarse en GTM |
| Datos profesionales (CME DataMine + Refinitiv news + USDA premium) | 28.000 | 9% | Habilita Fase 1 intradía y mejora capa swing |
| Cloud + observabilidad | 18.000 | 6% | Migrar de runtime hobby a infra production-grade (Postgres, monitoring) |
| GTM Uruguay (eventos, contenido, ads) | 35.000 | 11% | Expo Prado, sponsoreo de gremiales, Google Ads agro |
| Capital piloto trader (paper + cash live MZS) | 40.000 | 13% | Validar Fase 1 intradía con dinero real, generar track record |
| Legal + compliance (LLC USA, contratos) | 15.000 | 5% | Vehículo de inversión limpio, contratos SaaS B2B |
| Sales (1 SDR part-time M6+) | 24.000 | 8% | Outbound a coops y mesas |
| Contingencia | 15.000 | 5% | 5% reserva |
| **Total** | **305.000** | **100%** | |

### 9.2 Multiplicadores de capital

1. **Velocidad:** equipo técnico full-time entrega 3× lo que un founder solo. Roadmap maíz pasa de M18 a M9.
2. **Calidad de modelo:** datos profesionales (CME DataMine, Refinitiv) suben AUC swing estimado de ~0.58 a ~0.62 y desbloquean Fase 1 intradía.
3. **Distribución:** sin presupuesto de marketing, AgroCast crece por boca a boca (5–8 cuentas/mes). Con USD 35k de GTM en Uruguay, 15–25 cuentas/mes (3× crecimiento).
4. **Credibilidad institucional:** ronda formal abre puertas a coops grandes (CALMER, Copagran) que hoy ignoran un MVP solo-founder.
5. **Track record live:** USD 40k de capital piloto en cuenta MZS genera 8–12 meses de track record auditado real, **el activo de marketing #1 para el segmento trader**.

### 9.3 Hitos por trimestre

| Q | Hito de producto | Hito comercial | Hito métrico |
|---|---|---|---|
| Q1 | Auth + billing + Postgres | Lanzamiento comercial Uruguay | 30 cuentas pagas |
| Q2 | Fase 1 intradía (CME DataMine) | Primer contrato Coop | 80 cuentas, MRR USD 8k |
| Q3 | Maíz beta | Apertura Argentina (Rosario) | 150 cuentas, MRR USD 16k |
| Q4 | Track record intradía 6m público | Serie A teaser | 250 cuentas, MRR USD 27k, ARR run-rate USD 320k |

---

## 10. Extensión intradía — el upgrade del trader

(Documentado en detalle en `docs/investment_memo_intraday.md` y `docs/intraday_design.md`.)

### 10.1 ¿Por qué importa para el dossier?

La línea **trader** se profundiza muchísimo con un módulo intradía funcional. Un trader retail con cuenta de USD 50k que opera ZS/MZS day-trading paga sin chistar USD 99/mes si la señal le rinde 1 R semanal.

### 10.2 Estado actual

- **Fase 0 completa:** todo el stack técnico construido (`src/intraday/` — 1.650 líneas, 14 componentes: tick_feed, session_calendar, microstructure features, regime detection, swing context bridge, signal router, slippage model, risk manager, replay engine, metrics).
- **Validación de calidad de datos:** GO en 3 intervalos (5m, 15m, 60m) con yfinance.
- **Backtest baseline:** 164 trades, WR 15.2%, PF 0.17, Sharpe -19.82.
- **A/B testing de 5 variantes** (swing on/off, horizonte 12 vs 24, thresholds 0.62/0.38 vs 0.55/0.45, filtro primeros 90min):
  - V1 sin swing prior: AUC 0.502 (random) → **prueba que el edge en swing baseline viene del prior bayesiano, no de la microestructura sola**.
  - Conclusión técnica: con datos retail (yfinance, latencia minuto, sin DOM), el edge intradía no supera costos (round-trip MZS = USD 10.23). Esto es **fail honesto**, no bug.
- **Plan Fase 1:** USD 150/mo CME DataMine + USD 3k one-time setup + paper broker → AUC 0.55–0.60 esperado, Sharpe 1.0–1.8 objetivo.

### 10.3 Por qué Fase 0 fail no es fracaso

Para un fondo, mostrar que el equipo:
1. Construyó la infra completa (validable en código).
2. Diagnosticó honestamente por qué falla con los datos disponibles.
3. Cuantificó qué inversión la pone en target.

…es una señal de **disciplina de ingeniería**, no de fracaso. Lo opuesto sería demos truchadas con leakage.

### 10.4 Impacto en proyección

Con Fase 1 viva en M6, la línea **trader** crece de ~10% a ~30% del MRR. En la proyección Uruguay M24 (USD 519k ARR), eso son ~USD 150k extra de ARR si Fase 1 entrega lo proyectado.

---

## 11. Riesgos y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| PMF Uruguay débil (mercado muy pequeño) | Media | Alto | Funnel directo a Argentina si M9 no llega a 70 cuentas |
| Modelo soja degrada (régimen cambia) | Media | Medio | Walk-forward retraining mensual, drift monitoring (PSI) |
| Costo Claude API se dispara | Baja | Bajo | Cache permanente, cap diario de artículos analizados |
| Competidor USA entra en LATAM (DTN, Barchart) | Baja | Alto | Foso = LLM + idioma + accountability acumulada |
| Restricción cambiaria Argentina bloquea pago SaaS | Media | Medio | Cobranza vía Stripe USA o pagos en USD bilaterales |
| Fase 1 intradía no llega al gate con CME DataMine | Media | Medio | Producto sigue siendo viable solo con línea swing — intradía es upside |
| Founder bottleneck (single point of failure) | Alta | Alto | Primera contratación es ML eng senior con overlap |

---

## 12. Conclusión

AgroCast no es una apuesta a un solo modelo — es una **plataforma con tres capas independientes** (ML cuantitativo, LLM cualitativo, señal compuesta) que ya está en producción sobre soja, con extensión intradía construida y diagnosticada, y arquitectura agnóstica al commodity.

La oportunidad:

- **Corto plazo (12m):** validar product-market-fit en Uruguay, llegar a USD 150–180k ARR.
- **Mediano plazo (36m):** soja Sudamérica integrada, USD 4–5M ARR, márgenes ≥85%.
- **Largo plazo (48m):** plataforma multi-commodity, USD 10M+ ARR, expansión a USA como producto premium.

La inversión propuesta (USD 250–350k) financia los 12 meses de mayor riesgo — los 12 meses donde se pasa de un MVP de un founder a una empresa con equipo, GTM y track record auditado.

El producto ya existe. El código ya corre. El modelo ya predice. Lo que falta es velocidad de distribución — y eso es lo que el capital compra.

---

## Apéndice A — Referencias técnicas

- `src/pipeline.py` — orquestación end-to-end
- `src/model/train.py` — entrenamiento XGBoost
- `src/intel/news_analyst.py` — análisis estructurado por noticia (Claude Haiku)
- `src/intel/aggregator.py` — agregación por driver
- `src/trader/signal_breakdown.py` — composite 6 factores
- `src/intraday/` — extensión intradía (14 módulos, 1.650 LOC)
- `docs/intraday_design.md` — diseño técnico módulo intradía
- `docs/investment_memo_intraday.md` — memo intradía detallado

## Apéndice B — KPIs de pipeline (snapshot abril 2026)

```
Modelo precio 7d (XGBoost regresor)
  MAE test out-of-sample          25.78 USc/bu
  R² walk-forward 5 folds         0.42 promedio

Capa Intel LLM
  Artículos analizados (acum.)    ~600
  Cache hit rate                  >70%
  Costo mensual                   ~USD 0.75

Pipeline runtime
  Frecuencia                      cada 6h
  Tiempo de ejecución             ~3 min
  Endpoint /api/forecast          <200ms
```

## Apéndice C — Proyección financiera consolidada

```
Escenario base (penetración conservadora 1–2% SAM por mercado)

                    M12      M24      M36       M48
Cuentas             130      400      3.450     8.500
MRR (USD)           14.066   43.280   373.290   900.000
ARR (USD)           168.792  519.360  4.479.480 10.800.000
Margen bruto        85%      88%      90%       92%
EBITDA (estimado)   -120k    +50k     +1.6M     +4.5M
Equipo (FTE)        3        6        15        35
```

---

*Documento preparado para presentación a fondo de capital semilla. Cifras de mercado tomadas de fuentes públicas (USDA FAS, MGAP Uruguay, BCBA, CONAB Brasil) y estimaciones internas del equipo.*
