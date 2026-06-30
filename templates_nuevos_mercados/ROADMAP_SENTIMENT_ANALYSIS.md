# ROADMAP — Análisis de Sentimiento de Noticias por Instrumento

> **Objetivo:** Agregar un tercer agente (Sentiment Agent) o enriquecer el contexto
> fundamental de cada robot para detectar shocks macro/micro que el análisis técnico
> puro no puede capturar a tiempo.
>
> **Estado actual (mayo 2026):** La arquitectura de 2 agentes (Trend + Risk) trabaja
> únicamente con precios/indicadores técnicos y un contexto fundamental hardcodeado
> (`_build_fundamental_context()`). Las noticias de alto impacto —hackeos, decisiones
> del Fed, datos de empleo, recortes de OPEC— generan shocks que los patrones de
> precio no anticipan.
>
> **Prioridad:** Implementar después de terminar TODOS los robots individuales y el
> Portfolio Orchestrator. Luego iterar instrumento por instrumento.

---

## Arquitectura propuesta

```
┌─────────────────────────────────────────────────────────┐
│                     live_runner.py                      │
│                                                         │
│  1. fetch_mt5_bars()    ← datos técnicos (actual)       │
│  2. build_features()    ← microstructure (actual)       │
│  3. [NUEVO] fetch_sentiment()  ← noticias en tiempo real│
│  4. call_trend_agent(summary, fund_ctx, sentiment_ctx)  │
│  5. call_risk_agent(summary, trend, fund_ctx, sent_ctx) │
│  6. synthesize_signal()                                 │
└─────────────────────────────────────────────────────────┘
```

### Opción A (Recomendada — mínimo esfuerzo): Enriquecer el contexto
- Agregar `fetch_sentiment()` en `live_runner.py` → retorna dict
- Pasar `sentiment_ctx` como argumento extra a `call_trend_agent()`
- El Trend Agent ya tiene instrucciones para interpretar noticias
- **Sin modificar la arquitectura de agentes**

### Opción B (Más poderosa): Tercer agente dedicado
- Nuevo `call_sentiment_agent(news_headlines, instrument)` en `agents.py`
- Retorna: `{"sentiment": "BULLISH"|"BEARISH"|"NEUTRAL", "shock_detected": bool, "shock_type": "...", "confidence": "HIGH"|"MEDIUM"|"LOW", "reasoning": "..."}`
- `synthesize_signal()` veta la señal si `shock_detected=True` y `confidence=HIGH`
- Costo: +2 LLM calls por ciclo (aumentar `MAX_LLM_CALLS_PER_DAY`)

---

## Por instrumento: Fuentes y Shocks a detectar

---

### 🟠 BTCUSD — Bitcoin

**Shocks que el técnico no captura:**
- Aprobación/rechazo de ETFs (SEC)
- Hackeos de exchanges importantes (Mt.Gox 2.0, etc.)
- Regulación por país (China ban, US crypto bill)
- Movimiento de ballenas on-chain (>1,000 BTC)
- Halving (ya sabido pero impacto gradual)
- Quiebras de custodios (FTX-style)
- Datos macro: CPI, NFP, FOMC (correlación inversa con DXY)

**Fuentes de datos:**
| Fuente | Endpoint | Gratuito |
|--------|----------|----------|
| CryptoPanic API | `https://cryptopanic.com/api/v1/posts/?currencies=BTC` | ✅ (free tier) |
| Alternative.me Fear & Greed | `https://api.alternative.me/fng/` | ✅ siempre |
| Glassnode (on-chain) | whale alerts, exchange inflows | ⚠ de pago |
| CoinMetrics | flujos on-chain | ⚠ de pago |
| NewsAPI | `q=Bitcoin&language=en` | ✅ dev key |
| Whale Alert Telegram | bot / API | ✅ free tier |

**Señales de veto automático sugeridas:**
- Fear & Greed < 15 (Extreme Fear) → no abrir LONG
- Fear & Greed > 85 (Extreme Greed) → no abrir SHORT
- Headline con "hack", "exploit", "bankrupt", "ban" → veto inmediato si HIGH confidence

---

### 🔷 ETHUSD — Ethereum

**Shocks que el técnico no captura:**
- Exploits de protocolos DeFi (Aave, Uniswap, bridges)
- Ataques a bridges cross-chain (Ronin, Wormhole precedentes)
- Hard forks o upgrades inesperados (retrasos, bugs en testnets)
- ETH/BTC ratio breakdown súbito (alt-season o BTC dominance)
- Regulación de staking (SEC vs staking como security)
- ETF de ETH: flujos diarios ETHA/FETH
- Gas fees spike = congestión = demanda (corto plazo bullish)
- L2 exploits que perjudican confianza en el ecosistema

**Fuentes de datos:**
| Fuente | Endpoint | Gratuito |
|--------|----------|----------|
| CryptoPanic API | `?currencies=ETH` | ✅ |
| DeFiLlama TVL | `https://api.llama.fi/tvl/ethereum` | ✅ siempre |
| Etherscan Gas | `https://api.etherscan.io/api?module=gastracker` | ✅ (API key) |
| Alternative.me Fear & Greed | mismo índice que BTC | ✅ |
| L2Beat | TVL de L2s | ✅ (scraping) |
| Rekt.news RSS | alertas de hacks DeFi | ✅ |

**Señales de veto automático sugeridas:**
- DeFi TVL caída >10% en 24h → no abrir LONG
- "exploit" o "hack" en noticias ETH últimas 2h → veto
- ETH/BTC ratio cayendo >3% en sesión → favorecer SHORT

---

### 🥇 XAUUSD — Oro

**Shocks que el técnico no captura:**
- FOMC: decisiones de tasas (el mayor driver del oro)
- NFP: datos de empleo (viernes 8:30 ET)
- CPI/PPI: inflación (gold como hedge)
- Conflictos geopolíticos (safe haven bid)
- Compras de bancos centrales (China, Rusia, India)
- DXY spike/crash (correlación inversa fuerte)
- Yields del T10 (correlación inversa con oro)

**Fuentes de datos:**
| Fuente | Endpoint | Gratuito |
|--------|----------|----------|
| Investing.com Economic Calendar | scraping/API | ⚠ scraping |
| Alpha Vantage News | `?function=NEWS_SENTIMENT&tickers=GLD` | ✅ API key |
| FRED (Federal Reserve) | datos macro con delay | ✅ siempre |
| ForexFactory Calendar | scraping | ✅ |
| NewsAPI | `q=gold+Federal+Reserve` | ✅ dev key |
| Benzinga Pro | noticias en tiempo real | ⚠ de pago |

**Señales de veto automático sugeridas:**
- FOMC day (±2h de la decisión) → reducir tamaño a 50% o flat
- NFP Friday 8:00-9:30 ET → no abrir nuevas posiciones
- DXY >0.5% en 1h → veto a LONG oro

---

### 🛢 WTI_N6 / BRENT_N6 — Petróleo

**Shocks que el técnico no captura:**
- EIA Weekly Petroleum Status Report (miércoles 10:30 ET)
- API Weekly Statistical Bulletin (martes ~16:30 ET)
- Reuniones OPEC+ y recortes de producción
- Huracanes en el Golfo de México (supply disruption)
- Tensiones geopolíticas en Medio Oriente
- SPR releases (Strategic Petroleum Reserve)
- Datos económicos de China (mayor demandante)

**Fuentes de datos:**
| Fuente | Endpoint | Gratuito |
|--------|----------|----------|
| EIA API | `https://api.eia.gov/v2/petroleum/` | ✅ API key |
| Alpha Vantage News | `?tickers=USO,OIL` | ✅ API key |
| NewsAPI | `q=OPEC+oil+crude` | ✅ dev key |
| ForexFactory | calendario EIA/API | ✅ scraping |

**Señales de veto automático sugeridas:**
- EIA inventory release en próximas 2h → no abrir nuevas posiciones
- OPEC meeting day → solo scalping, sin swings
- Inventarios >+5M barriles (bearish sorpresa) → veto LONG

---

### 📈 US500 — S&P 500

**Shocks que el técnico no captura:**
- Earnings de FAANG/Mag7 (Apple, NVDA, Meta, Alphabet, Amazon, MSFT, Tesla)
- FOMC / Powell speeches
- NFP / CPI / PCE datos macro
- Geopolítica: guerras, tensiones China-Taiwan
- Bank stress (SVB 2023 style)
- VIX spike > 25 (fear level)
- Circuit breakers / trading halts

**Fuentes de datos:**
| Fuente | Endpoint | Gratuito |
|--------|----------|----------|
| Alpha Vantage News | `?tickers=SPY,QQQ` | ✅ API key |
| FRED | datos macro con delay | ✅ |
| Benzinga | earnings calendar | ⚠ de pago |
| Yahoo Finance | `?=^VIX` para VIX en tiempo real | ✅ scraping |
| NewsAPI | `q=S&P500+Federal+Reserve+earnings` | ✅ dev key |

**Señales de veto automático sugeridas:**
- VIX > 25 → reducir tamaño, no SHORT en panic
- FOMC day → flat antes de decisión
- Earnings de cualquier Mag7 en próximas 4h → no abrir posición

---

### 💻 USTEC — Nasdaq 100

**Shocks similares a US500 + tech-específicos:**
- NVDA earnings (mayor peso en USTEC actualmente)
- AI news: OpenAI, Anthropic, Google Gemini releases
- Regulación tech: antitrust, privacidad
- Tasas de interés (tech growth stocks muy sensibles a yields)
- Chip stocks: TSMC earnings, Samsung, Intel

**Fuentes de datos:** Mismos que US500, filtrar por tech tickers.

---

### 🇬🇧 UK100 — FTSE 100

**Shocks que el técnico no captura:**
- Bank of England decisiones de tasas (8 veces/año)
- UK CPI / GDP / PMI datos
- GBP/USD movimientos fuertes (FTSE correlaciona inversamente con GBP)
- Brexit aftermath: acuerdos comerciales
- Energía: FTSE pesado en oil companies (BP, Shell)

**Fuentes de datos:**
| Fuente | Endpoint | Gratuito |
|--------|----------|----------|
| Alpha Vantage News | `?tickers=EWU` (UK ETF) | ✅ |
| BBC Business RSS | noticias UK | ✅ |
| NewsAPI | `q=FTSE+Bank+of+England+UK` | ✅ dev key |

---

## Plan de implementación

### Fase 1 — Fear & Greed + Crypto headlines (BTC + ETH)
```python
# nuevo archivo: src/sentiment/fear_greed.py
import requests

def get_fear_greed() -> dict:
    r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
    d = r.json()["data"][0]
    return {"value": int(d["value"]), "label": d["value_classification"]}
    # Ej: {"value": 45, "label": "Fear"}

def get_crypto_headlines(currency="BTC", n=10) -> list[str]:
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token=TU_TOKEN&currencies={currency}&kind=news"
    r = requests.get(url, timeout=5)
    return [p["title"] for p in r.json().get("results", [])[:n]]
```

Agregar en `live_runner._build_fundamental_context()`:
```python
from sentiment.fear_greed import get_fear_greed, get_crypto_headlines
fg = get_fear_greed()
headlines = get_crypto_headlines("BTC")  # o "ETH"
return {
    ...,
    "fear_greed_index": fg["value"],    # 0=Extreme Fear, 100=Extreme Greed
    "fear_greed_label": fg["label"],
    "recent_headlines": headlines[:5],   # últimas 5 noticias
}
```

### Fase 2 — Economic calendar veto (Oro, Índices, Oil)
```python
# nuevo archivo: src/sentiment/econ_calendar.py
# Parsea ForexFactory o Investing.com para eventos de alto impacto
# Retorna lista de eventos en próximas N horas
def get_upcoming_high_impact(hours_ahead=2) -> list[dict]:
    ...
    # {"time": "08:30 ET", "event": "NFP", "impact": "HIGH", "currency": "USD"}
```

### Fase 3 — Sentiment Agent (LLM) para análisis contextual
```python
SENTIMENT_AGENT_SYSTEM = """You are a market sentiment analyst...
Given recent news headlines and market context, assess:
- Overall sentiment (BULLISH/BEARISH/NEUTRAL)
- Whether a market shock is in progress or imminent
- Confidence level

Output ONLY valid JSON: {
  "sentiment": "BULLISH"|"BEARISH"|"NEUTRAL",
  "shock_detected": true|false,
  "shock_type": null|"REGULATORY"|"HACK"|"MACRO"|"GEOPOLITICAL"|"EARNINGS",
  "shock_severity": null|"MINOR"|"MODERATE"|"SEVERE",
  "confidence": "HIGH"|"MEDIUM"|"LOW",
  "veto_trade": true|false,
  "reasoning": "..."
}"""
```

---

## APIs a registrar (claves necesarias)

| API | URL | Costo | Uso |
|-----|-----|-------|-----|
| CryptoPanic | cryptopanic.com/developers/api | Gratis (100 req/día) | BTC, ETH headlines |
| Alternative.me | api.alternative.me/fng | Siempre gratis | Fear & Greed index |
| Alpha Vantage | alphavantage.co/support/#api-key | Gratis (25 req/día) | Stocks + Gold news |
| EIA | eia.gov/opendata | Gratis | Oil inventories |
| NewsAPI | newsapi.org/register | Gratis (100 req/día dev) | All instruments |
| Etherscan | etherscan.io/apis | Gratis | ETH gas tracker |

Guardar todas las claves en el mismo `.env` del proyecto:
```
CRYPTOPANIC_TOKEN=...
ALPHA_VANTAGE_KEY=...
EIA_API_KEY=...
NEWS_API_KEY=...
ETHERSCAN_API_KEY=...
```

---

## Notas importantes

1. **No bloquear el loop principal.** Todas las llamadas a APIs externas deben tener
   `timeout=5` y estar en `try/except`. Si falla → continuar sin el sentimiento.

2. **Caché de 15 minutos.** No llamar la API de noticias en cada ciclo (cada 15 min
   ya es suficiente). Guardar timestamp de último fetch.

3. **El técnico siempre tiene prioridad.** Sentiment es señal de veto, no de entrada.
   "Noticias bullish" sin setup técnico = FLAT igual.

4. **Costo LLM.** Si usas Sentiment Agent (opción B), sube `MAX_LLM_CALLS_PER_DAY`
   de 8 a 12 por robot.

5. **Empezar simple.** Fear & Greed + CryptoPanic headlines pasados como texto al
   Trend Agent ya da el 80% del valor. El Sentiment Agent completo puede esperar.

---

*Documento creado: 2026-05-24 | Actualizar al implementar cada fase.*
