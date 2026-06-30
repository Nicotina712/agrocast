"""FASE 2 (MODO SOMBRA) — Debate multi-agente sobre cada gap, adaptado de
src/intel/multi_agent_debate.py (AgroCast). El veredicto SE REGISTRA pero NO veta:
tras 20-30 gaps se compara veredicto vs resultado (accountability) y recien ahi
se decide si darle poder de veto.

Flujo: noticias del ticker (GDELT, cache 2h) -> 1 llamada Haiku con debate
estructurado Bull/Bear/Riesgo -> Juez (Fund Manager) -> veredicto JSON.
Costo ~$0.01-0.03 por gap (Haiku). Si GDELT/LLM fallan -> verdict=UNAVAILABLE
(el robot opera igual: es sombra).
"""
import os, sys, json
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in [_HERE, _MVP_ROOT, os.path.join(_MVP_ROOT, "templates_nuevos_mercados")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_MVP_ROOT, ".env"))
except Exception:
    pass

# Perfiles por ticker: terminos de busqueda de noticias
TICKER_PROFILES = {
    "AAPL.NAS": "Apple AAPL iPhone earnings",
    "NVDA.NAS": "Nvidia NVDA AI chips earnings",
    "TSLA.NAS": "Tesla TSLA Musk deliveries earnings",
    "MSFT.NAS": "Microsoft MSFT Azure earnings",
    "AMZN.NAS": "Amazon AMZN AWS earnings",
    "AMD.NAS":  "AMD Advanced Micro Devices chips earnings",
    "NFLX.NAS": "Netflix NFLX subscribers earnings",
    "COIN.NAS": "Coinbase COIN crypto SEC earnings",
    "MSTR.NAS": "MicroStrategy MSTR bitcoin Saylor",
    "GOOG.NAS": "Google Alphabet GOOG search AI earnings",
}

MODEL = "claude-haiku-4-5-20251001"

_PROMPT = """Sos el panel de debate de un fondo. Hoy {ticker} abrio con un GAP de {gap:+.1f}%
(cierre ayer -> apertura hoy). El robot mecanico va a COMPRAR (long) porque:
- gaps alcistas >3% tienden a driftear (PEAD)
- gaps bajistas <-3% tienden a revertir (panico sobre-reaccionado)

NOTICIAS RECIENTES del ticker (ultimas 24h, puede estar vacio):
{news}

Conduci un debate en 3 roles y emite veredicto:
1. ANALISTA ALCISTA: mejor caso para que el LONG funcione.
2. ANALISTA BAJISTA: mejor caso para que el LONG falle (¿la causa del gap es terminal?
   ¿fraude/quiebra/regulacion existencial? ¿sympathy vacio sin sustancia?).
3. RIESGO: ¿que tan confiable es la informacion disponible?
JUEZ (Fund Manager): sintetiza y decide.

Responde SOLO con JSON:
{{"gap_cause": "earnings|guidance|regulatorio|legal|producto|macro|sympathy|desconocido",
  "bull_thesis": "1 oracion",
  "bear_thesis": "1 oracion",
  "verdict": "CONFIRM|VETO",
  "confidence": 0.0-1.0,
  "reasoning": "1-2 oraciones del juez"}}"""


def _fetch_news(ticker: str) -> list:
    try:
        from news_intelligence.gdelt_client import fetch_news
        q = TICKER_PROFILES.get(ticker, ticker.split(".")[0])
        arts = fetch_news(q, timespan="1d", max_records=8) or []
        return [str(a.get("title", ""))[:140] for a in arts[:8]]
    except Exception:
        return []


def shadow_debate(ticker: str, gap_pct: float) -> dict:
    """Devuelve el veredicto del debate (solo registro, NO decide). Nunca lanza."""
    out = {"ticker": ticker, "gap_pct": round(gap_pct, 2),
           "ts": datetime.now().isoformat(), "mode": "shadow"}
    try:
        titles = _fetch_news(ticker)
        out["news_count"] = len(titles)
        news_txt = "\n".join(f"- {t}" for t in titles) if titles else "(sin noticias disponibles)"
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=MODEL, max_tokens=500,
            messages=[{"role": "user",
                       "content": _PROMPT.format(ticker=ticker, gap=gap_pct, news=news_txt)}])
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].replace("json", "", 1).strip()
        verdict = json.loads(raw)
        out.update({k: verdict.get(k) for k in
                    ("gap_cause", "bull_thesis", "bear_thesis", "verdict", "confidence", "reasoning")})
    except Exception as e:
        out["verdict"] = "UNAVAILABLE"
        out["error"] = str(e)[:120]
    return out
