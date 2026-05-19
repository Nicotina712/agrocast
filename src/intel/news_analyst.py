"""
src/intel/news_analyst.py
Agente analista de noticias basado en Claude Haiku.

A diferencia de VADER (léxico, ciego al contexto), este agente:
  - Identifica el driver concreto (china_demand, weather_br, supply, policy, …)
  - Estima dirección, magnitud y horizonte del impacto
  - Extrae datos cuantitativos (volúmenes, precios objetivo, países)
  - Devuelve confianza para poder ponderar el agregado

Cada artículo se analiza UNA sola vez: cache permanente por hash(url|title).
Si ANTHROPIC_API_KEY no está disponible o falla la llamada, se cae a VADER
(modo degradado) para no romper el pipeline.
"""

import hashlib
import json
import os
import time
from datetime import datetime

# Ensure .env is loaded so ANTHROPIC_API_KEY is available
try:
    from dotenv import load_dotenv as _load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.exists(_env_path):
        _load_dotenv(_env_path, override=True)
except ImportError:
    pass
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_DIR    = os.path.join(_PROJECT_ROOT, "data", "intel_cache")
_INDEX_PATH   = os.path.join(_CACHE_DIR, "index.json")

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 600

VALID_DRIVERS = {
    "china_demand", "weather_br", "weather_ar", "weather_us",
    "supply_global", "usda_report", "policy_ar", "policy_us",
    "policy_br", "macro_usd", "macro_oil", "logistics", "biofuels",
    "geopolitics", "other",
}
VALID_IMPACTS  = {"bullish", "bearish", "neutral"}
VALID_HORIZONS = {"1d", "7d", "30d", "90d"}


def _key(url: str, title: str) -> str:
    raw = f"{url or ''}|{title or ''}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:16]


def _cache_path(key: str) -> str:
    return os.path.join(_CACHE_DIR, f"{key}.json")


def _load_cached(key: str) -> Optional[dict]:
    p = _cache_path(key)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cached(key: str, data: dict) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_cache_path(key), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _vader_fallback(title: str, body: str) -> dict:
    """Modo degradado cuando Claude no está disponible."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        score = SentimentIntensityAnalyzer().polarity_scores(f"{title} {body}")["compound"]
    except Exception:
        score = 0.0

    if score > 0.15:
        impact = "bullish"
    elif score < -0.15:
        impact = "bearish"
    else:
        impact = "neutral"

    return {
        "price_impact":   impact,
        "magnitude":      min(5, max(1, int(round(abs(score) * 5)))) if score else 1,
        "horizon":        "7d",
        "drivers":        ["other"],
        "confidence":     0.25,
        "extracted_data": {},
        "key_quote":      "",
        "rationale":      f"VADER fallback (compound={score:+.2f})",
        "_source":        "vader_fallback",
    }


_PROMPT_TEMPLATE = """Sos un analista de mercados de commodities especializado en soja CBOT.

Analizá esta noticia y devolvé SOLO JSON válido (sin markdown, sin explicaciones extra):

{{
  "price_impact": "bullish" | "bearish" | "neutral",
  "magnitude": 1-5,
  "horizon": "1d" | "7d" | "30d" | "90d",
  "drivers": ["china_demand"|"weather_br"|"weather_ar"|"weather_us"|"supply_global"|"usda_report"|"policy_ar"|"policy_us"|"policy_br"|"macro_usd"|"macro_oil"|"logistics"|"biofuels"|"geopolitics"|"other"],
  "confidence": 0.0-1.0,
  "extracted_data": {{"volume_mmt": null, "price_target_usc": null, "country": null, "yield_change_pct": null}},
  "key_quote": "frase corta de la noticia que justifica el impacto",
  "rationale": "una oración explicando por qué movería el precio de la soja"
}}

Reglas:
- magnitude=1 ruido, 3 noticia relevante, 5 evento de mercado mayor (WASDE shock, sequía severa, sanciones).
- Si la noticia NO afecta el precio CBOT de soja (ej. fútbol, tecnología sin link), devolvé impact=neutral, magnitude=1, confidence<=0.3, drivers=["other"].
- horizon es cuándo se materializa el impacto: '1d' notica del día, '7d' próxima semana, '30d' próximo mes, '90d' trimestral.
- Si no hay datos cuantitativos, dejá los campos de extracted_data en null.

NOTICIA:
Título: {title}
Fuente: {source}
Cuerpo: {body}
"""


def _validate(parsed: dict) -> dict:
    """Recorta y normaliza la respuesta del modelo a campos válidos."""
    out = {
        "price_impact":   parsed.get("price_impact", "neutral"),
        "magnitude":      parsed.get("magnitude", 1),
        "horizon":        parsed.get("horizon", "7d"),
        "drivers":        parsed.get("drivers", ["other"]) or ["other"],
        "confidence":     parsed.get("confidence", 0.5),
        "extracted_data": parsed.get("extracted_data", {}) or {},
        "key_quote":      (parsed.get("key_quote") or "")[:300],
        "rationale":      (parsed.get("rationale") or "")[:400],
        "_source":        "claude",
    }

    if out["price_impact"] not in VALID_IMPACTS:
        out["price_impact"] = "neutral"
    if out["horizon"] not in VALID_HORIZONS:
        out["horizon"] = "7d"
    try:
        out["magnitude"] = int(out["magnitude"])
        out["magnitude"] = min(5, max(1, out["magnitude"]))
    except Exception:
        out["magnitude"] = 1
    try:
        out["confidence"] = float(out["confidence"])
        out["confidence"] = min(1.0, max(0.0, out["confidence"]))
    except Exception:
        out["confidence"] = 0.5

    if not isinstance(out["drivers"], list):
        out["drivers"] = ["other"]
    out["drivers"] = [d for d in out["drivers"] if d in VALID_DRIVERS] or ["other"]

    return out


def analyze_article(
    title: str,
    body: str = "",
    url: str = "",
    source: str = "",
    use_cache: bool = True,
) -> dict:
    """
    Analiza un artículo y devuelve dict estructurado. Cachea por url|title.

    Si Claude no está disponible (sin API key, error, etc.) usa VADER.
    """
    title = (title or "").strip()
    body  = (body or "").strip()

    if not title and not body:
        return _vader_fallback("", "")

    key = _key(url, title)

    if use_cache:
        cached = _load_cached(key)
        if cached is not None:
            return cached

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        result = _vader_fallback(title, body)
        result["_cached_at"] = datetime.utcnow().isoformat()
        _save_cached(key, result)
        return result

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        prompt = _PROMPT_TEMPLATE.format(
            title=title[:300],
            source=source or "desconocida",
            body=(body or title)[:1500],
        )

        msg = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()

        # El modelo a veces envuelve en ```json ... ```
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip("` \n")

        parsed = json.loads(text)
        result = _validate(parsed)
        result["_cached_at"] = datetime.utcnow().isoformat()
        result["_model"]     = MODEL

        _save_cached(key, result)
        return result

    except Exception as e:
        print(f"[news_analyst] Claude error ({e}) — usando VADER")
        result = _vader_fallback(title, body)
        result["_cached_at"] = datetime.utcnow().isoformat()
        result["_error"]     = str(e)[:120]
        _save_cached(key, result)
        return result


def analyze_batch(articles: list, max_articles: int = 30, sleep_s: float = 0.3) -> list:
    """
    Analiza una lista de artículos. Respeta cache, limita batch para controlar costo.

    Cada artículo debe tener: title, description (o body), url, source.
    """
    out = []
    api_calls = 0

    for art in articles[:max_articles]:
        title  = art.get("title", "")
        body   = art.get("description") or art.get("body") or ""
        url    = art.get("url", "")
        src    = art.get("source", "")

        key    = _key(url, title)
        cached = _load_cached(key)

        if cached is not None:
            analysis = cached
        else:
            analysis = analyze_article(title, body, url, src)
            api_calls += 1
            if sleep_s > 0:
                time.sleep(sleep_s)

        out.append({**art, "intel": analysis})

    print(f"[news_analyst] batch: {len(out)} artículos ({api_calls} llamadas API, "
          f"{len(out) - api_calls} desde cache)")
    return out


if __name__ == "__main__":
    # Smoke test
    sample = analyze_article(
        title="China resumes large-scale soybean purchases from US after trade meeting",
        body="Beijing booked 1.5 MMT of US soybeans this week, the biggest weekly purchase since 2023.",
        url="https://example.com/test",
        source="Reuters",
    )
    print(json.dumps(sample, indent=2, ensure_ascii=False))
