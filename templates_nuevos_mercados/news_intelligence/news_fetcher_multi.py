"""
Fetcher de noticias multi-instrumento para el portfolio de robots.
Reutiliza la infraestructura de AgroCast adaptando queries y prompts por instrumento.

Fuentes: GDELT (gratis), NewsAPI (si key disponible), RSS feeds financieros.

Uso:
  python news_fetcher_multi.py                     # fetch todos
  python news_fetcher_multi.py --instrument UK100  # solo uno
  python news_fetcher_multi.py --dry-run           # sin LLM
"""

import os, sys, json, time, hashlib, argparse
from datetime import datetime, timezone
from pathlib import Path

_HERE     = Path(__file__).resolve().parent
_MVP_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_MVP_ROOT))

from dotenv import load_dotenv
load_dotenv(_MVP_ROOT / ".env", override=True)

from templates_nuevos_mercados.news_intelligence.instrument_profiles import INSTRUMENT_PROFILES
from templates_nuevos_mercados.news_intelligence.gdelt_client import get_client as _get_gdelt

_DATA_DIR  = _MVP_ROOT / "data" / "news_portfolio"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_DIR = _MVP_ROOT / "data" / "intel_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

RSS_FEEDS = [
    # Finanzas generales
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.ft.com/rss/home",
    "https://www.marketwatch.com/rss/topstories",
    # Mercados y economia global
    "https://finance.yahoo.com/news/rssindex",
    "https://www.investing.com/rss/news.rss",
    # Crypto (BTCUSD / ETHUSD)
    "https://cointelegraph.com/rss",
    "https://coindesk.com/arc/outboundfeeds/rss/",
    # Energia y materias primas (BRENT / WTI / Corn)
    "https://oilprice.com/rss/main",
    # Asia (HK50)
    "https://www.scmp.com/rss/5/feed",          # South China Morning Post (50 items)
]


def _gdelt_fetch(query: str, max_records: int = 50) -> list:
    """Usa el cliente robusto con retry, backoff y cache."""
    try:
        return _get_gdelt().fetch(query, timespan="1d", max_records=max_records)
    except Exception as e:
        print(f"  [GDELT] Error inesperado: {e}")
        return []


def _newsapi_fetch(query: str, max_articles: int = 15) -> list:
    api_key = os.getenv("NEWS_API_KEY", "")
    if not api_key:
        return []
    import urllib.request, urllib.parse
    params = {"q": query, "language": "en", "sortBy": "publishedAt",
              "pageSize": str(max_articles), "apiKey": api_key}
    url = "https://newsapi.org/v2/everything?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
        return [
            {"title": a.get("title",""), "url": a.get("url",""),
             "source": (a.get("source") or {}).get("name",""),
             "publishedAt": a.get("publishedAt",""),
             "description": a.get("description","") or a.get("title","")}
            for a in data.get("articles", []) if a.get("title")
        ]
    except Exception as e:
        print(f"  [NewsAPI] Error: {e}")
        return []


def _rss_fetch(keyword_filter: list = None) -> list:
    import urllib.request, re
    articles = []
    for feed_url in RSS_FEEDS:
        try:
            req = urllib.request.Request(feed_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as r:
                content = r.read().decode("utf-8", errors="ignore")
            items = re.findall(r"<item>(.*?)</item>", content, re.DOTALL)
            for item in items[:30]:
                t_m = re.search(r"<title>(.*?)</title>", item, re.DOTALL)
                l_m = re.search(r"<link>(.*?)</link>",  item, re.DOTALL)
                d_m = re.search(r"<description>(.*?)</description>", item, re.DOTALL)
                p_m = re.search(r"<pubDate>(.*?)</pubDate>", item, re.DOTALL)
                if not t_m:
                    continue
                t = re.sub(r"<[^>]+>", "", t_m.group(1)).strip()
                d = re.sub(r"<[^>]+>", "", (d_m.group(1) if d_m else "")).strip()
                if keyword_filter and not any(k.lower() in f"{t} {d}".lower() for k in keyword_filter):
                    continue
                articles.append({
                    "title": t, "url": l_m.group(1).strip() if l_m else "",
                    "source": feed_url, "publishedAt": p_m.group(1).strip() if p_m else "",
                    "description": d[:400],
                })
        except Exception as e:
            print(f"  [RSS] {feed_url[:50]} Error: {e}")
    return articles


def _dedup(articles: list) -> list:
    seen, out = set(), []
    for a in articles:
        key = a.get("title","").lower().strip()[:80]
        if key and key not in seen:
            seen.add(key); out.append(a)
    return out


def _vader_fallback(title: str, body: str) -> dict:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        score = SentimentIntensityAnalyzer().polarity_scores(f"{title} {body}")["compound"]
    except Exception:
        score = 0.0
    impact = "bullish" if score > 0.15 else ("bearish" if score < -0.15 else "neutral")
    return {"price_impact": impact, "magnitude": max(1,min(5,int(abs(score)*5))),
            "horizon": "7d", "drivers": ["other"], "confidence": 0.2,
            "key_quote": "", "rationale": f"VADER fallback (score={score:+.2f})",
            "_source": "vader_fallback"}


def _build_prompt(profile: dict, title: str, body: str, source: str) -> str:
    drivers_str = "|".join(profile["drivers"])
    return (
        f"{profile['sentiment_prompt']}\n\n"
        f"Analiza esta noticia y devuelve SOLO JSON valido (sin markdown):\n\n"
        f"{{\n"
        f'  "price_impact": "bullish" | "bearish" | "neutral",\n'
        f'  "magnitude": 1-5,\n'
        f'  "horizon": "1d" | "7d" | "30d" | "90d",\n'
        f'  "drivers": ["{drivers_str}"],\n'
        f'  "confidence": 0.0-1.0,\n'
        f'  "key_quote": "frase de la noticia",\n'
        f'  "rationale": "por que moveria el precio"\n'
        f"}}\n\n"
        f"Reglas: magnitude=1 ruido, 3 relevante, 5 shock. "
        f"Si no afecta el instrumento: neutral, magnitude=1, confidence<=0.3.\n\n"
        f"NOTICIA:\nTitulo: {title[:300]}\nFuente: {source}\nCuerpo: {body[:1200]}"
    )


def _analyze_for_instrument(articles: list, profile: dict) -> list:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    out, api_calls, cached_hits = [], 0, 0

    for art in articles:
        title = art.get("title","").strip()
        body  = art.get("description","").strip() or title
        url   = art.get("url","")
        cache_key  = hashlib.sha1(f"{profile['name']}|{url}|{title}".encode("utf-8","ignore")).hexdigest()[:16]
        cache_path = _CACHE_DIR / f"{cache_key}.json"

        if cache_path.exists():
            try:
                intel = json.loads(cache_path.read_text(encoding="utf-8"))
                cached_hits += 1
                out.append({**art, "intel": intel}); continue
            except Exception:
                pass

        if api_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                prompt = _build_prompt(profile, title, body, art.get("source",""))
                msg    = client.messages.create(
                    model="claude-haiku-4-5-20251001", max_tokens=600,
                    messages=[{"role":"user","content":prompt}])
                text = msg.content[0].text.strip()
                if text.startswith("```"):
                    text = text.split("```",2)[1]
                    if text.startswith("json"): text = text[4:]
                    text = text.strip("` \n")
                intel = json.loads(text)
                intel["_source"]     = "claude"
                intel["_cached_at"]  = datetime.utcnow().isoformat()
                intel["_instrument"] = [k for k,v in INSTRUMENT_PROFILES.items() if v["name"]==profile["name"]][0]
                cache_path.write_text(json.dumps(intel, ensure_ascii=False), encoding="utf-8")
                api_calls += 1
                time.sleep(0.3)
            except Exception as e:
                intel = _vader_fallback(title, body)
                intel["_error"] = str(e)[:100]
        else:
            intel = _vader_fallback(title, body)

        valid_drivers = set(profile["drivers"])
        if "drivers" in intel:
            intel["drivers"] = [d for d in intel["drivers"] if d in valid_drivers] or ["other"]
        out.append({**art, "intel": intel})

    print(f"  -> Analizados: {len(out)} ({api_calls} LLM, {cached_hits} cache)")
    return out


def fetch_for_instrument(instrument: str, dry_run: bool = False) -> list:
    profile = INSTRUMENT_PROFILES.get(instrument)
    if not profile:
        print(f"[news_fetcher] Sin perfil para {instrument}")
        return []
    print(f"\n[{instrument}] Fetching noticias ({profile['name']})...")
    keywords = profile["keywords"]
    query    = " OR ".join(f'"{k}"' if " " in k else k for k in keywords[:8])

    articles  = _gdelt_fetch(query)
    articles += _newsapi_fetch(" OR ".join(keywords[:5]))
    articles += _rss_fetch(keyword_filter=keywords)
    articles  = _dedup(articles)
    print(f"  -> {len(articles)} articulos unicos")

    if dry_run:
        print("  [dry-run] Saltando LLM")
        return articles
    if not articles:
        return []
    return _analyze_for_instrument(articles, profile)


def save_results(instrument: str, articles: list) -> Path:
    out_path = _DATA_DIR / f"{instrument}_latest.json"
    data = {"instrument": instrument, "fetched_at": datetime.now(timezone.utc).isoformat(),
            "count": len(articles), "articles": articles}
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> Guardado: {out_path.name}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    instruments = [args.instrument] if args.instrument else list(INSTRUMENT_PROFILES.keys())
    print(f"\n=== Portfolio News Fetcher ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ===")
    for inst in instruments:
        arts = fetch_for_instrument(inst, dry_run=args.dry_run)
        if arts:
            save_results(inst, arts)
    print("\n=== Fetch completado ===")


if __name__ == "__main__":
    main()
