"""
news_engine.py
Motor de noticias para AgroCast PRO.
Obtiene, puntúa y analiza noticias de múltiples fuentes.
"""

import os
import feedparser
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# ✅ MEJORA: API key SOLO desde variable de entorno, sin fallback hardcodeado.
# Ejecutar: export NEWS_API_KEY="tu_key"
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
if not NEWS_API_KEY:
    print("⚠️  NEWS_API_KEY no configurada. NewsAPI no funcionará.")

# -----------------------------
# QUERY
# -----------------------------
QUERY = (
    "soybean OR soybeans OR soja OR "
    "soybean oil OR soy meal OR "
    "USDA OR crop report OR harvest OR "
    "Argentina soy OR Brazil soy OR China soybean imports OR "
    "corn price OR wheat price OR crude oil OR diesel price"
)

# ✅ MEJORA: TOPICS y WEIGHTS juntos como lista de tuplas para garantizar
# que cada topic tiene exactamente un peso y no se desincronicen.
TOPIC_CONFIG = [
    ("weather", [
        "drought", "rain", "flood", "frost", "storm", "weather", "climate",
        # ES
        "sequia", "sequía", "lluvia", "helada", "tormenta", "clima", "niña", "niño", "la nina",
    ], 0.40),
    ("china", [
        "china", "import", "demand", "trade", "tariff", "beijing",
        # ES
        "importacion", "importación", "demanda", "aranceles", "embargo",
    ], 0.35),
    ("macro", [
        "inflation", "interest", "economy", "fed", "dollar", "gdp",
        # ES
        "inflacion", "inflación", "tasa", "economia", "economía", "dolar", "dólar",
        "retenciones", "cepo", "devaluacion", "devaluación", "tipo de cambio",
    ], 0.25),
]

# Derivados para uso rápido
TOPICS  = {name: words  for name, words, _ in TOPIC_CONFIG}
WEIGHTS = {name: weight for name, _, weight in TOPIC_CONFIG}


# -----------------------------
# RELEVANCE SCORE
# -----------------------------
def soy_relevance_score(text: str) -> int:
    """
    Devuelve un score de relevancia para noticias de soja.
    Valores típicos: 0 (irrelevante) → 5+ (muy relevante).
    """
    text = text.lower()

    BLACKLIST = {
        "python", "pypi", "geocif", "software",
        "library", "api", "release", "version",
    }
    if any(b in text for b in BLACKLIST):
        return 0

    score = 0

    STRONG = {
        # EN
        "soybean", "soybeans", "soybean oil", "soy meal",
        "usda", "harvest", "crop", "argentina", "brazil", "china",
        "grain", "export",
        # ES
        "soja", "oleaginosa", "cosecha", "exportacion", "exportación",
        "retenciones", "cepo", "bolsa rosario", "bcr", "ciara", "cec",
        "mercosur", "campo argentino", "campaña", "acopio", "fob rosario",
        "wasde", "informe usda",
    }
    MEDIUM = {
        # EN
        "agriculture", "commodities", "farming", "market",
        # ES
        "agro", "agricultura", "granos", "mercado", "precio", "maiz",
        "maíz", "trigo", "produccion", "producción", "camion", "camión",
    }

    for w in STRONG:
        if w in text:
            score += 2

    for w in MEDIUM:
        if w in text:
            score += 1

    # Energía suma señal (correlación con costo de producción)
    if "oil" in text or "diesel" in text or "gasoil" in text:
        score += 1

    # Penalizar si no menciona soja directamente (EN o ES)
    if not any(w in text for w in ("soybean", "soybeans", "soja", "oleaginosa")):
        score -= 1

    return score


# -----------------------------
# NEWSAPI
# -----------------------------
def fetch_news(from_date: str, to_date: str, page_size: int = 50) -> list:
    """Obtiene artículos de NewsAPI."""
    if not NEWS_API_KEY:
        print("⚠️  Saltando NewsAPI: NEWS_API_KEY no configurada")
        return []

    params = {
        "q":        QUERY,
        "from":     from_date,
        "to":       to_date,
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": page_size,
        "apiKey":   NEWS_API_KEY,
    }

    try:
        res = requests.get(
            "https://newsapi.org/v2/everything",
            params=params,
            timeout=10,
        )
        res.raise_for_status()
        articles = res.json().get("articles", [])
        print(f"📰 NewsAPI: {len(articles)}")
        return articles

    except requests.exceptions.HTTPError as e:
        print(f"❌ NewsAPI HTTP error {e.response.status_code}: {e}")
        return []
    except Exception as e:
        print(f"❌ NewsAPI error: {e}")
        return []


# -----------------------------
# GDELT
# -----------------------------
def fetch_gdelt_news() -> list:
    """Obtiene artículos de GDELT (con manejo robusto de errores)."""
    import json as _json

    # Query simple — GDELT responde mejor a consultas cortas y claras
    # sort=DateDesc (case-sensitive), maxrecords=25 (más estable que 50)
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        "?query=soybean+OR+soja+OR+%22soy+export%22+OR+%22crop+report%22"
        "&mode=artlist"
        "&format=json"
        "&maxrecords=25"
        "&sort=DateDesc"
        "&timespan=1week"
    )

    # Retry con backoff: GDELT API a veces hace hangs prolongados.
    last_err = None
    res = None
    for attempt in range(3):
        try:
            res = requests.get(
                url,
                timeout=(10, 45),  # connect, read
                headers={"User-Agent": "Mozilla/5.0 AgroCastBot/1.0"},
            )
            break
        except (requests.ConnectTimeout, requests.ReadTimeout, requests.ConnectionError) as e:
            last_err = e
            print(f"⚠️  GDELT timeout (intento {attempt + 1}/3): {type(e).__name__}")
            if attempt < 2:
                import time as _t; _t.sleep(2 ** attempt)
                continue
            return []

    try:
        if res is None:
            return []

        if res.status_code == 429:
            print("⚠️  GDELT rate limit (429) — se omite")
            return []

        if res.status_code != 200:
            print(f"⚠️  GDELT status: {res.status_code}")
            return []

        body = res.text.strip()
        if not body:
            print("⚠️  GDELT respuesta vacía")
            return []

        # Parsear JSON con manejo explícito de error (GDELT a veces devuelve HTML)
        try:
            data = _json.loads(body)
        except _json.JSONDecodeError:
            print("⚠️  GDELT respuesta no es JSON (posible error del servidor)")
            return []

        articles = data.get("articles", [])
        print(f"🌍 GDELT: {len(articles)}")

        return [
            {
                "title":       a.get("title", ""),
                "description": a.get("seendate", ""),
                "url":         a.get("url"),
                "source":      a.get("sourcecountry", "GDELT"),
                "publishedAt": a.get("seendate"),
            }
            for a in articles
            if a.get("title")
        ]

    except Exception as e:
        print(f"⚠️  GDELT error: {e}")
        return []


# -----------------------------
# RSS
# -----------------------------
def fetch_rss_news() -> list:
    """Obtiene artículos de feeds RSS agrícolas y de commodities."""
    import socket as _socket

    feeds = [
        # Fuentes originales (EN)
        "https://www.agweb.com/rss",
        "https://www.farmprogress.com/rss.xml",
        "https://www.feedstuffs.com/rss.xml",
        # USDA NASS — reportes oficiales de cultivos (soybeans, grains)
        "https://www.nass.usda.gov/Newsroom/Syndication/Todays_Reports/index.php",
        "https://www.nass.usda.gov/Newsroom/Syndication/News/index.php",
        # Argentina — granos, mercados, campo
        "https://www.infocampo.com.ar/feed",
        "https://www.todoagro.com.ar/categoria/agricultura/feed/",
        "https://www.lanacion.com.ar/arc/outboundfeeds/rss/?outputType=xml",
        # Uruguay — diversificación de cobertura local (mercado piloto)
        # Algunos sitios no exponen RSS directo; usamos los que sí.
        "https://www.elobservador.com.uy/rss/agro.xml",       # El Observador / Agro
        "https://www.busqueda.com.uy/rss",                    # Búsqueda (semanario)
        "https://www.elpais.com.uy/rss/rurales/index.xml",    # El País Rurales
        # Brasil — productor de soja #1 del mundo
        "https://www.noticiasagricolas.com.br/rss/noticias/soja.xml",  # Notícias Agrícolas (soja)
        "https://www.canalrural.com.br/feed/",                          # Canal Rural
    ]

    articles = []
    prev_timeout = _socket.getdefaulttimeout()

    try:
        _socket.setdefaulttimeout(8)   # 8s por feed — evita cuelgues

        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)

                for entry in feed.entries[:20]:
                    title = entry.get("title", "").strip()
                    if not title:
                        continue

                    articles.append({
                        "title":       title,
                        "description": entry.get("summary", ""),
                        "url":         entry.get("link"),
                        "source":      feed_url,
                        "publishedAt": entry.get("published", ""),
                    })

            except Exception as e:
                print(f"⚠️  RSS error ({feed_url}): {e}")
    finally:
        _socket.setdefaulttimeout(prev_timeout)   # restaurar siempre

    print(f"📰 RSS: {len(articles)}")
    return articles


# -----------------------------
# SENTIMENT
# -----------------------------
def get_sentiment(text: str) -> float:
    """
    Devuelve el compound score de VADER: -1.0 (negativo) → +1.0 (positivo).
    """
    try:
        return analyzer.polarity_scores(text)["compound"]
    except Exception as e:
        # ✅ MEJORA: loguear el error en lugar de silenciarlo
        print(f"⚠️  Sentiment error: {e}")
        return 0.0


# -----------------------------
# TOPIC
# -----------------------------
def classify_topic(text: str) -> str:
    """
    Clasifica el texto en uno de los temas definidos en TOPIC_CONFIG.
    Devuelve el tema con más palabras clave encontradas; 'macro' como fallback.
    """
    text = text.lower()

    # ✅ MEJORA: elegir el topic con más coincidencias en lugar del primero
    best_topic = "macro"
    best_count = 0

    for topic, words, _ in TOPIC_CONFIG:
        count = sum(1 for w in words if w in text)
        if count > best_count:
            best_count = count
            best_topic = topic

    return best_topic


# -----------------------------
# IMPACT
# -----------------------------
def compute_impact(topic: str, sentiment: float) -> float:
    """
    Calcula el impacto de una noticia en el precio de la soja.
    Rango: -0.5 (muy bajista) → +0.5 (muy alcista).
    """
    weight = WEIGHTS.get(topic, 0.2)
    impact = weight * sentiment * 2
    return round(max(min(impact, 0.5), -0.5), 4)
