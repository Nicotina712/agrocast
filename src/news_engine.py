"""
src/news_engine.py
Motor de noticias (versión legacy en src/).
La versión activa está en MVP lectura de noticias/news_engine.py.
Este archivo se mantiene para imports de pipeline_news_impact.py.
"""

import os
import feedparser
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
if not NEWS_API_KEY:
    print("⚠️  NEWS_API_KEY no configurada. NewsAPI no funcionará.")

QUERY = (
    "soybean OR soybeans OR soja OR "
    "soybean oil OR soy meal OR "
    "USDA OR crop report OR harvest OR "
    "Argentina soy OR Brazil soy OR China soybean imports OR "
    "corn price OR wheat price OR crude oil OR diesel price"
)

TOPIC_CONFIG = [
    ("weather", ["drought", "rain", "flood", "frost", "storm", "weather", "climate"], 0.40),
    ("china",   ["china", "import", "demand", "trade", "tariff", "beijing"],          0.35),
    ("macro",   ["inflation", "interest", "economy", "fed", "dollar", "gdp"],         0.25),
]

TOPICS  = {name: words  for name, words, _ in TOPIC_CONFIG}
WEIGHTS = {name: weight for name, _, weight in TOPIC_CONFIG}


def soy_relevance_score(text: str) -> int:
    text = text.lower()
    BLACKLIST = {"python", "pypi", "geocif", "software", "library", "api", "release", "version"}
    if any(b in text for b in BLACKLIST):
        return 0
    score = 0
    STRONG = {"soybean", "soybeans", "soja", "soybean oil", "soy meal",
              "usda", "harvest", "crop", "argentina", "brazil", "china", "grain", "export"}
    MEDIUM = {"agriculture", "commodities", "farming", "market"}
    for w in STRONG:
        if w in text: score += 2
    for w in MEDIUM:
        if w in text: score += 1
    if "oil" in text or "diesel" in text: score += 1
    if not any(w in text for w in ("soybean", "soybeans", "soja")): score -= 1
    return score


def fetch_news(from_date: str, to_date: str, page_size: int = 50) -> list:
    if not NEWS_API_KEY:
        return []
    params = {"q": QUERY, "from": from_date, "to": to_date, "language": "en",
              "sortBy": "publishedAt", "pageSize": page_size, "apiKey": NEWS_API_KEY}
    try:
        res = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
        res.raise_for_status()
        articles = res.json().get("articles", [])
        print(f"📰 NewsAPI: {len(articles)}")
        return articles
    except Exception as e:
        print(f"❌ NewsAPI error: {e}")
        return []


def fetch_gdelt_news() -> list:
    import json as _json
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        "?query=soybean+OR+soja+OR+%22soy+export%22+OR+%22crop+report%22"
        "&mode=artlist&format=json&maxrecords=25&sort=DateDesc&timespan=1week"
    )
    try:
        res = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 AgroCastBot/1.0"})
        if res.status_code != 200:
            return []
        body = res.text.strip()
        if not body:
            return []
        try:
            data = _json.loads(body)
        except _json.JSONDecodeError:
            return []
        articles = data.get("articles", [])
        print(f"🌍 GDELT: {len(articles)}")
        return [{"title": a.get("title", ""), "description": a.get("seendate", ""),
                 "url": a.get("url"), "source": a.get("sourcecountry", "GDELT"),
                 "publishedAt": a.get("seendate")} for a in articles if a.get("title")]
    except Exception as e:
        print(f"⚠️  GDELT error: {e}")
        return []


def fetch_rss_news() -> list:
    import socket as _socket
    feeds = ["https://www.agweb.com/rss", "https://www.farmprogress.com/rss.xml",
             "https://www.feedstuffs.com/rss.xml"]
    articles = []
    prev = _socket.getdefaulttimeout()
    try:
        _socket.setdefaulttimeout(8)
        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:20]:
                    title = entry.get("title", "").strip()
                    if title:
                        articles.append({"title": title, "description": entry.get("summary", ""),
                                         "url": entry.get("link"), "source": feed_url,
                                         "publishedAt": entry.get("published", "")})
            except Exception as e:
                print(f"⚠️  RSS error ({feed_url}): {e}")
    finally:
        _socket.setdefaulttimeout(prev)
    print(f"📰 RSS: {len(articles)}")
    return articles


def get_sentiment(text: str) -> float:
    try:
        return analyzer.polarity_scores(text)["compound"]
    except Exception as e:
        print(f"⚠️  Sentiment error: {e}")
        return 0.0


def classify_topic(text: str) -> str:
    text = text.lower()
    best_topic, best_count = "macro", 0
    for topic, words, _ in TOPIC_CONFIG:
        count = sum(1 for w in words if w in text)
        if count > best_count:
            best_count, best_topic = count, topic
    return best_topic


def compute_impact(topic: str, sentiment: float) -> float:
    weight = WEIGHTS.get(topic, 0.2)
    return round(max(min(weight * sentiment * 2, 0.5), -0.5), 4)
