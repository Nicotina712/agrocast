import sys
import os
import json
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.news_engine import (
    fetch_news,
    fetch_gdelt_news,
    fetch_rss_news,
    get_sentiment,
    classify_topic,
    compute_impact,
    soy_relevance_score
)

MEMORY_FILE = "news_memory.json"


class MultiNewsFetcher:

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.memory_path = os.path.join(self.base_dir, MEMORY_FILE)

    # -----------------------------
    # 🧠 MEMORIA
    # -----------------------------
    def load_memory(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r") as f:
                return json.load(f)
        return []

    def save_memory(self, data):
        with open(self.memory_path, "w") as f:
            json.dump(data, f)

    # -----------------------------
    # 📈 FORECAST BASE (FIX)
    # -----------------------------
    def load_forecast(self):

        path = os.path.join(self.base_dir, "artifacts", "forecast.csv")

        if not os.path.exists(path):
            print("⚠️ No hay forecast.csv")
            return []

        df = pd.read_csv(path)

        if df.empty:
            print("⚠️ Forecast vacío")
            return []

        forecast = []

        for _, row in df.iterrows():

            try:
                val = float(row["Soybeans"])
            except:
                continue

            forecast.append({
                "Date": str(row["Date"])[:10],
                "Soybeans": val
            })

        print(f"📈 Forecast cargado: {len(forecast)} puntos")

        return forecast

    # -----------------------------
    # 🔍 FILTRO
    # -----------------------------
    def filter_articles(self, articles, threshold=2):

        filtered = []

        for a in articles:
            text = f"{a.get('title','')} {a.get('description','')}"
            score = soy_relevance_score(text)

            if score >= threshold:
                filtered.append((a, score))

        if len(filtered) < 3:
            print("⚠️ Pocas noticias relevantes → bajando threshold")

            for a in articles:
                text = f"{a.get('title','')} {a.get('description','')}"
                score = soy_relevance_score(text)

                if score >= 1:
                    filtered.append((a, score))

        return filtered

    # -----------------------------
    # 📊 MARKET INDEX
    # -----------------------------
    def compute_market_index(self, articles):

        if not articles:
            return {
                "index": 0,
                "signal_es": "NEUTRAL ⚖",
                "confidence": 0,
                "sentiment": 0,
                "volume": 0
            }

        impacts = [a["impact"] for a in articles]
        sentiments = [a["sentiment"] for a in articles]

        index = (sum(impacts) / len(impacts)) * 0.7

        if index > 0.05:
            signal = "ALCISTA 📈"
        elif index < -0.05:
            signal = "BAJISTA 📉"
        else:
            signal = "NEUTRAL ⚖"

        return {
            "index": round(index, 3),
            "signal_es": signal,
            "confidence": round(min(len(articles)/10, 1), 2),
            "sentiment": round(sum(sentiments)/len(sentiments), 3),
            "volume": len(articles)
        }

    # -----------------------------
    # 🚨 ALERTAS
    # -----------------------------
    def generate_alerts(self, articles, memory):

        alerts = []
        seen_titles = [m["title"] for m in memory]

        for a in articles:

            if abs(a["impact"]) > 0.25 and a["relevance"] >= 3:

                if a["title"] in seen_titles:
                    continue

                if a["impact"] > 0:
                    alerts.append(f"🚨 Evento alcista: {a['title'][:80]}")
                else:
                    alerts.append(f"🚨 Evento bajista: {a['title'][:80]}")

        return alerts[:3]

    # -----------------------------
    # 🚀 TRADING SIGNAL
    # -----------------------------
    def generate_trading_signal(self, market, forecast):

        if not forecast:
            return {"signal": "HOLD", "score": 0, "reason": "Sin datos"}

        prices = [f["Soybeans"] for f in forecast[:5]]

        trend = (prices[-1] - prices[0]) / prices[0]

        market_index = market.get("index", 0)

        score = (market_index * 0.6) + (trend * 0.4)

        if score > 0.03:
            signal = "🟢 BUY"
        elif score < -0.03:
            signal = "🔴 SELL"
        else:
            signal = "⚖ HOLD"

        reasons = []

        if market_index < -0.1:
            reasons.append("Noticias negativas")
        elif market_index > 0.1:
            reasons.append("Noticias positivas")

        if trend > 0.02:
            reasons.append("Tendencia alcista")
        elif trend < -0.02:
            reasons.append("Tendencia bajista")

        if not reasons:
            reasons.append("Mercado sin dirección clara")

        return {
            "signal": signal,
            "score": round(score, 3),
            "trend": round(trend * 100, 2),
            "reason": " | ".join(reasons)
        }

    # -----------------------------
    # 🔄 MAIN
    # -----------------------------
    def update_news(self):

        today = datetime.utcnow().strftime("%Y-%m-%d")
        week_ago = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

        newsapi = fetch_news(week_ago, today)
        gdelt = fetch_gdelt_news()
        rss = fetch_rss_news()

        raw_news = newsapi + gdelt + rss

        print(f"🧠 TOTAL RAW: {len(raw_news)}")

        all_articles = []

        for a in raw_news:

            source = a.get("source")

            if isinstance(source, dict):
                source_name = source.get("name")
            else:
                source_name = source

            all_articles.append({
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "url": a.get("url"),
                "source": source_name,
                "publishedAt": a.get("publishedAt")
            })

        filtered = self.filter_articles(all_articles)

        titles_seen = set()
        processed = []

        for art, rel in filtered:

            title = art["title"]
            norm_title = title.lower().strip()[:80]

            if norm_title in titles_seen:
                continue

            titles_seen.add(norm_title)

            text = f"{art['title']} {art.get('description','')}"

            sentiment = get_sentiment(text)
            topic = classify_topic(text)
            impact = compute_impact(topic, sentiment)

            processed.append({
                "title": title,
                "url": art["url"],
                "source": art["source"],
                "publishedAt": art["publishedAt"],
                "sentiment": sentiment,
                "topic": topic,
                "impact": impact,
                "relevance": rel
            })

        memory = self.load_memory()
        combined = (memory + processed)[-50:]
        self.save_memory(combined)

        market = self.compute_market_index(processed)
        alerts = self.generate_alerts(processed, memory)
        forecast = self.load_forecast()

        signal = self.generate_trading_signal(market, forecast)

        return {
            "articles": combined[::-1],
            "market": market,
            "alerts": alerts,
            "forecast": forecast,
            "signal": signal
        }