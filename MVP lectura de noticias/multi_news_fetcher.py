"""
multi_news_fetcher.py
Orquestador de fuentes de noticias para AgroCast PRO.
Combina NewsAPI + GDELT + RSS, deduplica, puntúa y mantiene memoria.
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta

from news_engine import (
    fetch_news,
    fetch_gdelt_news,
    fetch_rss_news,
    get_sentiment,
    classify_topic,
    compute_impact,
    soy_relevance_score,
)

MEMORY_FILE = "news_memory.json"
MEMORY_MAX  = 50


class MultiNewsFetcher:

    def __init__(self):
        self.base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.memory_path = os.path.join(self.base_dir, MEMORY_FILE)

    # -----------------------------
    # 🧠 MEMORIA
    # -----------------------------
    def load_memory(self) -> list:
        if os.path.exists(self.memory_path):
            # ✅ MEJORA: context manager para no dejar el archivo abierto
            with open(self.memory_path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print("⚠️  Memoria corrupta, reiniciando")
                    return []
        return []

    def save_memory(self, data: list) -> None:
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    # -----------------------------
    # 📈 FORECAST BASE
    # -----------------------------
    def load_forecast(self) -> list:
        path = os.path.join(self.base_dir, "artifacts", "forecast.csv")

        if not os.path.exists(path):
            return []

        df = pd.read_csv(path)
        forecast = []

        for _, row in df.iterrows():
            try:
                val = float(row["Soybeans"])
            except (ValueError, KeyError):
                continue  # ✅ Saltar filas malformadas

            forecast.append({
                "date":     str(row["Date"])[:10],
                "base":     round(val, 2),
                "adjusted": round(val, 2),
            })

        return forecast

    # -----------------------------
    # 🔍 FILTRO
    # -----------------------------
    def filter_articles(self, articles: list, threshold: int = 2) -> list[tuple]:
        """
        Devuelve lista de (article, score) con score >= threshold.
        Si hay menos de 3 resultados, baja el threshold a 1.

        ✅ MEJORA: se puntúan todos los artículos de una sola pasada para
        evitar duplicados al relajar el threshold.
        """
        scored = []
        for a in articles:
            text  = f"{a.get('title', '')} {a.get('description', '')}"
            score = soy_relevance_score(text)
            if score >= 1:  # Calcular siempre, filtrar después
                scored.append((a, score))

        # Aplicar threshold principal
        filtered = [(a, s) for a, s in scored if s >= threshold]

        # Si hay pocas noticias, bajar threshold (sin re-puntuar)
        if len(filtered) < 3:
            print("⚠️  Pocas noticias relevantes → bajando threshold a 1")
            filtered = scored  # Ya están puntuados con score >= 1

        return filtered

    # -----------------------------
    # 📊 MARKET INDEX
    # -----------------------------
    def compute_market_index(self, articles: list) -> dict:
        if not articles:
            return {"index": 0, "signal_es": "NEUTRAL ⚖", "confidence": 0}

        impacts = [a["impact"] for a in articles if a.get("impact") is not None]

        if not impacts:
            return {"index": 0, "signal_es": "NEUTRAL ⚖", "confidence": 0}

        index = (sum(impacts) / len(impacts)) * 0.7  # Suavizado

        if index > 0.05:
            signal = "ALCISTA 📈"
        elif index < -0.05:
            signal = "BAJISTA 📉"
        else:
            signal = "NEUTRAL ⚖"

        return {
            "index":     round(index, 3),
            "signal_es": signal,
            "confidence": round(min(len(articles) / 10, 1.0), 2),
        }

    # -----------------------------
    # 🚨 ALERTAS
    # -----------------------------
    def generate_alerts(self, articles: list, memory: list) -> list:
        # ✅ MEJORA: lookup O(1) con set en lugar de O(n) con lista
        seen_titles = {m["title"] for m in memory}
        alerts      = []

        for a in articles:
            if abs(a.get("impact", 0)) > 0.25 and a.get("relevance", 0) >= 3:
                if a["title"] in seen_titles:
                    continue

                direction = "alcista" if a["impact"] > 0 else "bajista"
                alerts.append(f"🚨 Evento {direction}: {a['title'][:80]}")

        return alerts[:3]

    # -----------------------------
    # 🔄 MAIN
    # -----------------------------
    def update_news(self) -> dict:
        today    = datetime.utcnow().strftime("%Y-%m-%d")
        week_ago = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

        # Multi-fuente
        raw_news = fetch_news(week_ago, today) + fetch_gdelt_news() + fetch_rss_news()
        print(f"🧠 TOTAL RAW: {len(raw_news)}")

        # Normalizar estructura
        all_articles = []
        for a in raw_news:
            source = a.get("source")
            source_name = source.get("name") if isinstance(source, dict) else source

            all_articles.append({
                "title":       a.get("title", ""),
                "description": a.get("description", ""),
                "url":         a.get("url"),
                "source":      source_name,
                "publishedAt": a.get("publishedAt"),
            })

        # Filtrar
        try:
            filtered = self.filter_articles(all_articles, threshold=2)
        except Exception as e:
            print(f"❌ Error filtrando noticias: {e}")
            filtered = []

        # Deduplicar + procesar
        titles_seen = set()
        processed   = []

        for art, rel in filtered:
            title      = (art.get("title") or "").strip()
            norm_title = title.lower()[:80]

            if not title or norm_title in titles_seen:
                continue

            titles_seen.add(norm_title)

            text      = f"{title} {art.get('description', '')}"
            sentiment = get_sentiment(text)
            topic     = classify_topic(text)
            impact    = compute_impact(topic, sentiment)

            processed.append({
                "title":       title,
                "url":         art["url"],
                "source":      art["source"],
                "publishedAt": art["publishedAt"],
                "sentiment":   sentiment,
                "topic":       topic,
                "impact":      impact,
                "relevance":   rel,
            })

        # Memoria
        memory   = self.load_memory()
        combined = (memory + processed)[-MEMORY_MAX:]
        self.save_memory(combined)

        # Índice de mercado y alertas
        market = self.compute_market_index(processed)
        alerts = self.generate_alerts(processed, memory)

        # 🧠 Capa LLM: análisis estructurado por artículo + agregación por driver.
        # El cache por URL hace que sólo los artículos nuevos paguen llamada API.
        try:
            import sys as _sys
            _sys.path.insert(0, self.base_dir)
            from src.intel.aggregator import run_full_pipeline as _run_intel
            _run_intel(processed, max_articles=25)
        except Exception as _e:
            print(f"⚠️  Intel layer omitida: {_e}")

        # 🚨 Enriquecer alertas con la capa LLM (más precisa que VADER).
        # Lee news_intel.json y promueve los hits de driver con magnitude>=3
        # y confidence>=0.6 a alertas visibles. Cubre el caso "VADER nunca
        # produce alertas porque sus sentimientos son demasiado suaves".
        try:
            import json as _json
            intel_path = os.path.join(self.base_dir, "data", "news_intel.json")
            if os.path.exists(intel_path):
                with open(intel_path, "r", encoding="utf-8") as _f:
                    intel = _json.load(_f)
                seen_titles = {m["title"] for m in memory}
                llm_alerts = []
                driver_label_es = {
                    "china_demand": "China demand",  "weather_br": "Clima BR",
                    "weather_ar":   "Clima AR",      "weather_us": "Clima US",
                    "supply_global": "Oferta global","usda_report": "USDA",
                    "policy_ar":    "Política AR",   "policy_us":  "Política US",
                    "policy_br":    "Política BR",   "macro_usd":  "USD",
                    "macro_oil":    "Petróleo",      "logistics":  "Logística",
                    "biofuels":     "Biocombustibles", "geopolitics": "Geopolítica",
                }
                for driver, items in (intel.get("top_per_driver") or {}).items():
                    for it in items or []:
                        if it.get("magnitude", 0) < 3:
                            continue
                        if it.get("confidence", 0) < 0.6:
                            continue
                        title = (it.get("title") or "").strip()
                        if not title or title in seen_titles:
                            continue
                        seen_titles.add(title)  # de-dup entre drivers
                        is_bull = it.get("impact") == "bullish"
                        arrow = "📈 ALCISTA" if is_bull else ("📉 BAJISTA" if it.get("impact") == "bearish" else "⚪ NEUTRAL")
                        label = driver_label_es.get(driver, driver)
                        m = it.get("magnitude", 0)
                        llm_alerts.append(
                            f"{arrow} M{m} · {label} — {title[:90]}"
                        )
                # Merge: las alertas LLM van primero (más precisas), luego las VADER.
                alerts = (llm_alerts + alerts)[:5]
        except Exception as _e:
            print(f"⚠️  Alertas LLM no integradas: {_e}")

        # Forecast
        # ✅ MEJORA: bloque de ajuste eliminado (era código muerto).
        # El ajuste lo hace el modelo en pipeline.py.
        forecast = self.load_forecast()

        return {
            "articles":      combined[::-1],
            "market":        market,
            "alerts":        alerts,
            "forecast":      forecast,
            "fresh_count":   len(processed),     # ← batch fresco real
            "raw_count":     len(raw_news),      # ← total bruto antes de filtrar
            "filtered_count": len(filtered),      # ← post-filtro de relevancia
        }
