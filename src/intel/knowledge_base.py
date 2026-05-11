"""
RAG Knowledge Base for the Intelligence Engine.
Indexes:
  1. shock_catalog.csv — 1093 historical price shocks with outcomes
  2. event_memory.csv  — 1424 detected events with narrative drivers
  3. Embedded market literature — academic/institutional knowledge about commodity markets

Uses TF-IDF (sklearn) for retrieval — no GPU, no new dependencies.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_BASE, "..", ".."))
_ARTIFACTS = os.path.join(_ROOT, "artifacts")
_DATA = os.path.join(_ROOT, "data")

MARKET_LITERATURE = [
    {
        "id": "lit_oil_commodity_correlation",
        "category": "cross_commodity",
        "text": (
            "Oil-commodity correlation during shocks. Crude oil price shocks propagate to "
            "agricultural commodities through three channels: (1) input costs — fertilizer, fuel, "
            "and transport costs rise with oil; (2) biofuel substitution — high oil makes ethanol/biodiesel "
            "competitive, diverting corn/soy from feed; (3) macro risk-off flows — commodity funds liquidate "
            "across the board. Historical correlation soy-oil: 0.55-0.65 during shocks. "
            "Typical soy reaction to oil drop >5%: -1.5% to -3% within 48h. "
            "Oil-driven rallies in soy tend to fade 60-70% within 30 days unless fundamental confirmation follows."
        ),
    },
    {
        "id": "lit_china_demand_cycle",
        "category": "fundamentals",
        "text": (
            "China soybean demand cycle and crush margins. China imports 60% of global soybean trade. "
            "Crush margin is the primary driver of Chinese import pace. When crush margin exceeds "
            "$20/ton, Chinese crushers accelerate purchases aggressively. Key cycle: "
            "margins rise → buying surge (2-4 week lag) → port congestion → temporary pause → "
            "margins fall → buying slows. CNY depreciation against USD reduces Chinese purchasing power, "
            "bearish for soybean demand. African Swine Fever outbreaks reduce soymeal demand. "
            "Typical soy price impact of China demand surge: +3-8% over 4-6 weeks."
        ),
    },
    {
        "id": "lit_wasde_surprise_mechanics",
        "category": "events",
        "text": (
            "WASDE report surprise mechanics. The USDA WASDE report is the single highest-impact "
            "recurring event for soybean prices. Key variables: US ending stocks, world ending stocks, "
            "Brazilian/Argentine production estimates. Market impact rules: "
            "Stocks surprise <-10M bu → bullish +1.5-3% within 24h. "
            "Stocks surprise >+10M bu → bearish -1-2.5% within 24h. "
            "Post-WASDE moves that align with prior trend have 75% persistence at 7d. "
            "Post-WASDE moves against prior trend fade 60% at 7d. "
            "Trading volume spikes 200-400% on report day. Position squaring in 48h before report is common."
        ),
    },
    {
        "id": "lit_weather_premium",
        "category": "weather",
        "text": (
            "Weather premium in soybean markets. Weather risk premium builds during critical growth "
            "stages: US planting (Apr-May), US pollination (Jul-Aug), Brazilian planting (Oct-Nov), "
            "Argentine filling (Jan-Feb). Key dynamics: "
            "Drought during US pollination can reduce yields 20-40%, price impact +15-30%. "
            "La Niña typically brings dryness to Argentina/southern Brazil, bullish for soy. "
            "El Niño brings excess rain to Argentina, bearish for quality but sometimes bullish for yield uncertainty. "
            "Weather premium tends to be front-loaded: prices jump on forecast, then adjust as actual conditions clarify. "
            "75% of weather-driven rallies peak within 3 weeks of initial scare."
        ),
    },
    {
        "id": "lit_trade_war_pattern",
        "category": "geopolitical",
        "text": (
            "Trade war patterns in soybean markets. US-China trade tensions have outsized impact on soybeans "
            "because China is the dominant buyer of US soy. Historical pattern: "
            "Tariff announcement → immediate -5% to -15% (US soy, varies by severity). "
            "Brazilian premium widens as China shifts sourcing. US basis collapses. "
            "Resolution/de-escalation → recovery rally of 50-80% of initial drop within 2-4 weeks. "
            "Key indicator: Chinese booking pace for US vs Brazilian soy (USDA export inspections). "
            "During 2018-2019 trade war, US soy lost $2/bu (~18%) and took 18 months to recover."
        ),
    },
    {
        "id": "lit_speculative_positioning",
        "category": "positioning",
        "text": (
            "Speculative positioning and soybean prices. CFTC Commitments of Traders (COT) data reveals "
            "managed money positioning. Key signals: "
            "Net long >200K contracts → crowded long, vulnerable to liquidation. "
            "Net short >100K contracts → extreme pessimism, contrarian buy signal. "
            "Rate of change matters more than level: rapid build of 50K+ contracts in 4 weeks "
            "signals conviction move. Funds typically anticipate fundamentals by 2-4 weeks. "
            "When funds are max long AND commercial hedgers are max short, reversal risk is highest. "
            "Open interest expansion with rising prices = healthy trend. Declining OI with rising prices = short covering rally, likely to fade."
        ),
    },
    {
        "id": "lit_south_american_supply",
        "category": "fundamentals",
        "text": (
            "South American supply dynamics. Brazil and Argentina together produce ~55% of world soybeans. "
            "Brazilian harvest peaks Feb-Apr, Argentine harvest peaks Apr-Jun. "
            "Key risk factors: Argentine export policy (retenciones, cepo cambiario), "
            "Brazilian logistics bottleneck at Santos/Paranaguá ports, Paraguay river levels affecting barge traffic. "
            "Uruguay basis reflects local supply/demand: premium signals tight local supply or strong export demand. "
            "When Argentine farmers hoard soybeans (common under inflationary/political uncertainty), "
            "global supply tightens and prices rally. Farmer selling pace in Brazil/Argentina is "
            "a leading indicator — slow selling = bullish, rapid selling = bearish for prices."
        ),
    },
    {
        "id": "lit_carry_basis_theory",
        "category": "market_structure",
        "text": (
            "Carry, basis, and term structure in soybean futures. "
            "Contango (front < back) = ample supply, no urgency. "
            "Backwardation (front > back) = tight supply, demand for immediate delivery. "
            "Basis = local cash price - nearby futures. Positive basis = local premium (tight local supply). "
            "Negative basis = local discount (ample supply, logistic bottleneck). "
            "Carry-to-full signal: when storage costs exceed the contango spread, it signals oversupply. "
            "Inverse carry (backwardation > storage cost) is the strongest bullish fundamental signal. "
            "Roll yield: in persistent contango, rolling long futures loses money each month."
        ),
    },
    {
        "id": "lit_momentum_reversion",
        "category": "technical",
        "text": (
            "Momentum and mean reversion in commodity futures. "
            "CFA Institute (2025) found momentum is consistently the #1 predictive feature for commodity returns. "
            "Short-term momentum (5-20 days) predicts continuation. "
            "Long-term momentum (60-120 days) predicts continuation but with higher reversal risk. "
            "Skewness is powerful at short horizons — negative skew signals crash risk. "
            "Mean reversion dominates when: RSI >70 or <30, price >2 standard deviations from 60d mean, "
            "or after event-driven spikes without fundamental confirmation. "
            "Cross-sectional momentum (soy vs corn vs wheat relative performance) outperforms time-series momentum."
        ),
    },
    {
        "id": "lit_news_precedes_data",
        "category": "methodology",
        "text": (
            "News precedes fundamental data in commodity price discovery. "
            "Academic research (arXiv 2508.06497, 2025) shows that removing news embeddings from "
            "commodity shock prediction models drops AUC from 0.94 to 0.46. "
            "News volume and persistence of coverage predict better than sentiment tone (MDPI 2025). "
            "The information flow: speculation/rumor → news coverage → data confirmation → price adjustment. "
            "By the time official data arrives (WASDE, export inspections, crop progress), "
            "60-80% of the price move has already occurred. "
            "Implication: reading the news intelligently IS prediction. "
            "The skill is distinguishing noise from signal — volume of coverage on a topic is a better "
            "indicator than individual article sentiment."
        ),
    },
]


class KnowledgeBase:
    def __init__(self):
        self._docs = []
        self._vectorizer = None
        self._tfidf_matrix = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        self._docs = []
        self._load_shock_catalog()
        self._load_event_memory()
        self._load_literature()
        self._build_index()
        self._loaded = True

    def _load_shock_catalog(self):
        path = os.path.join(_ARTIFACTS, "shock_catalog.csv")
        if not os.path.exists(path):
            return
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                shock_type = row.get("shock_type", "unknown")
                direction = row.get("shock_direction", "")
                date = str(row.get("Date", ""))[:10]
                price = row.get("Soybeans", 0)
                ret_5d = row.get("ret_5d_at_shock", 0)
                ret_30d = row.get("ret_30d_pct", 0)
                persistence = row.get("persistence_30d_pct", 0)
                survived = row.get("survived_30d", False)
                spike_pct = row.get("spike_size_pct", 0)
                oil_5d = row.get("oil_5d_at_shock", 0)

                text = (
                    f"Historical shock on {date}: {shock_type} {direction}. "
                    f"Soybean price ${price:.0f}. Initial move {spike_pct:.1f}%. "
                    f"5-day return {ret_5d*100:.1f}%. 30-day return {ret_30d:.1f}%. "
                    f"Persistence at 30d: {persistence:.0f}%. Survived: {survived}. "
                    f"Oil 5d change: {oil_5d*100:.1f}%."
                )
                self._docs.append({
                    "id": f"shock_{date}_{shock_type}",
                    "category": "shock_history",
                    "date": date,
                    "text": text,
                    "metadata": {
                        "shock_type": shock_type,
                        "direction": direction,
                        "spike_pct": spike_pct,
                        "ret_30d": ret_30d,
                        "survived": survived,
                    },
                })
        except Exception as e:
            print(f"[KB] Error loading shock_catalog: {e}")

    def _load_event_memory(self):
        path = os.path.join(_ARTIFACTS, "event_memory.csv")
        if not os.path.exists(path):
            return
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                date = str(row.get("date", ""))[:10]
                price = row.get("price", 0)
                event_type = row.get("event_type", "")
                direction = row.get("direction", "")
                primary = row.get("primary_driver", "")
                secondary = row.get("secondary_driver", "")
                narrative = row.get("narrative_strength", 0)
                speculation = row.get("speculation_level", 0)
                fade_risk = row.get("fade_risk", 0)
                outcome_7d = row.get("outcome_7d_pct", 0)
                outcome_30d = row.get("outcome_30d_pct", 0)
                faded = row.get("fade_occurred_7d", False)

                text = (
                    f"Event {date}: {event_type} {direction}. Price ${price:.0f}. "
                    f"Primary driver: {primary}. Secondary: {secondary}. "
                    f"Narrative strength: {narrative:.2f}. Speculation level: {speculation:.2f}. "
                    f"Fade risk: {fade_risk:.2f}. "
                    f"Outcome 7d: {outcome_7d:.1f}%. Outcome 30d: {outcome_30d:.1f}%. "
                    f"Faded at 7d: {faded}."
                )
                self._docs.append({
                    "id": f"event_{date}_{event_type}",
                    "category": "event_history",
                    "date": date,
                    "text": text,
                    "metadata": {
                        "event_type": event_type,
                        "direction": direction,
                        "primary_driver": primary,
                        "outcome_7d": outcome_7d,
                        "outcome_30d": outcome_30d,
                        "faded": faded,
                    },
                })
        except Exception as e:
            print(f"[KB] Error loading event_memory: {e}")

    def _load_literature(self):
        for entry in MARKET_LITERATURE:
            self._docs.append({
                "id": entry["id"],
                "category": entry["category"],
                "date": "reference",
                "text": entry["text"],
                "metadata": {"type": "literature"},
            })

    def _build_index(self):
        if not self._docs:
            return
        texts = [d["text"] for d in self._docs]
        self._vectorizer = TfidfVectorizer(
            max_features=8000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 8, category: str = None) -> list[dict]:
        if not self._loaded:
            self.load()
        if self._vectorizer is None or self._tfidf_matrix is None:
            return []

        query_vec = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        if category:
            for i, doc in enumerate(self._docs):
                if doc["category"] != category:
                    similarities[i] = 0.0

        top_indices = similarities.argsort()[-top_k:][::-1]
        results = []
        for idx in top_indices:
            if similarities[idx] < 0.01:
                continue
            doc = self._docs[idx].copy()
            doc["relevance"] = round(float(similarities[idx]), 4)
            results.append(doc)

        return results

    def get_analogous_shocks(self, shock_type: str, direction: str, top_k: int = 5) -> list[dict]:
        query = f"{shock_type} {direction} shock soybean price movement outcome"
        results = self.search(query, top_k=top_k * 2, category="shock_history")
        filtered = [
            r for r in results
            if r.get("metadata", {}).get("shock_type") == shock_type
            and r.get("metadata", {}).get("direction") == direction
        ]
        return filtered[:top_k] if filtered else results[:top_k]

    def get_literature_context(self, topics: list[str]) -> list[dict]:
        query = " ".join(topics)
        return self.search(query, top_k=4, category=None)

    def get_stats(self) -> dict:
        if not self._loaded:
            self.load()
        categories = {}
        for doc in self._docs:
            cat = doc["category"]
            categories[cat] = categories.get(cat, 0) + 1
        return {
            "total_documents": len(self._docs),
            "categories": categories,
            "index_features": self._vectorizer.max_features if self._vectorizer else 0,
        }


_kb_instance = None

def get_knowledge_base() -> KnowledgeBase:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
        _kb_instance.load()
    return _kb_instance


if __name__ == "__main__":
    kb = get_knowledge_base()
    print(f"Knowledge Base loaded: {kb.get_stats()}")
    print("\n--- Search: 'oil crash impact on soybean' ---")
    for r in kb.search("oil crash impact on soybean prices", top_k=5):
        print(f"  [{r['category']}] rel={r['relevance']:.3f} | {r['text'][:100]}...")
    print("\n--- Analogous shocks: oil_driven up ---")
    for r in kb.get_analogous_shocks("oil_driven", "up", top_k=3):
        print(f"  {r['date']} | spike={r['metadata']['spike_pct']:.1f}% → 30d={r['metadata']['ret_30d']:.1f}%")
