"""
Tier-1 sentiment classifier for mass news processing.
Attempts FinBERT (ProsusAI/finbert) if torch is installed;
falls back to VADER + keyword heuristics otherwise.
Returns structured sentiment + escalation flag for Tier-2 (Claude Sonnet).
"""

import re
from dataclasses import dataclass, asdict

_USE_FINBERT = False
_finbert_pipeline = None

try:
    from transformers import pipeline as hf_pipeline
    import torch  # noqa: F401
    _USE_FINBERT = True
except ImportError:
    pass

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except ImportError:
    _vader = None


HIGH_IMPACT_KEYWORDS = [
    "wasde", "usda", "opec", "tariff", "trade war", "embargo",
    "drought", "flood", "la niña", "la nina", "el niño", "el nino",
    "china", "crush margin", "bird flu", "african swine fever",
    "argentina", "brazil", "export ban", "retenciones", "derechos de exportación",
    "oil crash", "oil shock", "recession", "fed rate", "interest rate",
    "black sea", "ukraine", "russia", "iran", "sanctions",
    "crop failure", "yield shock", "planting delay",
    "biden", "trump", "xi jinping", "lula", "milei",
]

COMMODITY_KEYWORDS = [
    "soybean", "soy", "soja", "corn", "maiz", "wheat", "trigo",
    "oilseed", "meal", "crush", "basis", "futures", "cbot",
    "harvest", "planting", "acreage", "stocks", "exports",
]


@dataclass
class SentimentResult:
    text: str
    headline: str
    sentiment: str        # "bullish" | "bearish" | "neutral"
    confidence: float     # 0.0 - 1.0
    score: float          # -1.0 to 1.0
    method: str           # "finbert" | "vader_keywords"
    escalate: bool        # True → send to Claude Sonnet tier-2
    escalation_reason: str
    high_impact_topics: list


def _load_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None and _USE_FINBERT:
        _finbert_pipeline = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            top_k=None,
            truncation=True,
            max_length=512,
        )
    return _finbert_pipeline


def _classify_finbert(text: str) -> tuple[str, float, float]:
    pipe = _load_finbert()
    if pipe is None:
        raise RuntimeError("FinBERT not available")
    results = pipe(text[:512])[0]
    label_map = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}
    scores = {r["label"]: r["score"] for r in results}
    best_label = max(scores, key=scores.get)
    sentiment = label_map.get(best_label, "neutral")
    confidence = scores[best_label]
    net_score = scores.get("positive", 0) - scores.get("negative", 0)
    return sentiment, confidence, net_score


def _classify_vader(text: str) -> tuple[str, float, float]:
    if _vader is None:
        return "neutral", 0.5, 0.0
    scores = _vader.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.15:
        sentiment = "bullish"
    elif compound <= -0.15:
        sentiment = "bearish"
    else:
        sentiment = "neutral"
    confidence = min(abs(compound) * 1.5, 1.0)
    return sentiment, confidence, compound


def _detect_high_impact(text: str) -> list[str]:
    text_lower = text.lower()
    return [kw for kw in HIGH_IMPACT_KEYWORDS if kw in text_lower]


def _is_commodity_relevant(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in COMMODITY_KEYWORDS)


def _should_escalate(
    sentiment: str,
    confidence: float,
    high_impact: list[str],
    is_relevant: bool,
) -> tuple[bool, str]:
    reasons = []
    if confidence < 0.60:
        reasons.append(f"low_confidence({confidence:.2f})")
    if len(high_impact) >= 2:
        reasons.append(f"multi_impact({','.join(high_impact[:3])})")
    if any(kw in high_impact for kw in ["wasde", "opec", "tariff", "trade war",
                                         "embargo", "drought", "export ban"]):
        reasons.append("critical_topic")
    if is_relevant and sentiment != "neutral" and confidence < 0.75:
        reasons.append("ambiguous_commodity_signal")

    escalate = len(reasons) > 0
    return escalate, "; ".join(reasons) if reasons else ""


def classify_article(headline: str, body: str = "") -> SentimentResult:
    text = f"{headline}. {body}" if body else headline
    high_impact = _detect_high_impact(text)
    is_relevant = _is_commodity_relevant(text)

    if _USE_FINBERT:
        try:
            sentiment, confidence, score = _classify_finbert(text)
            method = "finbert"
        except Exception:
            sentiment, confidence, score = _classify_vader(text)
            method = "vader_keywords"
    else:
        sentiment, confidence, score = _classify_vader(text)
        method = "vader_keywords"

    if high_impact and method == "vader_keywords":
        confidence = max(confidence, 0.45)

    escalate, reason = _should_escalate(sentiment, confidence, high_impact, is_relevant)

    return SentimentResult(
        text=text[:200],
        headline=headline[:150],
        sentiment=sentiment,
        confidence=round(confidence, 3),
        score=round(score, 4),
        method=method,
        escalate=escalate,
        escalation_reason=reason,
        high_impact_topics=high_impact,
    )


def classify_batch(articles: list[dict]) -> list[dict]:
    results = []
    for art in articles:
        headline = art.get("title", art.get("headline", ""))
        body = art.get("summary", art.get("body", art.get("text", "")))
        result = classify_article(headline, body)
        results.append(asdict(result))
    return results


if __name__ == "__main__":
    test_articles = [
        {"title": "USDA WASDE report shows surprise drop in soybean ending stocks",
         "summary": "Ending stocks fell to 130M bushels, well below trade expectations of 185M."},
        {"title": "Oil prices surge 8% after OPEC announces emergency cuts",
         "summary": "Brent crude jumped above $85 as Saudi Arabia leads coordinated production cuts."},
        {"title": "China increases soybean imports amid strong crush margins",
         "summary": "Chinese crushers ramped up purchases with margins at 3-month highs."},
        {"title": "Weather remains favorable for US corn belt planting",
         "summary": "Normal rainfall and temperatures support timely planting across the Midwest."},
    ]
    for r in classify_batch(test_articles):
        print(f"  [{r['sentiment']:>7}] conf={r['confidence']:.2f} esc={r['escalate']} | {r['headline'][:60]}")
        if r["escalation_reason"]:
            print(f"           → reason: {r['escalation_reason']}")
