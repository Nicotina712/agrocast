"""
src/intel/aggregator.py
Agrega los análisis estructurados de news_analyst en señales por driver.

Output: dict con un score por driver (china_demand, weather_br, supply_global, …)
y un consolidado global. Estos se inyectan al modelo ML como features
desagregadas (mucha más señal que un único 'news_sentiment' promedio).

Score por driver:
    score = sum(direction * magnitude * confidence) / N_articulos_driver
    direction: bullish=+1, bearish=-1, neutral=0
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Iterable

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUTPUT_PATH  = os.path.join(_PROJECT_ROOT, "data", "news_intel.json")
_HISTORY_PATH = os.path.join(_PROJECT_ROOT, "data", "news_intel_history.csv")

DRIVERS = [
    "china_demand", "weather_br", "weather_ar", "weather_us",
    "supply_global", "usda_report", "policy_ar", "policy_us",
    "policy_br", "macro_usd", "macro_oil", "logistics", "biofuels",
    "geopolitics",
]

_HORIZON_WEIGHT = {"1d": 1.0, "7d": 0.8, "30d": 0.5, "90d": 0.25}


def _direction(impact: str) -> int:
    return {"bullish": 1, "bearish": -1}.get(impact, 0)


def aggregate(articles_with_intel: Iterable[dict]) -> dict:
    """
    Recibe artículos ya enriquecidos con campo 'intel' (de news_analyst).
    Devuelve features por driver + consolidado.
    """
    by_driver_score = defaultdict(list)
    by_driver_count = defaultdict(int)
    top_articles    = defaultdict(list)

    total_signed     = 0.0
    total_weight     = 0.0
    n_articles       = 0
    n_high_impact    = 0

    for art in articles_with_intel:
        intel = art.get("intel") or {}
        if not intel:
            continue

        n_articles += 1

        direction  = _direction(intel.get("price_impact", "neutral"))
        magnitude  = float(intel.get("magnitude", 1))
        confidence = float(intel.get("confidence", 0.5))
        horizon_w  = _HORIZON_WEIGHT.get(intel.get("horizon", "7d"), 0.8)

        contribution = direction * magnitude * confidence * horizon_w

        total_signed += contribution
        total_weight += magnitude * confidence * horizon_w

        if magnitude >= 4 and confidence >= 0.6:
            n_high_impact += 1

        for driver in intel.get("drivers", []) or []:
            if driver == "other":
                continue
            by_driver_score[driver].append(contribution)
            by_driver_count[driver] += 1

            top_articles[driver].append({
                "title":      art.get("title", "")[:120],
                "impact":     intel.get("price_impact"),
                "magnitude":  intel.get("magnitude"),
                "confidence": intel.get("confidence"),
                "url":        art.get("url"),
            })

    # Score por driver: promedio normalizado a [-1, +1]
    driver_features = {}
    for d in DRIVERS:
        scores = by_driver_score.get(d, [])
        if scores:
            raw = sum(scores) / len(scores)
            # Normalizar a [-1, 1]: max teórico = 1*5*1*1 = 5
            driver_features[f"news_{d}_signal"] = round(max(-1.0, min(1.0, raw / 5.0)), 4)
            driver_features[f"news_{d}_count"]  = by_driver_count[d]
        else:
            driver_features[f"news_{d}_signal"] = 0.0
            driver_features[f"news_{d}_count"]  = 0

    composite = total_signed / total_weight if total_weight > 0 else 0.0
    composite = round(max(-1.0, min(1.0, composite / 5.0)), 4)

    # Top 3 artículos por driver (para UI)
    top_per_driver = {
        d: sorted(top_articles[d],
                  key=lambda a: (a.get("magnitude") or 0) * (a.get("confidence") or 0),
                  reverse=True)[:3]
        for d in DRIVERS if top_articles[d]
    }

    result = {
        "generated_at":   datetime.utcnow().isoformat(),
        "n_articles":     n_articles,
        "n_high_impact":  n_high_impact,
        "composite":      composite,     # [-1, +1] consolidado
        "drivers":        driver_features,
        "top_per_driver": top_per_driver,
    }
    return result


def save_intel(intel: dict, path: str = _OUTPUT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(intel, f, ensure_ascii=False, indent=2)
    _append_history(intel)


def _append_history(intel: dict) -> None:
    """Append daily snapshot to CSV so el modelo pueda usarlo como serie."""
    try:
        import csv
        today = datetime.utcnow().strftime("%Y-%m-%d")
        row = {"Date": today,
               "intel_composite": intel.get("composite", 0.0),
               "intel_n_articles": intel.get("n_articles", 0),
               "intel_n_high_impact": intel.get("n_high_impact", 0)}
        for k, v in (intel.get("drivers") or {}).items():
            row[k] = v

        os.makedirs(os.path.dirname(_HISTORY_PATH), exist_ok=True)

        existing = []
        if os.path.exists(_HISTORY_PATH):
            with open(_HISTORY_PATH, "r", encoding="utf-8", newline="") as f:
                existing = [r for r in csv.DictReader(f) if r.get("Date") != today]

        all_keys = set(row.keys())
        for r in existing:
            all_keys.update(r.keys())
        cols = ["Date"] + sorted(k for k in all_keys if k != "Date")

        with open(_HISTORY_PATH, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in existing:
                w.writerow({c: r.get(c, "") for c in cols})
            w.writerow({c: row.get(c, "") for c in cols})
    except Exception as e:
        print(f"[intel] no se pudo persistir history: {e}")


def load_intel(path: str = _OUTPUT_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def run_full_pipeline(articles: list, max_articles: int = 25) -> dict:
    """
    Pipeline completo: analiza con LLM + agrega + persiste.
    """
    from src.intel.news_analyst import analyze_batch

    enriched = analyze_batch(articles, max_articles=max_articles)
    intel    = aggregate(enriched)
    save_intel(intel)
    print(f"[intel] composite={intel['composite']:+.3f}  "
          f"n={intel['n_articles']}  high_impact={intel['n_high_impact']}")
    return intel
