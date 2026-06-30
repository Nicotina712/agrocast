"""
News Monitor — Position conflict checker.

Logica Option B: si el robot tiene una posicion SHORT abierta y llegan
noticias BULLISH de alta magnitud/confianza, devuelve senal de cierre anticipado.
Y viceversa para LONG + BEARISH.

Uso tipico en live_runner.py (dentro del position guard):
    from templates_nuevos_mercados.news_intelligence.news_monitor import check_news_conflict
    conflict = check_news_conflict("UK100", position_type="SELL")
    if conflict["exit"]:
        # cerrar posicion
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_MVP_ROOT = Path(__file__).resolve().parent.parent.parent
_NEWS_DIR = _MVP_ROOT / "data" / "news_portfolio"

# Umbrales para activar cierre anticipado
MAG_THRESHOLD      = 3      # magnitud minima del articulo conflictivo (1-5)
CONF_THRESHOLD     = 0.60   # confianza minima del LLM (0-1)
MIN_CONFLICTS      = 2      # minimo de articulos conflictivos para activar salida
NEWS_MAX_AGE_SECS  = 14400  # 4 horas — noticias mas viejas se ignoran


def check_news_conflict(instrument: str, position_type: str) -> dict:
    """
    Revisa si las noticias vigentes contradicen la posicion abierta.

    Args:
        instrument:    Ej: "UK100", "BTCUSD", "XAUUSD"
        position_type: "BUY" (LONG) o "SELL" (SHORT)

    Returns:
        {
            "exit":       True/False,
            "reason":     str explicando por que se debe cerrar,
            "conflicts":  lista de articulos conflictivos,
            "score":      float indicador de urgencia (0.0-1.0),
            "data_age_h": horas desde el ultimo fetch de noticias
        }
    """
    news_file = _NEWS_DIR / f"{instrument}_latest.json"

    # Sin archivo de noticias = no hacer nada
    if not news_file.exists():
        return {"exit": False, "reason": "no_news_file", "conflicts": [], "score": 0.0, "data_age_h": None}

    try:
        data = json.loads(news_file.read_text(encoding="utf-8"))
    except Exception as e:
        return {"exit": False, "reason": f"read_error: {e}", "conflicts": [], "score": 0.0, "data_age_h": None}

    # Verificar frescura del dato
    fetched_str = data.get("fetched_at", "")
    data_age_h = None
    if fetched_str:
        try:
            fetched_at = datetime.fromisoformat(fetched_str)
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=timezone.utc)
            age_secs   = (datetime.now(timezone.utc) - fetched_at).total_seconds()
            data_age_h = round(age_secs / 3600, 1)
            if age_secs > NEWS_MAX_AGE_SECS:
                return {
                    "exit":       False,
                    "reason":     f"stale_data ({data_age_h}h old > {NEWS_MAX_AGE_SECS/3600}h max)",
                    "conflicts":  [],
                    "score":      0.0,
                    "data_age_h": data_age_h,
                }
        except Exception:
            pass

    articles = data.get("articles", [])
    if not articles:
        return {"exit": False, "reason": "no_articles", "conflicts": [], "score": 0.0, "data_age_h": data_age_h}

    # Direccion que contradice la posicion
    # SHORT open → buscar noticias BULLISH (mercado sube = posicion pierde)
    # LONG open  → buscar noticias BEARISH (mercado baja = posicion pierde)
    pt = position_type.upper()
    if pt in ("SELL", "SHORT"):
        conflict_direction = "bullish"
    elif pt in ("BUY", "LONG"):
        conflict_direction = "bearish"
    else:
        return {"exit": False, "reason": f"unknown_position_type: {position_type}", "conflicts": [], "score": 0.0, "data_age_h": data_age_h}

    # Filtrar articulos conflictivos de alta conviction
    conflicts = []
    for art in articles:
        intel = art.get("intel", {})
        if intel.get("_source") == "vader_fallback":
            continue  # ignorar fallbacks VADER, baja confianza
        if (
            intel.get("price_impact") == conflict_direction
            and intel.get("magnitude", 0) >= MAG_THRESHOLD
            and intel.get("confidence", 0.0) >= CONF_THRESHOLD
        ):
            conflicts.append({
                "title":     art.get("title", "")[:100],
                "magnitude": intel.get("magnitude"),
                "confidence": intel.get("confidence"),
                "rationale": intel.get("rationale", "")[:150],
                "source":    art.get("source", ""),
                "publishedAt": art.get("publishedAt", ""),
            })

    n = len(conflicts)
    if n < MIN_CONFLICTS:
        return {
            "exit":       False,
            "reason":     f"insuficientes_conflictos ({n}/{MIN_CONFLICTS} requeridos, dir={conflict_direction})",
            "conflicts":  conflicts,
            "score":      round(n / MIN_CONFLICTS, 2),
            "data_age_h": data_age_h,
        }

    # Score de urgencia: promedio ponderado de magnitud x confianza
    score = min(1.0, sum(c["magnitude"] * c["confidence"] for c in conflicts) / (MIN_CONFLICTS * 5))
    top   = sorted(conflicts, key=lambda x: x["magnitude"] * x["confidence"], reverse=True)[0]

    reason = (
        f"{n} noticias {conflict_direction.upper()} (mag>={MAG_THRESHOLD}, conf>={CONF_THRESHOLD}) "
        f"contradicen {pt}. Top: '{top['title']}' "
        f"(mag={top['magnitude']}, conf={top['confidence']:.2f})"
    )

    return {
        "exit":       True,
        "reason":     reason,
        "conflicts":  conflicts,
        "score":      round(score, 3),
        "data_age_h": data_age_h,
    }


def get_news_bias(instrument: str) -> dict:
    """
    Devuelve el sesgo general de noticias para un instrumento.
    Util como filtro pre-entrada (Option A futura).

    Returns:
        {
            "bias":       "bullish" | "bearish" | "neutral" | "mixed",
            "bull_score": float,
            "bear_score": float,
            "data_age_h": float
        }
    """
    news_file = _NEWS_DIR / f"{instrument}_latest.json"
    if not news_file.exists():
        return {"bias": "unknown", "bull_score": 0.0, "bear_score": 0.0, "data_age_h": None}

    try:
        data = json.loads(news_file.read_text(encoding="utf-8"))
    except Exception:
        return {"bias": "unknown", "bull_score": 0.0, "bear_score": 0.0, "data_age_h": None}

    fetched_str = data.get("fetched_at", "")
    data_age_h  = None
    if fetched_str:
        try:
            fetched_at = datetime.fromisoformat(fetched_str)
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=timezone.utc)
            data_age_h = round((datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600, 1)
        except Exception:
            pass

    articles = data.get("articles", [])
    bull_score = bear_score = 0.0
    for art in articles:
        intel = art.get("intel", {})
        if intel.get("_source") == "vader_fallback":
            continue
        mag  = intel.get("magnitude", 0)
        conf = intel.get("confidence", 0.0)
        if intel.get("price_impact") == "bullish" and mag >= 2:
            bull_score += mag * conf
        elif intel.get("price_impact") == "bearish" and mag >= 2:
            bear_score += mag * conf

    total = bull_score + bear_score
    if total == 0:
        bias = "neutral"
    elif bull_score / total > 0.65:
        bias = "bullish"
    elif bear_score / total > 0.65:
        bias = "bearish"
    else:
        bias = "mixed"

    return {
        "bias":       bias,
        "bull_score": round(bull_score, 2),
        "bear_score": round(bear_score, 2),
        "data_age_h": data_age_h,
    }
