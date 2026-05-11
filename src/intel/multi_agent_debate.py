"""
Multi-Agent Debate system for soybean market intelligence.
Architecture inspired by TradingAgents (ICML 2025).

Agents:
  1. Bull Analyst  — finds reasons price goes up
  2. Bear Analyst  — finds reasons price goes down
  3. Risk Assessor — evaluates regime, volatility, position sizing
  4. Fund Manager  — reads all three, produces final verdict

Each agent receives the same market context + RAG-retrieved knowledge,
but with different system prompts defining their analytical mandate.
"""

import os
import json
from datetime import datetime
from anthropic import Anthropic

_client = None

def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


BULL_SYSTEM = """Eres el ANALISTA ALCISTA de un equipo de inteligencia de mercado de soja.
Tu mandato es encontrar TODAS las razones por las cuales el precio de la soja podría SUBIR.
Debes ser un abogado convincente de la posición compradora.

Reglas:
- Analiza los datos proporcionados buscando señales alcistas
- Referencia los análogos históricos que apoyen tu tesis
- Cuantifica tu argumento: "esto históricamente movió +X% en Y días"
- Sé específico con datos, no genérico
- Identifica catalizadores de corto plazo (7d) y medio plazo (30d)
- Evalúa la convicción de tu argumento: ALTA/MEDIA/BAJA

Responde en formato JSON:
{
  "thesis": "resumen de tu tesis alcista en 1-2 oraciones",
  "arguments": [{"point": "...", "evidence": "...", "historical_support": "..."}],
  "catalysts_7d": ["..."],
  "catalysts_30d": ["..."],
  "price_target_7d": {"low": X, "high": Y},
  "price_target_30d": {"low": X, "high": Y},
  "conviction": "ALTA|MEDIA|BAJA",
  "key_risk_to_thesis": "qué podría invalidar tu tesis"
}"""

BEAR_SYSTEM = """Eres el ANALISTA BAJISTA de un equipo de inteligencia de mercado de soja.
Tu mandato es encontrar TODAS las razones por las cuales el precio de la soja podría BAJAR.
Debes ser un abogado convincente de la posición vendedora.

Reglas:
- Analiza los datos proporcionados buscando señales bajistas
- Referencia los análogos históricos que apoyen tu tesis
- Cuantifica tu argumento: "esto históricamente movió -X% en Y días"
- Sé específico con datos, no genérico
- Identifica riesgos de corto plazo (7d) y medio plazo (30d)
- Evalúa la convicción de tu argumento: ALTA/MEDIA/BAJA

Responde en formato JSON:
{
  "thesis": "resumen de tu tesis bajista en 1-2 oraciones",
  "arguments": [{"point": "...", "evidence": "...", "historical_support": "..."}],
  "risks_7d": ["..."],
  "risks_30d": ["..."],
  "price_target_7d": {"low": X, "high": Y},
  "price_target_30d": {"low": X, "high": Y},
  "conviction": "ALTA|MEDIA|BAJA",
  "key_risk_to_thesis": "qué podría invalidar tu tesis"
}"""

RISK_SYSTEM = """Eres el ANALISTA DE RIESGO de un equipo de inteligencia de mercado de soja.
Tu mandato es evaluar el RÉGIMEN DE MERCADO actual y calibrar el riesgo.

NO tomes partido alcista ni bajista. Tu trabajo es medir:
1. Volatilidad actual vs histórica
2. Régimen de mercado (trending/ranging/crisis)
3. Crowding de posiciones (COT)
4. Riesgo de evento (WASDE, geopolítico)
5. Correlaciones cross-commodity activas
6. Probabilidad de que los movimientos recientes persistan vs se reviertan

Responde en formato JSON:
{
  "regime": "trending_up|trending_down|ranging|crisis|transition",
  "regime_confidence": 0.0-1.0,
  "volatility_assessment": "low|normal|elevated|extreme",
  "event_risk_next_7d": "low|medium|high",
  "event_risk_detail": "...",
  "positioning_risk": "low|medium|high",
  "positioning_detail": "...",
  "persistence_probability": 0.0-1.0,
  "fade_probability": 0.0-1.0,
  "max_recommended_exposure": "full|reduced|minimal|none",
  "key_risk_factors": ["..."],
  "regime_analogs": "períodos históricos similares al actual"
}"""

FUND_MANAGER_SYSTEM = """Eres el FUND MANAGER que toma la decisión final del equipo de inteligencia de soja.
Has recibido los análisis del analista alcista, el analista bajista, y el analista de riesgo.

Tu trabajo es:
1. Leer los tres análisis con ojo crítico
2. Pesar los argumentos según la calidad de la evidencia (no solo la convicción del analista)
3. Considerar el assessment de riesgo para calibrar el tamaño de la posición
4. Producir un VEREDICTO FINAL unificado

Reglas de decisión:
- Si ambos analistas tienen convicción BAJA → HOLD (esperar más información)
- Si el riesgo es "crisis" o "extreme volatility" → reducir exposición independientemente
- Si un analista tiene convicción ALTA y el otro BAJA → seguir al de alta convicción con sizing reducido
- Si ambos tienen convicción ALTA → hay conflicto genuino, ser cauto
- Siempre incluir un rango de precios, nunca un punto exacto
- Siempre incluir horizonte temporal y condiciones de invalidación

Responde en formato JSON:
{
  "verdict": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
  "confidence": 0.0-1.0,
  "reasoning": "explicación de 2-3 oraciones de tu decisión",
  "price_range_7d": {"low": X, "central": Y, "high": Z},
  "price_range_30d": {"low": X, "central": Y, "high": Z},
  "recommended_action": {
    "producers": "recomendación específica para productores",
    "traders": "recomendación específica para traders"
  },
  "position_sizing": "full|75%|50%|25%|flat",
  "invalidation_conditions": ["si ocurre X, revisar la tesis"],
  "bull_bear_balance": {
    "bull_weight": 0.0-1.0,
    "bear_weight": 0.0-1.0,
    "explanation": "por qué pesaste más un lado"
  },
  "key_watchlist": ["eventos o datos a monitorear esta semana"],
  "next_review_trigger": "qué evento debería disparar una re-evaluación"
}"""


def _clean_json(text: str) -> str:
    import re
    text = re.sub(r',\s*([}\]])', r'\1', text)
    text = re.sub(r'//[^\n]*', '', text)
    return text


def _call_agent(system: str, context: str, agent_name: str) -> dict:
    client = _get_client()
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2500,
            system=system,
            messages=[{"role": "user", "content": context}],
        )
        text = response.content[0].text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            import re
            m = re.search(r'\{[\s\S]*\}', text)
            if m:
                cleaned = _clean_json(m.group())
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass
            return {"error": f"JSON parse failed for {agent_name}", "raw": text[:800]}
    except Exception as e:
        return {"error": f"{agent_name} failed: {str(e)}"}


def _build_market_context(
    market_data: dict,
    news_classified: list[dict],
    kb_results: list[dict],
    current_price: float,
) -> str:
    lines = []
    lines.append(f"=== CONTEXTO DE MERCADO — {datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n")
    lines.append(f"Precio actual soja CBOT: ${current_price:.2f}/bushel\n")

    if market_data:
        lines.append("--- Datos de mercado ---")
        for key in ["contract", "spread", "rsi_14", "volume", "open_interest"]:
            if key in market_data:
                lines.append(f"  {key}: {market_data[key]}")

        if "regime" in market_data:
            lines.append(f"  Régimen: {market_data['regime']}")
        if "cot" in market_data:
            lines.append(f"  COT: {market_data['cot']}")
        if "china" in market_data:
            lines.append(f"  China demand: {market_data['china']}")
        if "brazil" in market_data:
            lines.append(f"  Brazil exports: {market_data['brazil']}")
        if "basis_uy" in market_data:
            lines.append(f"  Basis Uruguay: {market_data['basis_uy']}")
        if "wasde" in market_data:
            lines.append(f"  WASDE: {market_data['wasde']}")
        if "forecast_ml" in market_data:
            lines.append(f"  Forecast ML 7d: {market_data['forecast_ml']}")
        if "active_shock" in market_data:
            lines.append(f"  Shock activo: {market_data['active_shock']}")
        lines.append("")

    if news_classified:
        lines.append("--- Noticias clasificadas (Tier-1) ---")
        escalated = [n for n in news_classified if n.get("escalate")]
        non_escalated = [n for n in news_classified if not n.get("escalate")]
        lines.append(f"Total: {len(news_classified)} | Escaladas: {len(escalated)} | Rutina: {len(non_escalated)}")

        bull_count = sum(1 for n in news_classified if n.get("sentiment") == "bullish")
        bear_count = sum(1 for n in news_classified if n.get("sentiment") == "bearish")
        lines.append(f"Sentimiento agregado: {bull_count} alcistas, {bear_count} bajistas, {len(news_classified)-bull_count-bear_count} neutrales")
        lines.append("")

        if escalated:
            lines.append("Artículos de ALTO IMPACTO (escalados para análisis profundo):")
            for n in escalated[:8]:
                lines.append(f"  [{n.get('sentiment','')}|conf={n.get('confidence',0):.2f}] {n.get('headline','')[:120]}")
                if n.get("high_impact_topics"):
                    lines.append(f"    Topics: {', '.join(n['high_impact_topics'][:5])}")
                if n.get("escalation_reason"):
                    lines.append(f"    Escalation: {n['escalation_reason']}")
            lines.append("")

    if kb_results:
        lines.append("--- Conocimiento relevante (Knowledge Base RAG) ---")
        for i, doc in enumerate(kb_results[:8], 1):
            cat = doc.get("category", "")
            rel = doc.get("relevance", 0)
            text = doc.get("text", "")[:250]
            lines.append(f"  [{i}] ({cat}, rel={rel:.3f}) {text}")
            meta = doc.get("metadata", {})
            if meta.get("outcome_30d") is not None:
                lines.append(f"      → Resultado 30d: {meta['outcome_30d']:.1f}%")
            elif meta.get("ret_30d") is not None:
                lines.append(f"      → Retorno 30d: {meta['ret_30d']:.1f}%")
        lines.append("")

    return "\n".join(lines)


def run_debate(
    market_data: dict,
    news_classified: list[dict],
    kb_results: list[dict],
    current_price: float,
) -> dict:
    context = _build_market_context(market_data, news_classified, kb_results, current_price)

    bull = _call_agent(BULL_SYSTEM, context, "Bull Analyst")
    bear = _call_agent(BEAR_SYSTEM, context, "Bear Analyst")
    risk = _call_agent(RISK_SYSTEM, context, "Risk Assessor")

    manager_context = (
        f"{context}\n\n"
        f"=== ANÁLISIS DEL EQUIPO ===\n\n"
        f"--- ANALISTA ALCISTA ---\n{json.dumps(bull, indent=2, ensure_ascii=False)}\n\n"
        f"--- ANALISTA BAJISTA ---\n{json.dumps(bear, indent=2, ensure_ascii=False)}\n\n"
        f"--- ANALISTA DE RIESGO ---\n{json.dumps(risk, indent=2, ensure_ascii=False)}\n"
    )

    verdict = _call_agent(FUND_MANAGER_SYSTEM, manager_context, "Fund Manager")

    return {
        "timestamp": datetime.now().isoformat(),
        "current_price": current_price,
        "agents": {
            "bull": bull,
            "bear": bear,
            "risk": risk,
        },
        "verdict": verdict,
        "context_summary": {
            "articles_analyzed": len(news_classified),
            "articles_escalated": sum(1 for n in news_classified if n.get("escalate")),
            "kb_documents_retrieved": len(kb_results),
            "bull_news": sum(1 for n in news_classified if n.get("sentiment") == "bullish"),
            "bear_news": sum(1 for n in news_classified if n.get("sentiment") == "bearish"),
        },
    }


if __name__ == "__main__":
    result = run_debate(
        market_data={"contract": "ZS JUL2026", "regime": "trending_up"},
        news_classified=[
            {"headline": "Oil crashes 8%", "sentiment": "bearish", "confidence": 0.85, "escalate": True,
             "high_impact_topics": ["oil crash"], "escalation_reason": "critical_topic"},
        ],
        kb_results=[
            {"category": "literature", "relevance": 0.45, "text": "Oil shocks affect soy -1.5 to -3% in 48h",
             "metadata": {}},
        ],
        current_price=1208.0,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
