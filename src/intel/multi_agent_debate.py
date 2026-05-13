"""
Multi-Agent Debate system for soybean market intelligence.
Architecture inspired by TradingAgents (ICML 2025).

Agents:
  1. Bull Analyst       — finds reasons price goes up
  2. Bear Analyst       — finds reasons price goes down
  3. Risk Assessor      — evaluates regime, volatility, position sizing
  4. Technical Analyst  — reads price chart, identifies key levels and setup
  5. Fund Manager       — reads all four, produces final verdict

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

TECHNICAL_ANALYST_SYSTEM = """Eres el ANALISTA TÉCNICO CHARTISTA de un equipo de inteligencia de mercado de soja CBOT.
Tu mandato es leer el gráfico de precios e identificar el setup técnico actual.

NO tomas partido en el debate fundamental. Tu trabajo es describir lo que el GRÁFICO dice:
1. Tendencia primaria (UP/DOWN/SIDEWAYS) y su fortaleza
2. Niveles clave de soporte y resistencia (últimos 3 swings relevantes)
3. Señales de momentum: RSI divergencias, cruces de medias, posición Bollinger
4. Setup actual: ¿está el precio en zona de entrada, distribución o compresión?
5. Target técnico medido (measured move) y nivel de invalidación
6. Confirmación o divergencia con el volumen/OI si está disponible

Reglas:
- Cita niveles de precio concretos, no rangos vagos
- Indica si el setup es de alta, media o baja confiabilidad técnica
- Cuando el técnico contradice el fundamental, dilo explícitamente
- Usa terminología estándar: soporte/resistencia, breakout/breakdown, divergencia alcista/bajista, compresión de Bollinger, cruce dorado/cruce de la muerte

Responde en formato JSON:
{
  "primary_trend": "UP|DOWN|SIDEWAYS",
  "trend_strength": "strong|moderate|weak",
  "key_support": [{"level": X, "type": "swing_low|MA200|bollinger_lower|prior_resistance", "strength": "major|minor"}],
  "key_resistance": [{"level": X, "type": "swing_high|MA50|bollinger_upper|prior_support", "strength": "major|minor"}],
  "momentum_signals": ["RSI en X (sobrevendido/sobrecomprado/neutro)", "precio vs MA20/50/200", "..."],
  "bollinger_position": "upper_band|middle|lower_band|squeeze",
  "setup": "descripción del setup técnico actual en 1-2 oraciones",
  "price_target_bullish": X,
  "price_target_bearish": X,
  "technical_invalidation": X,
  "confirmation_needed": "qué señal técnica confirmaría la dirección",
  "fundamental_alignment": "CONFIRMS|CONTRADICTS|NEUTRAL",
  "confidence": 0.0-1.0,
  "key_level_to_watch": X
}"""

FUND_MANAGER_SYSTEM = """Eres el FUND MANAGER que toma la decisión final del equipo de inteligencia de soja.
Has recibido los análisis del analista alcista, el analista bajista, el analista de riesgo, y el analista técnico chartista.

Tu trabajo es:
1. Leer los cuatro análisis con ojo crítico
2. Pesar los argumentos según la calidad de la evidencia (no solo la convicción del analista)
3. Considerar el assessment de riesgo para calibrar el tamaño de la posición
4. Usar el análisis técnico como árbitro cuando fundamental y técnico están alineados, y como señal de cautela cuando divergen
5. Producir un VEREDICTO FINAL unificado

Reglas de decisión:
- Si ambos analistas fundamentales tienen convicción BAJA → HOLD (esperar más información)
- Si el riesgo es "crisis" o "extreme volatility" → reducir exposición independientemente
- Si un analista tiene convicción ALTA y el otro BAJA → seguir al de alta convicción con sizing reducido
- Si ambos tienen convicción ALTA → hay conflicto genuino, ser cauto
- Si el técnico CONFIRMA el fundamental dominante → aumentar conviction 1 nivel
- Si el técnico CONTRADICE el fundamental dominante → reducir sizing un nivel
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


def _load_technical_context() -> dict:
    """
    Compute technical indicators from raw_market.csv for the Technical Analyst agent.
    Returns a dict with price series, indicators, and support/resistance levels.
    """
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        return {}

    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_path = os.path.join(_ROOT, "data", "raw_market.csv")
    if not os.path.exists(raw_path):
        return {}

    try:
        df = pd.read_csv(raw_path, parse_dates=["Date"]).sort_values("Date").dropna(subset=["Soybeans"])
        s  = df["Soybeans"]
        if len(s) < 50:
            return {}

        price   = float(s.iloc[-1])
        ma20    = float(s.rolling(20).mean().iloc[-1])
        ma50    = float(s.rolling(50).mean().iloc[-1]) if len(s) >= 50  else None
        ma200   = float(s.rolling(200).mean().iloc[-1]) if len(s) >= 200 else None

        # RSI(14)
        delta = s.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / (loss + 1e-8)
        rsi   = float((100 - 100 / (1 + rs)).iloc[-1])

        # Bollinger Bands(20, 2)
        bb_std   = float(s.rolling(20).std().iloc[-1])
        bb_upper = round(ma20 + 2 * bb_std, 2)
        bb_lower = round(ma20 - 2 * bb_std, 2)
        bb_pct   = round((price - bb_lower) / (bb_upper - bb_lower + 1e-8) * 100, 1)

        # Momentum
        ret_5d  = round((price / float(s.iloc[-6])  - 1) * 100, 2) if len(s) >= 6  else None
        ret_20d = round((price / float(s.iloc[-21]) - 1) * 100, 2) if len(s) >= 21 else None

        # Support/resistance: local swing highs and lows in last 60 bars
        window = s.iloc[-60:].reset_index(drop=True)
        swing_highs = []
        swing_lows  = []
        for i in range(2, len(window) - 2):
            if window[i] > window[i-1] and window[i] > window[i-2] and window[i] > window[i+1] and window[i] > window[i+2]:
                swing_highs.append(round(float(window[i]), 2))
            if window[i] < window[i-1] and window[i] < window[i-2] and window[i] < window[i+1] and window[i] < window[i+2]:
                swing_lows.append(round(float(window[i]), 2))

        # Keep the 3 most recent, deduplicated within 0.5%
        def _dedup(levels: list[float], tol: float = 0.005) -> list[float]:
            out = []
            for lv in sorted(set(levels), reverse=True):
                if not out or abs(lv / out[-1] - 1) > tol:
                    out.append(lv)
            return out[:3]

        resistances = _dedup([h for h in swing_highs if h > price])
        supports    = _dedup([lo for lo in reversed(sorted(swing_lows)) if lo < price])

        # Add MA levels as support/resistance
        for ma_val, label in [(ma50, "MA50"), (ma200, "MA200")]:
            if ma_val is None:
                continue
            if ma_val > price:
                resistances.append(round(ma_val, 2))
            else:
                supports.append(round(ma_val, 2))

        return {
            "price":       round(price, 2),
            "ma20":        round(ma20, 2),
            "ma50":        round(ma50, 2) if ma50 else None,
            "ma200":       round(ma200, 2) if ma200 else None,
            "rsi":         round(rsi, 1),
            "bb_upper":    bb_upper,
            "bb_lower":    bb_lower,
            "bb_pct":      bb_pct,
            "ret_5d_pct":  ret_5d,
            "ret_20d_pct": ret_20d,
            "resistances": sorted(resistances)[:3],
            "supports":    sorted(supports, reverse=True)[:3],
        }
    except Exception:
        return {}


def _format_technical_context(tc: dict) -> str:
    if not tc:
        return ""
    lines = ["--- Datos técnicos (para Analista Técnico) ---"]
    lines.append(f"  Precio actual: {tc.get('price')}")
    lines.append(f"  MA20: {tc.get('ma20')} | MA50: {tc.get('ma50')} | MA200: {tc.get('ma200')}")
    lines.append(f"  RSI(14): {tc.get('rsi'):.1f}" if tc.get("rsi") else "  RSI: N/A")
    lines.append(f"  Bollinger: upper={tc.get('bb_upper')} lower={tc.get('bb_lower')} %B={tc.get('bb_pct')}%")
    lines.append(f"  Momentum: 5d={tc.get('ret_5d_pct'):+.2f}% 20d={tc.get('ret_20d_pct'):+.2f}%" if tc.get("ret_5d_pct") is not None else "")
    lines.append(f"  Resistencias clave: {tc.get('resistances')}")
    lines.append(f"  Soportes clave: {tc.get('supports')}")
    lines.append("")
    return "\n".join(l for l in lines if l is not None)


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
    technical_data: dict | None = None,
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

    if technical_data:
        lines.append(_format_technical_context(technical_data))

    return "\n".join(lines)


def run_debate(
    market_data: dict,
    news_classified: list[dict],
    kb_results: list[dict],
    current_price: float,
) -> dict:
    technical_data = _load_technical_context()
    context = _build_market_context(
        market_data, news_classified, kb_results, current_price, technical_data
    )

    bull = _call_agent(BULL_SYSTEM, context, "Bull Analyst")
    bear = _call_agent(BEAR_SYSTEM, context, "Bear Analyst")
    risk = _call_agent(RISK_SYSTEM, context, "Risk Assessor")
    tech = _call_agent(TECHNICAL_ANALYST_SYSTEM, context, "Technical Analyst")

    manager_context = (
        f"{context}\n\n"
        f"=== ANÁLISIS DEL EQUIPO ===\n\n"
        f"--- ANALISTA ALCISTA ---\n{json.dumps(bull, indent=2, ensure_ascii=False)}\n\n"
        f"--- ANALISTA BAJISTA ---\n{json.dumps(bear, indent=2, ensure_ascii=False)}\n\n"
        f"--- ANALISTA DE RIESGO ---\n{json.dumps(risk, indent=2, ensure_ascii=False)}\n\n"
        f"--- ANALISTA TÉCNICO CHARTISTA ---\n{json.dumps(tech, indent=2, ensure_ascii=False)}\n"
    )

    verdict = _call_agent(FUND_MANAGER_SYSTEM, manager_context, "Fund Manager")

    return {
        "timestamp": datetime.now().isoformat(),
        "current_price": current_price,
        "agents": {
            "bull":      bull,
            "bear":      bear,
            "risk":      risk,
            "technical": tech,
        },
        "verdict": verdict,
        "context_summary": {
            "articles_analyzed":     len(news_classified),
            "articles_escalated":    sum(1 for n in news_classified if n.get("escalate")),
            "kb_documents_retrieved": len(kb_results),
            "bull_news":             sum(1 for n in news_classified if n.get("sentiment") == "bullish"),
            "bear_news":             sum(1 for n in news_classified if n.get("sentiment") == "bearish"),
            "technical_data_loaded": bool(technical_data),
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
