"""
QuantAgent-lite — 2 LLM agents specialized in price action (no news/sentiment).

Inspired by QuantAgent (arXiv:2509.09995, 2025):
  - Trend Agent:  reads OHLC patterns, identifies trend/momentum/setup
  - Risk Agent:   evaluates volatility regime, position sizing, stop placement

Both agents receive ONLY price data (OHLC bars + technical indicators).
No news, no sentiment, no fundamentals — pure price action reasoning.

Target: 60m bars during RTH (08:30-13:20 CT), run every ~4h.
"""

import os
import json
from datetime import datetime

try:
    from dotenv import load_dotenv
    _env_candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        os.path.join(os.path.dirname(__file__), "..", "..", "MVP lectura de noticias", ".env"),
    ]
    for _ep in _env_candidates:
        if os.path.exists(_ep):
            load_dotenv(_ep, override=True)
            break
except ImportError:
    pass

from anthropic import Anthropic

_client = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


# ─── System prompts ──────────────────────────────────────────────────────

TREND_AGENT_SYSTEM = """Eres el TREND AGENT de un sistema de trading intradiario de futuros de soja (ZS) en CBOT.

Tu UNICO input es datos de precio (OHLCV + indicadores tecnicos). NO recibes noticias, sentimiento ni fundamentales.
Tu trabajo: identificar la tendencia actual, momentum, y setup de trading basandote exclusivamente en price action.

Analiza:
1. Tendencia: ¿alcista, bajista o lateral? ¿fuerza de la tendencia?
2. Momentum: RSI, EMA cross, retornos recientes — ¿acelerando o desacelerando?
3. Estructura: ¿soportes/resistencias clave? ¿compresion/expansion de rango?
4. Setup: ¿hay un trade setup claro? (breakout, pullback, reversal, range-bound)
5. Timing: ¿momento de la sesion favorable? (primera/ultima 30min, VWAP position)

Reglas:
- Se ESPECIFICO con niveles de precio, no generico
- Cuantifica: "momentum decreciendo: RSI cayo de X a Y en N barras"
- Si no hay setup claro, di FLAT — no forces un trade
- Prioriza el riesgo: mejor perder un trade que entrar en uno malo

Responde en JSON:
{
  "trend": "UP|DOWN|LATERAL",
  "trend_strength": "strong|moderate|weak",
  "momentum": "accelerating|steady|decelerating|reversing",
  "key_levels": {
    "resistance": [level1, level2],
    "support": [level1, level2],
    "vwap": level
  },
  "setup": {
    "type": "breakout|pullback|reversal|range|compression|none",
    "description": "...",
    "direction": "LONG|SHORT|FLAT",
    "entry_zone": {"low": X, "high": Y},
    "confidence": "HIGH|MEDIUM|LOW"
  },
  "reasoning": "1-3 sentences explaining your read"
}"""


RISK_AGENT_SYSTEM = """Eres el RISK AGENT de un sistema de trading intradiario de futuros de soja (ZS) en CBOT.

Tu UNICO input es datos de precio (OHLCV + indicadores tecnicos). NO recibes noticias, sentimiento ni fundamentales.
Tu trabajo: evaluar el riesgo actual, definir stops/targets, y dimensionar la posicion.

Contexto de mercado:
- ZS Mini = 1000 bushels por contrato (MZS), tick = 0.125 cents = $1.25
- ZS Full = 5000 bushels por contrato, tick = 0.25 cents = $12.50
- Operamos MZS (micro) para sizing flexible
- Capital de referencia: $10,000

Analiza:
1. Volatilidad: ATR actual vs historico, ¿expansion o contraccion?
2. Regimen: ¿trending, mean-reverting, o caotico?
3. Risk/Reward: dado el setup del Trend Agent, ¿donde poner SL y TP?
4. Position sizing: basado en ATR y capital, ¿cuantos contratos?
5. Session timing: ¿la sesion tiene suficiente tiempo restante?

Reglas:
- Stop loss SIEMPRE basado en estructura (soporte/resistencia), NO en porcentaje fijo
- R:R minimo 1.5:1 para tomar el trade
- Si la volatilidad es anomala (vol_zscore > 2), reducir sizing a la mitad
- Ultimos 30 min de sesion (mins_to_close < 30): NO abrir posiciones nuevas
- Max perdida por trade: 2% del capital ($200 en $10k)

Responde en JSON:
{
  "volatility_regime": "low|normal|elevated|extreme",
  "atr_assessment": "ATR actual X vs media Y — expansion/contraccion",
  "trade_viable": true|false,
  "veto_reason": "razon si trade_viable=false, null si true",
  "stop_loss": {"price": X, "type": "structure|atr|vwap", "risk_usd": Y},
  "take_profit": {"price": X, "risk_reward": "N:1"},
  "position_size": {"contracts_mzs": N, "risk_pct": X, "rationale": "..."},
  "session_risk": "low|moderate|high",
  "max_hold_bars": N,
  "reasoning": "1-3 sentences"
}"""


def _format_bars_for_llm(bars_summary: dict) -> str:
    """Format price data into a compact text context for the agents."""
    lines = []
    lines.append("=== DATOS DE PRECIO — ZS FUTUROS SOJA 60min ===\n")

    # Current state
    cs = bars_summary.get("current_state", {})
    lines.append(f"Precio actual: {cs.get('close', '?')}")
    lines.append(f"Sesion: {cs.get('session_date', '?')} | Hora CT: {cs.get('hour_ct', '?')}:00")
    lines.append(f"Minutos para cierre RTH: {cs.get('mins_to_close', '?')}")
    lines.append(f"Es RTH: {'Si' if cs.get('is_rth') else 'No'}")
    lines.append("")

    # Indicators
    ind = bars_summary.get("indicators", {})
    lines.append("--- INDICADORES ACTUALES ---")
    lines.append(f"RSI(14): {ind.get('rsi_14', '?')}")
    lines.append(f"ATR(14): {ind.get('atr_14', '?')}")
    lines.append(f"ATR media 30 barras: {ind.get('atr_mean_30', '?')}")
    lines.append(f"EMA(9): {ind.get('ema_fast', '?')} | EMA(26): {ind.get('ema_slow', '?')}")
    lines.append(f"EMA cross (fast-slow)/price: {ind.get('ema_cross', '?')}")
    lines.append(f"Realized vol 30 bars: {ind.get('realized_vol_30', '?')}")
    lines.append(f"Vol z-score 30: {ind.get('vol_zscore_30', '?')}")
    lines.append(f"VWAP sesion: {ind.get('vwap_session', '?')}")
    lines.append(f"Dist a VWAP: {ind.get('vwap_dist', '?')}")
    lines.append(f"Body%: {ind.get('body_pct', '?')} | Upper wick: {ind.get('upper_wick', '?')} | Lower wick: {ind.get('lower_wick', '?')}")
    lines.append(f"Cum delta proxy (5): {ind.get('cum_delta_5', '?')} | (20): {ind.get('cum_delta_20', '?')}")
    lines.append("")

    # Returns
    rets = bars_summary.get("returns", {})
    lines.append("--- RETORNOS ---")
    lines.append(f"Ret 1 bar: {rets.get('ret_1', '?')}")
    lines.append(f"Ret 3 bars: {rets.get('ret_3', '?')}")
    lines.append(f"Ret 12 bars: {rets.get('ret_12', '?')}")
    lines.append(f"Ret sesion (open to current): {rets.get('session_return', '?')}")
    lines.append("")

    # Recent bars
    recent = bars_summary.get("recent_bars", [])
    if recent:
        lines.append("--- ULTIMAS 12 BARRAS (60min, mas reciente primero) ---")
        lines.append(f"{'Hora CT':>8} {'Open':>8} {'High':>8} {'Low':>8} {'Close':>8} {'Vol':>8} {'Body%':>6}")
        for b in recent:
            lines.append(
                f"{b.get('hour_ct','?'):>8} "
                f"{b.get('open','?'):>8} {b.get('high','?'):>8} "
                f"{b.get('low','?'):>8} {b.get('close','?'):>8} "
                f"{b.get('volume','?'):>8} {b.get('body_pct','?'):>6}"
            )
        lines.append("")

    # Session range
    sr = bars_summary.get("session_range", {})
    if sr:
        lines.append("--- RANGO DE SESION HOY ---")
        lines.append(f"Open: {sr.get('open', '?')} | High: {sr.get('high', '?')} | Low: {sr.get('low', '?')}")
        lines.append(f"Rango: {sr.get('range', '?')} ({sr.get('range_pct', '?')}%)")
        lines.append("")

    # Multi-day context
    md = bars_summary.get("multi_day", {})
    if md:
        lines.append("--- CONTEXTO MULTI-DIA (ultimos 5 dias) ---")
        for day in md:
            lines.append(
                f"  {day.get('date','?')}: O={day.get('open','?')} H={day.get('high','?')} "
                f"L={day.get('low','?')} C={day.get('close','?')} "
                f"ret={day.get('daily_return','?')}%"
            )

    return "\n".join(lines)


def call_trend_agent(bars_summary: dict, track_record: str = "") -> dict:
    """Call the Trend Agent with price data and return parsed JSON."""
    context = _format_bars_for_llm(bars_summary)
    if track_record:
        context += f"\n\n{track_record}"
    client = _get_client()

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            system=TREND_AGENT_SYSTEM,
            messages=[{"role": "user", "content": context}],
        )
        text = response.content[0].text.strip()
        # Extract JSON from response
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "JSON parse failed", "raw": text[:500]}
    except Exception as e:
        return {"error": str(e)}


def call_risk_agent(bars_summary: dict, trend_output: dict, track_record: str = "") -> dict:
    """Call the Risk Agent with price data + Trend Agent's analysis."""
    context = _format_bars_for_llm(bars_summary)
    context += f"\n\n=== ANALISIS DEL TREND AGENT ===\n{json.dumps(trend_output, indent=2, ensure_ascii=False)}"
    if track_record:
        context += f"\n\n{track_record}"

    client = _get_client()

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=RISK_AGENT_SYSTEM,
            messages=[{"role": "user", "content": context}],
        )
        text = response.content[0].text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "JSON parse failed", "raw": text[:500]}
    except Exception as e:
        return {"error": str(e)}
