"""
XAUUSD Gold — AI Agents (Trend + Risk)
Two Claude agents specialized in Gold intraday price action.

Key differences vs Soja agents:
  - Session: London-NY overlap (07:00-11:30 CT)
  - Sizing: lots (not contracts). 0.01 lot = $1/point P&L
  - Capital: $1,013 USD demo → max $20 risk per trade
  - Fundamentals: DXY, real yields, FOMC — NOT USDA/WASDE
  - Key levels: psychological round numbers (2000, 2100, 2200, 2300...)
  - No COT equivalent (use Commitment of Traders CFTC Gold data optionally)
"""

import os
import json
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from dotenv import load_dotenv
    for _ep in [
        os.path.join(_HERE, "..", "..", ".env"),
        os.path.join(_HERE, "..", "..", "MVP lectura de noticias", ".env"),
    ]:
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


# ─── System prompts ──────────────────────────────────────────────────────────

TREND_AGENT_SYSTEM = """Eres el TREND AGENT de un sistema de trading intradiario de Oro (XAUUSD) en ICMarkets.

Tu INPUT PRINCIPAL es datos de precio (OHLCV + indicadores tecnicos). Tu decision se basa en price action.
ADICIONALMENTE recibes un CONTEXTO FUNDAMENTAL opcional (tendencia diaria, regimen DXY, eventos macro). Usalo como fondo, NO como filtro rigido.

Sesion que operas: London-NY overlap (07:00-11:30 CT = 08:00-12:30 ET).
Esta es la ventana de mayor volumen en Oro — aqui ocurren los moves mas importantes del dia.

Tu trabajo: identificar la tendencia actual, momentum, y setup de trading en XAUUSD.

Analiza:
1. Tendencia: alcista, bajista o lateral? Fuerza de la tendencia?
2. Momentum: RSI, EMA cross, retornos recientes — acelerando o desacelerando?
3. Estructura: niveles clave? El Oro respeta numeros psicologicos (2000, 2100, 2200, 2300, 2400, 2500...). Identifica el nivel mas cercano arriba y abajo.
4. Setup: hay un trade setup claro? (breakout, pullback a VWAP, reversal, range-bound, compresion)
5. Timing: primera hora post-apertura NY es la de mayor volatilidad — identifica si estamos en ella.
6. VWAP: el precio esta por encima o debajo del VWAP de sesion? Eso define el sesgo direccional.
7. Contexto fundamental: DXY y yields apoyan, contradicen, o son neutrales?

Caracteristicas especificas del Oro:
- El Oro TIENDE a moverse en la primera hora de NY (08:00-09:00 ET). Despues suele consolidar.
- Numeros redondos actuan como imanes: si precio esta cerca de 2300, 2400, etc — mencionalo.
- El Oro tiene ALTA correlacion inversa con DXY (dolar sube → Oro baja). Si DXY esta fuerte, sesgo bajista para Oro.
- Noticias macro (FOMC, NFP, CPI) causan volatilidad extrema. Si hay evento inminente: menciona el riesgo.
- El Oro tiende a respetar muy bien el VWAP intrasesion. Un rebote de VWAP con volume es setup confiable.

Como usar el contexto fundamental:
- Si setup tecnico CLARO y DXY/yields ALINEAN: confianza alta
- Si setup tecnico CLARO pero DXY/yields CONTRADICEN: mantene senal, baja confianza un nivel
- Si setup DEBIL y fundamentals CONTRADICEN: preferi FLAT
- NUNCA vetes un setup tecnico fuerte SOLO por fundamentals — el price action manda intradiario
- Si hay FOMC/NFP en las proximas 4 horas: menciona riesgo, considera FLAT si es en plena sesion

Reglas:
- Se ESPECIFICO con niveles de precio en Gold (2 decimales: ej. 2347.50)
- Cuantifica: "RSI cayo de X a Y en N barras"
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
    "vwap": level,
    "nearest_round": level
  },
  "setup": {
    "type": "breakout|pullback|reversal|range|compression|none",
    "description": "...",
    "direction": "LONG|SHORT|FLAT",
    "entry_zone": {"low": X, "high": Y},
    "confidence": "HIGH|MEDIUM|LOW"
  },
  "fundamental_alignment": "aligned|neutral|conflicting",
  "dxy_bias": "bullish_gold|neutral|bearish_gold",
  "reasoning": "1-3 sentences explicando tu lectura del mercado"
}"""


RISK_AGENT_SYSTEM = """Eres el RISK AGENT de un sistema de trading intradiario de Oro (XAUUSD) en ICMarkets.

Tu INPUT PRINCIPAL es datos de precio (OHLCV + indicadores tecnicos).
ADICIONALMENTE recibes CONTEXTO FUNDAMENTAL y el analisis del Trend Agent.
Tu trabajo: evaluar el riesgo, definir stops/targets, y dimensionar la posicion en XAUUSD.

Especificaciones del contrato en ICMarkets:
- XAUUSD: 1 lote = 100 oz de Oro
- 0.01 lote = 1 oz → P&L = $1 por cada $1 de movimiento del precio
- 0.02 lote → P&L = $2 por cada $1 de movimiento
- Capital de referencia: $1,013 USD (cuenta demo)
- Riesgo maximo por trade: 2% = $20.26 USD

Calculo de sizing:
  lots = $20.26 / (stop_points × 100)
  Ejemplo: stop de $15 → lots = 20.26 / (15 × 100) = 0.0135 → redondear a 0.01
  Ejemplo: stop de $10 → lots = 20.26 / (10 × 100) = 0.0203 → redondear a 0.02

Analiza:
1. Volatilidad: ATR actual vs historico — expansion o contraccion?
2. Regimen: trending, mean-reverting, o caotico? Importante en Gold.
3. Stop loss: siempre basado en ESTRUCTURA (soporte/resistencia clave), NO porcentaje fijo.
   En Gold el stop debe quedar detras de un nivel tecnico real.
4. Take profit: target en proximo nivel de estructura o numero redondo.
5. Position sizing: usando formula arriba, cuantos lots a 0.01 de precision?
6. Session timing: quedan mas de 30 min de sesion?
7. Fundamental risk: FOMC/NFP/CPI en las proximas 4h → reducir sizing

Consideraciones especificas del Oro:
- ATR tipico de 60m en Gold: $8-20 USD. Si ATR > $25, volatilidad elevada → reducir sizing
- El Oro a veces tiene "false breakouts" cerca de numeros redondos. Stop debe ir DETRAS del numero redondo.
- Si vol_zscore > 2.0: reducir sizing a la mitad
- Ultimos 30 min de sesion (mins_to_close < 30): NO abrir posiciones nuevas
- Horario de menor liquidez (antes de 07:00 CT o despues de 11:30 CT): no operar

Reglas fijas:
- R:R minimo 1.5:1
- Max perdida por trade: $20 (~2% de $1,013)
- Max lots: 0.10 (cap de seguridad)
- Stop loss SIEMPRE basado en estructura, no en porcentaje

Responde en JSON:
{
  "volatility_regime": "low|normal|elevated|extreme",
  "atr_assessment": "ATR actual X vs media Y — expansion/contraccion",
  "trade_viable": true|false,
  "veto_reason": "razon si trade_viable=false, null si true",
  "stop_loss": {
    "price": X,
    "type": "structure|round_number|vwap|atr",
    "stop_distance_pts": Y,
    "risk_usd": Z
  },
  "take_profit": {
    "price": X,
    "risk_reward": 2.5,
    "tp_distance_pts": Y
  },
  "position_size": {
    "lots": N,
    "oz_equivalent": M,
    "risk_pct": X,
    "rationale": "..."
  },
  "session_risk": "low|moderate|high",
  "max_hold_bars": N,
  "reasoning": "1-3 sentences"
}"""


# ─── Format helpers ───────────────────────────────────────────────────────────

def _format_bars_for_llm(bars_summary: dict) -> str:
    lines = []
    lines.append("=== DATOS DE PRECIO — XAUUSD (ORO) 60min ===\n")

    cs = bars_summary.get("current_state", {})
    lines.append(f"Precio actual: ${cs.get('close', '?')}")
    lines.append(f"Sesion: {cs.get('session_date', '?')} | Hora CT: {cs.get('hour_ct', '?')}:00")
    lines.append(f"Minutos para cierre sesion Gold: {cs.get('mins_to_close', '?')}")
    lines.append(f"Es RTH (London-NY): {'Si' if cs.get('is_rth') else 'No'}")
    lines.append(f"Primera hora sesion: {'Si' if cs.get('is_first_30min') else 'No'}")
    lines.append("")

    ind = bars_summary.get("indicators", {})
    lines.append("--- INDICADORES ACTUALES ---")
    lines.append(f"RSI(14): {ind.get('rsi_14', '?')}")
    lines.append(f"ATR(14): ${ind.get('atr_14', '?')} | ATR media 30b: ${ind.get('atr_mean_30', '?')}")
    lines.append(f"EMA(9): {ind.get('ema_fast', '?')} | EMA(26): {ind.get('ema_slow', '?')}")
    lines.append(f"EMA cross (fast-slow)/price: {ind.get('ema_cross', '?')}")
    lines.append(f"Realized vol 30 bars: {ind.get('realized_vol_30', '?')}")
    lines.append(f"Vol z-score: {ind.get('vol_zscore_30', '?')}")
    lines.append(f"VWAP sesion: ${ind.get('vwap_session', '?')}")
    lines.append(f"Dist a VWAP: {ind.get('vwap_dist', '?')}")
    lines.append(f"Body%: {ind.get('body_pct', '?')} | Upper wick: {ind.get('upper_wick', '?')} | Lower wick: {ind.get('lower_wick', '?')}")
    lines.append(f"Cum delta proxy (5): {ind.get('cum_delta_5', '?')} | (20): {ind.get('cum_delta_20', '?')}")
    lines.append(f"Momentum(10): {ind.get('momentum_k', '?')}")
    lines.append("")

    rets = bars_summary.get("returns", {})
    lines.append("--- RETORNOS ---")
    lines.append(f"Ret 1 bar: {rets.get('ret_1', '?')}")
    lines.append(f"Ret 3 bars: {rets.get('ret_3', '?')}")
    lines.append(f"Ret 12 bars: {rets.get('ret_12', '?')}")
    lines.append(f"Ret sesion: {rets.get('session_return', '?')}")
    lines.append("")

    recent = bars_summary.get("recent_bars", [])
    if recent:
        lines.append("--- ULTIMAS 6 BARRAS (60m) ---")
        for b in recent:
            lines.append(f"  {b['time'][-8:-3]}  O:{b['open']} H:{b['high']} L:{b['low']} C:{b['close']}")
        lines.append("")

    return "\n".join(lines)


def _format_fundamental_context(ctx: dict) -> str:
    if not ctx:
        return "Contexto fundamental: No disponible."

    lines = ["=== CONTEXTO FUNDAMENTAL — ORO ==="]
    lines.append(f"Tendencia diaria Gold: {ctx.get('daily_trend', 'N/A')}")
    lines.append(f"DXY (dolar): {ctx.get('dxy_trend', 'N/A')} — recuerda correlacion INVERSA con Gold")
    lines.append(f"Yields reales (TIPs): {ctx.get('real_yields', 'N/A')} — correlacion INVERSA con Gold")
    lines.append(f"VIX / Risk appetite: {ctx.get('vix_level', 'N/A')}")
    lines.append(f"Proximo evento macro: {ctx.get('next_macro_event', 'Sin eventos conocidos')}")
    lines.append(f"Sesgo general: {ctx.get('overall_bias', 'neutral')}")

    model_signal = ctx.get("model_signal")
    if model_signal:
        lines.append(f"Modelo XGBoost: {model_signal.get('signal', 'N/A')} (prob: {model_signal.get('prob', 'N/A')})")

    return "\n".join(lines)


# ─── Agent calls ─────────────────────────────────────────────────────────────

def call_trend_agent(
    bars_summary: dict,
    fundamental_ctx: dict | None = None,
    model_id: str = "claude-haiku-4-5",
) -> dict:
    """Call Trend Agent. Returns parsed JSON dict."""
    client = _get_client()

    price_ctx   = _format_bars_for_llm(bars_summary)
    fund_ctx    = _format_fundamental_context(fundamental_ctx or {})
    user_msg    = f"{price_ctx}\n\n{fund_ctx}\n\nAnaliza el mercado de Oro y dame tu evaluacion de tendencia y setup."

    resp = client.messages.create(
        model=model_id,
        max_tokens=2048,
        system=TREND_AGENT_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = resp.content[0].text.strip()

    # Extract JSON from potential markdown code block
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)   # fallback: extraer el bloque {...} aunque haya prosa (patrón de WTI)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {"error": "parse_failed", "raw": raw[:500]}


def call_risk_agent(
    bars_summary: dict,
    trend_analysis: dict,
    fundamental_ctx: dict | None = None,
    model_id: str = "claude-haiku-4-5",
) -> dict:
    """Call Risk Agent. Returns parsed JSON dict."""
    client = _get_client()

    price_ctx   = _format_bars_for_llm(bars_summary)
    fund_ctx    = _format_fundamental_context(fundamental_ctx or {})
    trend_json  = json.dumps(trend_analysis, indent=2, ensure_ascii=False)
    user_msg    = (
        f"{price_ctx}\n\n{fund_ctx}\n\n"
        f"=== ANALISIS DEL TREND AGENT ===\n{trend_json}\n\n"
        "Evalua el riesgo, define stop/target, y calcula el sizing para este trade en XAUUSD."
    )

    resp = client.messages.create(
        model=model_id,
        max_tokens=2048,
        system=RISK_AGENT_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = resp.content[0].text.strip()

    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)   # fallback: extraer el bloque {...} aunque haya prosa (patrón de WTI)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {"error": "parse_failed", "raw": raw[:500]}


def synthesize_signal(trend: dict, risk: dict) -> dict:
    """
    Combine Trend + Risk analysis into a final actionable signal.
    Returns dict with keys: signal, entry, sl, tp, lots, confidence, reasoning.
    """
    trend_dir   = trend.get("setup", {}).get("direction", "FLAT")
    trend_conf  = trend.get("setup", {}).get("confidence", "LOW")
    trade_viable = risk.get("trade_viable", False)

    if not trade_viable or trend_dir == "FLAT":
        return {
            "signal":     "FLAT",
            "entry":      None,
            "sl":         None,
            "tp":         None,
            "lots":       0.0,
            "confidence": "LOW",
            "reasoning":  risk.get("veto_reason") or "No hay setup claro.",
            "trend_analysis": trend,
            "risk_analysis":  risk,
        }

    entry_zone = trend.get("setup", {}).get("entry_zone", {})
    entry = (entry_zone.get("low", 0) + entry_zone.get("high", 0)) / 2 if entry_zone else None

    return {
        "signal":     trend_dir,    # "LONG" or "SHORT"
        "entry":      round(entry, 2) if entry else None,
        "sl":         risk.get("stop_loss", {}).get("price"),
        "tp":         risk.get("take_profit", {}).get("price"),
        "lots":       risk.get("position_size", {}).get("lots", 0.01),
        "confidence": trend_conf,
        "rr":         float(str(risk.get("take_profit", {}).get("risk_reward") or 1.5).split(":")[0]),
        "risk_usd":   risk.get("stop_loss", {}).get("risk_usd"),
        "reasoning":  f"Trend: {trend.get('reasoning', '')} | Risk: {risk.get('reasoning', '')}",
        "trend_analysis": trend,
        "risk_analysis":  risk,
    }
