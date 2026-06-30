"""
USTEC Nasdaq 100 Robot — AI Agents
Tech index: NVDA/AI driven, rate-sensitive growth stocks.
"""

import os, json, re, sys

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _MVP_ROOT)

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

import anthropic

MODEL_ID = "claude-haiku-4-5"
_client  = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


TREND_AGENT_SYSTEM = """You are a professional Nasdaq 100 (USTEC) intraday trader and technical analyst.
You specialize in 5-minute chart analysis during US market hours (09:30-16:00 ET).

Your expertise:
- USTEC intraday momentum: tech index moves faster than S&P, high beta to NVDA/AAPL
- NVDA weight: ~10% of USTEC, single stock can move the index significantly
- AI sentiment: OpenAI, Anthropic, Google news can spike tech sector instantly
- Rate sensitivity: tech/growth stocks inversely correlated with Treasury yields
- VIX: VIX spike > 20 = headwind for longs; VIX crush = bullish
- Opening range (first 30min) breakout: strong predictor for day direction
- Mag7 momentum: AAPL/MSFT/NVDA/GOOGL/META/AMZN/TSLA drive 50%+ of index
- Key psychological levels: 17000, 18000, 19000, 20000, 21000, 22000
- Pre-market gaps: USTEC gaps more than S&P on tech news

Output ONLY valid JSON:
{
  "trend": "UP" | "DOWN" | "SIDEWAYS",
  "trend_strength": "STRONG" | "MODERATE" | "WEAK",
  "trend_reasoning": "...",
  "key_levels": {"support": [<price>, ...], "resistance": [<price>, ...]},
  "vwap_position": "ABOVE" | "BELOW" | "AT",
  "momentum": "ACCELERATING" | "STABLE" | "DECELERATING" | "REVERSING",
  "ustec_regime": "BULL_TREND" | "RANGE" | "DISTRIBUTION" | "BEAR_TREND" | "AI_BUBBLE",
  "ai_sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
  "risk_on_off": "RISK_ON" | "RISK_OFF" | "NEUTRAL",
  "setup": {
    "type": "BREAKOUT" | "VWAP_PULLBACK" | "TREND_CONTINUATION" | "REVERSAL" | "NONE",
    "direction": "LONG" | "SHORT" | "NONE",
    "confidence": "HIGH" | "MEDIUM" | "LOW",
    "entry_zone": [<low_price>, <high_price>],
    "invalidation": <price>,
    "reasoning": "..."
  }
}"""


RISK_AGENT_SYSTEM = """You are a professional risk manager for USTEC Nasdaq 100 intraday trading.

Account parameters:
- Capital: $1,013 USD (demo account, ICMarkets)
- Max risk per trade: 2% = ~$20.26 USD
- Instrument: USTEC CFD (1 lot = $1/pt P&L)
- ATR on 5m: typically 15-50 pts depending on session
- Avoid stops < 15 pts (tech wicks aggressively on news)
- Lot sizing: lots = $20.26 / stop_pts
  - E.g., 20pt stop → lots = 20.26/20 = 1.01 lots
  - E.g., 40pt stop → lots = 20.26/40 = 0.51 lots

USTEC-specific risk:
- First 30 min (08:30-09:00 CT): wider stops, higher vol
- Last 30 min (14:30-15:00 CT): avoid new positions
- NVDA earnings: can move USTEC ±3-5% in afterhours
- FOMC days: binary risk, flat preferred
- AI news (OpenAI/Anthropic/Google): instant gap moves
- Avoid stops < 15 pts

Output ONLY valid JSON:
{
  "volatility_regime": "LOW" | "NORMAL" | "HIGH" | "EXTREME",
  "atr_5m": <value>,
  "stop_distance": <recommended_stop_in_pts>,
  "take_profit_distance": <recommended_tp_in_pts>,
  "rr_ratio": <risk_reward_ratio>,
  "lots": <position_size>,
  "risk_usd": <actual_risk_in_usd>,
  "trade_viable": true | false,
  "veto_reason": null | "...",
  "entry": <price>,
  "sl": <stop_loss_price>,
  "tp": <take_profit_price>,
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "notes": "..."
}"""


def _call_claude(system: str, user_msg: str, model: str = MODEL_ID) -> dict:
    client = _get_client()
    msg = client.messages.create(
        model=model, max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m: return json.loads(m.group(0))
        return {"error": f"JSON parse failed: {raw[:200]}"}


def call_trend_agent(bars_summary: dict, fundamental_ctx: dict, model_id: str = MODEL_ID) -> dict:
    user_msg = f"""Analyze USTEC 5-minute chart for intraday trading signal.

PRICE DATA SUMMARY:
{json.dumps(bars_summary, indent=2, default=str)}

FUNDAMENTAL / MACRO CONTEXT:
{json.dumps(fundamental_ctx, indent=2, default=str)}

Identify trend, USTEC regime, key levels, and best intraday setup (if any).
Output only valid JSON."""
    return _call_claude(TREND_AGENT_SYSTEM, user_msg, model_id)


def call_risk_agent(bars_summary: dict, trend_analysis: dict, fundamental_ctx: dict, model_id: str = MODEL_ID) -> dict:
    current_price = bars_summary.get("current_state", {}).get("close", 0)
    user_msg = f"""Evaluate risk and size a USTEC trade.

CURRENT PRICE: {current_price:,.2f}
PRICE DATA SUMMARY:
{json.dumps(bars_summary, indent=2, default=str)}

TREND AGENT ANALYSIS:
{json.dumps(trend_analysis, indent=2, default=str)}

FUNDAMENTAL CONTEXT:
{json.dumps(fundamental_ctx, indent=2, default=str)}

Determine entry, SL, TP, position size, and viability.
Output only valid JSON."""
    return _call_claude(RISK_AGENT_SYSTEM, user_msg, model_id)


def synthesize_signal(trend: dict, risk: dict) -> dict:
    setup     = trend.get("setup", {})
    direction = setup.get("direction", "NONE")
    viable    = risk.get("trade_viable", False)

    if direction == "NONE" or not viable:
        return {
            "signal": "FLAT", "entry": None, "sl": None, "tp": None,
            "lots": None, "confidence": "LOW", "rr": None, "risk_usd": None,
            "reasoning": risk.get("veto_reason") or setup.get("reasoning", "No setup."),
            "ustec_regime": trend.get("ustec_regime"),
            "ai_sentiment": trend.get("ai_sentiment"),
            "risk_on_off":  trend.get("risk_on_off"),
        }

    signal   = "LONG" if direction == "LONG" else "SHORT"
    entry    = risk.get("entry")
    sl       = risk.get("sl")
    tp       = risk.get("tp")
    lots     = risk.get("lots", 0.5)
    rr       = risk.get("rr_ratio")
    risk_usd = risk.get("risk_usd")

    if rr is not None and float(rr) < 1.5:
        return {
            "signal": "FLAT", "entry": entry, "sl": sl, "tp": tp,
            "lots": lots, "confidence": "LOW", "rr": rr, "risk_usd": risk_usd,
            "reasoning": f"R:R {rr} < 1.5 minimum — skip.",
        }

    conf_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    final_conf = min(
        [setup.get("confidence", "LOW"), risk.get("confidence", "LOW")],
        key=lambda c: conf_rank.get(c, 0)
    )

    reasoning = (
        f"{setup.get('reasoning', '')} | "
        f"Regime: {risk.get('volatility_regime')} | "
        f"ATR5m: {risk.get('atr_5m','?')}pts | "
        f"{trend.get('ustec_regime','')} / {trend.get('ai_sentiment','')} / {trend.get('risk_on_off','')}"
    )

    return {
        "signal":       signal,
        "entry":        entry, "sl": sl, "tp": tp,
        "lots":         lots, "confidence": final_conf,
        "rr":           rr, "risk_usd": risk_usd,
        "reasoning":    reasoning,
        "ustec_regime": trend.get("ustec_regime"),
        "ai_sentiment": trend.get("ai_sentiment"),
        "risk_on_off":  trend.get("risk_on_off"),
    }
