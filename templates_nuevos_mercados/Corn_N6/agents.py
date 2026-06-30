"""
Corn_N6 CBOT Corn Robot — AI Agents
USDA/weather driven. CBOT grain session focus.
NOTE: JSON keys 'oil_regime'/'opec_dynamic' are reused as generic regime/supply-bias
fields so the shared live_runner + dashboard stay compatible across instruments.
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


TREND_AGENT_SYSTEM = """You are a professional CBOT corn (Corn_N6) intraday trader and analyst.
You specialize in 15-minute chart analysis during the CBOT grain day session (08:30-13:20 CT).

Your expertise:
- Corn price drivers: USDA reports (WASDE, Crop Progress, Grain Stocks, Prospective Plantings)
- Weather: Midwest growing-season weather is the dominant intraday catalyst (drought = bullish, ideal rain = bearish)
- Planting/harvest seasonality: spring planting pace, summer pollination, fall harvest pressure
- Export demand: USDA export sales (Thursday), China/Mexico buying, competing Brazilian/Ukrainian supply
- Ethanol demand and crude-oil correlation (corn feedstock)
- USD index: strong dollar pressures US grain export competitiveness
- Key psychological levels: round numbers per bushel (e.g., 400, 425, 450, 475, 500 cents)
- ATR 15m: a few cents/bushel depending on the session and report flow
- Grains whipsaw hard on USDA report releases (07:30 CT report days) then trend

Output ONLY valid JSON
(NOTE: 'oil_regime' here = market regime; 'opec_dynamic' here = supply-side bias from USDA/weather):
{
  "trend": "UP" | "DOWN" | "SIDEWAYS",
  "trend_strength": "STRONG" | "MODERATE" | "WEAK",
  "trend_reasoning": "...",
  "key_levels": {"support": [<price>, ...], "resistance": [<price>, ...]},
  "vwap_position": "ABOVE" | "BELOW" | "AT",
  "momentum": "ACCELERATING" | "STABLE" | "DECELERATING" | "REVERSING",
  "oil_regime": "BULL_TREND" | "RANGE" | "BEAR_TREND" | "SUPPLY_SHOCK" | "DEMAND_DRIVEN",
  "opec_dynamic": "HAWKISH" | "DOVISH" | "NEUTRAL",
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


RISK_AGENT_SYSTEM = """You are a professional risk manager for CBOT corn (Corn_N6) intraday trading.

Account parameters:
- Capital: $1,013 USD (demo account, ICMarkets)
- Max risk per trade: 3% = ~$30.39 USD
- Instrument: Corn_N6 CFD (priced in cents/bushel)
- ATR on 15m: a few cents/bushel typically
- Avoid stops < 1.5 cents (grains wick on weather/USDA headlines)
- Lot sizing: lots = $30.39 / stop_distance

Corn-specific risk:
- USDA report days (WASDE, Crop Progress, Grain Stocks, ~07:30 CT): DO NOT open within 30min of release
- Weather scares (drought/frost/flood headlines): can gap several cents instantly
- Export-sales Thursday: demand surprises move price
- Low liquidity after 13:20 CT close: avoid new entries
- Weekend risk: grains can gap on weather over the weekend

Output ONLY valid JSON:
{
  "volatility_regime": "LOW" | "NORMAL" | "HIGH" | "EXTREME",
  "atr_5m": <value>,
  "stop_distance": <recommended_stop_in_usd>,
  "take_profit_distance": <recommended_tp_in_usd>,
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
    user_msg = f"""Analyze Corn_N6 15-minute chart for intraday trading signal.

PRICE DATA SUMMARY:
{json.dumps(bars_summary, indent=2, default=str)}

FUNDAMENTAL / MACRO CONTEXT:
{json.dumps(fundamental_ctx, indent=2, default=str)}

Identify trend, market regime, key levels, and best intraday setup.
Output only valid JSON."""
    return _call_claude(TREND_AGENT_SYSTEM, user_msg, model_id)


def call_risk_agent(bars_summary: dict, trend_analysis: dict, fundamental_ctx: dict, model_id: str = MODEL_ID) -> dict:
    current_price = bars_summary.get("current_state", {}).get("close", 0)
    user_msg = f"""Evaluate risk and size a Corn_N6 trade.

CURRENT PRICE: {current_price:,.3f}
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
            "reasoning":    risk.get("veto_reason") or setup.get("reasoning", "No setup."),
            "oil_regime":   trend.get("oil_regime"),
            "opec_dynamic": trend.get("opec_dynamic"),
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
        f"ATR5m: ${risk.get('atr_5m','?')} | "
        f"{trend.get('oil_regime','')} / {trend.get('opec_dynamic','')} / {trend.get('risk_on_off','')}"
    )

    return {
        "signal":       signal,
        "entry":        entry, "sl": sl, "tp": tp,
        "lots":         lots, "confidence": final_conf,
        "rr":           rr, "risk_usd": risk_usd,
        "reasoning":    reasoning,
        "oil_regime":   trend.get("oil_regime"),
        "opec_dynamic": trend.get("opec_dynamic"),
        "risk_on_off":  trend.get("risk_on_off"),
    }
