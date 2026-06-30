"""
WTI_N6 Crude Oil Robot — AI Agents
EIA/OPEC driven. NYMEX session focus.
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


TREND_AGENT_SYSTEM = """You are a professional WTI crude oil (WTI_N6) intraday trader and analyst.
You specialize in 15-minute chart analysis during the prime session (06:00-16:00 CT).

Your expertise:
- WTI crude oil price drivers: supply/demand, EIA inventories, OPEC+ policy
- EIA Weekly Petroleum Status Report: Wednesday 10:30 ET — biggest weekly catalyst
  - Surprise draw (more than expected): bullish; Surprise build: bearish
- OPEC+ production decisions: multi-week trend driver
- Geopolitical risk premium: Middle East tensions = instant spike
- DXY inverse correlation: USD strength → oil weakness
- China demand: PMI/manufacturing data affects oil demand outlook
- Key psychological levels: $65, $70, $75, $80, $85, $90 per barrel
- ATR 15m: $0.40-1.50 depending on volatility session
- WTI often whipsaws ±$1 around EIA release then trends

Output ONLY valid JSON:
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


RISK_AGENT_SYSTEM = """You are a professional risk manager for WTI crude oil (WTI_N6) intraday trading.

Account parameters:
- Capital: $1,013 USD (demo account, ICMarkets)
- Max risk per trade: 2% = ~$20.26 USD
- Instrument: WTI_N6 CFD (approx $1/unit per lot P&L)
- ATR on 15m: $0.40-1.50 typically
- Avoid stops < $0.50 (oil wicks aggressively)
- Lot sizing: lots = $20.26 / stop_usd
  - E.g., $1.00 stop → lots = 20.26 → capped at MAX 2.0
  - E.g., $2.00 stop → lots = 10.13 → capped at 2.0

WTI-specific risk:
- EIA Wednesday 10:30 ET: DO NOT open new positions within 30min of report
- Geopolitical headlines: can cause $2-5 gap instantly
- OPEC meeting days: wide stops or flat
- Low liquidity after 13:30 CT: avoid new entries
- Weekend risk: oil can gap $1-3 on Monday open

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
    user_msg = f"""Analyze WTI_N6 15-minute chart for intraday trading signal.

PRICE DATA SUMMARY:
{json.dumps(bars_summary, indent=2, default=str)}

FUNDAMENTAL / MACRO CONTEXT:
{json.dumps(fundamental_ctx, indent=2, default=str)}

Identify trend, oil regime, key levels, and best intraday setup.
Output only valid JSON."""
    return _call_claude(TREND_AGENT_SYSTEM, user_msg, model_id)


def call_risk_agent(bars_summary: dict, trend_analysis: dict, fundamental_ctx: dict, model_id: str = MODEL_ID) -> dict:
    current_price = bars_summary.get("current_state", {}).get("close", 0)
    user_msg = f"""Evaluate risk and size a WTI_N6 trade.

CURRENT PRICE: ${current_price:,.3f}
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
