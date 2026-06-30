"""
UK100 FTSE 100 Robot — AI Agents
London session, BOE driven, GBP/oil sensitive.
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


TREND_AGENT_SYSTEM = """You are a professional FTSE 100 (UK100) intraday trader and analyst.
You specialize in 5-minute chart analysis during the London session (08:00-16:30 BST = 02:00-11:30 CT).

Your expertise:
- FTSE 100 composition: ~15% energy (BP/Shell), ~25% financials, ~10% miners, ~10% pharma (GSK/AZ)
- GBP/USD inverse: GBP stronger → FTSE lower (multinationals earn in foreign currencies)
  Example: GBP +0.5% → FTSE -0.3% typically
- Oil price sensitivity: BP+Shell = major weight, oil up → FTSE up
- Bank of England: 8 meetings/year, same impact as FOMC on FTSE
- UK macro: CPI, GDP, retail sales — BOE reaction matters more than the data itself
- London open (08:00 BST): highest volume, gap fills, directional breakouts
- US pre-market: FTSE often tracks US futures 02:00-08:30 CT
- Key psychological levels: 7500, 8000, 8500, 9000
- European risk sentiment: DAX/CAC correlation ~0.85

Output ONLY valid JSON:
{
  "trend": "UP" | "DOWN" | "SIDEWAYS",
  "trend_strength": "STRONG" | "MODERATE" | "WEAK",
  "trend_reasoning": "...",
  "key_levels": {"support": [<price>, ...], "resistance": [<price>, ...]},
  "vwap_position": "ABOVE" | "BELOW" | "AT",
  "momentum": "ACCELERATING" | "STABLE" | "DECELERATING" | "REVERSING",
  "uk100_regime": "BULL_TREND" | "RANGE" | "DISTRIBUTION" | "BEAR_TREND" | "RECOVERY",
  "gbp_impact": "GBP_STRONG" | "GBP_WEAK" | "NEUTRAL",
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


RISK_AGENT_SYSTEM = """You are a professional risk manager for UK100 (FTSE 100) intraday trading.

Account parameters:
- Capital: $1,013 USD (demo account, ICMarkets)
- Max risk per trade: 2% = ~$20.26 USD
- Instrument: UK100 CFD (1 lot = $1/pt P&L approximately)
- ATR on 5m: 10-40 pts depending on London session
- Avoid stops < 15 pts
- Lot sizing: lots = $20.26 / stop_pts
  - E.g., 25pt stop → lots = 20.26/25 = 0.81 lots
  - E.g., 40pt stop → lots = 20.26/40 = 0.51 lots

UK100-specific risk:
- London open (02:00-02:30 CT): highest vol, wider stops needed
- BOE meeting days (8x/year): binary risk, flat or small size
- US futures open (08:30 CT): FTSE can gap/reverse when Wall St opens
- GBP data releases: instant volatility spike
- Avoid last 30 min of London session (11:00-11:30 CT): thin liquidity

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
    user_msg = f"""Analyze UK100 5-minute chart for intraday trading signal.

PRICE DATA SUMMARY:
{json.dumps(bars_summary, indent=2, default=str)}

FUNDAMENTAL / MACRO CONTEXT:
{json.dumps(fundamental_ctx, indent=2, default=str)}

Identify trend, UK100 regime, key levels, and best intraday setup.
Output only valid JSON."""
    return _call_claude(TREND_AGENT_SYSTEM, user_msg, model_id)


def call_risk_agent(bars_summary: dict, trend_analysis: dict, fundamental_ctx: dict, model_id: str = MODEL_ID) -> dict:
    current_price = bars_summary.get("current_state", {}).get("close", 0)
    user_msg = f"""Evaluate risk and size a UK100 trade.

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
            "reasoning":   risk.get("veto_reason") or setup.get("reasoning", "No setup."),
            "uk100_regime": trend.get("uk100_regime"),
            "gbp_impact":   trend.get("gbp_impact"),
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
        f"{trend.get('uk100_regime','')} / {trend.get('gbp_impact','')} / {trend.get('risk_on_off','')}"
    )

    return {
        "signal":       signal,
        "entry":        entry, "sl": sl, "tp": tp,
        "lots":         lots, "confidence": final_conf,
        "rr":           rr, "risk_usd": risk_usd,
        "reasoning":    reasoning,
        "uk100_regime": trend.get("uk100_regime"),
        "gbp_impact":   trend.get("gbp_impact"),
        "risk_on_off":  trend.get("risk_on_off"),
    }
