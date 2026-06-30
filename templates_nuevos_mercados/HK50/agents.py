"""
HK50 Hang Seng Index Robot — AI Agents
Two-agent system specialized for HK50 (Hang Seng) CFD intraday trading.
NOTE: JSON key 'sp500_regime' is reused as a generic index-regime field so the shared
live_runner + dashboard stay compatible across instruments.

HK50 / Hang Seng fundamentals that matter:
  - China macro: PMI, GDP, retail sales, credit/TSF data drive sentiment
  - PBOC policy: rate/RRR cuts, liquidity injections = bullish
  - Property sector stress: developer defaults (Evergrande-type) = bearish
  - Index megacaps: Tencent, Alibaba, Meituan, HSBC, AIA dominate the index
  - US-China relations: tariffs, tech export controls, delisting risk
  - CNH/HKD and capital flows; Stock Connect (northbound/southbound) flows
  - Session: HKT 09:30-12:00 + 13:00-16:00 (lunch break midday)
  - Key levels: 16000, 17000, 18000, 19000, 20000, 22000 psychological
  - HSI regime: BULL_TREND, RANGE, DISTRIBUTION, BEAR_TREND, RECOVERY
"""

import os
import json
import re
import sys

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


# ─── System Prompts ───────────────────────────────────────────────────────────

TREND_AGENT_SYSTEM = """You are a professional Hang Seng Index (HK50) intraday trader and technical analyst.
You specialize in 30-minute chart analysis during the Hong Kong cash session (HKT 09:30-16:00, lunch 12:00-13:00).

Your expertise:
- HK50 intraday momentum and trend patterns on 30m charts
- China macro sensitivity: PMI, GDP, retail sales, credit/TSF data move sentiment
- PBOC policy: rate/RRR cuts and liquidity injections are bullish catalysts
- Property-sector stress: developer defaults / credit events are bearish
- Index megacaps: Tencent, Alibaba, Meituan, HSBC, AIA drive the index
- Stock Connect flows (northbound/southbound) signal mainland appetite
- Opening range: first 30-60min sets the tone; lunch break can cause gaps
- US-China relations: tariffs, tech export controls, delisting headlines
- CNH/HKD moves and capital flows
- Key psychological levels: 16000, 17000, 18000, 19000, 20000, 22000

Output ONLY valid JSON with this structure
(NOTE: 'sp500_regime' here = generic HSI/index regime field):
{
  "trend": "UP" | "DOWN" | "SIDEWAYS",
  "trend_strength": "STRONG" | "MODERATE" | "WEAK",
  "trend_reasoning": "...",
  "key_levels": {"support": [<price>, ...], "resistance": [<price>, ...]},
  "vwap_position": "ABOVE" | "BELOW" | "AT",
  "momentum": "ACCELERATING" | "STABLE" | "DECELERATING" | "REVERSING",
  "sp500_regime": "BULL_TREND" | "RANGE" | "DISTRIBUTION" | "BEAR_TREND" | "RECOVERY",
  "risk_on_off": "RISK_ON" | "RISK_OFF" | "NEUTRAL",
  "setup": {
    "type": "BREAKOUT" | "VWAP_PULLBACK" | "TREND_CONTINUATION" | "REVERSAL" | "ORB" | "NONE",
    "direction": "LONG" | "SHORT" | "NONE",
    "confidence": "HIGH" | "MEDIUM" | "LOW",
    "entry_zone": [<low_price>, <high_price>],
    "invalidation": <price>,
    "reasoning": "..."
  }
}"""


RISK_AGENT_SYSTEM = """You are a professional risk manager for Hang Seng Index (HK50) intraday trading.
You specialize in position sizing and trade management for HK50 CFDs.

Account parameters:
- Capital: $1,013 USD (demo account, ICMarkets)
- Max risk per trade: 3% = ~$30.39 USD
- Instrument: HK50 CFD (≈ $1/pt P&L on demo)
- Lot sizing: lots = $30.39 / stop_distance_pts
  - E.g., 150pt stop → lots = 30.39/150 = 0.20 lots
  - E.g., 100pt stop → lots = 30.39/100 = 0.30 lots
  - capped at MAX 2.0 lots

HK50-specific risk considerations:
- ATR on 30m: typically 80-300 pts depending on session and volatility
- Avoid stops < 40 pts (HSI wicks aggressively)
- First 30-60min after open: high volatility, wider stops needed
- Lunch break (HKT 12:00-13:00): liquidity gap, avoid straddling it
- Last 30min of session: avoid new signals
- High-impact: China data releases, PBOC decisions, property-sector headlines, US-China news
- Round number magnets: 16000, 17000, 18000, 19000, 20000, 22000

Output ONLY valid JSON with this structure:
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


# ─── Agent callers ────────────────────────────────────────────────────────────

def _call_claude(system: str, user_msg: str, model: str = MODEL_ID) -> dict:
    client = _get_client()
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
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
        if m:
            return json.loads(m.group(0))
        return {"error": f"JSON parse failed: {raw[:200]}"}


def call_trend_agent(bars_summary: dict, fundamental_ctx: dict, model_id: str = MODEL_ID) -> dict:
    user_msg = f"""Analyze HK50 (Hang Seng) 30-minute chart for intraday trading signal.

PRICE DATA SUMMARY:
{json.dumps(bars_summary, indent=2, default=str)}

FUNDAMENTAL / MACRO CONTEXT:
{json.dumps(fundamental_ctx, indent=2, default=str)}

Identify the current trend, HSI regime, key levels, and the best intraday setup (if any).
Output only valid JSON."""
    return _call_claude(TREND_AGENT_SYSTEM, user_msg, model_id)


def call_risk_agent(bars_summary: dict, trend_analysis: dict, fundamental_ctx: dict, model_id: str = MODEL_ID) -> dict:
    current_price = bars_summary.get("current_state", {}).get("close", 0)
    user_msg = f"""Evaluate risk and size a HK50 trade based on trend analysis.

CURRENT PRICE: {current_price:,.2f}
PRICE DATA SUMMARY:
{json.dumps(bars_summary, indent=2, default=str)}

TREND AGENT ANALYSIS:
{json.dumps(trend_analysis, indent=2, default=str)}

FUNDAMENTAL CONTEXT:
{json.dumps(fundamental_ctx, indent=2, default=str)}

Determine entry, stop-loss, take-profit, position size, and whether the trade is viable.
Output only valid JSON."""
    return _call_claude(RISK_AGENT_SYSTEM, user_msg, model_id)


def synthesize_signal(trend: dict, risk: dict) -> dict:
    """Combine trend + risk agent outputs into final actionable signal."""
    setup     = trend.get("setup", {})
    direction = setup.get("direction", "NONE")
    viable    = risk.get("trade_viable", False)

    if direction == "NONE" or not viable:
        return {
            "signal":       "FLAT",
            "entry":        None, "sl": None, "tp": None,
            "lots":         None, "confidence": "LOW",
            "rr":           None, "risk_usd": None,
            "reasoning":    risk.get("veto_reason") or setup.get("reasoning", "No setup."),
            "sp500_regime": trend.get("sp500_regime"),
            "risk_on_off":  trend.get("risk_on_off"),
        }

    signal   = "LONG" if direction == "LONG" else "SHORT"
    entry    = risk.get("entry")
    sl       = risk.get("sl")
    tp       = risk.get("tp")
    lots     = risk.get("lots", 1.0)
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
        f"{trend.get('sp500_regime','')} / {trend.get('risk_on_off','')}"
    )

    return {
        "signal":       signal,
        "entry":        entry, "sl": sl, "tp": tp,
        "lots":         lots, "confidence": final_conf,
        "rr":           rr, "risk_usd": risk_usd,
        "reasoning":    reasoning,
        "sp500_regime": trend.get("sp500_regime"),
        "risk_on_off":  trend.get("risk_on_off"),
    }
