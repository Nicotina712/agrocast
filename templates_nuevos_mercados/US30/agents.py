"""
US30 Dow Jones Robot — AI Agents
Two-agent system specialized for US30 (Dow Jones Industrial Average) CFD intraday trading.

US30 fundamentals that matter:
  - Dow Jones 30 industrial stocks: heavier weight on financials/industrials vs tech
  - VIX level: fear index, inverse correlation with equities
  - Fed policy (FOMC): rate decisions drive multi-week trends
  - ISM Manufacturing: key for industrial sector stocks
  - Financial sector: JPMorgan, Goldman Sachs, Bank of America
  - DXY trend: USD strength affects multinational earnings
  - Key round levels: 38k, 40k, 42k, 44k, 45k psychological
  - NFP, GDP, retail sales: macro data releases move the index
  - S&P500 / NASDAQ correlation: US30 typically less volatile than NDX
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

TREND_AGENT_SYSTEM = """You are a professional US30 (Dow Jones Industrial Average) intraday trader and technical analyst.
You specialize in 5-minute chart analysis during NYSE regular market hours (09:30-16:00 ET).

Your expertise:
- Dow Jones 30 price action, intraday momentum, and trend patterns on 5m charts
- US30 composition: heavier weighting on financials (JPMorgan, Goldman, Visa), industrials (Boeing, Caterpillar, 3M), and energy vs technology
- VWAP and session behavior: opening range breakout, 09:30-10:30 ET first hour volatility
- VIX correlation: VIX > 25 = high fear, avoid long setups; VIX > 30 = extreme, reduce risk
- Key psychological round levels: 38,000 / 40,000 / 42,000 / 44,000 / 45,000 points
- FOMC meeting impact: rate decisions can move US30 by 300-600 pts in minutes
- NFP Fridays: 08:30 ET release causes strong initial spike then reversal
- S&P500 divergence: when SPX and US30 diverge, expect mean reversion
- Pre-market futures gap: large gaps often fill within first 2h of trading
- ATR 5m: typically 30-100 pts depending on news/volatility

Output ONLY valid JSON with this structure:
{
  "trend": "UP" | "DOWN" | "SIDEWAYS",
  "trend_strength": "STRONG" | "MODERATE" | "WEAK",
  "trend_reasoning": "...",
  "key_levels": {"support": [<price>, ...], "resistance": [<price>, ...]},
  "vwap_position": "ABOVE" | "BELOW" | "AT",
  "momentum": "ACCELERATING" | "STABLE" | "DECELERATING" | "REVERSING",
  "us30_regime": "BULL_TREND" | "RANGE" | "DISTRIBUTION" | "BEAR_TREND" | "RECOVERY",
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


RISK_AGENT_SYSTEM = """You are a professional risk manager for US30 (Dow Jones Industrial Average) intraday trading.
You specialize in position sizing and trade management for US30 CFDs.

Account parameters:
- Capital: $1,013 USD (demo account, ICMarkets)
- Max risk per trade: 2% = ~$20.26 USD
- Instrument: US30 CFD (1 lot = $1/pt P&L approximation)
- Lot sizing: lots = $20.26 / stop_distance_pts
  - E.g., 50pt stop  → lots = 20.26/50  = 0.40 lots
  - E.g., 100pt stop → lots = 20.26/100 = 0.20 lots
  - E.g., 30pt stop  → lots = 20.26/30  = 0.67 → cap at MAX_LOTS=1.0

US30-specific risk considerations:
- ATR on 5m: typically 30-100 pts depending on volatility/news
- Dow Jones is an industrial-heavy index, financials sector is the largest weight
- Avoid stops < 30 pts (US30 wicks aggressively around round numbers)
- High-impact: FOMC, NFP (08:30 ET Fridays), CPI/PPI, ISM Manufacturing
- NO weekend trading (equity index follows NYSE hours)
- Opening first 5min extremely volatile: wait for 09:35-09:40 ET before entries
- Last 30min of session (15:30-16:00 ET): liquidity drops, avoid new positions
- VIX spike > 25: reduce position size or avoid longs

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
    user_msg = f"""Analyze US30 (Dow Jones) 5-minute chart for intraday trading signal.

PRICE DATA SUMMARY:
{json.dumps(bars_summary, indent=2, default=str)}

FUNDAMENTAL / MACRO CONTEXT:
{json.dumps(fundamental_ctx, indent=2, default=str)}

Identify the current trend, US30-specific regime, key levels, and the best intraday setup (if any).
Remember: US30 key round levels are 38k, 40k, 42k, 44k, 45k — price gravitates to these.
Output only valid JSON."""
    return _call_claude(TREND_AGENT_SYSTEM, user_msg, model_id)


def call_risk_agent(bars_summary: dict, trend_analysis: dict, fundamental_ctx: dict, model_id: str = MODEL_ID) -> dict:
    current_price = bars_summary.get("current_state", {}).get("close", 0)
    user_msg = f"""Evaluate risk and size a US30 (Dow Jones) trade based on trend analysis.

CURRENT PRICE: {current_price:,.0f} pts
PRICE DATA SUMMARY:
{json.dumps(bars_summary, indent=2, default=str)}

TREND AGENT ANALYSIS:
{json.dumps(trend_analysis, indent=2, default=str)}

FUNDAMENTAL CONTEXT:
{json.dumps(fundamental_ctx, indent=2, default=str)}

Determine entry, stop-loss, take-profit, position size, and whether the trade is viable.
Min stop: 30 pts. Max lots: 1.0. Capital: $1,013. Max risk: $20.26 (2%).
Output only valid JSON."""
    return _call_claude(RISK_AGENT_SYSTEM, user_msg, model_id)


def synthesize_signal(trend: dict, risk: dict) -> dict:
    """Combine trend + risk agent outputs into final actionable signal."""
    setup     = trend.get("setup", {})
    direction = setup.get("direction", "NONE")
    viable    = risk.get("trade_viable", False)

    if direction == "NONE" or not viable:
        return {
            "signal":      "FLAT",
            "entry":       None, "sl": None, "tp": None,
            "lots":        None, "confidence": "LOW",
            "rr":          None, "risk_usd": None,
            "reasoning":   risk.get("veto_reason") or setup.get("reasoning", "No setup."),
            "us30_regime": trend.get("us30_regime"),
            "risk_on_off": trend.get("risk_on_off"),
        }

    signal   = "LONG" if direction == "LONG" else "SHORT"
    entry    = risk.get("entry")
    sl       = risk.get("sl")
    tp       = risk.get("tp")
    lots     = risk.get("lots", 0.3)
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
        f"ATR5m: {risk.get('atr_5m','?')} pts | "
        f"{trend.get('us30_regime','')} / {trend.get('risk_on_off','')}"
    )

    return {
        "signal":      signal,
        "entry":       entry, "sl": sl, "tp": tp,
        "lots":        lots, "confidence": final_conf,
        "rr":          rr, "risk_usd": risk_usd,
        "reasoning":   reasoning,
        "us30_regime": trend.get("us30_regime"),
        "risk_on_off": trend.get("risk_on_off"),
    }
