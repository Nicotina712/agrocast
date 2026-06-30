"""
BTCUSD Bitcoin Robot — AI Agents
Two-agent system: Trend Agent + Risk Agent, specialized for Bitcoin CFD.

Bitcoin fundamentals that matter:
  - Halving cycle (last: April 2024, next: ~April 2028)
  - Institutional flows: ETF (IBIT, FBTC, GBTC) daily inflows/outflows
  - BTC Dominance: high dominance = altcoins weak, BTC strong
  - DXY correlation: negative (strong dollar → BTC pressure)
  - Risk-on/risk-off: BTC moves with NASDAQ in risk-off events
  - On-chain: whale accumulation, exchange reserves, miner selling
  - Key levels: $70k, $75k, $80k, $85k, $90k, $100k psychological levels
  - 24/7 market: gaps from Asian/European sessions matter
"""

import os
import json
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
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
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


# ─── System Prompts ───────────────────────────────────────────────────────────

TREND_AGENT_SYSTEM = """You are a professional Bitcoin (BTCUSD) intraday trader and technical analyst.
You specialize in 5-minute chart analysis for the London-NY session overlap and US trading hours.

Your expertise:
- Bitcoin 24/7 price action and intraday momentum patterns
- Identifying trend continuations vs reversals on BTC
- Reading VWAP and session open behavior
- Understanding crypto-specific microstructure (funding rates, liquidation clusters)
- Psychological round number levels: $90k, $95k, $100k, $105k, $110k
- Detecting accumulation/distribution patterns via volume
- BTC's relationship with NASDAQ (risk-on/risk-off correlation)
- DXY inverse correlation: rising dollar pressure on BTC
- Bitcoin halving cycle context (last halving: April 2024, in bull-cycle phase)

Output ONLY valid JSON with this structure:
{
  "trend": "UP" | "DOWN" | "SIDEWAYS",
  "trend_strength": "STRONG" | "MODERATE" | "WEAK",
  "trend_reasoning": "...",
  "key_levels": {"support": [<price>, ...], "resistance": [<price>, ...]},
  "vwap_position": "ABOVE" | "BELOW" | "AT",
  "momentum": "ACCELERATING" | "STABLE" | "DECELERATING" | "REVERSING",
  "btc_regime": "BULL_TREND" | "CONSOLIDATION" | "DISTRIBUTION" | "CAPITULATION",
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


RISK_AGENT_SYSTEM = """You are a professional risk manager for Bitcoin (BTCUSD) intraday trading.
You specialize in position sizing, stop placement, and trade management for BTC CFDs.

Account parameters:
- Capital: $1,013 USD (demo account, ICMarkets)
- Max risk per trade: 2% = ~$20.26 USD
- Instrument: BTCUSD CFD (1 lot = 1 BTC, 0.01 lot = 0.01 BTC)
- P&L: at 0.01 lot, a $1,000 BTC move = $10 profit/loss
- Lot sizing formula: lots = $20.26 / stop_distance_usd
  - E.g., $1,500 stop → lots = 20.26/1500 = 0.013 → round to 0.01
  - E.g., $800 stop  → lots = 20.26/800  = 0.025 → round to 0.03 (max 0.05)

Bitcoin-specific risk considerations:
- Volatility regime: BTC ATR(14) on 5m can be $300–1,500+ depending on conditions
- High-impact events: FOMC, NFP, macro news can cause $2,000–5,000 moves instantly
- Weekend risk: lower liquidity, sudden gap moves common
- Funding rate spikes: perpetual futures funding can cause rapid price resets
- Liquidation cascades: stop clusters at round numbers ($100k, $105k etc.)
- Avoid entries with <$500 stop (too tight, high whipsaw probability on BTC)
- Never risk more than $20 per trade regardless of setup quality

Output ONLY valid JSON with this structure:
{
  "volatility_regime": "LOW" | "NORMAL" | "HIGH" | "EXTREME",
  "atr_5m": <value>,
  "stop_distance": <recommended_stop_in_usd_points>,
  "take_profit_distance": <recommended_tp_in_usd_points>,
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

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON object in text
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            return json.loads(m.group(0))
        return {"error": f"JSON parse failed: {raw[:200]}"}


def call_trend_agent(bars_summary: dict, fundamental_ctx: dict, model_id: str = MODEL_ID) -> dict:
    user_msg = f"""Analyze BTCUSD 5-minute chart for intraday trading signal.

PRICE DATA SUMMARY:
{json.dumps(bars_summary, indent=2, default=str)}

FUNDAMENTAL / MACRO CONTEXT:
{json.dumps(fundamental_ctx, indent=2, default=str)}

Identify the current trend, key levels, and the best intraday setup (if any).
Output only valid JSON."""
    return _call_claude(TREND_AGENT_SYSTEM, user_msg, model_id)


def call_risk_agent(bars_summary: dict, trend_analysis: dict, fundamental_ctx: dict, model_id: str = MODEL_ID) -> dict:
    current_price = bars_summary.get("current_state", {}).get("close", 0)
    user_msg = f"""Evaluate risk and size a BTCUSD trade based on trend analysis.

CURRENT PRICE: ${current_price:,.2f}
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
    setup = trend.get("setup", {})
    direction = setup.get("direction", "NONE")
    trade_viable = risk.get("trade_viable", False)

    if direction == "NONE" or not trade_viable:
        return {
            "signal":     "FLAT",
            "entry":      None,
            "sl":         None,
            "tp":         None,
            "lots":       None,
            "confidence": "LOW",
            "rr":         None,
            "risk_usd":   None,
            "reasoning":  risk.get("veto_reason") or setup.get("reasoning", "No setup."),
        }

    # Map direction
    signal = "LONG" if direction == "LONG" else "SHORT"

    entry = risk.get("entry")
    sl    = risk.get("sl")
    tp    = risk.get("tp")
    lots  = risk.get("lots", 0.01)
    rr    = risk.get("rr_ratio")
    risk_usd = risk.get("risk_usd")

    # Guard: minimum 1.5 R:R
    if rr is not None and float(rr) < 1.5:
        return {
            "signal":     "FLAT",
            "entry":      entry,
            "sl":         sl,
            "tp":         tp,
            "lots":       lots,
            "confidence": "LOW",
            "rr":         rr,
            "risk_usd":   risk_usd,
            "reasoning":  f"R:R {rr} < 1.5 minimum — skip.",
        }

    # Confidence: take the lower of trend + risk
    trend_conf = setup.get("confidence", "LOW")
    risk_conf  = risk.get("confidence", "LOW")
    conf_rank  = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    final_conf = min([trend_conf, risk_conf], key=lambda c: conf_rank.get(c, 0))

    reasoning = (
        f"{setup.get('reasoning', '')} | "
        f"Regime: {risk.get('volatility_regime')} | "
        f"ATR5m: ${risk.get('atr_5m', '?')} | "
        f"{trend.get('btc_regime', '')} / {trend.get('risk_on_off', '')}"
    )

    return {
        "signal":     signal,
        "entry":      entry,
        "sl":         sl,
        "tp":         tp,
        "lots":       lots,
        "confidence": final_conf,
        "rr":         rr,
        "risk_usd":   risk_usd,
        "reasoning":  reasoning,
        "btc_regime": trend.get("btc_regime"),
        "risk_on_off": trend.get("risk_on_off"),
    }
