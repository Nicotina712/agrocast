"""
US500 S&P 500 Robot — AI Agents
Two-agent system specialized for S&P 500 CFD intraday trading.

S&P 500 fundamentals that matter:
  - VIX (Fear Index): VIX spike = sell, VIX crush = buy signal
  - FOMC sensitivity: rate hikes = S&P down, rate cuts = S&P up
  - Mag7 earnings impact: AAPL/NVDA/MSFT/GOOGL/META/AMZN/TSLA drive index
  - NFP (8:30 ET first Friday): high volatility catalyst
  - CPI/PPI: inflation data moves entire market
  - DXY (Dollar Index): inverse correlation with equities
  - Sector rotation: tech/growth vs value/defensive tells market health
  - Opening range (first 30min): key breakout reference
  - Key levels: 4800, 5000, 5200, 5500, 5800, 6000 psychological
  - S&P regime: BULL_TREND, RANGE, DISTRIBUTION, BEAR_TREND, RECOVERY
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

TREND_AGENT_SYSTEM = """You are a professional S&P 500 (US500) intraday trader and technical analyst.
You specialize in 5-minute chart analysis during regular US equity market hours (09:30-16:00 ET).

Your expertise:
- S&P 500 intraday momentum and trend patterns on 5m charts
- VIX correlation: VIX spike = bearish, VIX crush = bullish
- FOMC sensitivity: rate hikes negative, rate cuts positive for equities
- Mag7 earnings impact: AAPL/NVDA/MSFT/GOOGL/META/AMZN/TSLA drive 30%+ of index
- Opening range breakout (ORB): first 30min sets the tone for the day
- VWAP and session VWAP behavior specific to equities
- Pre-market gaps: gap-and-go vs gap-fill setups
- Sector rotation signals: tech/growth vs value/defensive
- Key psychological levels: 4800, 5000, 5200, 5500, 5800, 6000
- NFP (8:30 ET first Friday), CPI, GDP: high-impact macro events
- DXY inverse correlation: strong dollar = S&P headwind

Output ONLY valid JSON with this structure:
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


RISK_AGENT_SYSTEM = """You are a professional risk manager for S&P 500 (US500) intraday trading.
You specialize in position sizing and trade management for US500 CFDs.

Account parameters:
- Capital: $1,013 USD (demo account, ICMarkets)
- Max risk per trade: 2% = ~$20.26 USD
- Instrument: US500 CFD (1 lot = $1/pt P&L)
- Lot sizing: lots = $20.26 / stop_distance_pts
  - E.g., 15pt stop → lots = 20.26/15 = 1.35 lots
  - E.g., 20pt stop → lots = 20.26/20 = 1.01 lots
  - E.g., 10pt stop → lots = 20.26/10 = 2.03 lots (max 5.0)

US500-specific risk considerations:
- ATR on 5m: typically 5-20 pts depending on session and volatility
- Avoid stops < 8 pts (index wicks aggressively, especially at open/close)
- First 30min (09:30-10:00 ET): high volatility, wider stops needed
- Last 30min (15:30-16:00 ET): avoid new signals
- High-impact: FOMC meetings, NFP (8:30 ET first Friday), CPI, major earnings
- VIX > 25: reduce size, VIX > 30: consider no new longs
- Opening gaps > 20 pts: wait for confirmation before fading
- Round number magnets: 4800, 5000, 5200, 5500, 5800, 6000

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
    user_msg = f"""Analyze US500 (S&P 500) 5-minute chart for intraday trading signal.

PRICE DATA SUMMARY:
{json.dumps(bars_summary, indent=2, default=str)}

FUNDAMENTAL / MACRO CONTEXT:
{json.dumps(fundamental_ctx, indent=2, default=str)}

Identify the current trend, S&P 500 regime, key levels, and the best intraday setup (if any).
Output only valid JSON."""
    return _call_claude(TREND_AGENT_SYSTEM, user_msg, model_id)


def call_risk_agent(bars_summary: dict, trend_analysis: dict, fundamental_ctx: dict, model_id: str = MODEL_ID) -> dict:
    current_price = bars_summary.get("current_state", {}).get("close", 0)
    user_msg = f"""Evaluate risk and size a US500 trade based on trend analysis.

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
