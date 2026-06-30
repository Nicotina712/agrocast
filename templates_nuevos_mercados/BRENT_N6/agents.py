"""
BRENT_N6 Brent Crude Oil Robot — AI Agents
Two-agent system specialized for Brent Crude CFD intraday trading.

Brent fundamentals that matter:
  - EIA inventory: Wednesday 10:30 ET — crude drawdown/build moves Brent $0.50-2.00
  - OPEC+ production decisions: spare capacity drives the baseline
  - Geopolitical risk premium: Middle East, Russia supply disruptions
  - Brent/WTI spread: typically $3-5, widens on North Sea supply issues
  - North Sea supply: Brent is the North Sea benchmark
  - USD index: Brent denominated in USD, inverse correlation
  - London session (03:00-11:00 CT) has highest Brent volume
  - ATR 5m: $0.30-1.80 depending on session volatility
  - Key levels: $70, $75, $80, $85, $90, $95 per barrel
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

TREND_AGENT_SYSTEM = """You are a professional Brent Crude Oil (BRENT_N6) intraday trader and technical analyst.
You specialize in 5-minute chart analysis during London and NYMEX trading sessions.

Your expertise:
- Brent crude intraday 5m price action — global oil benchmark
- London session (03:00-11:00 CT) has the highest Brent volume and tightest spreads
- OPEC+ spare capacity drives baseline supply; geopolitical premium on top
- Geopolitical risk premium (Middle East, Russia) more impactful for Brent than WTI
- Brent/WTI spread: normally $3-5; widens on North Sea supply disruptions
- EIA inventory data (Wednesday 10:30 ET): crude drawdown bullish, build bearish
- USD index: Brent denominated in USD — strong DXY → Brent headwind
- ATR 5m: $0.30-1.80 depending on session; London open can be volatile
- Stops minimum: $0.60 (too tight below this, wicks wash out)
- Key psychological levels: $70, $75, $80, $85, $90, $95 per barrel

Output ONLY valid JSON with this structure:
{
  "trend": "UP" | "DOWN" | "SIDEWAYS",
  "trend_strength": "STRONG" | "MODERATE" | "WEAK",
  "trend_reasoning": "...",
  "key_levels": {"support": [<price>, ...], "resistance": [<price>, ...]},
  "vwap_position": "ABOVE" | "BELOW" | "AT",
  "momentum": "ACCELERATING" | "STABLE" | "DECELERATING" | "REVERSING",
  "oil_regime": "BULL_TREND" | "RANGE" | "BEAR_TREND" | "SUPPLY_SHOCK" | "DEMAND_DRIVEN",
  "brent_wti_spread": "WIDENING" | "NARROWING" | "STABLE",
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


RISK_AGENT_SYSTEM = """You are a professional risk manager for Brent Crude Oil (BRENT_N6) intraday trading.
You specialize in position sizing and trade management for Brent CFDs.

Account parameters:
- Capital: $1,013 USD (demo account, ICMarkets)
- Max risk per trade: 2% = ~$20.26 USD
- Instrument: BRENT_N6 CFD (1 lot = $1/pt approx)
- Lot sizing: lots = $20.26 / stop_usd
  - E.g., $1.00 stop → lots = 20.26/1.00 = 2.0 lots (max)
  - E.g., $1.50 stop → lots = 20.26/1.50 = 1.4 lots
  - E.g., $2.00 stop → lots = 20.26/2.00 = 1.0 lots

Brent-specific risk considerations:
- ATR on 5m: typically $0.30-1.80 depending on session
- London open (03:00 CT): can gap/spike $0.50-1.00 in first 15 min
- EIA Wednesday 10:30 ET: avoid new positions 30 min either side
- OPEC+ announcements: binary event, do not trade during surprise announcements
- Stop minimum: $0.60 (wicks wash out tighter stops)
- Brent/WTI spread trades can cause sudden correlation breaks
- Geopolitical events: risk premium can collapse 3-5% in minutes on de-escalation

Output ONLY valid JSON with this structure:
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
    user_msg = f"""Analyze BRENT_N6 5-minute chart for intraday trading signal.

PRICE DATA SUMMARY:
{json.dumps(bars_summary, indent=2, default=str)}

FUNDAMENTAL / MACRO CONTEXT:
{json.dumps(fundamental_ctx, indent=2, default=str)}

Identify the current trend, Brent oil regime, key levels ($70-$95 range), and the best intraday setup (if any).
Output only valid JSON."""
    return _call_claude(TREND_AGENT_SYSTEM, user_msg, model_id)


def call_risk_agent(bars_summary: dict, trend_analysis: dict, fundamental_ctx: dict, model_id: str = MODEL_ID) -> dict:
    current_price = bars_summary.get("current_state", {}).get("close", 0)
    user_msg = f"""Evaluate risk and size a BRENT_N6 trade based on trend analysis.

CURRENT PRICE: ${current_price:,.2f}
PRICE DATA SUMMARY:
{json.dumps(bars_summary, indent=2, default=str)}

TREND AGENT ANALYSIS:
{json.dumps(trend_analysis, indent=2, default=str)}

FUNDAMENTAL CONTEXT:
{json.dumps(fundamental_ctx, indent=2, default=str)}

Determine entry, stop-loss (minimum $0.60), take-profit, position size (lots = $20.26 / stop_usd, max 2.0), and whether the trade is viable.
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
            "oil_regime":   trend.get("oil_regime"),
            "brent_wti_spread": trend.get("brent_wti_spread"),
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
        f"{trend.get('oil_regime','')} / Spread: {trend.get('brent_wti_spread','')} / {trend.get('risk_on_off','')}"
    )

    return {
        "signal":           signal,
        "entry":            entry, "sl": sl, "tp": tp,
        "lots":             lots, "confidence": final_conf,
        "rr":               rr, "risk_usd": risk_usd,
        "reasoning":        reasoning,
        "oil_regime":       trend.get("oil_regime"),
        "brent_wti_spread": trend.get("brent_wti_spread"),
        "risk_on_off":      trend.get("risk_on_off"),
    }
