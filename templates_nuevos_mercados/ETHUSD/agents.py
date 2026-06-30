"""
ETHUSD Ethereum Robot — AI Agents
Two-agent system specialized for Ethereum CFD intraday trading.

ETH fundamentals that matter:
  - ETH/BTC ratio: ETH outperforms BTC in alt-season, underperforms in BTC dominance phases
  - DeFi TVL: Total Value Locked across Ethereum DeFi (Uniswap, Aave, Compound)
  - Staking yield: ETH staking APR (~3-5%) creates demand floor
  - Gas fees (Gwei): high gas = network congestion = ETH demand
  - Layer 2 activity: Arbitrum, Optimism, Base — L2 growth = ETH ecosystem bullish
  - ETF flows: ETH spot ETF (ETHA, FETH) daily inflows/outflows
  - Upcoming upgrades: Ethereum roadmap (Pectra, Fusaka etc.)
  - Key levels: $1.8k, $2k, $2.2k, $2.5k, $3k, $3.5k, $4k psychological
  - Correlation with BTC: high (0.85+), ETH typically amplifies BTC moves ×1.3-1.8
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

TREND_AGENT_SYSTEM = """You are a professional Ethereum (ETHUSD) intraday trader and technical analyst.
You specialize in 30-minute chart analysis during US trading hours and crypto prime sessions.

Your expertise:
- Ethereum 24/7 price action, intraday momentum, and trend patterns on 30m charts
- ETH/BTC ratio dynamics: when ETH leads, when BTC leads
- VWAP and session behavior specific to ETH (high beta to BTC, amplified moves)
- DeFi event-driven moves: protocol exploits, governance votes, major unlocks
- Ethereum upgrade cycle impact on sentiment
- Key psychological levels: $1.8k, $2k, $2.2k, $2.5k, $3k, $3.5k, $4k
- Liquidation clusters: ETH perp funding rate spikes, short/long squeeze setups
- Correlation with BTC: ETH typically moves 1.3-1.8× BTC moves
- Layer 2 activity as ETH demand proxy
- ETH staking yield creates a demand floor (~$1,800 break-even for validators)

CRITICAL TRADING RULES — must be respected in every signal:
- Each 30m bar represents 30 minutes of price action. Normal wicks can span $30-80+.
- NEVER go LONG when risk_on_off=RISK_OFF and eth_btc_dynamic=BTC_LEADING simultaneously.
  In this condition ETH amplifies BTC downside 1.3-1.8×. Counter-trend longs will be stopped.
- NEVER go LONG when eth_regime=DISTRIBUTION. Sell into strength, not buy into weakness.
- Only generate HIGH or MEDIUM confidence setups. LOW confidence = output NONE direction.
- Prefer trend-following over counter-trend. Oversold RSI alone is NOT a valid entry reason
  when the broader regime is bearish — oversold can stay oversold for many bars.

Output ONLY valid JSON with this structure:
{
  "trend": "UP" | "DOWN" | "SIDEWAYS",
  "trend_strength": "STRONG" | "MODERATE" | "WEAK",
  "trend_reasoning": "...",
  "key_levels": {"support": [<price>, ...], "resistance": [<price>, ...]},
  "vwap_position": "ABOVE" | "BELOW" | "AT",
  "momentum": "ACCELERATING" | "STABLE" | "DECELERATING" | "REVERSING",
  "eth_regime": "BULL_TREND" | "CONSOLIDATION" | "DISTRIBUTION" | "ALT_SEASON" | "BTC_DOMINANCE",
  "eth_btc_dynamic": "ETH_LEADING" | "BTC_LEADING" | "CORRELATED",
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


RISK_AGENT_SYSTEM = """You are a professional risk manager for Ethereum (ETHUSD) intraday trading.
You specialize in position sizing and trade management for ETH CFDs on 30-minute charts.

Account parameters:
- Capital: $1,013 USD (demo account, ICMarkets)
- Max risk per trade: 2% = ~$20.26 USD
- Instrument: ETHUSD CFD (1 lot = 1 ETH, 0.01 lot = 0.01 ETH)
- P&L: at 0.10 lot, a $100 ETH move = $10 profit/loss
- Lot sizing: lots = $20.26 / stop_distance_usd
  - E.g., $100 stop → lots = 20.26/100 = 0.20 lots
  - E.g., $200 stop → lots = 20.26/200 = 0.10 lots
  - E.g., $50  stop → lots = 20.26/50  = 0.40 lots (max 0.50)

ETH-specific risk considerations (30m timeframe):
- ATR on 30m: typically $30-150+ depending on volatility session
- Each 30m bar can wick $30-80 in normal conditions. Never place SL < 1.5× ATR30m.
- ETH amplifies BTC moves: if BTC dumps 2%, ETH may dump 3-4%
- Minimum stop distance: 1.5× ATR30m (hard floor — wicks will hit anything tighter)
- High-impact: FOMC, NFP, ETH-specific (protocol hacks, bridge exploits, SEC news)
- Weekend risk: lower liquidity, gap moves
- Funding rate spikes on perps cascade to spot price
- ETH/BTC ratio breaks can cause rapid reversion

Output ONLY valid JSON with this structure:
{
  "volatility_regime": "LOW" | "NORMAL" | "HIGH" | "EXTREME",
  "atr_30m": <value>,
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
    user_msg = f"""Analyze ETHUSD 5-minute chart for intraday trading signal.

PRICE DATA SUMMARY:
{json.dumps(bars_summary, indent=2, default=str)}

FUNDAMENTAL / MACRO CONTEXT:
{json.dumps(fundamental_ctx, indent=2, default=str)}

Identify the current trend, ETH-specific regime, key levels, and the best intraday setup (if any).
Output only valid JSON."""
    return _call_claude(TREND_AGENT_SYSTEM, user_msg, model_id)


def call_risk_agent(bars_summary: dict, trend_analysis: dict, fundamental_ctx: dict, model_id: str = MODEL_ID) -> dict:
    current_price = bars_summary.get("current_state", {}).get("close", 0)
    user_msg = f"""Evaluate risk and size an ETHUSD trade based on trend analysis.

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
    """Combine trend + risk agent outputs into final actionable signal.

    Extra filters applied here (code-level, cannot be overridden by LLM):
      F1 — Regime veto: no LONG when RISK_OFF + BTC_LEADING
      F2 — Distribution veto: no LONG when eth_regime=DISTRIBUTION
      F3 — ATR floor: SL must be >= 1.5 × ATR30m from entry
      F4 — R:R minimum: 1.5
    """
    setup     = trend.get("setup", {})
    direction = setup.get("direction", "NONE")
    viable    = risk.get("trade_viable", False)

    def _flat(reason):
        return {
            "signal": "FLAT", "entry": None, "sl": None, "tp": None,
            "lots": None, "confidence": "LOW", "rr": None, "risk_usd": None,
            "reasoning": reason,
            "eth_regime":  trend.get("eth_regime"),
            "eth_btc":     trend.get("eth_btc_dynamic"),
            "risk_on_off": trend.get("risk_on_off"),
        }

    if direction == "NONE" or not viable:
        return _flat(risk.get("veto_reason") or setup.get("reasoning", "No setup."))

    signal   = "LONG" if direction == "LONG" else "SHORT"
    entry    = risk.get("entry")
    sl       = risk.get("sl")
    tp       = risk.get("tp")
    lots     = risk.get("lots", 0.10)
    rr       = risk.get("rr_ratio")
    risk_usd = risk.get("risk_usd")
    atr      = risk.get("atr_30m") or risk.get("atr_5m") or 0   # accept both field names

    # ── F1: Regime veto — LONG blocked in RISK_OFF + BTC_LEADING ─────────────
    if (signal == "LONG"
            and trend.get("risk_on_off") == "RISK_OFF"
            and trend.get("eth_btc_dynamic") == "BTC_LEADING"):
        return _flat(
            f"[F1] LONG blocked: RISK_OFF + BTC_LEADING — ETH amplifies BTC downside ×1.3-1.8."
        )

    # ── F2: Distribution veto — LONG blocked in DISTRIBUTION regime ──────────
    if signal == "LONG" and trend.get("eth_regime") == "DISTRIBUTION":
        return _flat(
            f"[F2] LONG blocked: eth_regime=DISTRIBUTION — no buying into distribution phase."
        )

    # ── F4: R:R minimum ───────────────────────────────────────────────────────
    if rr is not None and float(rr) < 1.5:
        return _flat(f"R:R {rr} < 1.5 minimum — skip.")

    # ── F3: ATR floor — SL must be >= 1.5 × ATR30m ───────────────────────────
    ATR_MULT = 1.5
    if atr and entry and sl:
        min_sl_dist = ATR_MULT * float(atr)
        actual_dist = abs(float(entry) - float(sl))
        if actual_dist < min_sl_dist:
            # Widen SL to ATR floor instead of rejecting
            if signal == "LONG":
                sl = round(float(entry) - min_sl_dist, 2)
            else:
                sl = round(float(entry) + min_sl_dist, 2)
            # Recalculate lots with widened stop
            new_stop = abs(float(entry) - sl)
            if new_stop > 0:
                lots = round(max(0.01, min(0.50, round(20.26 / new_stop / 0.01) * 0.01)), 2)
            print(f"[F3-ATR] SL widened: {actual_dist:.1f} < {min_sl_dist:.1f} (1.5×ATR {atr}). "
                  f"New SL={sl}, lots={lots}")

    conf_rank  = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    final_conf = min(
        [setup.get("confidence", "LOW"), risk.get("confidence", "LOW")],
        key=lambda c: conf_rank.get(c, 0)
    )

    reasoning = (
        f"{setup.get('reasoning', '')} | "
        f"Regime: {risk.get('volatility_regime')} | "
        f"ATR30m: ${atr} | "
        f"{trend.get('eth_regime','')} / {trend.get('eth_btc_dynamic','')} / {trend.get('risk_on_off','')}"
    )

    return {
        "signal":     signal,
        "entry":      entry, "sl": sl, "tp": tp,
        "lots":       lots, "confidence": final_conf,
        "rr":         rr, "risk_usd": risk_usd,
        "reasoning":  reasoning,
        "eth_regime": trend.get("eth_regime"),
        "eth_btc":    trend.get("eth_btc_dynamic"),
        "risk_on_off": trend.get("risk_on_off"),
    }
