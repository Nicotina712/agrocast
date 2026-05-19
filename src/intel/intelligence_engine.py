"""
Intelligence Engine v2 — Orchestrator
Pipeline: FinBERT (Tier-1) → Claude Sonnet (Tier-2) → Multi-Agent Debate → Verdict

This is the main entry point for the "intelligent reading" system.
Instead of pure ML regression on historical prices, this system:
  1. Reads ALL incoming news via fast classifier (FinBERT/VADER)
  2. Escalates high-impact articles to Claude Sonnet for deep analysis
  3. Retrieves relevant historical analogs from Knowledge Base (RAG)
  4. Runs a Multi-Agent Debate (Bull/Bear/Risk → Fund Manager)
  5. Produces a unified verdict with price range, action, and confidence
"""

import gc
import os
import json
import time
from datetime import datetime

try:
    from dotenv import load_dotenv
    _env_candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        os.path.join(os.path.dirname(__file__), "..", "..", "MVP lectura de noticias", ".env"),
    ]
    for _ep in _env_candidates:
        if os.path.exists(_ep):
            load_dotenv(_ep, override=True)
except ImportError:
    pass

from .finbert_classifier import classify_batch
from .knowledge_base import get_knowledge_base
from .multi_agent_debate import run_debate
from .debate_repository import save_debate
from .ie_accountability import save_verdict_snapshot, get_feedback_for_debate, evaluate_verdicts

_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_BASE, "..", ".."))
_DATA = os.path.join(_ROOT, "data")
_ARTIFACTS = os.path.join(_ROOT, "artifacts")
_CACHE_FILE = os.path.join(_DATA, "intelligence_engine_verdict.json")
_CACHE_TTL = 3600


def _load_market_context() -> dict:
    ctx = {}

    try:
        with open(os.path.join(_DATA, "current_contract.json")) as f:
            cc = json.load(f)
            ctx["contract"] = cc.get("front_contract", "ZS")
            ctx["spread"] = cc.get("spread_to_next")
            ctx["current_price"] = cc.get("price", 0)
    except Exception:
        ctx["current_price"] = 0

    try:
        with open(os.path.join(_ARTIFACTS, "regime.json")) as f:
            reg = json.load(f)
            ctx["regime"] = reg.get("regime", "unknown")
            ctx["regime_method"] = reg.get("method", "")
    except Exception:
        pass

    try:
        with open(os.path.join(_DATA, "china_demand.json")) as f:
            china = json.load(f)
            cm = china.get("crush_margin", {})
            ctx["china"] = (
                f"crush_margin={cm.get('margin_usd_ton','?')} USD/ton ({cm.get('signal','?')}), "
                f"imports_yoy={china.get('imports_yoy_pct','?')}%, "
                f"demand_score={china.get('demand_score','?')}/100"
            )
    except Exception:
        pass

    try:
        with open(os.path.join(_DATA, "brazil_exports.json")) as f:
            br = json.load(f)
            ctx["brazil"] = (
                f"exports_ytd={br.get('exported_ytd_mmt','?')} MMT, "
                f"yoy={br.get('yoy_pct','?')}%, "
                f"pace={br.get('weekly_pace_mmt','?')} MMT/sem"
            )
    except Exception:
        pass

    try:
        with open(os.path.join(_DATA, "basis_uruguay.json")) as f:
            basis = json.load(f)
            if basis.get("basis_usd_ton") is not None:
                ctx["basis_uy"] = (
                    f"{basis['basis_usd_ton']:+.1f} USD/ton, "
                    f"signal={basis.get('signal','?')}, "
                    f"zscore={basis.get('basis_zscore','?')}"
                )
    except Exception:
        pass

    try:
        with open(os.path.join(_DATA, "wasde_official.json")) as f:
            w = json.load(f)
            ctx["wasde"] = (
                f"ending_stocks={w.get('ending_stocks_mbu','?')} Mbu, "
                f"surprise={w.get('surprise_signal','?')}, "
                f"as_of={w.get('as_of','?')}"
            )
    except Exception:
        pass

    try:
        with open(os.path.join(_ARTIFACTS, "active_shock.json")) as f:
            shock = json.load(f)
            if shock.get("active"):
                ctx["active_shock"] = (
                    f"type={shock.get('shock_type','?')}, "
                    f"direction={shock.get('direction','?')}, "
                    f"magnitude={shock.get('magnitude_pct','?')}%"
                )
    except Exception:
        pass

    try:
        import pandas as pd
        sig_path = os.path.join(_ARTIFACTS, "signals.csv")
        if os.path.exists(sig_path):
            df = pd.read_csv(sig_path)
            if not df.empty:
                last = df.iloc[-1]
                ctx["forecast_ml"] = (
                    f"signal={last.get('signal','?')}, "
                    f"confidence={last.get('confidence','?')}, "
                    f"expected_return={last.get('expected_return','?')}"
                )
    except Exception:
        pass

    try:
        cot_path = os.path.join(_DATA, "cot_soybeans.csv")
        if os.path.exists(cot_path):
            import pandas as pd
            df = pd.read_csv(cot_path)
            if not df.empty:
                last = df.iloc[-1]
                net_spec = last.get("NonComm_Long", 0) - last.get("NonComm_Short", 0)
                ctx["cot"] = f"net_speculative={net_spec:.0f} contracts"
    except Exception:
        pass

    # ── ML accountability stats ──
    try:
        from ..trader.accountability import get_accountability_records
        acc = get_accountability_records()
        summary = acc.get("summary", {})
        if summary.get("dir_accuracy_7d") is not None:
            ctx["ml_accountability"] = (
                f"accuracy_dir_7d={summary['dir_accuracy_7d']}% "
                f"(n={summary.get('n_evaluated_7d', 0)}), "
                f"mae_7d={summary.get('mae_pct_7d', '?')}%, "
                f"accuracy_dir_30d={summary.get('dir_accuracy_30d', '?')}%"
            )
    except Exception:
        pass

    # ── LLM synthesis track record ──
    try:
        from .llm_accountability import get_summary as llm_summary
        llm = llm_summary()
        if llm.get("hit_rate") is not None:
            ctx["llm_track_record"] = (
                f"hit_rate={llm['hit_rate']}% "
                f"(verified={llm['verified']}, pending={llm['pending']})"
            )
    except Exception:
        pass

    # ── Cross-market data (oil, dollar, corn) for Fund Manager ──
    try:
        mc_path = os.path.join(_DATA, "multi_commodity.json")
        if os.path.exists(mc_path):
            with open(mc_path, encoding="utf-8") as f:
                ctx["multi_commodity"] = json.load(f)
    except Exception:
        pass

    # ── Narrative forecast (event + ranges + decision classifier) ──
    try:
        nf_path = os.path.join(_ARTIFACTS, "narrative_forecast", "latest.json")
        if os.path.exists(nf_path):
            with open(nf_path, encoding="utf-8") as f:
                ctx["narrative_forecast"] = json.load(f)
    except Exception:
        pass

    # ── Signal breakdown (composite score + factors) ──
    try:
        sb_path = os.path.join(_DATA, "signal_breakdown.json")
        if os.path.exists(sb_path):
            with open(sb_path, encoding="utf-8") as f:
                ctx["signal_breakdown"] = json.load(f)
    except Exception:
        pass

    # ── Drift monitor ──
    try:
        dm_path = os.path.join(_ARTIFACTS, "drift_monitor.json")
        if os.path.exists(dm_path):
            with open(dm_path, encoding="utf-8") as f:
                ctx["drift_monitor"] = json.load(f)
    except Exception:
        pass

    return ctx


def _load_recent_articles() -> list[dict]:
    articles = []

    news_memory_path = os.path.join(_ROOT, "news_memory.json")
    try:
        if os.path.exists(news_memory_path):
            with open(news_memory_path) as f:
                data = json.load(f)
            if isinstance(data, list):
                for art in data:
                    articles.append({
                        "title": art.get("title", ""),
                        "summary": art.get("summary", art.get("text", "")),
                        "source": art.get("source", ""),
                        "topic": art.get("topic", ""),
                    })
    except Exception:
        pass

    try:
        from ..news_engine import load_cached_news
        cached = load_cached_news()
        if cached:
            for art in cached.get("articles", [])[:15]:
                if not any(a["title"] == art.get("title") for a in articles):
                    articles.append({
                        "title": art.get("title", ""),
                        "summary": art.get("summary", ""),
                        "source": art.get("source", ""),
                    })
    except Exception:
        pass

    return articles[:30]


def _build_rag_query(market_ctx: dict, classified_news: list[dict]) -> str:
    parts = []

    shock = market_ctx.get("active_shock")
    if shock:
        parts.append(f"active shock: {shock}")

    escalated = [n for n in classified_news if n.get("escalate")]
    for n in escalated[:3]:
        topics = n.get("high_impact_topics", [])
        if topics:
            parts.append(" ".join(topics))
        parts.append(n.get("headline", "")[:80])

    regime = market_ctx.get("regime", "")
    if regime:
        parts.append(f"regime {regime}")

    if not parts:
        parts.append("soybean market current conditions price outlook")

    return " ".join(parts)


def run_intelligence_engine(force: bool = False) -> dict:
    # ── Cost gate: 1x/day during market hours ──
    if not force:
        try:
            from .market_hours import can_run_llm
            if not can_run_llm("intelligence_engine"):
                # Return cached verdict
                if os.path.exists(_CACHE_FILE):
                    with open(_CACHE_FILE, encoding="utf-8") as f:
                        cached = json.load(f)
                    cached["from_cache"] = True
                    cached["gate_reason"] = "already_ran_today_or_market_closed"
                    return cached
        except ImportError:
            pass

    if not force and os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, encoding="utf-8") as f:
                cached = json.load(f)
            ts = cached.get("timestamp", "")
            if ts:
                cached_time = datetime.fromisoformat(ts)
                age = (datetime.now() - cached_time).total_seconds()
                if age < _CACHE_TTL:
                    cached["from_cache"] = True
                    cached["cache_age_seconds"] = int(age)
                    return cached
        except Exception:
            pass

    t0 = time.time()

    print("[IE] Loading market context...")
    market_ctx = _load_market_context()
    current_price = market_ctx.get("current_price", 0)

    print("[IE] Loading and classifying articles (Tier-1)...")
    articles = _load_recent_articles()
    classified = classify_batch(articles) if articles else []

    escalated_count = sum(1 for c in classified if c.get("escalate"))
    print(f"[IE] Classified {len(classified)} articles, {escalated_count} escalated to Tier-2")

    print("[IE] Searching Knowledge Base (RAG)...")
    kb = get_knowledge_base()
    rag_query = _build_rag_query(market_ctx, classified)
    kb_results = kb.search(rag_query, top_k=8)
    print(f"[IE] Retrieved {len(kb_results)} relevant documents from KB")

    # ── Load feedback from previous verdicts ──
    print("[IE] Loading verdict history feedback...")
    try:
        evaluate_verdicts()  # verify any matured verdicts first
        feedback_text = get_feedback_for_debate()
    except Exception as e:
        print(f"[IE] Feedback load failed (non-blocking): {e}")
        feedback_text = ""

    print("[IE] Running Multi-Agent Debate...")
    debate_result = run_debate(
        market_data=market_ctx,
        news_classified=classified,
        kb_results=kb_results,
        current_price=current_price,
        feedback_text=feedback_text,
    )

    elapsed = time.time() - t0
    debate_result["execution_time_seconds"] = round(elapsed, 1)
    debate_result["from_cache"] = False
    debate_result["pipeline"] = {
        "tier1_method": classified[0].get("method", "unknown") if classified else "none",
        "tier1_articles": len(classified),
        "tier1_escalated": escalated_count,
        "rag_documents": len(kb_results),
        "rag_query": rag_query[:200],
        "kb_stats": kb.get_stats(),
    }

    try:
        os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(debate_result, f, indent=2, ensure_ascii=False, default=str)
        print(f"[IE] Verdict cached to {_CACHE_FILE}")
    except Exception as e:
        print(f"[IE] Cache write failed: {e}")

    try:
        saved = save_debate(debate_result)
        print(f"[IE] Debate saved to repository: verdict={saved.get('verdict')} ts={saved.get('timestamp')}")
    except Exception as e:
        print(f"[IE] Debate repository save failed: {e}")

    # ── Save verdict to history for accountability ──
    try:
        snap = save_verdict_snapshot(debate_result)
        if snap:
            print(f"[IE] Verdict snapshot saved to history: {snap['verdict']}")
    except Exception as e:
        print(f"[IE] Verdict snapshot failed (non-blocking): {e}")

    # ── Mark as ran for cost gate ──
    try:
        from .market_hours import mark_llm_ran
        mark_llm_ran("intelligence_engine")
    except Exception:
        pass

    gc.collect()
    print(f"[IE] Intelligence Engine complete in {elapsed:.1f}s")
    return debate_result


if __name__ == "__main__":
    result = run_intelligence_engine(force=True)
    verdict = result.get("verdict", {})
    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict.get('verdict', '?')}")
    print(f"Confidence: {verdict.get('confidence', '?')}")
    print(f"Reasoning: {verdict.get('reasoning', '?')}")
    print(f"7d range: {verdict.get('price_range_7d', '?')}")
    print(f"30d range: {verdict.get('price_range_30d', '?')}")
    print(f"Action - Producers: {verdict.get('recommended_action', {}).get('producers', '?')}")
    print(f"Action - Traders: {verdict.get('recommended_action', {}).get('traders', '?')}")
    print(f"Position sizing: {verdict.get('position_sizing', '?')}")
    print(f"Execution time: {result.get('execution_time_seconds', '?')}s")
