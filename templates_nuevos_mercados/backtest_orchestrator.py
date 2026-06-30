"""
Portfolio Orchestrator — Impact Backtest
========================================
Question this answers: if we put a PORTFOLIO ORCHESTRATOR on top of the 9
independent robots (per memory/portfolio_orchestrator_research.md), does it
actually improve the book's risk-adjusted return — and by how much?

The research doc is explicit that the orchestrator's value is NOT better
individual signals (the robots already have their LLM), but managing JOINT
exposure: correlation, concentration, global drawdown and regime. So the only
honest way to backtest it is at the PORTFOLIO level:

  1. Generate every robot's trades once (reusing backtest_portfolio, optimized
     configs). Each trade's entry/exit/outcome is FIXED — the orchestrator can
     only DROP a trade, REDUCE its size, or PAUSE entries during a window. It
     can never turn a loss into a win. (Same discipline as backtest_cluster_policy.)
  2. Replay the whole portfolio chronologically under competing policies and
     compare Sharpe / MaxDD / Calmar / P&L.

Policies compared:
  • NO-CAP        every signal at full size (ceiling, no portfolio mgmt)
  • BASELINE      hard cluster_cap only  ← what the robots do live today
  • MECH-ORCH     mechanical orchestrator: cluster_cap + global daily-loss
                  circuit breaker + correlation/concentration cap + regime gate
  • LLM-ORCH      real claude-sonnet-4-5 orchestrator: at each decision point
                  with joint exposure, the model returns APPROVE / VETO /
                  REDUCE_50. Gated (only called when there is exposure to manage)
                  and disk-cached for determinism + cost control.

Overfitting discipline: any tunable threshold (daily-loss limit, correlation K)
is reported with an OUT-OF-SAMPLE split — pick/learn on the first --train-frac
of the timeline, measure impact on the rest. This is the same honest test that
killed the day-of-week risk-weighting idea.

Read-only on the live system. Writes only its own cache/report in this folder.
Touches nothing in the soja codebase.

Usage:
  # robot Python (has MT5 + anthropic):
  PY="C:\\Users\\Lenovo\\AppData\\Local\\Programs\\Python\\Python312\\python.exe"
  $PY backtest_orchestrator.py --selftest          # validate engine, no MT5/LLM
  $PY backtest_orchestrator.py --no-llm            # mechanical layers only
  $PY backtest_orchestrator.py                     # full, incl. LLM orchestrator
  $PY backtest_orchestrator.py --bars 6000 --daily-loss 30 --corr-k 2
"""

import os, sys, json, hashlib, argparse, warnings
from datetime import datetime
from collections import defaultdict
import heapq

warnings.filterwarnings("ignore")

# NOTE: stdout UTF-8 wrapping is handled by backtest_portfolio on import.
# Do NOT wrap here too — double-wrapping the same buffer closes it prematurely.

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(_HERE)
for p in [_HERE, _MVP_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Load ANTHROPIC_API_KEY from .env (same locations the robots use) — otherwise
# anthropic.Anthropic() has no key and every orchestrator call throws.
try:
    from dotenv import load_dotenv
    for _ep in [os.path.join(_MVP_ROOT, ".env"),
                os.path.join(_MVP_ROOT, "MVP lectura de noticias", ".env")]:
        if os.path.exists(_ep):
            load_dotenv(_ep, override=True)
            break
except ImportError:
    pass

import numpy as np
import pandas as pd

# ── instrument role classification (from the research doc) ───────────────────
# Risk-on bloc: crypto + US indices move together, +0.90 in risk-off.
RISK_ON     = {"BTCUSD", "ETHUSD", "US500", "USTEC", "US30"}
SAFE_HAVEN  = {"XAUUSD"}
ENERGY      = {"WTI_N6", "BRENT_N6"}
# UK100 is treated as isolated (low corr with US) — neither risk-on bloc nor safe haven.

_CACHE_PATH = os.path.join(_HERE, "orchestrator_llm_cache.json")


# ── trade-intent generation (reuses optimized robot configs) ─────────────────

def _gen_intents(bars_override=None):
    """One trade list per instrument, tagged with cluster + role. Each dict is a
    trade 'intent' the orchestrator may accept/scale/drop."""
    import backtest_portfolio as bp
    from portfolio_guard import _get_cluster_name

    intents = []
    per_sym = {}
    for inst in bp.INSTRUMENTS:
        sym = inst["sym"]
        cfg = bp.OPTIMIZED_CONFIGS.get(sym, {})
        _, trades = bp.backtest_instrument(
            inst, n_bars=bars_override or 5000, verbose=False,
            tf_override=cfg.get("tf"), ema_fast_ov=cfg.get("ef"), ema_slow_ov=cfg.get("es"),
            sl_mult_ov=cfg.get("sl"), tp_mult_ov=cfg.get("tp"), cooldown_ov=cfg.get("cd"),
            rsi_lo_long=cfg.get("rll", 38), rsi_hi_long=cfg.get("rhl", 65),
            rsi_lo_short=cfg.get("rls", 35), rsi_hi_short=cfg.get("rhs", 62),
        )
        per_sym[sym] = len(trades)
        cl = _get_cluster_name(sym)
        for t in trades:
            intents.append(dict(
                sym=sym, cluster=cl,
                risk_on=(sym in RISK_ON), safe=(sym in SAFE_HAVEN),
                entry_time=pd.Timestamp(t["entry_time"]),
                exit_time=pd.Timestamp(t["exit_time"]),
                direction=t["direction"], rr=t["rr"],
                risk_usd=t["risk_usd"], pnl_usd=t["pnl_usd"], outcome=t["outcome"],
            ))
    return intents, per_sym


# ── regime series (macro risk barometer from US500, backward-looking only) ───

def _build_regime_series(n_bars=4000, tf="1h"):
    """Classify each timestamp as BULL / BEAR / CRISIS using US500 only.
    Everything is strictly backward-looking (EMA + trailing vol z-score) so
    there is no lookahead. Weekends (no US500 bars) carry forward Friday."""
    import backtest_portfolio as bp
    bars = bp._mt5_bars("US500", tf, n_bars)
    if bars is None or len(bars) < 250:
        return None
    c = bars["close"]
    ret = c.pct_change()
    ema = c.ewm(span=200, adjust=False).mean()
    trend_up = c > ema
    vol = ret.rolling(48).std()
    vmean = vol.rolling(200, min_periods=50).mean()
    vstd = vol.rolling(200, min_periods=50).std()
    crisis = vol > (vmean + 1.5 * vstd)
    regime = pd.Series("BULL", index=bars.index)
    regime[~trend_up] = "BEAR"
    regime[crisis.fillna(False)] = "CRISIS"
    return regime


def _regime_at(regime, t):
    if regime is None:
        return "BULL"
    idx = regime.index
    pos = idx.searchsorted(t, side="right") - 1
    if pos < 0:
        return "BULL"
    return str(regime.iloc[pos])


# ── event-driven portfolio replay engine ─────────────────────────────────────

def replay(intents, decide, regime=None):
    """Process candidate entries in time order. Before each, flush exits of
    already-selected trades to update realized P&L / peak / daily P&L. `decide`
    returns (take: bool, scale: float, reason: str). Selected trades carry a
    scaled P&L. Returns (selected_trades, debug_counts)."""
    entries = sorted(intents, key=lambda x: x["entry_time"])
    open_heap = []           # (exit_time, seq, trade_dict)  — currently open selected
    seq = 0
    selected = []
    realized = 0.0
    peak = 0.0
    daily = defaultdict(float)   # UTC date -> realized P&L closed that day
    counts = defaultdict(int)

    for cand in entries:
        t = cand["entry_time"]
        # flush exits that have closed by time t
        while open_heap and open_heap[0][0] <= t:
            xt, _, tr = heapq.heappop(open_heap)
            realized += tr["pnl_scaled"]
            peak = max(peak, realized)
            daily[xt.normalize()] += tr["pnl_scaled"]

        open_now = [tr for (_, _, tr) in open_heap]
        ctx = dict(
            t=t, realized=realized, peak=peak,
            drawdown=realized - peak,                  # <= 0
            daily_pnl=daily[t.normalize()],            # realized P&L so far today
            open_now=open_now,
            regime=_regime_at(regime, t),
        )
        take, scale, reason = decide(cand, ctx)
        counts[reason] += 1
        if not take or scale <= 0:
            continue
        tr = {**cand, "scale": scale,
              "risk_scaled": cand["risk_usd"] * scale,
              "pnl_scaled":  cand["pnl_usd"] * scale}
        selected.append(tr)
        heapq.heappush(open_heap, (cand["exit_time"], seq, tr)); seq += 1

    return selected, dict(counts)


# ── metrics on a selected (scaled) trade list ─────────────────────────────────

def metrics(selected):
    if not selected:
        return dict(n=0, wr=0.0, pnl=0.0, dd=0.0, sharpe=0.0, calmar=0.0, exposure=0.0)
    merged = sorted(selected, key=lambda t: t["exit_time"])
    pnls = [t["pnl_scaled"] for t in merged]
    wins = [p for p in pnls if p > 0]
    cum = np.cumsum(pnls); peak = np.maximum.accumulate(cum)
    dd = float((cum - peak).min())
    sharpe = (np.mean(pnls) / np.std(pnls) * np.sqrt(252)) if len(pnls) > 1 and np.std(pnls) > 0 else 0.0
    calmar = (sum(pnls) / abs(dd)) if dd != 0 else 0.0
    exposure = sum(t["scale"] for t in merged) / len(merged)   # avg fraction of full size deployed
    return dict(n=len(merged), wr=round(len(wins)/len(merged)*100, 1),
                pnl=round(float(sum(pnls)), 2), dd=round(dd, 2),
                sharpe=round(float(sharpe), 2), calmar=round(float(calmar), 2),
                exposure=round(exposure, 2))


# ── decision policies ─────────────────────────────────────────────────────────

def _cluster_busy(cand, ctx):
    cl = cand["cluster"]
    if cl is None:
        return False
    return any(o["cluster"] == cl and o["sym"] != cand["sym"] for o in ctx["open_now"])


def decide_no_cap(cand, ctx):
    return True, 1.0, "take"


def decide_baseline(cand, ctx):
    """Hard cluster_cap — current live behaviour."""
    if _cluster_busy(cand, ctx):
        return False, 0.0, "cluster_cap_drop"
    return True, 1.0, "take"


def make_decide_mech(daily_loss, corr_k):
    """Mechanical orchestrator: cluster_cap + daily-loss circuit breaker +
    correlation/concentration cap + regime gate. Layers compose; scales multiply."""
    def decide(cand, ctx):
        # 1. cluster_cap (kept — proven good in backtest_cluster_policy)
        if _cluster_busy(cand, ctx):
            return False, 0.0, "cluster_cap_drop"
        # 2. global circuit breaker: if today's realized loss breached, halt new entries
        if ctx["daily_pnl"] <= -abs(daily_loss):
            return False, 0.0, "circuit_breaker_drop"
        scale = 1.0
        # 3. regime gate
        if cand["risk_on"]:
            if ctx["regime"] == "CRISIS":
                return False, 0.0, "regime_crisis_drop"
            if ctx["regime"] == "BEAR":
                scale *= 0.5
        # 4. correlation / concentration cap: too many risk-on bets same direction
        if cand["risk_on"]:
            same = sum(1 for o in ctx["open_now"]
                       if o["risk_on"] and o["direction"] == cand["direction"])
            if same >= corr_k:
                scale *= 0.5
        reason = "take" if scale == 1.0 else "reduced"
        return True, scale, reason
    return decide


# ── LLM orchestrator ──────────────────────────────────────────────────────────

_ORCH_SYSTEM = """You are a PORTFOLIO ORCHESTRATOR for a book of 9 independent intraday trading robots
(crypto: BTCUSD ETHUSD; US indices: US500 USTEC US30; energy: WTI_N6 BRENT_N6; gold: XAUUSD; FTSE: UK100).
Account ~ $1,013. Each robot already produced its own signal. Your ONLY job is to manage JOINT exposure:
correlation, concentration, global drawdown and market regime. You CANNOT improve an individual signal —
you can only APPROVE it, VETO it, or REDUCE its size to 50%.

Principles:
- Correlated risk-on names (BTC, ETH, US500, USTEC, US30) moving the SAME direction are ONE concentrated
  bet, not many independent ones. Trim duplicates.
- Energy (WTI, BRENT) are ~0.95 correlated to each other.
- XAUUSD is a safe haven (diversifier) — favour keeping it.
- In a BEAR/CRISIS regime, cut risk-on exposure. In BULL, allow it.
- If the book is already in meaningful daily drawdown, be defensive.
- Do not over-veto: vetoing everything earns nothing. Veto/reduce only when joint exposure is genuinely concentrated or the regime/drawdown warrants caution.

Output ONLY valid JSON: {"action": "APPROVE" | "VETO" | "REDUCE_50", "reason": "<short>"}"""


def _llm_context_key(cand, ctx):
    """Stable, bucketed key so the cache is deterministic and reused across runs."""
    open_summ = sorted(f"{o['sym']}:{o['direction']}" for o in ctx["open_now"])
    dd_bucket = int(ctx["drawdown"] // 20) * 20          # $20 buckets
    day_bucket = int(ctx["daily_pnl"] // 20) * 20
    payload = dict(sym=cand["sym"], dir=cand["direction"], cl=cand["cluster"],
                   risk_on=cand["risk_on"], rr=round(cand["rr"], 1),
                   regime=ctx["regime"], open=open_summ,
                   dd=dd_bucket, day=day_bucket)
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest(), payload


def make_decide_llm(cache, stats):
    """Real-LLM orchestrator. Gated: only call the model when there is exposure
    to manage (open positions, drawdown, or non-BULL regime); otherwise APPROVE
    at full size for free. Results cached on a bucketed context key."""
    import backtest_portfolio  # ensures stdout wrap + sys.path
    client = None

    def _get_client():
        nonlocal client
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        return client

    def decide(cand, ctx):
        # cluster_cap stays as a hard pre-filter? No — let the LLM own concentration.
        gate = (len(ctx["open_now"]) > 0) or (ctx["drawdown"] <= -20) or (ctx["regime"] != "BULL")
        if not gate:
            return True, 1.0, "auto_approve_no_exposure"

        key, payload = _llm_context_key(cand, ctx)
        if key in cache:
            res = cache[key]
            stats["cache_hits"] += 1
        else:
            user = ("New robot signal awaiting your portfolio decision:\n"
                    + json.dumps(payload, indent=2)
                    + "\n\nCurrently OPEN positions: " + (json.dumps(payload["open"]) or "[]")
                    + f"\nMarket regime: {payload['regime']}"
                    + f"\nBook drawdown bucket: ${payload['dd']}  | today's realized P&L bucket: ${payload['day']}"
                    + "\n\nReturn APPROVE / VETO / REDUCE_50 as JSON.")
            try:
                cl = _get_client()
                msg = cl.messages.create(model="claude-sonnet-4-5", max_tokens=150,
                                         system=_ORCH_SYSTEM,
                                         messages=[{"role": "user", "content": user}])
                raw = msg.content[0].text.strip()
                stats["calls"] += 1
                if hasattr(msg, "usage"):
                    stats["in_tok"] += getattr(msg.usage, "input_tokens", 0) or 0
                    stats["out_tok"] += getattr(msg.usage, "output_tokens", 0) or 0
                import re
                raw = re.sub(r"^```(?:json)?\s*", "", raw); raw = re.sub(r"\s*```$", "", raw)
                res = json.loads(raw)
                cache[key] = res                      # cache only successful calls
            except Exception as e:
                res = {"action": "APPROVE", "reason": f"llm_error:{type(e).__name__}"}
                stats["errors"] += 1
                stats.setdefault("last_error", str(e)[:200])

        action = (res.get("action") or "APPROVE").upper()
        if action == "VETO":
            return False, 0.0, "llm_veto"
        if action == "REDUCE_50":
            return True, 0.5, "llm_reduce"
        return True, 1.0, "llm_approve"
    return decide


def _load_cache():
    if os.path.exists(_CACHE_PATH):
        try:
            return json.load(open(_CACHE_PATH, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache):
    try:
        json.dump(cache, open(_CACHE_PATH, "w", encoding="utf-8"), indent=0)
    except Exception:
        pass


# ── reporting helpers ─────────────────────────────────────────────────────────

def _print_table(title, rows):
    print("\n" + "-" * 78)
    print(f"  {title}")
    print("-" * 78)
    print(f"  {'Policy':<22} {'Ops':>5} {'WR%':>6} {'NetP&L':>9} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'Expo':>5}")
    for name, m in rows:
        print(f"  {name:<22} {m['n']:>5} {m['wr']:>5.1f}% ${m['pnl']:>7,.0f} "
              f"${m['dd']:>6,.0f} {m['sharpe']:>7} {m['calmar']:>7} {m['exposure']:>5}")


def _split(intents, frac):
    seq = sorted(intents, key=lambda x: x["entry_time"])
    cut = int(len(seq) * frac)
    return seq[:cut], seq[cut:]


# ── self-test (no MT5, no LLM) ────────────────────────────────────────────────

def selftest():
    print("\n=== SELF-TEST: replay engine ===")
    T = pd.Timestamp("2026-01-05 10:00", tz="UTC")
    H = pd.Timedelta(hours=1)
    def mk(sym, cl, ro, d, ent, dur, pnl, rr=2.0):
        return dict(sym=sym, cluster=cl, risk_on=ro, safe=False,
                    entry_time=T+ent*H, exit_time=T+(ent+dur)*H,
                    direction=d, rr=rr, risk_usd=20.0, pnl_usd=pnl, outcome="X")
    ok = True

    # 1. cluster_cap drops a same-cluster overlap, keeps the non-overlap.
    ints = [mk("US500","equity",True,"LONG",0,5,10),
            mk("USTEC","equity",True,"LONG",1,5,10),   # overlaps US500 → drop
            mk("US30","equity",True,"LONG",6,2,10)]     # after US500 closed → keep
    sel,_ = replay(ints, decide_baseline)
    got = sorted(s["sym"] for s in sel)
    exp = ["US30","US500"]
    print(f"  cluster_cap: kept {got}  (expect {exp})  ->", "OK" if got==exp else "FAIL"); ok &= got==exp

    # 2. circuit breaker halts entries the rest of the day after the daily loss limit.
    #    Two losers close early, then a would-be entry same day must be dropped.
    ints = [mk("BTCUSD","crypto",True,"LONG",0,1,-20),
            mk("WTI_N6","energy",False,"LONG",0,1,-20),   # by t=2h, daily realized = -40
            mk("XAUUSD",None,False,"LONG",3,1,30)]          # same day → halted
    dec = make_decide_mech(daily_loss=30, corr_k=99)
    sel,cnt = replay(ints, dec)
    halted = "circuit_breaker_drop" in cnt
    print(f"  circuit_breaker: daily -40 vs limit -30 -> halt fired:", "OK" if halted else "FAIL"); ok &= halted

    # 3. correlation cap reduces size of the (k+1)-th concurrent risk-on same-dir bet.
    ints = [mk("BTCUSD","crypto",True,"LONG",0,10,10),
            mk("US500","equity",True,"LONG",1,10,10),
            mk("USTEC","equity2",True,"LONG",2,10,10)]   # 3rd concurrent risk-on LONG
    dec = make_decide_mech(daily_loss=9999, corr_k=2)
    sel,cnt = replay(ints, dec)
    reduced = any(s["scale"]==0.5 for s in sel)
    print(f"  correlation_cap (k=2): a concurrent risk-on LONG reduced to 0.5 ->",
          "OK" if reduced else "FAIL"); ok &= reduced

    # 4. no-cap takes everything.
    sel,_ = replay(ints, decide_no_cap)
    print(f"  no_cap: took all {len(sel)}/3 ->", "OK" if len(sel)==3 else "FAIL"); ok &= len(sel)==3

    print("=== SELF-TEST:", "ALL PASS ===" if ok else "FAILURES ABOVE ===")
    return ok


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=5000)
    ap.add_argument("--daily-loss", type=float, default=30.0, help="daily realized-loss circuit-breaker limit ($)")
    ap.add_argument("--corr-k", type=int, default=2, help="max concurrent risk-on same-dir before halving")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM orchestrator (mechanical only)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    print("\n" + "=" * 78)
    print("  PORTFOLIO ORCHESTRATOR — IMPACT BACKTEST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  {args.bars:,} bars/sym")
    print(f"  Circuit breaker: daily loss ${args.daily_loss:.0f}  |  corr-K {args.corr_k}  |  train {args.train_frac:.0%}")
    print("=" * 78)

    print("\n  Generating per-robot trades (optimized configs)...")
    intents, per_sym = _gen_intents(args.bars)
    if not intents:
        print("  No trades — MT5 connected?"); return
    print("  " + "  ".join(f"{k}:{v}" for k, v in per_sym.items()))
    span = (min(i["entry_time"] for i in intents), max(i["exit_time"] for i in intents))
    print(f"  Total intents: {len(intents)}  |  span {span[0].date()} -> {span[1].date()}")

    print("\n  Building regime series from US500 (backward-looking)...")
    regime = _build_regime_series()
    if regime is not None:
        from collections import Counter
        rc = Counter(_regime_at(regime, i["entry_time"]) for i in intents)
        print(f"  Regime distribution over entries: {dict(rc)}")
    else:
        print("  (regime unavailable — gate disabled, treated as BULL)")

    # policies
    mech = make_decide_mech(args.daily_loss, args.corr_k)
    policies = [
        ("NO-CAP (ceiling)",   decide_no_cap),
        ("BASELINE cluster_cap", decide_baseline),
        ("MECH-ORCH",          mech),
    ]

    llm_stats = dict(calls=0, cache_hits=0, in_tok=0, out_tok=0, errors=0)
    cache = _load_cache()
    if not args.no_llm:
        policies.append(("LLM-ORCH", make_decide_llm(cache, llm_stats)))

    # ── full-sample (in-sample, directional) ──
    full_rows = []
    for name, dec in policies:
        sel, _ = replay(intents, dec, regime)
        full_rows.append((name, metrics(sel)))
    _print_table("FULL SAMPLE (in-sample — directional only)", full_rows)

    # ── OUT-OF-SAMPLE: thresholds are 'set' on train, impact measured on test ──
    train, test = _split(intents, args.train_frac)
    print(f"\n  OOS split — train {len(train)} ({train[0]['entry_time'].date()}→{train[-1]['entry_time'].date()})"
          f" | test {len(test)} ({test[0]['entry_time'].date()}→{test[-1]['entry_time'].date()})")
    test_rows = []
    for name, dec in policies:
        sel, _ = replay(test, dec, regime)
        test_rows.append((name, metrics(sel)))
    _print_table("TEST SET (out-of-sample — the honest comparison)", test_rows)

    if not args.no_llm:
        _save_cache(cache)
        cost = llm_stats["in_tok"]/1e6*3.0 + llm_stats["out_tok"]/1e6*15.0   # sonnet $3/$15 per Mtok
        print(f"\n  LLM: {llm_stats['calls']} live calls, {llm_stats['cache_hits']} cache hits, "
              f"{llm_stats['errors']} errors | ~{llm_stats['in_tok']+llm_stats['out_tok']:,} tok | est ${cost:.3f}")
        print(f"  Cache: {len(cache)} entries -> {_CACHE_PATH}")
        if llm_stats["errors"]:
            print(f"  !! {llm_stats['errors']} LLM errors — last: {llm_stats.get('last_error','?')}")

    # ── verdict (based on TEST set) ──
    base = dict(test_rows)["BASELINE cluster_cap"]
    print("\n" + "=" * 78)
    print("  VERDICT (out-of-sample, vs live BASELINE cluster_cap)")
    print("=" * 78)
    for name, m in test_rows:
        if name == "BASELINE cluster_cap":
            continue
        dsh = m["sharpe"] - base["sharpe"]
        ddd = m["dd"] - base["dd"]      # >0 means shallower (better) drawdown
        dpl = m["pnl"] - base["pnl"]
        tag = "better" if (dsh > 0 and ddd >= 0) else ("mixed" if dsh > 0 or ddd > 0 else "worse")
        print(f"  {name:<22} ΔSharpe {dsh:+.2f} | ΔMaxDD ${ddd:+,.0f} | ΔP&L ${dpl:+,.0f}  -> {tag}")
    print("  Reminder: orchestrator can only veto/scale; it cannot improve a single trade's outcome.")
    print("  CAVEAT: small sample; circuit breaker is realized-P&L based (no intrabar MTM);")
    print("  LLM decisions are cached/bucketed; regime proxy = US500 trend+vol only.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
