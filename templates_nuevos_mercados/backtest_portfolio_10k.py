"""
Portfolio backtest simulation on a USD $10,000 account (as of today).

Reuses the orchestrator backtest engine (real per-instrument trades from
backtest_portfolio + the live cluster_cap policy). The only change vs. the live
config is the per-trade risk: P&L scales LINEARLY with risk_usd, so we just set
MAX_RISK_USD before generating intents and everything downstream is consistent.

Two risk settings are reported:
  - 2.0%  -> $200/trade  (current live risk %, applied to a $10k account)
  - 0.3%  -> $30/trade   (prop-firm-compatible de-risked setting)

Run with the MT5-enabled interpreter:
  C:\\Users\\Lenovo\\AppData\\Local\\Programs\\Python\\Python312\\python.exe backtest_portfolio_10k.py
"""
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import backtest_portfolio as bp
import backtest_orchestrator as bo

ACCOUNT = 10_000.0


def run_at_risk(risk_pct, bars=5000):
    risk_usd = round(ACCOUNT * risk_pct / 100.0, 2)
    bp.MAX_RISK_USD = risk_usd            # everything downstream uses this
    intents, per_sym = bo._gen_intents(bars_override=bars)

    no_cap, _   = bo.replay(intents, bo.decide_no_cap)
    baseline, _ = bo.replay(intents, bo.decide_baseline)

    m_nc = bo.metrics(no_cap)
    m_bl = bo.metrics(baseline)

    span_lo = min(i["entry_time"] for i in intents)
    span_hi = max(i["exit_time"]  for i in intents)
    return dict(risk_pct=risk_pct, risk_usd=risk_usd, per_sym=per_sym,
                n_intents=len(intents), span=(span_lo, span_hi),
                no_cap=m_nc, baseline=m_bl)


def _fmt_row(label, m):
    ret_pct = m["pnl"] / ACCOUNT * 100
    dd_pct  = m["dd"]  / ACCOUNT * 100
    return (f"  {label:<22} {m['n']:>4}  {m['wr']:>5.1f}%  "
            f"${m['pnl']:>9,.0f}  {ret_pct:>+6.1f}%  "
            f"${m['dd']:>8,.0f}  {dd_pct:>6.1f}%  "
            f"{m['sharpe']:>6.2f}  {m['calmar']:>6.2f}")


def main():
    print("\n" + "=" * 86)
    print(f"  PORTFOLIO BACKTEST — USD ${ACCOUNT:,.0f} account  ({len(bp.INSTRUMENTS)} instruments)")
    print("=" * 86)

    for risk_pct in (2.0, 0.3):
        r = run_at_risk(risk_pct)
        lo, hi = r["span"]
        print(f"\n  RISK {risk_pct:.1f}%/trade  =  ${r['risk_usd']:,.2f}/trade")
        print(f"  {r['n_intents']} trade intents | {lo.date()} -> {hi.date()}")
        print(f"  per-instrument: " + "  ".join(f"{k}:{v}" for k, v in r["per_sym"].items()))
        print("  " + "-" * 84)
        print(f"  {'Policy':<22} {'Ops':>4}  {'WR':>6}  {'NetP&L':>10}  {'Ret%':>6}  "
              f"{'MaxDD$':>9}  {'DD%':>6}  {'Sharpe':>6}  {'Calmar':>6}")
        print("  " + "-" * 84)
        print(_fmt_row("NO-CAP (ceiling)", r["no_cap"]))
        print(_fmt_row("BASELINE clstr_cap", r["baseline"]))

    print("\n" + "=" * 86)
    print("  NOTES")
    print("  - 5,000 bars/instrument; per-instrument OPTIMIZED_CONFIGS (same as live).")
    print("  - BASELINE cluster_cap = the policy actually running live (1 pos / correlated cluster).")
    print("  - In-sample on the recent window (optimistic): no slippage/commission/LLM cost modeled.")
    print("  - P&L and MaxDD scale linearly with risk; WR/Sharpe/Calmar are risk-invariant.")
    print("=" * 86 + "\n")


if __name__ == "__main__":
    main()
