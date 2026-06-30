"""
Analisis Opcion C para BTCUSD:
  Regla actual:  SHORT + CAPITULATION -> veto si conf != HIGH  (bloquea LOW y MEDIUM)
  Opcion C:      SHORT + CAPITULATION -> veto solo si conf == LOW  (permite MEDIUM y HIGH)

Ejecutar:
  py -3 templates_nuevos_mercados/_backtest_btc_option_c.py
"""

import sys, io, json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_MVP_ROOT = Path(__file__).resolve().parent.parent
_LOG_PATH = _MVP_ROOT / "artifacts" / "btcusd" / "live_log.jsonl"


def load_log(path):
    if not path.exists():
        print(f"Log no encontrado: {path}")
        return [], []
    entries = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except Exception:
            pass
    seen, deduped = set(), []
    for e in entries:
        key = (e.get("timestamp", "")[:16], e.get("type"), e.get("price"), e.get("signal"))
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    signals = [e for e in deduped if e.get("type") == "signal" and e.get("signal") not in ("FLAT", None)]
    cycles  = sorted([e for e in deduped if e.get("type") == "cycle"], key=lambda x: x.get("timestamp", ""))
    return signals, cycles


def simulate_trade(sig, cycles_after):
    direction = sig.get("signal")
    entry     = float(sig.get("entry") or 0)
    sl        = float(sig.get("sl")    or 0)
    tp        = float(sig.get("tp")    or 0)
    risk_usd  = float(sig.get("risk_usd") or 0)
    rr_raw    = str(sig.get("rr") or "0").split(":")[0].split("/")[0].strip()
    rr        = float(rr_raw) if rr_raw else 0

    if not entry or not sl or not tp or not cycles_after:
        return {"outcome": "no_data", "pnl_usd": 0}

    for i, cyc in enumerate(cycles_after):
        price = float(cyc.get("price") or 0)
        if not price:
            continue
        if direction == "LONG":
            if price <= sl: return {"outcome": "SL", "pnl_usd": -risk_usd}
            if price >= tp: return {"outcome": "TP", "pnl_usd": risk_usd * rr}
        else:
            if price >= sl: return {"outcome": "SL", "pnl_usd": -risk_usd}
            if price <= tp: return {"outcome": "TP", "pnl_usd": risk_usd * rr}

    last_price = float(cycles_after[-1].get("price", entry)) if cycles_after else entry
    move = last_price - entry if direction == "LONG" else entry - last_price
    sl_dist = abs(entry - sl)
    pnl_usd = (move / sl_dist * risk_usd) if sl_dist > 0 else 0
    return {"outcome": "OPEN", "pnl_usd": round(pnl_usd, 2)}


def main():
    signals, cycles = load_log(_LOG_PATH)

    if not signals:
        print("Sin senales BTCUSD para analizar.")
        return

    print(f"\n{'='*90}")
    print(f"  BTCUSD - Analisis Opcion C: SHORT+CAPITULATION solo vetado si conf=LOW")
    print(f"{'='*90}")
    print(f"  {'Fecha':16} | {'Sig':5} | {'Regime':16} | {'Conf':6} | {'RR':>4} | {'Outcome':6} | {'PnL':>8} | {'Baseline':9} | {'Opc-C':8} | Nota")
    print(f"  {'-'*85}")

    pnl_base  = 0.0
    pnl_orig  = 0.0  # regla actual: bloquea LOW+MEDIUM
    pnl_optc  = 0.0  # Opcion C:     bloquea solo LOW

    n_base = n_orig_vetoed = n_optc_vetoed = 0
    wins_base = wins_orig = wins_optc = 0

    rows = []
    for sig in signals:
        ts     = sig.get("timestamp", "")
        signal = sig.get("signal")
        conf   = sig.get("confidence", "LOW")
        regime = sig.get("btc_regime", "")
        rr_raw = str(sig.get("rr") or "0").split(":")[0].split("/")[0].strip()
        rr     = float(rr_raw) if rr_raw else 0

        cycles_after = [c for c in cycles if c.get("timestamp", "") > ts][:200]
        sim = simulate_trade(sig, cycles_after)
        outcome = sim["outcome"]
        pnl     = sim["pnl_usd"]

        # Es SHORT + CAPITULATION?
        is_target = (signal == "SHORT" and regime == "CAPITULATION")

        # Regla ORIGINAL: veto si conf != HIGH
        orig_vetoed = is_target and conf != "HIGH"
        # Opcion C: veto solo si conf == LOW
        optc_vetoed = is_target and conf == "LOW"

        # P&L acumulado
        pnl_base += pnl
        pnl_orig += 0.0 if orig_vetoed else pnl
        pnl_optc += 0.0 if optc_vetoed else pnl

        n_base += 1
        if outcome == "TP":
            wins_base += 1
            if not orig_vetoed: wins_orig += 1
            if not optc_vetoed: wins_optc += 1

        if orig_vetoed: n_orig_vetoed += 1
        if optc_vetoed: n_optc_vetoed += 1

        nota = ""
        if is_target:
            if orig_vetoed and not optc_vetoed:
                nota = "<-- DIFF: Orig veta, OpcC permite"
            elif orig_vetoed and optc_vetoed:
                nota = "<-- Ambas vetan (LOW)"
            elif not orig_vetoed:
                nota = "(HIGH, ninguna veta)"

        rows.append((ts[:16], signal, regime, conf, rr, outcome, pnl, orig_vetoed, optc_vetoed, nota))

    for ts, sig, reg, conf, rr, out, pnl, ov, cv, nota in rows:
        b_tag = "[VETO]" if ov else "[PASS]"
        c_tag = "[VETO]" if cv else "[PASS]"
        print(f"  {ts:16} | {sig:5} | {str(reg):16} | {conf:6} | {rr:>4.1f} | {out:6} | ${pnl:>+7.2f} | {b_tag:9} | {c_tag:8} | {nota}")

    n_orig_pass = n_base - n_orig_vetoed
    n_optc_pass = n_base - n_optc_vetoed

    wr_base = wins_base / n_base * 100 if n_base else 0
    wr_orig = wins_orig / n_orig_pass * 100 if n_orig_pass else 0
    wr_optc = wins_optc / n_optc_pass * 100 if n_optc_pass else 0

    print(f"\n{'='*90}")
    print(f"  RESUMEN COMPARATIVO BTCUSD")
    print(f"{'='*90}")
    print(f"  {'Escenario':30} | {'Trades':7} | {'Vetados':8} | {'WR%':6} | {'P&L Total':10}")
    print(f"  {'-'*65}")
    print(f"  {'Sin filtro (baseline)':30} | {n_base:7} | {'--':8} | {wr_base:5.1f}% | ${pnl_base:>+9.2f}")
    print(f"  {'Regla actual (veta LOW+MED)':30} | {n_orig_pass:7} | {n_orig_vetoed:8} | {wr_orig:5.1f}% | ${pnl_orig:>+9.2f}  delta ${pnl_orig-pnl_base:+.2f}")
    print(f"  {'Opcion C (veta solo LOW)':30} | {n_optc_pass:7} | {n_optc_vetoed:8} | {wr_optc:5.1f}% | ${pnl_optc:>+9.2f}  delta ${pnl_optc-pnl_base:+.2f}")

    print(f"\n  INTERPRETACION:")
    delta_orig = pnl_orig - pnl_base
    delta_optc = pnl_optc - pnl_base
    if delta_optc > delta_orig:
        print(f"  Opcion C es MEJOR que la regla actual (${delta_optc:+.2f} vs ${delta_orig:+.2f} vs baseline)")
        print(f"  Recomienda: bloquear SHORT+CAPITULATION solo con conf=LOW")
    else:
        print(f"  La regla actual y Opcion C producen resultados similares")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
