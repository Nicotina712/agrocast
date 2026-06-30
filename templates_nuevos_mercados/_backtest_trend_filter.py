"""
Backtest del filtro contrarian sobre datos historicos de live_log.

Para cada senal no-FLAT en el log:
  1. Simula el outcome usando los precios de ciclo posteriores (SL/TP hit)
  2. Aplica la regla del filtro contrarian (version testing)
  3. Compara: baseline (sin filtro) vs filtered (con filtro)

Uso:
  py -3 templates_nuevos_mercados/_backtest_trend_filter.py
  py -3 templates_nuevos_mercados/_backtest_trend_filter.py --robot UK100
  py -3 templates_nuevos_mercados/_backtest_trend_filter.py --csv
"""

import sys, io, json, argparse
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_MVP_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# CONFIGURACION DE ROBOTS
# ---------------------------------------------------------------------------
ROBOTS = {
    "UK100":    {"log": "artifacts/uk100/live_log.jsonl",    "regime_field": "uk100_regime"},
    "BTCUSD":   {"log": "artifacts/btcusd/live_log.jsonl",   "regime_field": "btc_regime"},
    "ETHUSD":   {"log": "artifacts/ethusd/live_log.jsonl",   "regime_field": "eth_regime"},
    "XAUUSD":   {"log": "artifacts/xauusd/live_log.jsonl",   "regime_field": None},
    "US30":     {"log": "artifacts/us30/live_log.jsonl",     "regime_field": "us30_regime"},
    "US500":    {"log": "artifacts/us500/live_log.jsonl",    "regime_field": "sp500_regime"},
    "USTEC":    {"log": "artifacts/ustec/live_log.jsonl",    "regime_field": "ustec_regime"},
    "HK50":     {"log": "artifacts/hk50/live_log.jsonl",     "regime_field": "sp500_regime"},
    "Corn_N6":  {"log": "artifacts/corn_n6/live_log.jsonl",  "regime_field": "oil_regime"},
    "BRENT_N6": {"log": "artifacts/brent_n6/live_log.jsonl", "regime_field": "oil_regime"},
    "WTI_N6":   {"log": "artifacts/wti_n6/live_log.jsonl",   "regime_field": "oil_regime"},
}

# ---------------------------------------------------------------------------
# REGLAS DEL FILTRO CONTRARIAN (version testing — NO implementada en vivo)
# ---------------------------------------------------------------------------
# Estructura: (signal, regime) -> descripcion del conflicto
# Si la combo esta en este dict, es counter-trend.
# El filtro la veta si confidence != "HIGH".
# Ademas: DISTRIBUTION + LONG + MEDIUM + RR < 2.5 -> veto

COUNTER_TREND_RULES = {
    # Indices y materias primas con BEAR/BULL trend
    ("LONG",  "BEAR_TREND"):    "LONG en tendencia bajista",
    ("SHORT", "BULL_TREND"):    "SHORT en tendencia alcista",
    # Indices: comprar en distribucion (techo) es arriesgado
    ("LONG",  "DISTRIBUTION"):  "LONG en fase de distribucion (techo)",
    # Cripto: shortar capitulacion es muy peligroso (rebote violento)
    ("SHORT", "CAPITULATION"):  "SHORT en capitulacion",
    # ETH: comprar cuando BTC domina
    ("LONG",  "BTC_DOMINANCE"): "LONG ETH en regimen BTC_DOMINANCE",
    # USTEC: shortar burbuja requiere alta conviccion
    ("SHORT", "AI_BUBBLE"):     "SHORT en regimen AI_BUBBLE",
}

DISTRIBUTION_LONG_MIN_RR = 2.5   # para DISTRIBUTION+LONG, permitir si RR >= esto con MEDIUM


def is_counter_trend_vetoed(signal: str, regime: str, confidence: str, rr: float) -> tuple:
    """
    Retorna (vetado: bool, razon: str)
    """
    if not regime or not signal or signal == "FLAT":
        return False, ""

    key = (signal, regime)
    if key not in COUNTER_TREND_RULES:
        return False, ""

    desc = COUNTER_TREND_RULES[key]

    # Caso especial DISTRIBUTION+LONG: permitir si RR alto aunque sea MEDIUM
    if regime == "DISTRIBUTION" and signal == "LONG":
        if confidence == "HIGH":
            return False, ""
        if confidence == "MEDIUM" and rr and rr >= DISTRIBUTION_LONG_MIN_RR:
            return False, ""
        return True, f"{desc} con conf={confidence} RR={rr} (necesita HIGH o RR>={DISTRIBUTION_LONG_MIN_RR})"

    # Resto: solo HIGH pasa
    if confidence == "HIGH":
        return False, ""
    return True, f"{desc} con conf={confidence} (necesita HIGH)"


# ---------------------------------------------------------------------------
# SIMULACION DE OUTCOME
# ---------------------------------------------------------------------------

def simulate_trade(signal_entry: dict, cycles_after: list) -> dict:
    """
    Simula el resultado de un trade usando los precios de ciclo posteriores.
    Retorna: {outcome, exit_price, pnl_usd, bars_held, exit_reason}
    """
    direction = signal_entry.get("signal")
    entry     = float(signal_entry.get("entry") or 0)
    sl        = float(signal_entry.get("sl")    or 0)
    tp        = float(signal_entry.get("tp")    or 0)
    risk_usd  = float(signal_entry.get("risk_usd") or 0)
    rr_raw    = str(signal_entry.get("rr") or "0").split(":")[0].split("/")[0].strip()
    rr        = float(rr_raw) if rr_raw else 0

    if not entry or not sl or not tp or not cycles_after:
        return {"outcome": "no_data", "exit_price": entry, "pnl_usd": 0, "bars_held": 0, "exit_reason": "missing data"}

    for i, cyc in enumerate(cycles_after):
        price = float(cyc.get("price") or 0)
        if not price:
            continue

        if direction == "LONG":
            if price <= sl:
                return {"outcome": "SL", "exit_price": sl, "pnl_usd": -risk_usd, "bars_held": i+1, "exit_reason": f"SL hit @ {sl}"}
            if price >= tp:
                return {"outcome": "TP", "exit_price": tp, "pnl_usd": risk_usd * rr, "bars_held": i+1, "exit_reason": f"TP hit @ {tp}"}
        else:  # SHORT
            if price >= sl:
                return {"outcome": "SL", "exit_price": sl, "pnl_usd": -risk_usd, "bars_held": i+1, "exit_reason": f"SL hit @ {sl}"}
            if price <= tp:
                return {"outcome": "TP", "exit_price": tp, "pnl_usd": risk_usd * rr, "bars_held": i+1, "exit_reason": f"TP hit @ {tp}"}

    # Session ended without hitting SL or TP -> close at last price
    last_price = float(cycles_after[-1].get("price", entry)) if cycles_after else entry
    if direction == "LONG":
        move = last_price - entry
    else:
        move = entry - last_price
    sl_dist = abs(entry - sl)
    pnl_usd = (move / sl_dist * risk_usd) if sl_dist > 0 else 0
    return {"outcome": "OPEN", "exit_price": last_price, "pnl_usd": round(pnl_usd, 2), "bars_held": len(cycles_after), "exit_reason": "session ended"}


# ---------------------------------------------------------------------------
# CARGA Y PROCESAMIENTO DE LOGS
# ---------------------------------------------------------------------------

def load_log(path: Path) -> tuple[list, list]:
    """Carga el live_log y retorna (signals, cycles) ordenados por timestamp."""
    if not path.exists():
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

    # Dedup por timestamp+type (el robot a veces corre doble)
    seen, deduped = set(), []
    for e in entries:
        key = (e.get("timestamp", "")[:16], e.get("type"), e.get("price"), e.get("signal"))
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    signals = [e for e in deduped if e.get("type") == "signal" and e.get("signal") not in ("FLAT", None)]
    cycles  = sorted([e for e in deduped if e.get("type") == "cycle"], key=lambda x: x.get("timestamp",""))
    return signals, cycles


def backtest_robot(robot: str, cfg: dict, verbose: bool = True) -> list:
    """Backtestea un robot. Retorna lista de trades con resultado baseline y filtered."""
    log_path     = _MVP_ROOT / cfg["log"]
    regime_field = cfg["regime_field"]
    signals, cycles = load_log(log_path)

    if not signals:
        return []

    results = []
    for sig in signals:
        ts      = sig.get("timestamp", "")
        signal  = sig.get("signal")
        entry   = sig.get("entry")
        sl      = sig.get("sl")
        tp      = sig.get("tp")
        conf    = sig.get("confidence", "LOW")
        rr_raw  = str(sig.get("rr") or "0").split(":")[0].split("/")[0].strip()
        rr      = float(rr_raw) if rr_raw else 0
        risk    = float(sig.get("risk_usd") or 0)
        regime  = sig.get(regime_field) if regime_field else None

        # Ciclos posteriores al momento de la senal (misma sesion — max 200 barras)
        cycles_after = [c for c in cycles if c.get("timestamp","") > ts][:200]

        # Simular outcome
        sim = simulate_trade(sig, cycles_after)

        # Aplicar filtro
        vetoed, veto_reason = is_counter_trend_vetoed(signal, regime, conf, rr)

        result = {
            "robot":       robot,
            "timestamp":   ts[:16],
            "signal":      signal,
            "entry":       entry,
            "sl":          sl,
            "tp":          tp,
            "confidence":  conf,
            "rr":          rr,
            "risk_usd":    risk,
            "regime":      regime,
            "outcome":     sim["outcome"],
            "pnl_usd":     sim["pnl_usd"],
            "bars_held":   sim["bars_held"],
            "exit_reason": sim["exit_reason"],
            # Filtro
            "filter_vetoed":  vetoed,
            "veto_reason":    veto_reason,
            # P&L con filtro (si vetado -> 0, no se tomo el trade)
            "pnl_filtered":   0.0 if vetoed else sim["pnl_usd"],
        }
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# ESTADISTICAS
# ---------------------------------------------------------------------------

def stats(trades: list, label: str) -> dict:
    if not trades:
        return {"label": label, "n": 0}
    n     = len(trades)
    wins  = [t for t in trades if t["outcome"] == "TP"]
    loss  = [t for t in trades if t["outcome"] == "SL"]
    open_ = [t for t in trades if t["outcome"] == "OPEN"]
    pnl   = sum(t["pnl_usd"] for t in trades)
    wr    = len(wins) / n * 100 if n else 0
    avg_r = pnl / (sum(t["risk_usd"] for t in trades) or 1)
    return {
        "label":  label,
        "n":      n,
        "wins":   len(wins),
        "losses": len(loss),
        "open":   len(open_),
        "win_rate": round(wr, 1),
        "total_pnl": round(pnl, 2),
        "avg_r":   round(avg_r, 3),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot",   help="Solo un robot")
    parser.add_argument("--csv",     action="store_true", help="Output CSV")
    parser.add_argument("--verbose", action="store_true", help="Detalle por trade")
    args = parser.parse_args()

    robots_to_run = {args.robot: ROBOTS[args.robot]} if args.robot and args.robot in ROBOTS else ROBOTS

    all_trades = []

    for robot, cfg in robots_to_run.items():
        trades = backtest_robot(robot, cfg)
        if not trades:
            continue
        all_trades.extend(trades)

        if args.verbose or args.robot:
            print(f"\n{'='*70}")
            print(f"  {robot} ({len(trades)} trades)")
            print(f"{'='*70}")
            for t in trades:
                filt = "[VETADO]" if t["filter_vetoed"] else "[PASS  ]"
                pnl_b = f"${t['pnl_usd']:+.2f}"
                pnl_f = f"${t['pnl_filtered']:+.2f}" if not t['filter_vetoed'] else "  skip "
                print(f"  {t['timestamp']} | {t['signal']:5} | {str(t['regime']):16} | conf={t['confidence']:6} | RR={t['rr']:.1f} | {t['outcome']:4} | base={pnl_b:8} | filt={pnl_f:8} | {filt} {t['veto_reason'][:50]}")

    if not all_trades:
        print("Sin trades para analizar.")
        return

    # --- Resumen por robot ---
    print(f"\n{'='*80}")
    print(f"  RESUMEN POR ROBOT")
    print(f"{'='*80}")
    print(f"{'Robot':12} | {'N':>4} | {'WR%':>6} | {'PnL Base':>10} | {'Vetados':>8} | {'PnL Filtro':>11} | {'Delta':>8} | {'Avg R':>7}")
    print("-"*80)

    robots_in_results = sorted(set(t["robot"] for t in all_trades))
    total_delta = 0
    for robot in robots_in_results:
        rt = [t for t in all_trades if t["robot"] == robot]
        baseline = [t for t in rt]
        filtered = [t for t in rt if not t["filter_vetoed"]]
        vetoed   = [t for t in rt if t["filter_vetoed"]]

        pnl_base = sum(t["pnl_usd"] for t in baseline)
        pnl_filt = sum(t["pnl_filtered"] for t in rt)
        delta    = pnl_filt - pnl_base
        total_delta += delta

        wr_base  = len([t for t in baseline if t["outcome"]=="TP"]) / len(baseline) * 100 if baseline else 0
        wr_filt  = len([t for t in filtered if t["outcome"]=="TP"]) / len(filtered) * 100 if filtered else 0
        avg_r    = pnl_filt / (sum(t["risk_usd"] for t in filtered) or 1)

        print(f"{robot:12} | {len(rt):>4} | {wr_filt:>5.1f}% | {pnl_base:>+10.2f} | {len(vetoed):>8} | {pnl_filt:>+11.2f} | {delta:>+8.2f} | {avg_r:>7.3f}")

    # --- Totales ---
    pnl_base_total = sum(t["pnl_usd"]      for t in all_trades)
    pnl_filt_total = sum(t["pnl_filtered"] for t in all_trades)
    vetoed_total   = sum(1 for t in all_trades if t["filter_vetoed"])
    wr_base_total  = len([t for t in all_trades if t["outcome"]=="TP"]) / len(all_trades) * 100
    filt_trades    = [t for t in all_trades if not t["filter_vetoed"]]
    wr_filt_total  = len([t for t in filt_trades if t["outcome"]=="TP"]) / len(filt_trades) * 100 if filt_trades else 0

    print("-"*80)
    print(f"{'TOTAL':12} | {len(all_trades):>4} | {wr_filt_total:>5.1f}% | {pnl_base_total:>+10.2f} | {vetoed_total:>8} | {pnl_filt_total:>+11.2f} | {pnl_filt_total-pnl_base_total:>+8.2f} |")

    # --- Analisis de trades vetados ---
    vetoed_list = [t for t in all_trades if t["filter_vetoed"]]
    if vetoed_list:
        print(f"\n{'='*80}")
        print(f"  TRADES QUE EL FILTRO HABRIA BLOQUEADO ({len(vetoed_list)})")
        print(f"{'='*80}")
        print(f"{'Robot':10} | {'Fecha':16} | {'Sig':5} | {'Regime':18} | {'Conf':6} | {'RR':>4} | {'Outcome':6} | {'PnL':>8} | Razon del veto")
        print("-"*105)
        for t in vetoed_list:
            print(f"{t['robot']:10} | {t['timestamp']:16} | {t['signal']:5} | {str(t['regime']):18} | {t['confidence']:6} | {t['rr']:>4.1f} | {t['outcome']:6} | ${t['pnl_usd']:>+7.2f} | {t['veto_reason'][:50]}")

        wins_v   = len([t for t in vetoed_list if t["outcome"] == "TP"])
        losses_v = len([t for t in vetoed_list if t["outcome"] == "SL"])
        pnl_v    = sum(t["pnl_usd"] for t in vetoed_list)
        print(f"\n  Trades vetados: {len(vetoed_list)} | Wins: {wins_v} | Losses: {losses_v} | P&L que habriamos evitado: ${pnl_v:+.2f}")
        print(f"  WR de los vetados: {wins_v/len(vetoed_list)*100:.1f}%  (si > 50% el filtro estaria cortando buenos trades)")

    # --- CSV ---
    if args.csv:
        csv_path = _MVP_ROOT / "data" / "backtest_trend_filter.csv"
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_trades[0].keys()))
            writer.writeheader()
            writer.writerows(all_trades)
        print(f"\nCSV guardado: {csv_path}")

    print(f"\n{'='*80}")
    print(f"  CONCLUSION:")
    delta = pnl_filt_total - pnl_base_total
    if delta > 0:
        print(f"  El filtro MEJORA el P&L total en ${delta:+.2f}")
    elif delta < 0:
        print(f"  El filtro REDUCE el P&L total en ${delta:.2f} (estaria cortando trades ganadores)")
    else:
        print(f"  El filtro no cambia el P&L total")
    print(f"  WR sin filtro: {wr_base_total:.1f}%  |  WR con filtro: {wr_filt_total:.1f}%")
    print(f"  Trades eliminados: {vetoed_total}/{len(all_trades)} ({vetoed_total/len(all_trades)*100:.1f}%)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
