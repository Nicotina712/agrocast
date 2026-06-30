"""
_restore_filtered_history.py
Lee el backup pre_fix y escribe en paper_trades.jsonl únicamente
los trades que habrían pasado los filtros del sistema mejorado:

  Fix #1 — Paper position guard: elimina trades abiertos mientras
            ya había una posición paper vigente (SL/TP no tocado)
  Fix #2 — MIN_SL_PCT: ajusta SL si era demasiado ajustado
            (no bloquea, solo corrige el SL en el registro)
  F1/F2   — Solo ETHUSD: bloquea LONG en RISK_OFF+BTC_LEADING
             o en regime DISTRIBUTION

Ejecutar UNA sola vez:  py -3 _restore_filtered_history.py
"""

import json, sys
from pathlib import Path

HERE = Path(__file__).parent
BASE = HERE.parent / "artifacts"

MIN_SL_PCT = {
    "btcusd":  0.80, "us500":   0.30, "ustec":   0.30,
    "us30":    0.35, "wti_n6":  0.20, "xauusd":  0.15,
    "ethusd":  0.80, "brent_n6":0.20, "uk100":   0.20,
}

INSTRUMENTS = [
    "btcusd", "ethusd", "us500", "ustec",
    "us30", "wti_n6", "brent_n6", "uk100", "xauusd",
]


def _apply_filters(sym: str, trades: list) -> list:
    """Devuelve solo los trades que habrían pasado los nuevos filtros,
    con el SL corregido donde corresponda."""
    passing  = []
    open_pos = None
    min_pct  = MIN_SL_PCT.get(sym, 0)

    for t in trades:
        t = dict(t)   # copia para no mutar el original
        sig   = t.get("signal")
        entry = float(t.get("entry") or 0)
        sl    = float(t.get("sl")    or 0)
        tp    = float(t.get("tp")    or 0)
        price = float(t.get("price_at_signal") or entry)

        # ── Paper position guard (Fix #1) ─────────────────────────────────
        if open_pos:
            ps    = open_pos["signal"]
            pe    = float(open_pos["entry"] or 0)
            ps_sl = float(open_pos["sl"]    or 0)
            ps_tp = float(open_pos["tp"]    or 0)
            if ps == "LONG":
                outcome = "WIN" if price >= ps_tp else "LOSS" if price <= ps_sl else "OPEN"
            else:
                outcome = "WIN" if price <= ps_tp else "LOSS" if price >= ps_sl else "OPEN"

            if outcome == "OPEN":
                # posición anterior aún vigente → este trade no se habría generado
                continue
            else:
                open_pos = None   # cerrada; puede continuar

        # ── ETHUSD: F1 + F2 ───────────────────────────────────────────────
        if sym == "ethusd" and sig == "LONG":
            # risk_on_off no siempre está como campo directo — leer del reasoning
            reasoning = t.get("reasoning", "")
            roff = t.get("risk_on_off") or (
                "RISK_OFF" if "RISK_OFF" in reasoning else
                "RISK_ON"  if "RISK_ON"  in reasoning else ""
            )
            btc  = t.get("eth_btc", "")
            reg  = t.get("eth_regime", "")
            if roff == "RISK_OFF" and btc == "BTC_LEADING":
                continue   # F1
            if reg == "DISTRIBUTION":
                continue   # F2

        # ── MIN_SL_PCT (Fix #2) — ajustar SL, no bloquear ─────────────────
        if entry and sl and sig not in (None, "FLAT"):
            sl_dist_pct = abs(entry - sl) / entry * 100
            if sl_dist_pct < min_pct:
                min_dist = entry * min_pct / 100
                if sig == "LONG":
                    new_sl = round(entry - min_dist, 5)
                else:
                    new_sl = round(entry + min_dist, 5)
                t["sl"]           = new_sl
                t["sl_enforced"]  = True
                t["sl_original"]  = sl

        # Pasa todos los filtros
        passing.append(t)
        if sig not in (None, "FLAT"):
            open_pos = t

    return passing


def main():
    print("=" * 65)
    print("_restore_filtered_history.py — restaurar trades filtrados")
    print("=" * 65)

    total_orig = total_pass = 0

    for sym in INSTRUMENTS:
        art = BASE / sym
        bk  = art / "paper_trades_pre_fix_20260526.jsonl"
        pt  = art / "paper_trades.jsonl"

        if not bk.exists():
            print(f"  {sym:12s}  sin backup — skip")
            continue

        trades   = [json.loads(l) for l in bk.read_text(encoding="utf-8").splitlines() if l.strip()]
        filtered = _apply_filters(sym, trades)

        total_orig += len(trades)
        total_pass += len(filtered)

        # Escribir trades filtrados
        out = "\n".join(json.dumps(t, ensure_ascii=False, default=str) for t in filtered)
        if out:
            out += "\n"
        pt.write_text(out, encoding="utf-8")

        blocked   = len(trades) - len(filtered)
        adjusted  = sum(1 for t in filtered if t.get("sl_enforced"))
        print(f"  {sym:12s}  {len(trades):2d} orig -> {len(filtered):2d} pasan "
              f"({blocked:2d} bloqueados, {adjusted} SL ajustados)")

    print()
    print(f"Total: {total_orig} trades originales -> {total_pass} trades limpios "
          f"({total_orig - total_pass} eliminados)")


if __name__ == "__main__":
    main()
