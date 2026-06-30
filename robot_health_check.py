"""Chequeo de salud de los robots live. Detecta 3 modos de falla que el chequeo por
proceso NO ve:
  - DOWN      : el robot no tiene proceso corriendo.
  - DUPLICADO : >1 proceso para el mismo robot (pisan magic/posiciones).
  - CRASHING  : proceso VIVO pero el loop revienta cada ciclo (loop_error/agent_error
                recientes y SIN cycle exitoso) -> el caso WTI/BTC del 06-29 (código viejo
                en memoria crasheando, invisible al chequeo por proceso).
  - STALE     : (warn) robot de ciclo sin ningún evento en >STALE_MIN (proceso colgado).
                Se suprime en finde para evitar falsos positivos (no loguea con mercado cerrado).

Alerta por Telegram si hay algo crítico. Pensado para Task Scheduler cada ~30 min.
Uso: python robot_health_check.py
"""
import os, sys, io, json, glob
from datetime import datetime, timedelta
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import psutil

_HERE = os.path.dirname(os.path.abspath(__file__))
_TPL  = os.path.join(_HERE, "templates_nuevos_mercados")

# robot -> carpeta de artifacts (igual que restart_robots)
ART = {
    'UK100':'uk100','STOXX50':'stoxx50','BTCUSD':'btcusd','XAUUSD':'xauusd','US30':'us30',
    'US500':'us500','USTEC':'ustec','CHINA50':'china50','WTI_N6':'wti_n6',
    'OILFADE':'oilfade','GAPENGINE':'gapengine','EVENTBREAK':'eventbreak',
}
# robots de ciclo (emiten evento 'cycle' regular). Los de evento loguean esporádico.
CYCLE_ROBOTS = {'UK100','STOXX50','BTCUSD','XAUUSD','US30','US500','USTEC','CHINA50','WTI_N6'}
ERR_TYPES = {'loop_error','agent_error','error'}
CRASH_ERRS_60   = 3     # >=3 errores en 60min...
STALE_MIN       = 150   # ...sin cycle exitoso = CRASHING; sin NINGÚN evento = STALE

def _proc_counts():
    counts = {}
    for p in psutil.process_iter(['cmdline']):
        cl = p.info['cmdline']
        if not cl: continue
        if any('live_runner.py' == os.path.basename(str(t)) for t in cl) and '--loop' in cl:
            try: cw = os.path.basename(p.cwd() or '')
            except Exception: continue
            counts[cw.lower()] = counts.get(cw.lower(), 0) + 1
    return counts

def _scan_log(path, now):
    """Devuelve (min desde último cycle exitoso, min desde último evento, #errores en 60min)."""
    last_cycle = None; last_any = None; errs60 = 0
    if not os.path.exists(path): return None, None, 0
    try:
        lines = open(path, encoding="utf-8").read().splitlines()[-400:]
    except Exception:
        return None, None, 0
    for l in lines:
        try:
            j = json.loads(l); ts = j.get("timestamp", "")
            t = datetime.fromisoformat(ts)
        except Exception:
            continue
        typ = j.get("type", "")
        last_any = t if (last_any is None or t > last_any) else last_any
        if typ == "cycle":
            last_cycle = t if (last_cycle is None or t > last_cycle) else last_cycle
        if typ in ERR_TYPES and (now - t).total_seconds() < 3600:
            errs60 += 1
    mc = (now - last_cycle).total_seconds()/60 if last_cycle else None
    ma = (now - last_any).total_seconds()/60 if last_any else None
    return mc, ma, errs60

def main():
    now = datetime.now()
    is_weekend = now.weekday() >= 5
    counts = _proc_counts()
    rows = []; critical = []; warn = []
    for robot, d in ART.items():
        n = counts.get(d.lower(), 0)
        mc, ma, errs = _scan_log(os.path.join(_HERE, "artifacts", d, "live_log.jsonl"), now)
        status = "OK"
        if n == 0:
            status = "DOWN"; critical.append(f"{robot}: sin proceso")
        elif n > 1:
            status = f"DUP×{n}"; critical.append(f"{robot}: {n} procesos (duplicado)")
        elif errs >= CRASH_ERRS_60 and (mc is None or mc > 60):
            status = "CRASHING"; critical.append(f"{robot}: vivo pero crashea ({errs} errores/60min, sin ciclo exitoso)")
        elif robot in CYCLE_ROBOTS and not is_weekend and (ma is None or ma > STALE_MIN):
            status = "STALE"; warn.append(f"{robot}: sin eventos hace {int(ma) if ma else '?'}min")
        cyc = f"{int(mc)}m" if mc is not None else "—"
        rows.append((robot, n, status, cyc, errs))

    print(f"=== SALUD ROBOTS {now:%Y-%m-%d %H:%M} {'(finde)' if is_weekend else ''} ===")
    print(f"{'robot':12}{'proc':5}{'estado':10}{'últ.ciclo':11}{'err60'}")
    for robot, n, status, cyc, errs in rows:
        flag = "  <--" if status not in ("OK",) else ""
        print(f"{robot:12}{n:<5}{status:10}{cyc:11}{errs}{flag}")

    if critical:
        msg = "🤖❗ SALUD ROBOTS — CRÍTICO:\n" + "\n".join("• " + c for c in critical)
        if warn: msg += "\n⚠️ " + "; ".join(warn)
        print("\n" + msg)
        try:
            sys.path.insert(0, os.path.join(_TPL, "news_intelligence"))
            from alert_engine import _send_telegram
            _send_telegram(msg)
            print("(Telegram enviado)")
        except Exception as e:
            print(f"(no se pudo enviar Telegram: {e})")
        return 1
    print("\nTodo OK" + (f" ({len(warn)} warnings: {'; '.join(warn)})" if warn else ""))
    return 0

if __name__ == "__main__":
    sys.exit(main())
