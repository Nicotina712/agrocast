"""
Portfolio status check + auto-restart for all 11 robots.
Uso: python _portfolio_status.py [--restart]
"""
import os, sys, json, subprocess, argparse
from datetime import datetime

_HERE         = os.path.dirname(os.path.abspath(__file__))
_ARTIFACTS    = os.path.join(os.path.dirname(_HERE), "artifacts")

ROBOTS = [
    {"name": "BTCUSD",   "dir": "BTCUSD",   "artifacts": "btcusd",   "cycle_min": 30},
    {"name": "ETHUSD",   "dir": "ETHUSD",   "artifacts": "ethusd",   "cycle_min": 15},
    {"name": "XAUUSD",   "dir": "XAUUSD",   "artifacts": "xauusd",   "cycle_min": 15},
    {"name": "BRENT_N6", "dir": "BRENT_N6", "artifacts": "brent_n6", "cycle_min": 15},
    {"name": "WTI_N6",   "dir": "WTI_N6",   "artifacts": "wti_n6",   "cycle_min": 15},
    {"name": "UK100",    "dir": "UK100",     "artifacts": "uk100",    "cycle_min": 15},
    {"name": "US30",     "dir": "US30",      "artifacts": "us30",     "cycle_min": 15},
    {"name": "US500",    "dir": "US500",     "artifacts": "us500",    "cycle_min": 30},
    {"name": "USTEC",    "dir": "USTEC",     "artifacts": "ustec",    "cycle_min": 30},
    {"name": "HK50",     "dir": "HK50",      "artifacts": "hk50",     "cycle_min": 30},
    {"name": "Corn_N6",  "dir": "Corn_N6",   "artifacts": "corn_n6",  "cycle_min": 15},
]

def is_pid_alive(pid):
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True)
        return str(pid) in result.stdout
    except:
        return False

def read_pid(artifacts_dir):
    pid_file = os.path.join(_ARTIFACTS, artifacts_dir, "robot.pid")
    try:
        return int(open(pid_file).read().strip())
    except:
        return None

def last_cycle(artifacts_dir):
    log = os.path.join(_ARTIFACTS, artifacts_dir, "live_log.jsonl")
    sig = os.path.join(_ARTIFACTS, artifacts_dir, "live_signal.json")
    ts = None
    if os.path.exists(log):
        try:
            lines = open(log, encoding="utf-8").readlines()
            if lines:
                obj = json.loads(lines[-1])
                ts = obj.get("timestamp")
        except:
            pass
    if not ts and os.path.exists(sig):
        try:
            obj = json.load(open(sig, encoding="utf-8"))
            ts = obj.get("timestamp")
        except:
            pass
    return ts

def last_signal(artifacts_dir):
    sig = os.path.join(_ARTIFACTS, artifacts_dir, "live_signal.json")
    try:
        obj = json.load(open(sig, encoding="utf-8"))
        return obj.get("signal", "?")
    except:
        return "?"

def restart_robot(robot):
    work_dir = os.path.join(_HERE, robot["dir"])
    try:
        # Kill old PID if exists
        old_pid = read_pid(robot["artifacts"])
        if old_pid and is_pid_alive(old_pid):
            subprocess.run(["taskkill", "/PID", str(old_pid), "/F"], capture_output=True)
        subprocess.Popen(
            ["py", "-3", "live_runner.py", "--loop"],
            cwd=work_dir,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        return True
    except Exception as e:
        print(f"  ERROR al reiniciar {robot['name']}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true", help="Reiniciar robots caidos")
    args = parser.parse_args()

    now = datetime.now()
    print(f"\n{'='*70}")
    print(f"  PORTFOLIO STATUS  —  {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    print(f"{'Robot':<14} {'PID':<8} {'Proceso':<8} {'Ultimo ciclo':<22} {'Edad':<10} {'Signal':<8} {'Estado'}")
    print(f"{'-'*70}")

    n_ok = 0
    n_stale = 0
    n_dead = 0
    restarted = []

    for r in ROBOTS:
        pid    = read_pid(r["artifacts"])
        alive  = is_pid_alive(pid) if pid else False
        ts     = last_cycle(r["artifacts"])
        signal = last_signal(r["artifacts"])

        if ts:
            try:
                dt  = datetime.fromisoformat(ts)
                age = int((now - dt).total_seconds() / 60)
                age_str = f"{age}min"
                stale_threshold = r["cycle_min"] * 3  # 3 missed cycles = stale
                is_stale = age > stale_threshold
            except:
                age_str = "?"
                is_stale = False
        else:
            age_str = "nunca"
            is_stale = True

        if not alive:
            status = "MUERTO"
            n_dead += 1
        elif is_stale:
            status = "STALE"
            n_stale += 1
        else:
            status = "OK"
            n_ok += 1

        pid_str = str(pid) if pid else "-"
        alive_str = "vivo" if alive else "muerto"
        ts_short = ts[:19] if ts else "nunca"
        print(f"{r['name']:<14} {pid_str:<8} {alive_str:<8} {ts_short:<22} {age_str:<10} {signal:<8} {status}")

        if args.restart and status in ("MUERTO", "STALE"):
            print(f"  -> Reiniciando {r['name']}...")
            if restart_robot(r):
                restarted.append(r["name"])

    print(f"\nResumen: {n_ok} OK | {n_stale} STALE | {n_dead} MUERTOS")
    if restarted:
        print(f"Reiniciados: {', '.join(restarted)}")
    print()

if __name__ == "__main__":
    main()
