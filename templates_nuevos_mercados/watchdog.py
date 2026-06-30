"""
Portfolio Watchdog — reinicia robots y dashboards si se caen.
Corre cada 5 minutos via Task Scheduler.

Uso:
  python watchdog.py          # revisa y reinicia lo que falte
  python watchdog.py --status # solo muestra estado, sin reiniciar
"""

import os, sys, json, subprocess, argparse, time
from datetime import datetime, timezone, timedelta

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(_HERE)
LOG_FILE  = os.path.join(_HERE, "watchdog.log")

# ─── Definición de robots ────────────────────────────────────────────────────

ROBOTS = [
    {"name": "BTCUSD",   "dir": "BTCUSD",   "script": "live_runner.py", "artifacts": "btcusd"},
    {"name": "ETHUSD",   "dir": "ETHUSD",   "script": "live_runner.py", "artifacts": "ethusd"},
    {"name": "XAUUSD",   "dir": "XAUUSD",   "script": "live_runner.py", "artifacts": "xauusd"},
    {"name": "WTI_N6",   "dir": "WTI_N6",   "script": "live_runner.py", "artifacts": "wti_n6"},
    {"name": "BRENT_N6", "dir": "BRENT_N6", "script": "live_runner.py", "artifacts": "brent_n6"},
    {"name": "UK100",    "dir": "UK100",    "script": "live_runner.py", "artifacts": "uk100"},
    {"name": "US500",    "dir": "US500",    "script": "live_runner.py", "artifacts": "us500"},
    {"name": "USTEC",    "dir": "USTEC",    "script": "live_runner.py", "artifacts": "ustec"},
    {"name": "US30",     "dir": "US30",     "script": "live_runner.py", "artifacts": "us30"},
    {"name": "HK50",     "dir": "HK50",     "script": "live_runner.py", "artifacts": "hk50"},
    {"name": "Corn_N6",  "dir": "Corn_N6",  "script": "live_runner.py", "artifacts": "corn_n6"},
]

DASHBOARDS = [
    {"name": "BTCUSD_dash",   "dir": "BTCUSD",   "port": 8081},
    {"name": "ETHUSD_dash",   "dir": "ETHUSD",   "port": 8082},
    {"name": "XAUUSD_dash",   "dir": "XAUUSD",   "port": 8080},
    {"name": "WTI_N6_dash",   "dir": "WTI_N6",   "port": 8086},
    {"name": "BRENT_N6_dash", "dir": "BRENT_N6", "port": 8087},
    {"name": "UK100_dash",    "dir": "UK100",    "port": 8088},
    {"name": "US500_dash",    "dir": "US500",    "port": 8083},
    {"name": "USTEC_dash",    "dir": "USTEC",    "port": 8084},
    {"name": "US30_dash",     "dir": "US30",     "port": 8085},
    {"name": "HK50_dash",     "dir": "HK50",     "port": 8091},
    {"name": "Corn_N6_dash",  "dir": "Corn_N6",  "port": 8089},
    {"name": "MASTER_dash",   "dir": ".",        "port": 8090},
]

ARTIFACTS_BASE = os.path.join(_MVP_ROOT, "artifacts")


def _log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _is_pid_alive(pid: int) -> bool:
    """Check if a PID is running on Windows."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True
        )
        return str(pid) in result.stdout
    except Exception:
        return False


def _is_port_open(port: int) -> bool:
    """Check if a TCP port is listening."""
    import socket
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except Exception:
        return False


def _read_pid(artifacts_dir: str) -> int | None:
    pid_file = os.path.join(ARTIFACTS_BASE, artifacts_dir, "robot.pid")
    try:
        return int(open(pid_file).read().strip())
    except Exception:
        return None


def _start_robot(robot: dict) -> bool:
    """Launch robot in a new window."""
    work_dir = os.path.join(_HERE, robot["dir"])
    try:
        proc = subprocess.Popen(
            ["py", "-3", robot["script"], "--loop"],
            cwd=work_dir,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        _log(f"  OK {robot['name']} iniciado (PID {proc.pid})")
        return True
    except Exception as e:
        _log(f"  FAIL {robot['name']} ERROR al iniciar: {e}")
        return False


def _start_dashboard(dash: dict) -> bool:
    """Launch dashboard in a minimized window."""
    work_dir = os.path.join(_HERE, dash["dir"])
    script = "master_dashboard.py" if dash["dir"] == "." else "dashboard.py"
    try:
        subprocess.Popen(
            ["py", "-3", script],
            cwd=work_dir,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        _log(f"  OK {dash['name']} (:{dash['port']}) iniciado")
        return True
    except Exception as e:
        _log(f"  FAIL {dash['name']} ERROR: {e}")
        return False


def check_robots(dry_run: bool = False) -> dict:
    results = {}
    restarted = 0

    for robot in ROBOTS:
        pid = _read_pid(robot["artifacts"])
        alive = _is_pid_alive(pid) if pid else False
        status = "OK" if alive else "DOWN"
        results[robot["name"]] = status

        if not alive:
            _log(f"[ROBOT] {robot['name']} caído (PID {pid}) — {'reiniciando...' if not dry_run else 'dry-run'}")
            if not dry_run:
                _start_robot(robot)
                restarted += 1
        else:
            pass  # silencioso si OK

    return results, restarted


def check_dashboards(dry_run: bool = False) -> dict:
    results = {}
    restarted = 0

    for dash in DASHBOARDS:
        up = _is_port_open(dash["port"])
        status = "OK" if up else "DOWN"
        results[dash["name"]] = status

        if not up:
            _log(f"[DASH]  {dash['name']} (:{dash['port']}) caído — {'reiniciando...' if not dry_run else 'dry-run'}")
            if not dry_run:
                _start_dashboard(dash)
                restarted += 1

    return results, restarted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true", help="Solo mostrar estado, no reiniciar")
    args = parser.parse_args()

    dry_run = args.status
    _log("=" * 55)
    _log(f"Watchdog run {'(dry-run)' if dry_run else ''}")

    robot_status, r_restarted = check_robots(dry_run)
    dash_status,  d_restarted = check_dashboards(dry_run)

    ok_robots = sum(1 for v in robot_status.values() if v == "OK")
    ok_dashes = sum(1 for v in dash_status.values()  if v == "OK")

    _log(f"Robots:     {ok_robots}/{len(ROBOTS)} OK | {r_restarted} reiniciados")
    _log(f"Dashboards: {ok_dashes}/{len(DASHBOARDS)} OK | {d_restarted} reiniciados")
    _log("Watchdog completado")

    if args.status:
        print("\n--- ROBOTS ---")
        for k, v in robot_status.items():
            icon = "[OK]" if v == "OK" else "[--]"
            print(f"  {icon} {k}: {v}")
        print("\n--- DASHBOARDS ---")
        for k, v in dash_status.items():
            icon = "[OK]" if v == "OK" else "[--]"
            print(f"  {icon} {k}: {v}")


if __name__ == "__main__":
    main()
