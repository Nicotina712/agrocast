"""
restart_dashboards.py
Mata dashboards existentes y levanta uno por robot + master dashboard.
"""
import subprocess, os, sys, time, socket, psutil
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT      = os.path.dirname(os.path.abspath(__file__))
TMPL      = os.path.join(ROOT, "templates_nuevos_mercados")
# Usar Python312 que tiene MetaTrader5 instalado (igual que restart_robots.py)
PYTHON    = r'C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe'

# (folder, dashboard_file, port)
DASHBOARDS = [
    ("XAUUSD",   "dashboard.py", 8080),
    ("BTCUSD",   "dashboard.py", 8081),
    ("STOXX50",  "dashboard.py", 8082),
    ("US500",    "dashboard.py", 8083),
    ("USTEC",    "dashboard.py", 8084),
    ("US30",     "dashboard.py", 8085),
    ("WTI_N6",   "dashboard.py", 8086),
    ("BRENT_N6", "dashboard.py", 8087),
    ("UK100",    "dashboard.py", 8088),
    ("Corn_N6",  "dashboard.py", 8089),
    ("CHINA50",  "dashboard.py", 8091),
]
MASTER = ("", "master_dashboard.py", 8090)  # cwd = TMPL

# Sbean_N6 — dashboard en src/quantagent/, puerto 8092
SBEAN_DASHBOARD = {
    "script": os.path.join(os.path.dirname(TMPL), "src", "quantagent", "dashboard.py"),
    "cwd":    os.path.dirname(TMPL),
    "port":   8092,
}

MY_PID = os.getpid()

def port_listening(port, timeout=0.3):
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        s.close()
        return True
    except:
        return False

def kill_on_port(port):
    """Kill any process listening on given port."""
    killed = []
    for proc in psutil.process_iter(["pid", "name"]):
        if proc.pid == MY_PID:
            continue
        try:
            for conn in proc.net_connections():
                if conn.laddr.port == port and conn.status in ("LISTEN", "LISTENING"):
                    proc.kill()
                    killed.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            pass
    return killed

print("=== RESTART DASHBOARDS ===\n")

# Matar los que esten escuchando en los puertos destino
all_ports = [p for _, _, p in DASHBOARDS] + [MASTER[2]]
print("--- Liberando puertos ---")
for port in all_ports:
    killed = kill_on_port(port)
    if killed:
        print(f"  :{port} — matados PIDs {killed}")
    else:
        print(f"  :{port} — libre")

time.sleep(2)

# Lanzar dashboards individuales
print("\n--- Lanzando dashboards individuales ---")
for folder, script, port in DASHBOARDS:
    cwd = os.path.join(TMPL, folder)
    script_path = os.path.join(cwd, script)
    if not os.path.exists(script_path):
        print(f"  {folder}: NO ENCONTRADO ({script_path})")
        continue
    log_file = open(os.path.join(cwd, "dashboard.log"), "w")
    proc = subprocess.Popen(
        [PYTHON, script],
        cwd=cwd,
        stdout=log_file,
        stderr=log_file,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    print(f"  {folder:<12} PID={proc.pid}  → http://localhost:{port}")

# Lanzar master dashboard
print("\n--- Lanzando master dashboard ---")
master_script = os.path.join(TMPL, MASTER[1])
if os.path.exists(master_script):
    log_file = open(os.path.join(TMPL, "master_dashboard.log"), "w")
    proc = subprocess.Popen(
        [PYTHON, MASTER[1]],
        cwd=TMPL,
        stdout=log_file,
        stderr=log_file,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    print(f"  master_dashboard  PID={proc.pid}  → http://localhost:{MASTER[2]}")
else:
    print(f"  master_dashboard: NO ENCONTRADO")

# Sbean_N6 RETIRADO 2026-06-11: robot sin edge viable (spread > edge), quantagent sin
# mantenimiento, -$37/30d. Solo se libera su puerto por si quedo algo escuchando.
kill_on_port(SBEAN_DASHBOARD["port"])

print("\nEsperando 5s para verificar...\n")
time.sleep(5)

print("--- Verificacion ---")
ok, fail = [], []
for name, port in [("master", 8090)] + [(f, p) for f, _, p in DASHBOARDS]:
    if port_listening(port):
        print(f"  [OK]    :{port}  {name}")
        ok.append(port)
    else:
        print(f"  [FALLO] :{port}  {name}")
        fail.append(port)

print(f"\nResultado: {len(ok)} OK / {len(fail)} fallidos")
if fail:
    print(f"Puertos con fallo: {fail}")
    print("Revisa los archivos .log en cada carpeta de robot.")
