"""Mata todas las instancias de live_runner y reinicia exactamente una por robot."""
import psutil, subprocess, os, time, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PYTHON = r'C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe'
MVP    = r'C:\Users\Lenovo\OneDrive\Escritorio\MVP\templates_nuevos_mercados'
ROOT   = Path(r'C:\Users\Lenovo\OneDrive\Escritorio\MVP')

ROBOTS = ['UK100','STOXX50','BTCUSD','XAUUSD','US30','US500','USTEC','CHINA50','WTI_N6','OILFADE','GAPENGINE','EVENTBREAK']  # RETIRADOS 2026-06-26: Corn_N6 (granos CFD sin edge/spread) + BRENT_N6 (sin edge mecánico intradía; WTI lidera price-discovery >80%, Brent es follower → momentum propio débil; redundante con WTI)
ART    = {
    'UK100':'artifacts/uk100','STOXX50':'artifacts/stoxx50','BTCUSD':'artifacts/btcusd',
    'XAUUSD':'artifacts/xauusd','US30':'artifacts/us30','US500':'artifacts/us500',
    'USTEC':'artifacts/ustec','CHINA50':'artifacts/china50','Corn_N6':'artifacts/corn_n6',
    'BRENT_N6':'artifacts/brent_n6','WTI_N6':'artifacts/wti_n6','OILFADE':'artifacts/oilfade','GAPENGINE':'artifacts/gapengine',
    'EVENTBREAK':'artifacts/eventbreak',
}

# Sbean_N6 — arquitectura diferente (src/quantagent), se gestiona por separado
SBEAN_CMD = [PYTHON, '-m', 'src.quantagent.live_runner', '--loop', '--execute']
SBEAN_CWD = str(ROOT)

MY_PID = os.getpid()

print(f'Mi PID: {MY_PID} (no me auto-mato)')
print('--- Matando live_runner duplicados ---')

# Matar TODOS los live_runner --loop (no solo los de ROBOTS): un robot retirado deja
# procesos zombie que siguen operando (incidente 06-13: ETH/HK50 retirados seguian
# tradeando). El kill se decide por cmdline; el cwd solo etiqueta (best-effort).
killed = {}
for proc in psutil.process_iter(['pid','cmdline','cwd']):
    try:
        if proc.info['pid'] == MY_PID:
            continue
        cmd = ' '.join(proc.info['cmdline'] or [])
        if 'live_runner.py' in cmd and '--loop' in cmd:
            try:
                cwd = proc.info['cwd'] or ''
                robot = cwd.replace(MVP + os.sep, '').replace(MVP + '/', '') or '?'
            except Exception:
                robot = '?'
            proc.kill()   # kill (no terminate): SIGTERM se ignoraba en algunos zombies
            killed.setdefault(robot, []).append(proc.info['pid'])
    except:
        pass

for name, pids in sorted(killed.items()):
    print(f'  {name}: matados {pids}')


# Matar Sbean si corre
for proc in psutil.process_iter(['pid','cmdline','cwd']):
    try:
        if proc.info['pid'] == MY_PID: continue
        cmd = ' '.join(proc.info['cmdline'] or [])
        if 'quantagent' in cmd and 'live_runner' in cmd:
            proc.terminate()
            print(f'  Sbean_N6: matado PID={proc.info["pid"]}')
    except: pass

print(f'\nEsperando 5s...')
time.sleep(5)

print('--- Iniciando 1 instancia por robot ---')
for name in ROBOTS:
    robot_dir = os.path.join(MVP, name)
    p = subprocess.Popen(
        [PYTHON, 'live_runner.py', '--loop'],
        cwd=robot_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    (ROOT / ART[name] / 'robot.pid').write_text(str(p.pid))
    print(f'  {name}: PID={p.pid}')


# Sbean_N6 RETIRADO 2026-06-11: sin edge mecanico viable (spread 0.165% > edge),
# arquitectura quantagent sin mantenimiento, -$37/30d. Carpeta src/quantagent conservada.

print('\nEsperando 20s para verificar...')
time.sleep(20)

print('\n--- Verificacion final ---')
from collections import defaultdict
count = defaultdict(list)
for proc in psutil.process_iter(['pid','cmdline','cwd']):
    try:
        cmd = ' '.join(proc.info['cmdline'] or [])
        cwd = proc.info['cwd'] or ''
        if 'live_runner.py' in cmd and '--loop' in cmd and MVP in cwd:
            robot = cwd.replace(MVP + os.sep, '').replace(MVP + '/', '')
            count[robot].append(proc.info['pid'])
    except:
        pass

all_ok = True
for name in ROBOTS:
    pids = count.get(name, [])
    n = len(pids)
    status = 'OK (1 instancia)' if n == 1 else f'!! {n} instancias' if n > 1 else '!! MUERTO'
    if n != 1: all_ok = False
    print(f'  {name:12}: {status} | PIDs={pids}')

print()
print('TODO OK - sin duplicados' if all_ok else 'ATENCION: revisar robots marcados')
