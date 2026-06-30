"""Chequeo de expiracion/validez de contratos (2026-06-16).
Para cada robot, verifica que su SYMBOL configurado EXISTA en el broker, tenga quote viva,
y -si es un futuro con vencimiento- que no este por expirar. Alerta (consola + Telegram) si:
  - el simbolo NO existe (caso WTI_N6 -> resolvia a soja),
  - no tiene quote (bid<=0),
  - expira en <= WARN_DAYS dias (hay que rolar al proximo contrato).
Pensado para Task Scheduler diario. NO opera ni modifica nada.
"""
import os, sys, glob, importlib, io
from datetime import datetime, timezone

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_TPL = os.path.join(_HERE, "templates_nuevos_mercados")
WARN_DAYS = 10

import MetaTrader5 as mt5

def _load_config_isolated(cfgpath, folder):
    """Carga config.py por RUTA con nombre de modulo unico (evita colision sys.modules['config'])."""
    import importlib.util
    d = os.path.dirname(cfgpath)
    added = d not in sys.path
    if added:
        sys.path.insert(0, d)
    # quitar cualquier 'config' cacheado para que sus imports internos resuelvan al de ESTA carpeta
    cached = sys.modules.pop("config", None)
    try:
        spec = importlib.util.spec_from_file_location(f"config_{folder}", cfgpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["config"] = mod          # por si config.py hace 'from config import ...'
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.modules.pop("config", None)
        if cached is not None:
            sys.modules["config"] = cached
        if added and d in sys.path:
            sys.path.remove(d)

def _robot_symbols():
    """Devuelve [(robot, symbol), ...] leyendo el SYMBOL de cada config + OIL_SYMBOLS si tiene."""
    out = []
    for cfgpath in sorted(glob.glob(os.path.join(_TPL, "*", "config.py"))):
        folder = os.path.basename(os.path.dirname(cfgpath))
        if folder in ("HK50", "ETHUSD"):   # retirados
            continue
        try:
            c = _load_config_isolated(cfgpath, folder)
            sym = getattr(c, "SYMBOL", None)
            if sym:
                out.append((folder, sym))
            for s in getattr(c, "OIL_SYMBOLS", []) or []:
                out.append((f"{folder}.OIL", s))
            for s in getattr(c, "BASKET", []) or []:
                out.append((f"{folder}.BASKET", s))
        except Exception as e:
            out.append((folder, f"<error config: {e}>"))
    return out

def main():
    if not mt5.initialize():
        print("MT5 init FALLO"); return
    alerts = []
    now = datetime.now(timezone.utc)
    print(f"CHEQUEO DE CONTRATOS ({now:%Y-%m-%d %H:%M} UTC):")
    seen = set()
    for robot, sym in _robot_symbols():
        if (robot, sym) in seen:
            continue
        seen.add((robot, sym))
        if sym.startswith("<error"):
            alerts.append(f"{robot}: {sym}"); print(f"  ⚠️ {robot:14} {sym}"); continue
        si = mt5.symbol_info(sym)
        if si is None:
            mt5.symbol_select(sym, True); si = mt5.symbol_info(sym)
        if si is None:
            alerts.append(f"{robot}: simbolo '{sym}' NO existe en el broker (revisar rolado)")
            print(f"  ❌ {robot:14} {sym:12} NO EXISTE"); continue
        tick = mt5.symbol_info_tick(sym)
        bid = tick.bid if tick else 0
        # expiracion (futuros)
        exp = getattr(si, "expiration_time", 0) or 0
        exp_str = ""
        if exp:
            exp_dt = datetime.fromtimestamp(exp, tz=timezone.utc)
            days = (exp_dt - now).days
            exp_str = f"exp {exp_dt:%Y-%m-%d} ({days}d)"
            if days <= WARN_DAYS:
                alerts.append(f"{robot}: '{sym}' expira en {days}d ({exp_dt:%Y-%m-%d}) — ROLAR al proximo contrato")
        # bid<=0 NO es alerta (puede ser mercado cerrado); solo nota visual.
        note = "" if bid > 0 else "(sin quote ahora — ¿mercado cerrado?)"
        flag = "✅"
        if bid <= 0:
            flag = "·"
        elif exp and (datetime.fromtimestamp(exp, tz=timezone.utc) - now).days <= WARN_DAYS:
            flag = "⏰"
        print(f"  {flag} {robot:14} {sym:12} bid={bid:<10} {exp_str} {note}")
    mt5.shutdown()

    if alerts:
        msg = "🛢️ CHEQUEO CONTRATOS — ACCION REQUERIDA:\n" + "\n".join("• " + a for a in alerts)
        print("\n" + msg)
        try:
            sys.path.insert(0, os.path.join(_TPL, "news_intelligence"))
            from alert_engine import _send_telegram
            _send_telegram(msg)
        except Exception as e:
            print(f"(no se pudo enviar Telegram: {e})")
    else:
        print("\nTodo OK — todos los contratos existen, con quote y sin vencimiento proximo.")

if __name__ == "__main__":
    main()
