"""Check MT5 symbol resolution and live bars for all robots."""
import sys, os

_HERE = os.path.dirname(os.path.abspath(__file__))
ROBOTS = ["BRENT_N6","BTCUSD","Corn_N6","ETHUSD","HK50","UK100","US30","US500","USTEC","WTI_N6","XAUUSD"]

results = []
for name in ROBOTS:
    robot_dir = os.path.join(_HERE, name)
    sys.path.insert(0, robot_dir)
    try:
        import importlib, config, mt5_bridge
        importlib.reload(config)
        importlib.reload(mt5_bridge)

        mt5_bridge.initialize()
        # resolve_symbol: try each fallback until MT5 returns bars
        sym = None
        candidates = [config.SYMBOL] + config.FALLBACK_SYMBOLS
        import MetaTrader5 as mt5
        for s in candidates:
            info = mt5.symbol_info(s)
            if info is not None:
                sym = s
                break
        if sym is None:
            results.append((name, "SYMBOL_FAIL", f"ninguno de {config.FALLBACK_SYMBOLS} encontrado"))
        else:
            tf = getattr(config, "TIMEFRAME", "60m")
            bars = mt5_bridge.fetch_mt5_bars(tf, 10)
            if bars is None or len(bars) == 0:
                results.append((name, "BARS_FAIL", f"symbol={sym} OK pero sin barras"))
            else:
                tick = mt5.symbol_info_tick(sym)
                bid  = round(tick.bid, 5) if tick else "?"
                ask  = round(tick.ask, 5) if tick else "?"
                sprd = round(tick.ask - tick.bid, 5) if tick else "?"
                results.append((name, "OK", f"symbol={sym} | tf={tf} | bars={len(bars)} | bid={bid} | ask={ask} | spread={sprd}"))
    except Exception as e:
        results.append((name, "ERROR", str(e)))
    finally:
        sys.path.pop(0)
        for mod in ["config","mt5_bridge","agents","microstructure","execution_tracker","retrainer","dashboard"]:
            sys.modules.pop(mod, None)

print("\n=== MT5 Symbol Check — All Robots ===")
for name, status, detail in results:
    icon = "OK" if status == "OK" else "FAIL"
    print(f"[{icon}] {name:<12} | {status:<12} | {detail}")
