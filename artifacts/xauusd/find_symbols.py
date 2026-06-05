import MetaTrader5 as mt5
mt5.initialize()
all_syms = mt5.symbols_get()
keywords = ["nas","sp5","us5","oil","wti","crude","brent","dow","ger","uk1","spx","ndx"]
results = []
for s in all_syms:
    n = s.name.lower()
    if any(k in n for k in keywords):
        bars = mt5.copy_rates_from_pos(s.name, mt5.TIMEFRAME_H1, 0, 3)
        if bars is not None and len(bars) > 0:
            results.append("{:<22} {:>12,.2f}".format(s.name, bars[-1]["close"]))
out = open(r"C:\Users\Lenovo\OneDrive\Escritorio\MVP\artifacts\xauusd\symbols_found.txt","w")
out.write("\n".join(sorted(results)))
out.close()
mt5.shutdown()
