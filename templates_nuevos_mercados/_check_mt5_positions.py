import sys
sys.path.insert(0, "C:/Users/Lenovo/OneDrive/Escritorio/MVP/templates_nuevos_mercados/XAUUSD")
import mt5_bridge
import MetaTrader5 as mt5

mt5_bridge.initialize()

positions = mt5.positions_get()
print(f"\n=== Posiciones abiertas en MT5: {len(positions) if positions else 0} ===")
if positions:
    for p in positions:
        side = "LONG" if p.type == 0 else "SHORT"
        print(f"  {p.symbol:<14} | {side} | vol={p.volume} | profit=${p.profit:.2f} | open={p.price_open} | ticket={p.ticket}")
else:
    print("  (ninguna)")

account = mt5.account_info()
if account:
    print(f"\n=== Cuenta ===")
    print(f"  Balance:    ${account.balance:,.2f}")
    print(f"  Equity:     ${account.equity:,.2f}")
    print(f"  Free margin:${account.margin_free:,.2f}")
    print(f"  Server:     {account.server}")
    print(f"  Login:      {account.login}")
