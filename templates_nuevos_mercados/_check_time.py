from datetime import datetime
from zoneinfo import ZoneInfo

ct  = datetime.now(ZoneInfo("America/Chicago"))
utc = datetime.now(ZoneInfo("UTC"))
print(f"UTC: {utc.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"CT:  {ct.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"Dia: {ct.strftime('%A')}")

# Check each robot's session
sessions = {
    "BRENT_N6":  (2,  0, 13, 30, False),
    "BTCUSD":    (3,  0, 22,  0, True),
    "Corn_N6":   (8, 30, 13, 20, False),
    "ETHUSD":    (3,  0, 22,  0, True),
    "HK50":      (9, 30, 16,  0, False),
    "UK100":     (2,  0, 10, 30, False),
    "US30":      (8, 30, 15,  0, False),
    "US500":     (8, 30, 15,  0, False),
    "USTEC":     (8, 30, 15,  0, False),
    "WTI_N6":    (8,  0, 13, 30, False),
    "XAUUSD":    (7,  0, 13,  0, False),
}

is_weekend = ct.weekday() >= 5  # Saturday=5, Sunday=6
ct_h = ct.hour + ct.minute / 60

print(f"\nEs fin de semana (Python weekday={ct.weekday()}): {is_weekend}")
print(f"\n{'Robot':<14} {'Sesion CT':<20} {'Weekends':<10} {'DEBERIA CICLAR'}")
print("-" * 60)
for name, (oh, om, ch, cm, wknd) in sessions.items():
    in_session = (ct_h >= oh + om/60) and (ct_h < ch + cm/60)
    should_run = in_session and (wknd or not is_weekend)
    flag = "SI ✓" if should_run else "no"
    print(f"{name:<14} {str(oh)+':'+str(om).zfill(2)+'-'+str(ch)+':'+str(cm).zfill(2):<20} {str(wknd):<10} {flag}")
