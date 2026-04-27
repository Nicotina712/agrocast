"""
src/alerts/whatsapp_bot.py
Alertas por WhatsApp para AgroCast PRO vía Twilio API.

Configuración en .env:
  TWILIO_ACCOUNT_SID  — Account SID de Twilio
  TWILIO_AUTH_TOKEN   — Auth Token de Twilio
  TWILIO_FROM_NUMBER  — número Twilio con prefijo "whatsapp:+..." (ej: whatsapp:+14155238886)
  WHATSAPP_TO_NUMBER  — número destino con prefijo "whatsapp:+..." (ej: whatsapp:+59899123456)

Para usar el Sandbox de Twilio (gratis para pruebas):
  1. Ir a https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn
  2. Enviar "join <código>" al número de Twilio desde tu WhatsApp
  3. Usar TWILIO_FROM_NUMBER=whatsapp:+14155238886 (sandbox)

Para producción:
  - Activar WhatsApp Business API en Twilio (requiere aprobación de Meta)
  - Usar templates aprobados para mensajes de alerta

Uso:
  python -m src.alerts.whatsapp_bot --test
  python -m src.alerts.whatsapp_bot --check-signals
"""

import os
import sys
import json
from datetime import datetime, date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TWILIO_SID    = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM   = os.getenv("TWILIO_FROM_NUMBER", "")   # whatsapp:+14155238886
WHATSAPP_TO   = os.getenv("WHATSAPP_TO_NUMBER", "")   # whatsapp:+59899XXXXXX

_LAST_WA_PATH = _PROJECT_ROOT / "data" / "last_whatsapp_signal.json"
MIN_CONFIDENCE = 0.30


def _is_configured() -> bool:
    return bool(TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM and WHATSAPP_TO)


def send_whatsapp(message: str) -> bool:
    """Envía un mensaje de WhatsApp vía Twilio. Retorna True si exitoso."""
    if not _is_configured():
        print("[WhatsApp] Twilio no configurado — ver src/alerts/whatsapp_bot.py")
        return False
    try:
        from twilio.rest import Client
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        msg = client.messages.create(
            body=message,
            from_=TWILIO_FROM,
            to=WHATSAPP_TO,
        )
        print(f"[WhatsApp] Mensaje enviado: SID={msg.sid}")
        return True
    except ImportError:
        print("[WhatsApp] twilio no instalado — ejecuta: pip install twilio")
        return False
    except Exception as e:
        print(f"[WhatsApp] Error: {e}")
        return False


def _load_last() -> dict:
    if _LAST_WA_PATH.exists():
        try:
            return json.loads(_LAST_WA_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_last(data: dict) -> None:
    _LAST_WA_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LAST_WA_PATH.write_text(json.dumps(data, indent=2))


def _signal_emoji(signal: str) -> str:
    return {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(signal, "⚪")


def check_and_alert_signal() -> bool:
    """
    Lee signals.csv y envía alerta WhatsApp si la señal cambió.
    Mismo criterio que telegram_bot.py pero para WhatsApp.
    """
    import pandas as pd

    signals_path = _PROJECT_ROOT / "artifacts" / "signals.csv"
    if not signals_path.exists():
        return False

    try:
        df = pd.read_csv(signals_path)
        if df.empty:
            return False

        last = df.iloc[-1]
        signal     = str(last.get("signal", "HOLD"))
        confidence = float(last.get("confidence", 0))
        exp_ret    = float(last.get("expected_return", 0))
        sig_date   = str(last.get("Date", ""))[:10]

        prob_up  = round((0.5 + exp_ret) * 100)
        conf_pct = round(confidence * 100)

        # Precio actual
        price = 0.0
        try:
            mkt = pd.read_csv(_PROJECT_ROOT / "data" / "raw_market.csv")
            price = float(mkt["Soybeans"].iloc[-1])
        except Exception:
            pass

        last_saved = _load_last()
        alert_today = last_saved.get("alert_date") == str(date.today())
        signal_changed = (last_saved.get("signal") != signal) or (last_saved.get("date") != sig_date)

        if signal_changed and confidence >= MIN_CONFIDENCE and not alert_today:
            emoji  = _signal_emoji(signal)
            prev   = last_saved.get("signal", "")
            change = f" (antes: {prev})" if prev and prev != signal else ""

            msg = (
                f"{emoji} *AgroCast — Señal {signal}*{change}\n\n"
                f"📅 Fecha: {sig_date}\n"
                f"💰 Precio soja: {price:.2f} USc/bu\n"
                f"📊 P(sube 7d): {prob_up}%\n"
                f"🎯 Confianza: {conf_pct}%\n\n"
                f"_AgroCast PRO · {datetime.now().strftime('%H:%M')}_"
            )
            ok = send_whatsapp(msg)
            if ok:
                _save_last({
                    "signal":     signal,
                    "confidence": confidence,
                    "date":       sig_date,
                    "alert_date": str(date.today()),
                })
            return ok
    except Exception as e:
        print(f"[WhatsApp] Error check_and_alert_signal: {e}")
    return False


def check_and_alert_price_target() -> bool:
    """Alerta cuando el precio local (USD/ton Uruguay) supera el objetivo."""
    import pandas as pd

    target_str = os.getenv("PRODUCER_PRICE_TARGET_USD_TON", "").strip()
    if not target_str:
        return False
    try:
        price_target = float(target_str)
    except ValueError:
        return False

    mkt_path = _PROJECT_ROOT / "data" / "raw_market.csv"
    if not mkt_path.exists():
        return False

    try:
        mkt = pd.read_csv(mkt_path)
        chicago_usc_bu = float(mkt["Soybeans"].iloc[-1])
    except Exception:
        return False

    BUSHELS_PER_TON = 36.744
    BASIS = float(os.getenv("URUGUAY_BASIS_USD_TON", "-25"))
    local_usd_ton = (chicago_usc_bu / 100.0) * BUSHELS_PER_TON + BASIS

    alert_file = _PROJECT_ROOT / "data" / "last_wa_price_alert.json"
    last_alert = {}
    if alert_file.exists():
        try:
            last_alert = json.loads(alert_file.read_text())
        except Exception:
            pass

    if local_usd_ton >= price_target and last_alert.get("alert_date") != str(date.today()):
        msg = (
            f"🎯 *AgroCast — Precio objetivo alcanzado*\n\n"
            f"💰 Precio local estimado: *USD {local_usd_ton:.1f}/ton*\n"
            f"🎯 Tu objetivo: USD {price_target:.1f}/ton\n"
            f"📈 Chicago: {chicago_usc_bu:.2f} USc/bu\n\n"
            f"*Considera ejecutar la venta.*\n"
            f"_AgroCast PRO · {datetime.now().strftime('%H:%M')}_"
        )
        ok = send_whatsapp(msg)
        if ok:
            alert_file.parent.mkdir(parents=True, exist_ok=True)
            alert_file.write_text(json.dumps({
                "alert_date":    str(date.today()),
                "price_usd_ton": round(local_usd_ton, 2),
                "target_usd_ton": price_target,
            }, indent=2))
        return ok
    return False


def test_connection() -> bool:
    """Envía mensaje de prueba para verificar configuración."""
    msg = (
        f"✅ *AgroCast PRO — WhatsApp conectado*\n\n"
        f"Bot configurado correctamente.\n"
        f"_{datetime.now().strftime('%Y-%m-%d %H:%M')}_"
    )
    return send_whatsapp(msg)


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv(_PROJECT_ROOT / ".env")
        TWILIO_SID   = os.getenv("TWILIO_ACCOUNT_SID", "")
        TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
        TWILIO_FROM  = os.getenv("TWILIO_FROM_NUMBER", "")
        WHATSAPP_TO  = os.getenv("WHATSAPP_TO_NUMBER", "")
    except ImportError:
        pass

    args = sys.argv[1:]
    if "--test" in args:
        test_connection()
    elif "--check-signals" in args:
        check_and_alert_signal()
    else:
        print("Uso: python -m src.alerts.whatsapp_bot [--test|--check-signals]")
        print()
        print("Variables de entorno necesarias:")
        print("  TWILIO_ACCOUNT_SID   — de console.twilio.com")
        print("  TWILIO_AUTH_TOKEN    — de console.twilio.com")
        print("  TWILIO_FROM_NUMBER   — whatsapp:+14155238886 (sandbox)")
        print("  WHATSAPP_TO_NUMBER   — whatsapp:+59899XXXXXX")
        print()
        print("Estado actual:")
        print(f"  TWILIO_SID:   {'OK' if TWILIO_SID else 'NO configurado'}")
        print(f"  TWILIO_TOKEN: {'OK' if TWILIO_TOKEN else 'NO configurado'}")
        print(f"  FROM:         {TWILIO_FROM or 'NO configurado'}")
        print(f"  TO:           {WHATSAPP_TO or 'NO configurado'}")
