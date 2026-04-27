"""
src/alerts/telegram_bot.py
Bot de Telegram para alertas de AgroCast PRO.

Usa la Bot API de Telegram directamente con requests (sin python-telegram-bot).

Configuración (en .env):
  TELEGRAM_BOT_TOKEN  — token del bot (de @BotFather)
  TELEGRAM_CHAT_ID    — chat ID del canal/grupo/usuario destino

Tipos de alerta:
  - Cambio de señal BUY/SELL/HOLD
  - Resultado del backtest semanal
  - Reporte WASDE detectado (precio move > umbral)
  - Error crítico del pipeline

Uso standalone (cron diario):
  python -m src.alerts.telegram_bot --check-signals
"""

import json
import os
import sys
from datetime import datetime, date
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Configuración ─────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")

_SIGNALS_PATH     = _PROJECT_ROOT / "artifacts" / "signals.csv"
_LAST_SIGNAL_PATH = _PROJECT_ROOT / "data" / "last_telegram_signal.json"
_ARTIFACTS_DIR    = _PROJECT_ROOT / "artifacts"

# Umbral mínimo de confianza para alertar (evita ruido en zona HOLD)
MIN_CONFIDENCE_ALERT = 0.30


def _send_message(text: str, parse_mode: str = "HTML") -> bool:
    """Envía un mensaje al chat configurado. Retorna True si exitoso."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no configurados")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": parse_mode,
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            return True
        print(f"[Telegram] Error {resp.status_code}: {resp.text[:200]}")
        return False
    except Exception as e:
        print(f"[Telegram] Excepción: {e}")
        return False


def _load_last_signal() -> dict:
    if _LAST_SIGNAL_PATH.exists():
        try:
            return json.loads(_LAST_SIGNAL_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_last_signal(data: dict) -> None:
    _LAST_SIGNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LAST_SIGNAL_PATH.write_text(json.dumps(data, indent=2))


# ── Formateo de alertas ───────────────────────────────────────────────

def _signal_emoji(signal: str) -> str:
    return {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(signal, "⚪")


def _format_signal_alert(
    signal: str,
    confidence: float,
    price: float,
    expected_return: float,
    prev_signal: str | None,
    date_str: str,
) -> str:
    emoji  = _signal_emoji(signal)
    change = f" <i>(antes: {prev_signal})</i>" if prev_signal and prev_signal != signal else ""
    conf_pct = int(confidence * 100)

    # expected_return = P(up) - 0.5  →  P(up) = 0.5 + expected_return
    prob_pct = round((0.5 + expected_return) * 100)

    lines = [
        f"{emoji} <b>AgroCast — Señal {signal}</b>{change}",
        f"",
        f"📅 <b>Fecha:</b> {date_str}",
        f"💰 <b>Precio soja:</b> {price:.2f} USc/bu",
        f"📊 <b>P(sube 7d):</b> {prob_pct}%",
        f"🎯 <b>Confianza:</b> {conf_pct}%",
        f"",
        f"<i>AgroCast PRO · {datetime.now().strftime('%H:%M')}</i>",
    ]
    return "\n".join(lines)


def _format_backtest_alert(result: dict) -> str:
    alpha = result.get("alpha", 0)
    emoji = "📈" if alpha > 0 else "📉"
    lines = [
        f"{emoji} <b>AgroCast — Reporte Semanal</b>",
        f"",
        f"📅 Período: {result.get('test_period', {}).get('start','?')} → {result.get('test_period', {}).get('end','?')}",
        f"💹 Retorno modelo: <b>{result.get('total_return', 0):.1f}%</b>",
        f"📊 Buy & Hold:     {result.get('bh_return', 0):.1f}%",
        f"🎯 Alpha:          <b>{alpha:.1f}%</b>",
        f"🏆 Win rate:       {result.get('win_rate', 0):.0f}%",
        f"⚡ Sharpe:         {result.get('sharpe', 0):.2f}",
        f"📉 Max Drawdown:   {result.get('max_drawdown', 0):.1f}%",
        f"🔢 Operaciones:    {result.get('n_trades', 0)}",
        f"",
        f"<i>AgroCast PRO · {date.today()}</i>",
    ]
    return "\n".join(lines)


def _format_wasde_alert(surprise_pct: float, price_change_pct: float) -> str:
    direction = "ALCISTA" if surprise_pct > 0 else "BAJISTA"
    emoji     = "🟢" if surprise_pct > 0 else "🔴"
    lines = [
        f"{emoji} <b>WASDE detectado — Sorpresa {direction}</b>",
        f"",
        f"📋 Reporte USDA publicado hoy",
        f"📊 Reacción del precio: {price_change_pct:+.1f}%",
        f"⚡ Surprise score: {surprise_pct:+.1f}%",
        f"",
        f"<i>AgroCast PRO · {datetime.now().strftime('%H:%M')}</i>",
    ]
    return "\n".join(lines)


def _format_pipeline_error(error_msg: str) -> str:
    return (
        f"🚨 <b>AgroCast — Error en pipeline</b>\n\n"
        f"<code>{error_msg[:500]}</code>\n\n"
        f"<i>{datetime.now().strftime('%Y-%m-%d %H:%M')}</i>"
    )


# ── Funciones públicas ────────────────────────────────────────────────

def check_and_alert_signal() -> bool:
    """
    Lee signals.csv, compara con la última señal guardada,
    y envía alerta si hay cambio de señal con confianza suficiente.

    Retorna True si se envió una alerta.
    """
    import pandas as pd

    if not _SIGNALS_PATH.exists():
        print("[Telegram] signals.csv no encontrado")
        return False

    signals = pd.read_csv(_SIGNALS_PATH)
    if signals.empty:
        return False

    last_row = signals.iloc[-1]
    current_signal     = str(last_row.get("signal", "HOLD"))
    current_confidence = float(last_row.get("confidence", 0))
    current_date       = str(last_row.get("Date", ""))[:10]
    expected_return    = float(last_row.get("expected_return", 0))

    # Precio real más reciente (del CSV de precios si disponible)
    price = 0.0
    price_path = _PROJECT_ROOT / "data" / "raw_market.csv"
    if price_path.exists():
        try:
            mkt = pd.read_csv(price_path)
            price = float(mkt["Soybeans"].iloc[-1])
        except Exception:
            pass

    last = _load_last_signal()
    prev_signal    = last.get("signal")
    prev_date      = last.get("date", "")
    alert_today    = last.get("alert_date") == str(date.today())

    # Alertar si: señal cambió Y confianza suficiente Y no alertamos hoy ya
    signal_changed = (current_signal != prev_signal) or (current_date != prev_date)

    if signal_changed and current_confidence >= MIN_CONFIDENCE_ALERT and not alert_today:
        msg = _format_signal_alert(
            signal=current_signal,
            confidence=current_confidence,
            price=price,
            expected_return=expected_return,
            prev_signal=prev_signal,
            date_str=current_date,
        )
        success = _send_message(msg)
        if success:
            _save_last_signal({
                "signal":     current_signal,
                "confidence": current_confidence,
                "date":       current_date,
                "alert_date": str(date.today()),
            })
            print(f"[Telegram] Alerta enviada: {current_signal} ({current_confidence*100:.0f}%)")
            return True
    else:
        print(f"[Telegram] Sin cambio: {current_signal} (conf={current_confidence*100:.0f}%) "
              f"— prev={prev_signal}, hoy alertado={alert_today}")
    return False


def send_backtest_report() -> bool:
    """Lee el último backtest y envía reporte semanal."""
    from src.model.backtest import run_backtest

    features_path = str(_PROJECT_ROOT / "data" / "features.csv")
    model_dir     = str(_PROJECT_ROOT / "src" / "model" / "artifacts")

    try:
        result = run_backtest(features_path, model_dir)
        msg    = _format_backtest_alert(result)
        return _send_message(msg)
    except Exception as e:
        print(f"[Telegram] Error en backtest: {e}")
        return False


def send_pipeline_error(error_msg: str) -> bool:
    """Alerta de error crítico en el pipeline."""
    return _send_message(_format_pipeline_error(error_msg))


def send_wasde_alert(surprise_pct: float, price_change_pct: float) -> bool:
    """Alerta cuando se detecta un WASDE con movimiento significativo."""
    return _send_message(_format_wasde_alert(surprise_pct, price_change_pct))


def check_and_alert_price_target() -> bool:
    """
    Alerta cuando el precio local estimado (USD/ton Uruguay) supera el objetivo
    configurado en .env como PRODUCER_PRICE_TARGET_USD_TON.

    Envía max 1 alerta por día. Guarda estado en data/last_price_alert.json.
    """
    import pandas as pd

    target_str = os.getenv("PRODUCER_PRICE_TARGET_USD_TON", "").strip()
    if not target_str:
        return False   # no configurado — no hacer nada

    try:
        price_target = float(target_str)
    except ValueError:
        return False

    # Precio actual Chicago (USD/bu) → USD/ton
    price_path = _PROJECT_ROOT / "data" / "raw_market.csv"
    if not price_path.exists():
        return False

    try:
        mkt = pd.read_csv(price_path)
        chicago_usd_bu = float(mkt["Soybeans"].iloc[-1])
    except Exception:
        return False

    BUSHELS_PER_TON = 36.744
    BASIS            = float(os.getenv("URUGUAY_BASIS_USD_TON", "-25"))
    # raw_market.csv precio en USc/bu → dividir por 100 para USD/bu
    local_usd_ton   = (chicago_usd_bu / 100.0) * BUSHELS_PER_TON + BASIS

    alert_file = _PROJECT_ROOT / "data" / "last_price_alert.json"
    last_alert = {}
    if alert_file.exists():
        try:
            last_alert = json.loads(alert_file.read_text())
        except Exception:
            pass

    # Enviar solo si precio supera objetivo Y no alertamos hoy ya
    if local_usd_ton >= price_target and last_alert.get("alert_date") != str(date.today()):
        msg = (
            f"🎯 <b>AgroCast — Precio objetivo alcanzado</b>\n\n"
            f"💰 Precio local estimado: <b>USD {local_usd_ton:.1f}/ton</b>\n"
            f"🎯 Tu objetivo: USD {price_target:.1f}/ton\n"
            f"📈 Chicago: {chicago_usd_bu:.2f} USc/bu\n\n"
            f"<b>Considera ejecutar la venta.</b>\n"
            f"<i>AgroCast PRO · {datetime.now().strftime('%H:%M')}</i>"
        )
        ok = _send_message(msg)
        if ok:
            alert_file.parent.mkdir(parents=True, exist_ok=True)
            alert_file.write_text(json.dumps({
                "alert_date": str(date.today()),
                "price_usd_ton": round(local_usd_ton, 2),
                "target_usd_ton": price_target,
            }, indent=2))
            print(f"[Telegram] Alerta precio: USD {local_usd_ton:.1f}/ton >= objetivo {price_target}")
            return True
    return False


def send_wasde_upcoming_alert(wasde_date, days_ahead: int) -> bool:
    """
    Envía alerta anticipada cuando el próximo WASDE está a 48h o menos.
    wasde_date: date object con la fecha del próximo reporte.
    """
    month_names = {
        1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
        7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"
    }
    month_name = month_names.get(wasde_date.month, str(wasde_date.month))
    msg = (
        f"📅 <b>Próximo Reporte WASDE en {days_ahead} día{'s' if days_ahead != 1 else ''}</b>\n\n"
        f"El reporte USDA WASDE de <b>{month_name} {wasde_date.year}</b> se publica "
        f"el <b>{wasde_date.strftime('%d/%m/%Y')}</b>.\n\n"
        f"Los reports WASDE mueven el precio en promedio <b>±10 USc/bu</b> al día siguiente.\n"
        f"Revisá el dashboard antes del reporte para tener el contexto completo.\n\n"
        f"<i>AgroCast PRO · {datetime.now().strftime('%Y-%m-%d %H:%M')}</i>"
    )
    return _send_message(msg)


def check_and_alert_wasde_upcoming() -> bool:
    """
    Verifica si el próximo WASDE está dentro de 48h y envía alerta si es así.
    Evita duplicados guardando la última alerta enviada.
    Retorna True si se envió una alerta.
    """
    import json
    from datetime import date, timedelta
    alert_path = os.path.join(_PROJECT_ROOT, "data", "wasde_alert_sent.json")

    # Calcular próximo WASDE (2do martes del mes siguiente o del mes actual)
    today = date.today()
    next_wasde = None
    for offset in range(0, 3):  # buscar en los próximos 3 meses
        year  = today.year + (today.month + offset - 1) // 12
        month = (today.month + offset - 1) % 12 + 1
        first = date(year, month, 1)
        days_until_tue = (1 - first.weekday()) % 7
        first_tue  = first + timedelta(days=days_until_tue)
        second_tue = first_tue + timedelta(weeks=1)
        if second_tue >= today:
            next_wasde = second_tue
            break

    if next_wasde is None:
        return False

    days_ahead = (next_wasde - today).days

    if days_ahead > 2:
        return False  # Más de 48h — no alerta aún

    # Verificar si ya mandamos alerta para esta fecha
    try:
        if os.path.exists(alert_path):
            saved = json.loads(open(alert_path).read())
            if saved.get("wasde_date") == next_wasde.isoformat():
                return False  # Ya alertamos
    except Exception:
        pass

    ok = send_wasde_upcoming_alert(next_wasde, days_ahead)
    if ok:
        os.makedirs(os.path.dirname(alert_path), exist_ok=True)
        open(alert_path, "w").write(json.dumps({"wasde_date": next_wasde.isoformat()}))

    return ok


def test_connection() -> bool:
    """Envía un mensaje de prueba para verificar la configuración."""
    msg = (
        f"✅ <b>AgroCast PRO — Conexión verificada</b>\n\n"
        f"Bot Telegram configurado correctamente.\n"
        f"<i>{datetime.now().strftime('%Y-%m-%d %H:%M')}</i>"
    )
    ok = _send_message(msg)
    if ok:
        print("[Telegram] Test OK — mensaje enviado")
    return ok


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Cargar .env
    try:
        from dotenv import load_dotenv
        load_dotenv(_PROJECT_ROOT / ".env")
        # Reload env vars after dotenv
        import importlib, src.alerts.telegram_bot as _self
        TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
        TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")
    except ImportError:
        pass

    args = sys.argv[1:]

    if "--test" in args:
        test_connection()
    elif "--check-signals" in args:
        check_and_alert_signal()
    elif "--backtest" in args:
        send_backtest_report()
    else:
        print("Uso: python -m src.alerts.telegram_bot [--test|--check-signals|--backtest]")
        print("")
        print("Variables de entorno necesarias:")
        print("  TELEGRAM_BOT_TOKEN  — token de @BotFather")
        print("  TELEGRAM_CHAT_ID    — chat ID del destino")
        print("")
        print("Estado actual:")
        print(f"  BOT_TOKEN: {'configurado' if TELEGRAM_BOT_TOKEN else 'NO configurado'}")
        print(f"  CHAT_ID:   {'configurado' if TELEGRAM_CHAT_ID else 'NO configurado'}")
