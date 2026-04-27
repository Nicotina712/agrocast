"""
src/alerts/weekly_brief.py
Brief semanal auto-generado en español via Claude API (Anthropic SDK).

Genera un resumen ejecutivo de ~400 palabras cada lunes con:
  - Performance de señales de la semana anterior
  - Top 3 drivers de mercado (qué señales movieron más)
  - Calendario de eventos próximos (WASDE, vencimientos)
  - Recomendación concreta para el productor

Entrega via Telegram + WhatsApp.
Requiere: ANTHROPIC_API_KEY en .env
"""

import json
import os
from datetime import date, datetime, timedelta

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STATE_PATH   = os.path.join(_PROJECT_ROOT, "data", "last_weekly_brief.json")


def _was_sent_this_week() -> bool:
    """Evita enviar más de un brief por semana."""
    if not os.path.exists(_STATE_PATH):
        return False
    try:
        with open(_STATE_PATH) as f:
            state = json.load(f)
        last = datetime.fromisoformat(state.get("sent_at", "2000-01-01"))
        return (datetime.now() - last).days < 6
    except Exception:
        return False


def _mark_sent():
    os.makedirs(os.path.dirname(_STATE_PATH), exist_ok=True)
    with open(_STATE_PATH, "w") as f:
        json.dump({"sent_at": datetime.now().isoformat()}, f)


def _load_context() -> dict:
    """Carga todos los datos disponibles para construir el brief."""
    ctx = {}

    # Señal actual y historial
    try:
        import pandas as pd
        sig_path = os.path.join(_PROJECT_ROOT, "artifacts", "signals.csv")
        if os.path.exists(sig_path):
            df = pd.read_csv(sig_path, parse_dates=["Date"])
            df = df.sort_values("Date")
            recent = df.tail(10)
            ctx["current_signal"] = df["Signal"].iloc[-1] if not df.empty else "HOLD"
            ctx["signal_history"] = recent[["Date", "Signal", "Confidence"]].to_dict("records")
            ctx["current_price"]  = float(df["Price"].iloc[-1]) if "Price" in df.columns else None
    except Exception:
        pass

    # Backtest metrics
    try:
        bt_path = os.path.join(_PROJECT_ROOT, "artifacts", "backtest_summary.json")
        if os.path.exists(bt_path):
            with open(bt_path) as f:
                ctx["backtest"] = json.load(f)
    except Exception:
        pass

    # COT data
    try:
        cot_path = os.path.join(_PROJECT_ROOT, "data", "cot_soybeans.csv")
        if os.path.exists(cot_path):
            import pandas as pd
            cot = pd.read_csv(cot_path, parse_dates=["Date"])
            cot = cot.sort_values("Date")
            last_cot = cot.iloc[-1]
            ctx["cot"] = {
                "date":           str(last_cot.get("Date", "")[:10]),
                "noncomm_net":    float(last_cot.get("cot_noncomm_net", 0)),
                "cot_index":      float(last_cot.get("cot_index", 50)),
                "commercial_net": float(last_cot.get("cot_commercial_net", 0)),
            }
    except Exception:
        pass

    # WASDE próximos
    try:
        from src.data.wasde_dates import get_wasde_dates
        wasde = get_wasde_dates()
        ctx["wasde_upcoming"] = wasde[:2] if wasde else []
    except Exception:
        pass

    # Argentina
    try:
        ar_path = os.path.join(_PROJECT_ROOT, "data", "argentina_supply.json")
        if os.path.exists(ar_path):
            with open(ar_path) as f:
                ctx["argentina"] = json.load(f)
    except Exception:
        pass

    # Brazil exports
    try:
        br_path = os.path.join(_PROJECT_ROOT, "data", "brazil_exports.json")
        if os.path.exists(br_path):
            with open(br_path) as f:
                ctx["brazil"] = json.load(f)
    except Exception:
        pass

    # China demand
    try:
        cn_path = os.path.join(_PROJECT_ROOT, "data", "china_demand.json")
        if os.path.exists(cn_path):
            with open(cn_path) as f:
                ctx["china"] = json.load(f)
    except Exception:
        pass

    # Basis Uruguay
    try:
        basis_path = os.path.join(_PROJECT_ROOT, "data", "basis_uruguay.json")
        if os.path.exists(basis_path):
            with open(basis_path) as f:
                ctx["basis"] = json.load(f)
    except Exception:
        pass

    # Accuracy reciente
    try:
        acc_path = os.path.join(_PROJECT_ROOT, "artifacts", "accuracy.json")
        if os.path.exists(acc_path):
            with open(acc_path) as f:
                ctx["accuracy"] = json.load(f)
    except Exception:
        pass

    return ctx


def _build_prompt(ctx: dict) -> str:
    today = date.today()
    week  = today.isocalendar()[1]

    return f"""Eres el analista jefe de AgroCast PRO, un sistema de inteligencia de mercado para productores de soja en Uruguay.

Es lunes {today.strftime('%d de %B de %Y')} (semana {week} del año).

Datos del mercado disponibles:
{json.dumps(ctx, indent=2, default=str, ensure_ascii=False)}

Redactá un BRIEF SEMANAL en español rioplatense (voseo), con el siguiente formato exacto:

---
🌱 **AgroCast PRO — Brief Semanal {today.strftime('%d/%m/%Y')}**

📊 **Señal actual:** [BUY/SELL/HOLD] con [X]% de confianza
💲 **Precio Chicago:** [precio] USc/bu

**¿Qué pasó la semana pasada?**
[2-3 oraciones sobre los movimientos de precio y qué drivers fueron más relevantes]

**Top 3 drivers esta semana:**
1. [Driver más importante con datos concretos]
2. [Segundo driver con datos]
3. [Tercer driver con datos]

**Señales clave del modelo:**
• COT (especuladores): [interpretación del posicionamiento]
• Argentina: [cepo/spread/retenciones y su impacto]
• Brasil: [pace de exportaciones]
• Basis Uruguay: [spread con Chicago]

**📅 Calendario de la semana:**
• [Eventos importantes: WASDE, reportes USDA, vencimientos]

**💡 Recomendación para el productor:**
[Una recomendación concreta y accionable en 2-3 oraciones. Incluí si conviene vender ahora o esperar, y por qué.]

---

Tono: profesional pero directo, sin jerga técnica innecesaria. Máximo 400 palabras. Basate SOLO en los datos proporcionados."""


def generate_weekly_brief(force: bool = False) -> str | None:
    """
    Genera y envía el brief semanal. Solo corre los lunes o si force=True.

    Retorna el texto del brief o None si no corresponde enviarlo.
    """
    today = date.today()
    is_monday = today.weekday() == 0

    if not force and not is_monday:
        return None

    if not force and _was_sent_this_week():
        print("[Brief] Ya enviado esta semana — omitiendo.")
        return None

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("[Brief] ANTHROPIC_API_KEY no configurada — brief no generado.")
        return None

    print("[Brief] Generando brief semanal con Claude API…")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        ctx    = _load_context()
        prompt = _build_prompt(ctx)

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",   # rápido y económico para texto
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        brief_text = message.content[0].text
        print(f"[Brief] Generado ({len(brief_text)} chars)")

        # Enviar por Telegram
        try:
            from src.alerts.telegram_bot import send_telegram
            send_telegram(brief_text)
            print("[Brief] Enviado por Telegram ✅")
        except Exception as e:
            print(f"[Brief] Telegram error: {e}")

        # Enviar por WhatsApp
        try:
            from src.alerts.whatsapp_bot import send_whatsapp
            send_whatsapp(brief_text)
            print("[Brief] Enviado por WhatsApp ✅")
        except Exception as e:
            print(f"[Brief] WhatsApp error: {e}")

        # Guardar copia local
        brief_path = os.path.join(_PROJECT_ROOT, "data", f"brief_{today.isoformat()}.txt")
        with open(brief_path, "w", encoding="utf-8") as f:
            f.write(brief_text)

        _mark_sent()
        return brief_text

    except Exception as e:
        print(f"[Brief] Error generando brief: {e}")
        return None
